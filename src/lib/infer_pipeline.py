from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

SUPPORTED_SCENARIOS = ("control", "guided", "feedback")
FEEDBACK_ANALYZER_MODEL = "gpt-5.4-mini"
FEEDBACK_ANALYZER_PROVIDER = "openai"
FEEDBACK_ANALYZER_API_KEY_ENV_VAR = "OPENAI_API_KEY"
FEEDBACK_ANALYZER_TEMPERATURE = 0.0
FEEDBACK_ANALYZER_MAX_TOKENS = 4000


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    provider: str
    runs: int
    temperature: float
    max_tokens: int
    api_key_env_var: str
    api_key: str
    feedback_analyzer_model: str
    feedback_analyzer_api_key_env_var: str
    feedback_analyzer_api_key: str | None
    feedback_analyzer_temperature: float
    feedback_analyzer_max_tokens: int
    config_path: Path
    env_path: Path


@dataclass(slots=True, frozen=True)
class PromptBundle:
    system_path: Path
    user_path: Path
    system: str
    user: str


@dataclass(slots=True, frozen=True)
class FeedbackPromptBundle:
    system_path: Path
    feedback_path: Path
    system: str
    feedback: str


@dataclass(slots=True, frozen=True)
class RunPaths:
    run_dir: Path
    step0_cs_path: Path
    feedback_md_path: Path
    final_cs_path: Path
    metadata_path: Path


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing pyproject.toml")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "item"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def render_template(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def parse_optional_int(raw: str | None) -> int | None:
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def strip_markdown_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if not lines:
        return stripped

    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    serialized_rows = frame.to_json(orient="records") or "[]"
    return cast(list[dict[str, Any]], json.loads(serialized_rows))


def load_runtime_config(repo_root: Path) -> RuntimeConfig:
    env_path = repo_root / ".env"
    config_path = repo_root / "config" / "default.yml"
    load_dotenv(env_path)

    config = yaml.safe_load(read_text(config_path)) or {}
    infer_config = config.get("infer", {})
    provider = infer_config.get("provider", "openrouter")
    api_key_env_var = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {api_key_env_var} is required for provider {provider}"
        )
    feedback_analyzer_api_key = os.getenv(FEEDBACK_ANALYZER_API_KEY_ENV_VAR)

    return RuntimeConfig(
        provider=provider,
        runs=int(infer_config.get("runs", 1)),
        temperature=float(infer_config.get("temperature", 0)),
        max_tokens=int(infer_config.get("max_tokens", 4000)),
        api_key_env_var=api_key_env_var,
        api_key=api_key,
        feedback_analyzer_model=FEEDBACK_ANALYZER_MODEL,
        feedback_analyzer_api_key_env_var=FEEDBACK_ANALYZER_API_KEY_ENV_VAR,
        feedback_analyzer_api_key=feedback_analyzer_api_key,
        feedback_analyzer_temperature=FEEDBACK_ANALYZER_TEMPERATURE,
        feedback_analyzer_max_tokens=FEEDBACK_ANALYZER_MAX_TOKENS,
        config_path=config_path,
        env_path=env_path,
    )


def build_config_snapshot(
    *,
    repo_root: Path,
    runtime_config: RuntimeConfig,
    source_matrix: pd.DataFrame,
    max_new_requests: int | None,
) -> dict[str, Any]:
    return {
        "started_at": datetime.now(UTC).isoformat(),
        "repo_root": repo_root.as_posix(),
        "config_path": runtime_config.config_path.relative_to(repo_root).as_posix(),
        "env_path": runtime_config.env_path.relative_to(repo_root).as_posix(),
        "provider": runtime_config.provider,
        "runs": runtime_config.runs,
        "temperature": runtime_config.temperature,
        "max_tokens": runtime_config.max_tokens,
        "api_key_env_var": runtime_config.api_key_env_var,
        "api_key_present": bool(runtime_config.api_key),
        "feedback_analyzer_model": runtime_config.feedback_analyzer_model,
        "feedback_analyzer_api_key_env_var": runtime_config.feedback_analyzer_api_key_env_var,
        "feedback_analyzer_api_key_present": bool(runtime_config.feedback_analyzer_api_key),
        "feedback_analyzer_temperature": runtime_config.feedback_analyzer_temperature,
        "feedback_analyzer_max_tokens": runtime_config.feedback_analyzer_max_tokens,
        "source_matrix_path": "artifacts/sources/source-matrix.json",
        "source_matrix_rows": len(source_matrix),
        "max_new_requests": max_new_requests,
        "supported_scenarios": list(SUPPORTED_SCENARIOS),
    }


def load_prompt_bundle(repo_root: Path, scenario: str) -> PromptBundle:
    prompts_root = repo_root / "data" / "soruce" / "prompts" / scenario
    return PromptBundle(
        system_path=prompts_root / "system.md",
        user_path=prompts_root / "user.md",
        system=read_text(prompts_root / "system.md"),
        user=read_text(prompts_root / "user.md"),
    )


def load_feedback_prompt_bundle(repo_root: Path) -> FeedbackPromptBundle:
    prompts_root = repo_root / "data" / "soruce" / "prompts" / "feedback"
    return FeedbackPromptBundle(
        system_path=prompts_root / "system.md",
        feedback_path=prompts_root / "feedback.md",
        system=read_text(prompts_root / "system.md"),
        feedback=read_text(prompts_root / "feedback.md"),
    )


def load_source_matrix(repo_root: Path) -> pd.DataFrame:
    matrix_path = repo_root / "artifacts" / "sources" / "source-matrix.json"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Source matrix not found: {matrix_path}")

    records = json.loads(read_text(matrix_path))
    source_matrix = pd.DataFrame(records)
    if source_matrix.empty:
        raise RuntimeError("Source matrix is empty")

    source_matrix.insert(0, "source_matrix_line", range(1, len(source_matrix) + 1))
    return source_matrix


def select_scenario_rows(source_matrix: pd.DataFrame, scenario: str) -> list[dict[str, Any]]:
    filtered = cast(pd.DataFrame, source_matrix[source_matrix["scenario"] == scenario].copy())
    return dataframe_records(filtered)


def build_messages(
    repo_root: Path,
    row: dict[str, Any],
    prompt_bundle: PromptBundle,
) -> list[ChatCompletionMessageParam]:
    source_text = read_text(repo_root / row["document_path"])
    replacements = {"SOURCE_TEXT": source_text}
    return [
        cast(
            ChatCompletionMessageParam,
            {"role": "system", "content": prompt_bundle.system},
        ),
        cast(
            ChatCompletionMessageParam,
            {"role": "user", "content": render_template(prompt_bundle.user, replacements)},
        ),
    ]


def build_feedback_analyzer_messages(step0_code: str) -> list[ChatCompletionMessageParam]:
    system_prompt = (
        "You are a static analyzer for semantic naming quality in C# domain models. "
        "Review only the provided C# file. Do not use any external task context."
    )
    user_prompt = (
        "Проанализируй только данный C# файл и выдай предупреждения в формате, близком к LSP.\n\n"
        "Нужны только предупреждения следующих классов, если они реально присутствуют:\n"
        "SMN001 — ambiguous type name\n"
        "SMN002 — ambiguous property name\n"
        "SMN003 — quantity without unit suffix\n"
        "SMN004 — duplicate semantic role under different names\n"
        "SMN005 — bool property without boolean naming pattern\n"
        "SMN006 — collection named as singular / singular named as collection\n"
        "SMN007 — generic placeholder name (Data, Info, Details, etc.)\n"
        "SMN008 — inconsistent synonym usage across model\n\n"
        "Формат ответа:\n"
        "- одна диагностика на строку\n"
        "- формат строки: step-0.cs(line,column): warning SMN###: message\n"
        "- после строки диагностики добавь две строки с отступом двумя пробелами:\n"
        "  symbol: <symbol name>\n"
        "  fix: <specific rename or rewrite suggestion>\n"
        "- не используй markdown fences\n"
        "- если предупреждений нет, верни ровно строку: No warnings.\n\n"
        "C# файл:\n"
        "---\n"
        f"{step0_code}\n"
        "---"
    )
    return [
        cast(ChatCompletionMessageParam, {"role": "system", "content": system_prompt}),
        cast(ChatCompletionMessageParam, {"role": "user", "content": user_prompt}),
    ]


def build_feedback_fix_messages(
    prompt_bundle: FeedbackPromptBundle,
    step0_code: str,
    analyzer_warnings: str,
) -> list[ChatCompletionMessageParam]:
    user_message = render_template(
        prompt_bundle.feedback,
        {
            "CSharpCode": step0_code,
            "AnalyzerWarnings": analyzer_warnings,
        },
    )
    return [
        cast(
            ChatCompletionMessageParam,
            {"role": "system", "content": prompt_bundle.system},
        ),
        cast(
            ChatCompletionMessageParam,
            {"role": "user", "content": user_message},
        ),
    ]


def infer_with_openrouter(
    *,
    api_key: str,
    model: str,
    messages: list[ChatCompletionMessageParam],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "llm-in-lsp-experiment",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    return {
        "id": payload.get("id"),
        "model": payload.get("model", model),
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage"),
        "output_text": choice["message"]["content"],
        "raw_response": payload,
    }


def infer_with_openai(
    *,
    api_key: str,
    model: str,
    messages: list[ChatCompletionMessageParam],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        timeout=300,
    )
    choice = completion.choices[0]
    return {
        "id": completion.id,
        "model": completion.model,
        "finish_reason": choice.finish_reason,
        "usage": completion.usage.model_dump() if completion.usage else None,
        "output_text": choice.message.content or "",
        "raw_response": completion.model_dump(),
    }


def run_inference_request(
    *,
    provider: str,
    api_key: str,
    model: str,
    messages: list[ChatCompletionMessageParam],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    if provider == "openrouter":
        return infer_with_openrouter(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "openai":
        return infer_with_openai(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def build_run_paths(repo_root: Path, row: dict[str, Any], run_number: int) -> RunPaths:
    model_slug = slugify(row["model"])
    run_dir = (
        repo_root
        / "artifacts"
        / "infer"
        / row["document_stem"]
        / row["scenario"]
        / model_slug
        / f"run{run_number}"
    )
    return RunPaths(
        run_dir=run_dir,
        step0_cs_path=run_dir / "step-0.cs",
        feedback_md_path=run_dir / "feedback.md",
        final_cs_path=run_dir / "final.cs",
        metadata_path=run_dir / "record.json",
    )


def load_existing_record(paths: RunPaths) -> dict[str, Any] | None:
    if not paths.metadata_path.exists():
        return None
    return cast(dict[str, Any], json.loads(read_text(paths.metadata_path)))


def build_summary_row(
    *,
    repo_root: Path,
    row: dict[str, Any],
    run_number: int,
    paths: RunPaths,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    summary = {
        "source_matrix_line": int(row["source_matrix_line"]),
        "scenario": row["scenario"],
        "model": row["model"],
        "document": row["document"],
        "run": run_number,
        "status": status,
        "run_dir": paths.run_dir.relative_to(repo_root).as_posix(),
        "step0_cs_path": paths.step0_cs_path.relative_to(repo_root).as_posix(),
        "feedback_md_path": paths.feedback_md_path.relative_to(repo_root).as_posix(),
        "final_cs_path": paths.final_cs_path.relative_to(repo_root).as_posix(),
        "metadata_path": paths.metadata_path.relative_to(repo_root).as_posix(),
    }
    if error is not None:
        summary["error"] = error
    return summary


def build_completed_record(
    *,
    repo_root: Path,
    row: dict[str, Any],
    run_number: int,
    runtime_config: RuntimeConfig,
    prompt_bundle: PromptBundle,
    messages: list[ChatCompletionMessageParam],
    response_payload: dict[str, Any],
    paths: RunPaths,
) -> dict[str, Any]:
    return {
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "provider": runtime_config.provider,
        "source_matrix_line": int(row["source_matrix_line"]),
        "scenario": row["scenario"],
        "model": row["model"],
        "run": run_number,
        "runs_config": runtime_config.runs,
        "document": row["document"],
        "document_stem": row["document_stem"],
        "document_path": row["document_path"],
        "run_dir": paths.run_dir.relative_to(repo_root).as_posix(),
        "step0_cs_path": paths.step0_cs_path.relative_to(repo_root).as_posix(),
        "feedback_md_path": paths.feedback_md_path.relative_to(repo_root).as_posix(),
        "final_cs_path": paths.final_cs_path.relative_to(repo_root).as_posix(),
        "metadata_path": paths.metadata_path.relative_to(repo_root).as_posix(),
        "request": {
            "temperature": runtime_config.temperature,
            "max_tokens": runtime_config.max_tokens,
            "system_prompt_path": prompt_bundle.system_path.relative_to(repo_root).as_posix(),
            "user_prompt_path": prompt_bundle.user_path.relative_to(repo_root).as_posix(),
            "messages": messages,
        },
        "response": response_payload,
    }


def persist_completed_run(
    *,
    repo_root: Path,
    row: dict[str, Any],
    run_number: int,
    runtime_config: RuntimeConfig,
    prompt_bundle: PromptBundle,
    messages: list[ChatCompletionMessageParam],
    response_payload: dict[str, Any],
    paths: RunPaths,
) -> dict[str, Any]:
    final_code = strip_markdown_code_fences(response_payload["output_text"])
    write_text(paths.final_cs_path, final_code + "\n")

    record = build_completed_record(
        repo_root=repo_root,
        row=row,
        run_number=run_number,
        runtime_config=runtime_config,
        prompt_bundle=prompt_bundle,
        messages=messages,
        response_payload=response_payload,
        paths=paths,
    )
    write_json(paths.metadata_path, record)
    return record


def build_feedback_completed_record(
    *,
    repo_root: Path,
    row: dict[str, Any],
    run_number: int,
    runtime_config: RuntimeConfig,
    guided_prompt_bundle: PromptBundle,
    feedback_prompt_bundle: FeedbackPromptBundle,
    step0_messages: list[ChatCompletionMessageParam],
    step0_response_payload: dict[str, Any] | None,
    feedback_messages: list[ChatCompletionMessageParam],
    feedback_response_payload: dict[str, Any] | None,
    revision_messages: list[ChatCompletionMessageParam],
    revision_response_payload: dict[str, Any],
    paths: RunPaths,
    reused_step0: bool,
    reused_feedback: bool,
) -> dict[str, Any]:
    return {
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "provider": runtime_config.provider,
        "source_matrix_line": int(row["source_matrix_line"]),
        "scenario": row["scenario"],
        "model": row["model"],
        "run": run_number,
        "runs_config": runtime_config.runs,
        "document": row["document"],
        "document_stem": row["document_stem"],
        "document_path": row["document_path"],
        "run_dir": paths.run_dir.relative_to(repo_root).as_posix(),
        "step0_cs_path": paths.step0_cs_path.relative_to(repo_root).as_posix(),
        "feedback_md_path": paths.feedback_md_path.relative_to(repo_root).as_posix(),
        "final_cs_path": paths.final_cs_path.relative_to(repo_root).as_posix(),
        "metadata_path": paths.metadata_path.relative_to(repo_root).as_posix(),
        "feedback_loop": {
            "step0": {
                "reused_from_disk": reused_step0,
                "temperature": runtime_config.temperature,
                "max_tokens": runtime_config.max_tokens,
                "system_prompt_path": (
                    guided_prompt_bundle.system_path.relative_to(repo_root).as_posix()
                ),
                "user_prompt_path": (
                    guided_prompt_bundle.user_path.relative_to(repo_root).as_posix()
                ),
                "messages": step0_messages,
                "response": step0_response_payload,
            },
            "analyzer": {
                "provider": FEEDBACK_ANALYZER_PROVIDER,
                "model": runtime_config.feedback_analyzer_model,
                "reused_from_disk": reused_feedback,
                "temperature": runtime_config.feedback_analyzer_temperature,
                "max_tokens": runtime_config.feedback_analyzer_max_tokens,
                "messages": feedback_messages,
                "response": feedback_response_payload,
            },
            "revision": {
                "provider": runtime_config.provider,
                "model": row["model"],
                "temperature": runtime_config.temperature,
                "max_tokens": runtime_config.max_tokens,
                "system_prompt_path": (
                    feedback_prompt_bundle.system_path.relative_to(repo_root).as_posix()
                ),
                "feedback_prompt_path": (
                    feedback_prompt_bundle.feedback_path.relative_to(repo_root).as_posix()
                ),
                "messages": revision_messages,
                "response": revision_response_payload,
            },
        },
    }


def persist_feedback_completed_run(
    *,
    repo_root: Path,
    row: dict[str, Any],
    run_number: int,
    runtime_config: RuntimeConfig,
    guided_prompt_bundle: PromptBundle,
    feedback_prompt_bundle: FeedbackPromptBundle,
    step0_messages: list[ChatCompletionMessageParam],
    step0_response_payload: dict[str, Any] | None,
    feedback_messages: list[ChatCompletionMessageParam],
    feedback_response_payload: dict[str, Any] | None,
    revision_messages: list[ChatCompletionMessageParam],
    revision_response_payload: dict[str, Any],
    paths: RunPaths,
    reused_step0: bool,
    reused_feedback: bool,
) -> dict[str, Any]:
    final_code = strip_markdown_code_fences(revision_response_payload["output_text"])
    write_text(paths.final_cs_path, final_code + "\n")

    record = build_feedback_completed_record(
        repo_root=repo_root,
        row=row,
        run_number=run_number,
        runtime_config=runtime_config,
        guided_prompt_bundle=guided_prompt_bundle,
        feedback_prompt_bundle=feedback_prompt_bundle,
        step0_messages=step0_messages,
        step0_response_payload=step0_response_payload,
        feedback_messages=feedback_messages,
        feedback_response_payload=feedback_response_payload,
        revision_messages=revision_messages,
        revision_response_payload=revision_response_payload,
        paths=paths,
        reused_step0=reused_step0,
        reused_feedback=reused_feedback,
    )
    write_json(paths.metadata_path, record)
    return record


def process_run(
    *,
    repo_root: Path,
    row: dict[str, Any],
    run_number: int,
    runtime_config: RuntimeConfig,
    prompt_bundle: PromptBundle,
    should_execute: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    paths = build_run_paths(repo_root, row, run_number)

    if paths.final_cs_path.exists() and paths.metadata_path.exists():
        cached_record = load_existing_record(paths)
        return (
            build_summary_row(
                repo_root=repo_root,
                row=row,
                run_number=run_number,
                paths=paths,
                status="cached",
            ),
            cached_record,
        )

    if not should_execute:
        return (
            build_summary_row(
                repo_root=repo_root,
                row=row,
                run_number=run_number,
                paths=paths,
                status="deferred",
            ),
            None,
        )

    messages = build_messages(repo_root, row, prompt_bundle)

    try:
        response_payload = run_inference_request(
            provider=runtime_config.provider,
            api_key=runtime_config.api_key,
            model=row["model"],
            messages=messages,
            temperature=runtime_config.temperature,
            max_tokens=runtime_config.max_tokens,
        )
    except Exception as exc:
        return (
            build_summary_row(
                repo_root=repo_root,
                row=row,
                run_number=run_number,
                paths=paths,
                status="failed",
                error=str(exc),
            ),
            None,
        )

    record = persist_completed_run(
        repo_root=repo_root,
        row=row,
        run_number=run_number,
        runtime_config=runtime_config,
        prompt_bundle=prompt_bundle,
        messages=messages,
        response_payload=response_payload,
        paths=paths,
    )
    return (
        build_summary_row(
            repo_root=repo_root,
            row=row,
            run_number=run_number,
            paths=paths,
            status="executed",
        ),
        record,
    )


def process_feedback_run(
    *,
    repo_root: Path,
    row: dict[str, Any],
    run_number: int,
    runtime_config: RuntimeConfig,
    guided_prompt_bundle: PromptBundle,
    feedback_prompt_bundle: FeedbackPromptBundle,
    should_execute: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    paths = build_run_paths(repo_root, row, run_number)

    if paths.final_cs_path.exists() and paths.metadata_path.exists():
        cached_record = load_existing_record(paths)
        return (
            build_summary_row(
                repo_root=repo_root,
                row=row,
                run_number=run_number,
                paths=paths,
                status="cached",
            ),
            cached_record,
        )

    if not should_execute:
        return (
            build_summary_row(
                repo_root=repo_root,
                row=row,
                run_number=run_number,
                paths=paths,
                status="deferred",
            ),
            None,
        )

    step0_messages = build_messages(repo_root, row, guided_prompt_bundle)
    reused_step0 = paths.step0_cs_path.exists()
    step0_response_payload: dict[str, Any] | None = None
    if reused_step0:
        step0_code = read_text(paths.step0_cs_path)
    else:
        try:
            step0_response_payload = run_inference_request(
                provider=runtime_config.provider,
                api_key=runtime_config.api_key,
                model=row["model"],
                messages=step0_messages,
                temperature=runtime_config.temperature,
                max_tokens=runtime_config.max_tokens,
            )
        except Exception as exc:
            return (
                build_summary_row(
                    repo_root=repo_root,
                    row=row,
                    run_number=run_number,
                    paths=paths,
                    status="failed",
                    error=f"step-0 generation failed: {exc}",
                ),
                None,
            )
        step0_code = strip_markdown_code_fences(step0_response_payload["output_text"])
        write_text(paths.step0_cs_path, step0_code + "\n")

    feedback_messages = build_feedback_analyzer_messages(step0_code)
    reused_feedback = paths.feedback_md_path.exists()
    feedback_response_payload: dict[str, Any] | None = None
    if reused_feedback:
        analyzer_warnings = read_text(paths.feedback_md_path)
    else:
        if not runtime_config.feedback_analyzer_api_key:
            return (
                build_summary_row(
                    repo_root=repo_root,
                    row=row,
                    run_number=run_number,
                    paths=paths,
                    status="failed",
                    error=(
                        f"feedback analyzer requires "
                        f"{runtime_config.feedback_analyzer_api_key_env_var}"
                    ),
                ),
                None,
            )
        try:
            feedback_response_payload = run_inference_request(
                provider=FEEDBACK_ANALYZER_PROVIDER,
                api_key=runtime_config.feedback_analyzer_api_key,
                model=runtime_config.feedback_analyzer_model,
                messages=feedback_messages,
                temperature=runtime_config.feedback_analyzer_temperature,
                max_tokens=runtime_config.feedback_analyzer_max_tokens,
            )
        except Exception as exc:
            return (
                build_summary_row(
                    repo_root=repo_root,
                    row=row,
                    run_number=run_number,
                    paths=paths,
                    status="failed",
                    error=f"feedback analyzer failed: {exc}",
                ),
                None,
            )
        analyzer_warnings = feedback_response_payload["output_text"].strip()
        write_text(paths.feedback_md_path, analyzer_warnings + "\n")

    revision_messages = build_feedback_fix_messages(
        feedback_prompt_bundle,
        step0_code,
        analyzer_warnings,
    )
    try:
        revision_response_payload = run_inference_request(
            provider=runtime_config.provider,
            api_key=runtime_config.api_key,
            model=row["model"],
            messages=revision_messages,
            temperature=runtime_config.temperature,
            max_tokens=runtime_config.max_tokens,
        )
    except Exception as exc:
        return (
            build_summary_row(
                repo_root=repo_root,
                row=row,
                run_number=run_number,
                paths=paths,
                status="failed",
                error=f"revision failed: {exc}",
            ),
            None,
        )

    record = persist_feedback_completed_run(
        repo_root=repo_root,
        row=row,
        run_number=run_number,
        runtime_config=runtime_config,
        guided_prompt_bundle=guided_prompt_bundle,
        feedback_prompt_bundle=feedback_prompt_bundle,
        step0_messages=step0_messages,
        step0_response_payload=step0_response_payload,
        feedback_messages=feedback_messages,
        feedback_response_payload=feedback_response_payload,
        revision_messages=revision_messages,
        revision_response_payload=revision_response_payload,
        paths=paths,
        reused_step0=reused_step0,
        reused_feedback=reused_feedback,
    )
    return (
        build_summary_row(
            repo_root=repo_root,
            row=row,
            run_number=run_number,
            paths=paths,
            status="executed",
        ),
        record,
    )


def rebuild_index(repo_root: Path) -> pd.DataFrame:
    infer_root = repo_root / "artifacts" / "infer"
    record_paths = sorted(infer_root.rglob("record.json"))
    records = [json.loads(read_text(path)) for path in record_paths]
    index_path = infer_root / "infer-results.json"
    write_json(index_path, records)
    return pd.DataFrame(records)
