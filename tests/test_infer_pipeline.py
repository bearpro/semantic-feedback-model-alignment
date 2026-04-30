from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.lib import analyze_pipeline, infer_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_source_matrix_has_stable_line_numbers() -> None:
    source_matrix = infer_pipeline.load_source_matrix(REPO_ROOT)
    assert len(source_matrix) == 54
    assert source_matrix.iloc[0]["source_matrix_line"] == 1
    assert source_matrix.iloc[-1]["source_matrix_line"] == 54


def test_render_template_replaces_named_placeholders() -> None:
    rendered = infer_pipeline.render_template(
        "alpha {{FIRST}} beta {{SECOND}}",
        {"FIRST": "one", "SECOND": "two"},
    )
    assert rendered == "alpha one beta two"


def test_strip_markdown_code_fences_returns_clean_code() -> None:
    fenced = "```csharp\npublic class A {}\n```"
    assert infer_pipeline.strip_markdown_code_fences(fenced) == "public class A {}"


def test_build_config_snapshot_exposes_runtime_values_without_hiding_paths() -> None:
    runtime_config = infer_pipeline.RuntimeConfig(
        provider="openrouter",
        runs=2,
        temperature=0.1,
        max_tokens=12000,
        api_key_env_var="OPENROUTER_API_KEY",
        api_key="secret",
        feedback_analyzer_model="gpt-5.4-mini",
        feedback_analyzer_api_key_env_var="OPENAI_API_KEY",
        feedback_analyzer_api_key="feedback-secret",
        feedback_analyzer_temperature=0.0,
        feedback_analyzer_max_tokens=4000,
        config_path=REPO_ROOT / "config" / "default.yml",
        env_path=REPO_ROOT / ".env",
    )
    source_matrix = infer_pipeline.load_source_matrix(REPO_ROOT)
    snapshot = infer_pipeline.build_config_snapshot(
        repo_root=REPO_ROOT,
        runtime_config=runtime_config,
        source_matrix=source_matrix,
        max_new_requests=5,
    )
    assert snapshot["provider"] == "openrouter"
    assert snapshot["runs"] == 2
    assert snapshot["source_matrix_rows"] == 54
    assert snapshot["max_new_requests"] == 5
    assert snapshot["api_key_present"] is True
    assert snapshot["feedback_analyzer_model"] == "gpt-5.4-mini"
    assert snapshot["feedback_analyzer_api_key_present"] is True


def test_build_run_paths_uses_document_scenario_model_and_run() -> None:
    row = {
        "document_stem": "framework-laptop-13-specs",
        "scenario": "guided",
        "model": "google/gemini-2.5-flash-lite",
    }
    paths = infer_pipeline.build_run_paths(REPO_ROOT, row, 3)
    assert paths.run_dir.relative_to(REPO_ROOT).as_posix() == (
        "artifacts/infer/framework-laptop-13-specs/guided/google-gemini-2-5-flash-lite/run3"
    )
    assert paths.step0_cs_path.name == "step-0.cs"
    assert paths.feedback_md_path.name == "feedback.md"
    assert paths.final_cs_path.name == "final.cs"
    assert paths.metadata_path.name == "record.json"


def test_discover_infer_artifacts_reads_existing_final_cs_files() -> None:
    artifact_df = analyze_pipeline.discover_infer_artifacts(REPO_ROOT)
    assert not artifact_df.empty
    assert "final_cs_path" in artifact_df.columns
    assert artifact_df["final_cs_path"].str.endswith("/final.cs").all()
    assert set(artifact_df["scenario"]) == {"control", "guided"}


def test_render_model_snapshot_svg_contains_document_and_scenario_legends() -> None:
    frame = pd.DataFrame(
        [
            {
                "document_stem": "alpha-doc",
                "scenario": "control",
                "model": "model/a",
                "run": 1,
                "defined_type_count": 3,
                "line_count": 30,
                "byte_count": 300,
                "parse_error_count": 0,
            },
            {
                "document_stem": "beta-doc",
                "scenario": "guided",
                "model": "model/b",
                "run": 1,
                "defined_type_count": 6,
                "line_count": 45,
                "byte_count": 450,
                "parse_error_count": 1,
            },
            {
                "document_stem": "alpha-doc",
                "scenario": "guided",
                "model": "model/a",
                "run": 2,
                "defined_type_count": 4,
                "line_count": 35,
                "byte_count": 350,
                "parse_error_count": 0,
            },
        ]
    )
    svg_markup = analyze_pipeline.render_model_snapshot_svg(frame)
    assert "<svg" in svg_markup
    assert "alpha-doc" in svg_markup
    assert "control" in svg_markup
    assert "guided" in svg_markup
    assert "model/a" in svg_markup


def test_process_feedback_run_persists_step0_feedback_and_final(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path
    document_path = repo_root / "data" / "soruce" / "documents" / "sample.md"
    document_path.parent.mkdir(parents=True, exist_ok=True)
    document_path.write_text("Sample source text", encoding="utf-8")

    runtime_config = infer_pipeline.RuntimeConfig(
        provider="openrouter",
        runs=1,
        temperature=0.1,
        max_tokens=12000,
        api_key_env_var="OPENROUTER_API_KEY",
        api_key="author-secret",
        feedback_analyzer_model="gpt-5.4-mini",
        feedback_analyzer_api_key_env_var="OPENAI_API_KEY",
        feedback_analyzer_api_key="feedback-secret",
        feedback_analyzer_temperature=0.0,
        feedback_analyzer_max_tokens=4000,
        config_path=repo_root / "config" / "default.yml",
        env_path=repo_root / ".env",
    )
    guided_prompt_bundle = infer_pipeline.PromptBundle(
        system_path=repo_root / "guided-system.md",
        user_path=repo_root / "guided-user.md",
        system="Return only C# code.",
        user="Source:\n{{SOURCE_TEXT}}",
    )
    feedback_prompt_bundle = infer_pipeline.FeedbackPromptBundle(
        system_path=repo_root / "feedback-system.md",
        feedback_path=repo_root / "feedback-template.md",
        system="Return only C# code.",
        feedback="Code:\n{{CSharpCode}}\nWarnings:\n{{AnalyzerWarnings}}",
    )
    row = {
        "source_matrix_line": 1,
        "scenario": "feedback",
        "model": "google/gemini-2.5-flash-lite",
        "document": "sample.md",
        "document_stem": "sample",
        "document_path": "data/soruce/documents/sample.md",
    }
    calls: list[dict[str, object]] = []

    def fake_run_inference_request(
        *,
        provider: str,
        api_key: str,
        model: str,
        messages,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, object]:
        calls.append(
            {
                "provider": provider,
                "api_key": api_key,
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if len(calls) == 1:
            return {"output_text": "```csharp\npublic class StepZero {}\n```"}
        if len(calls) == 2:
            return {
                "output_text": (
                    "step-0.cs(1,14): warning SMN001: Ambiguous type name\n"
                    "  symbol: StepZero\n"
                    "  fix: Rename to SampleModel"
                )
            }
        return {"output_text": "```csharp\npublic class SampleModel {}\n```"}

    monkeypatch.setattr(infer_pipeline, "run_inference_request", fake_run_inference_request)

    summary, record = infer_pipeline.process_feedback_run(
        repo_root=repo_root,
        row=row,
        run_number=1,
        runtime_config=runtime_config,
        guided_prompt_bundle=guided_prompt_bundle,
        feedback_prompt_bundle=feedback_prompt_bundle,
        should_execute=True,
    )

    assert summary["status"] == "executed"
    assert record is not None
    paths = infer_pipeline.build_run_paths(repo_root, row, 1)
    assert paths.step0_cs_path.read_text(encoding="utf-8").strip() == "public class StepZero {}"
    assert "warning SMN001" in paths.feedback_md_path.read_text(encoding="utf-8")
    assert paths.final_cs_path.read_text(encoding="utf-8").strip() == "public class SampleModel {}"
    stored_record = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    assert stored_record["feedback_loop"]["analyzer"]["model"] == "gpt-5.4-mini"
    assert len(calls) == 3

    calls.clear()
    summary, record = infer_pipeline.process_feedback_run(
        repo_root=repo_root,
        row=row,
        run_number=1,
        runtime_config=runtime_config,
        guided_prompt_bundle=guided_prompt_bundle,
        feedback_prompt_bundle=feedback_prompt_bundle,
        should_execute=True,
    )
    assert summary["status"] == "cached"
    assert record is not None
    assert len(calls) == 0
