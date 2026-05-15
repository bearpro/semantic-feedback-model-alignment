from __future__ import annotations

import argparse
import json
import os
import shutil
from itertools import product
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from src.lib import align_pipeline, infer_pipeline


def load_smoke_config(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(infer_pipeline.read_text(path)) or {})


def build_smoke_source_matrix(
    repo_root: Path,
    *,
    documents: list[str],
    models: list[str],
    scenarios: list[str],
) -> pd.DataFrame:
    records = []
    for scenario, model, document_path_str in product(scenarios, models, documents):
        document_path = Path(document_path_str)
        records.append(
            {
                "scenario": scenario,
                "model": model,
                "document": document_path.name,
                "document_stem": document_path.stem,
                "document_path": document_path.as_posix(),
            }
        )

    frame = pd.DataFrame(records).sort_values(
        ["scenario", "model", "document"],
        ignore_index=True,
    )
    missing_documents = [
        document
        for document in documents
        if not (repo_root / document).exists()
    ]
    if missing_documents:
        raise FileNotFoundError(f"Missing smoke document(s): {missing_documents}")
    return frame


def write_source_matrix(repo_root: Path, source_matrix: pd.DataFrame) -> Path:
    matrix_path = infer_pipeline.artifacts_root(repo_root) / "sources" / "source-matrix.json"
    infer_pipeline.write_json(
        matrix_path,
        infer_pipeline.dataframe_records(source_matrix),
    )
    return matrix_path


def fail_if_any_infer_failed(summaries: list[dict[str, Any]]) -> None:
    failures = [summary for summary in summaries if summary["status"] == "failed"]
    if failures:
        payload = [
            {
                "scenario": failure["scenario"],
                "model": failure["model"],
                "document": failure["document"],
                "run": failure["run"],
                "error": failure.get("error"),
            }
            for failure in failures
        ]
        raise RuntimeError(json.dumps(payload, indent=2, ensure_ascii=False))


def run_smoke(config_path: Path) -> dict[str, Any]:
    repo_root = infer_pipeline.find_repo_root(Path.cwd())
    config = load_smoke_config(config_path)
    smoke_config = cast(dict[str, Any], config.get("smoke", {}))
    artifact_root = Path(cast(str, smoke_config.get("artifact_root", "artifacts/smoke-e2e")))
    if not artifact_root.is_absolute():
        artifact_root = repo_root / artifact_root

    clean_override = os.getenv("SMOKE_E2E_CLEAN")
    clean = (
        clean_override.lower() in {"1", "true", "yes"}
        if clean_override is not None
        else bool(smoke_config.get("clean", True))
    )
    if clean and artifact_root.exists():
        shutil.rmtree(artifact_root)

    os.environ[infer_pipeline.ARTIFACTS_ROOT_ENV_VAR] = artifact_root.as_posix()
    runtime_config = infer_pipeline.load_runtime_config(repo_root, config_path=config_path)

    documents = cast(list[str], smoke_config["documents"])
    models = cast(list[str], smoke_config["models"])
    scenarios = cast(
        list[str],
        smoke_config.get("scenarios", list(infer_pipeline.SUPPORTED_SCENARIOS)),
    )
    source_matrix_seed = build_smoke_source_matrix(
        repo_root,
        documents=documents,
        models=models,
        scenarios=scenarios,
    )
    matrix_path = write_source_matrix(repo_root, source_matrix_seed)
    source_matrix = infer_pipeline.load_source_matrix(repo_root)

    prompt_bundles = {
        scenario: infer_pipeline.load_prompt_bundle(repo_root, scenario)
        for scenario in scenarios
    }
    guided_prompt_bundle = prompt_bundles["guided"]
    feedback_prompt_bundle = infer_pipeline.load_feedback_prompt_bundle(repo_root)

    summaries: list[dict[str, Any]] = []
    for scenario in scenarios:
        rows = infer_pipeline.select_scenario_rows(source_matrix, scenario)
        for row in rows:
            for run_number in range(1, runtime_config.runs + 1):
                if scenario == "feedback":
                    summary, _record = infer_pipeline.process_feedback_run(
                        repo_root=repo_root,
                        row=row,
                        run_number=run_number,
                        runtime_config=runtime_config,
                        guided_prompt_bundle=guided_prompt_bundle,
                        feedback_prompt_bundle=feedback_prompt_bundle,
                        should_execute=True,
                    )
                else:
                    summary, _record = infer_pipeline.process_run(
                        repo_root=repo_root,
                        row=row,
                        run_number=run_number,
                        runtime_config=runtime_config,
                        prompt_bundle=prompt_bundles[scenario],
                        should_execute=True,
                    )
                summaries.append(summary)
                print(
                    f"[infer] {summary['status']} "
                    f"{scenario} {summary['model']} run{run_number}: "
                    f"{summary['final_cs_path']}",
                    flush=True,
                )

    fail_if_any_infer_failed(summaries)
    infer_index = infer_pipeline.rebuild_index(repo_root)

    project_summary, model_index, elements = align_pipeline.prepare_alignment_artifacts(repo_root)
    pairs = align_pipeline.build_positive_pairs(
        model_index,
        same_condition_only=bool(
            cast(dict[str, Any], smoke_config.get("align", {})).get("same_condition_only", True)
        ),
        same_run_index_only=bool(
            cast(dict[str, Any], smoke_config.get("align", {})).get("same_run_index_only", True)
        ),
    )
    align_pipeline.write_pairs_csv(repo_root, pairs)

    align_config = cast(dict[str, Any], smoke_config.get("align", {}))
    tasks = align_pipeline.build_alignment_tasks(
        repo_root,
        pairs,
        projection_specs=align_pipeline.semantic_projection_specs(),
        backend_methods=align_pipeline.semantic_backend_methods(),
    )
    align_pipeline.append_alignment_candidates(repo_root, [], append=False)

    failures: list[dict[str, Any]] = []
    written_rows = 0
    max_workers = int(align_config.get("max_workers", 4))
    timeout_seconds = int(align_config.get("timeout_seconds", 240))
    for result in align_pipeline.iter_alignment_task_results_parallel(
        repo_root,
        tasks,
        timeout_seconds=timeout_seconds,
        max_workers=max_workers,
    ):
        task_rows = cast(list[dict[str, Any]], result["rows"])
        task_failures = cast(list[dict[str, Any]], result["failures"])
        if task_rows:
            align_pipeline.append_alignment_candidates(repo_root, task_rows, append=True)
            written_rows += len(task_rows)
        failures.extend(task_failures)
        print(
            f"[align] task {result['task_index']}/{result['total_tasks']} "
            f"{result['backend']}:{result['method']} rows={len(task_rows)}",
            flush=True,
        )

    failures_path = artifact_root / "align" / "failures.json"
    infer_pipeline.write_json(failures_path, failures)
    candidates = align_pipeline.load_alignment_candidates(repo_root)
    pair_scores, model_scores, scenario_scores, scenario_matrix_scores = (
        align_pipeline.score_alignment_candidates(model_index, elements, candidates)
    )
    score_summary = align_pipeline.write_score_outputs(
        repo_root,
        pair_scores,
        model_scores,
        scenario_scores,
        scenario_matrix_scores,
    )

    summary = {
        "artifact_root": infer_pipeline.relative_to_repo(repo_root, artifact_root),
        "config_path": infer_pipeline.relative_to_repo(repo_root, config_path),
        "source_matrix_path": infer_pipeline.relative_to_repo(repo_root, matrix_path),
        "source_matrix_rows": int(len(source_matrix)),
        "infer_records": int(len(infer_index)),
        "project": project_summary,
        "pair_count": int(len(pairs)),
        "task_count": int(len(tasks)),
        "candidate_rows": int(len(candidates)),
        "written_candidate_rows": written_rows,
        "failure_count": len(failures),
        "failures_json": infer_pipeline.relative_to_repo(repo_root, failures_path),
        "score": score_summary,
    }
    summary_path = artifact_root / "summary.json"
    infer_pipeline.write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated one-document e2e smoke test.")
    parser.add_argument(
        "--config",
        default="config/smoke.yml",
        type=Path,
        help="Smoke config path relative to the repository root.",
    )
    args = parser.parse_args()
    run_smoke(args.config)


if __name__ == "__main__":
    main()
