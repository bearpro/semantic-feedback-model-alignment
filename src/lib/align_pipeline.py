from __future__ import annotations

import concurrent.futures
import itertools
import json
import math
import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast

import pandas as pd

from src.lib import analyze_pipeline, infer_pipeline


@dataclass(slots=True, frozen=True)
class AlignPreparePaths:
    root_dir: Path
    extracted_dir: Path
    projections_dir: Path
    model_index_csv: Path
    elements_csv: Path
    column_maps_dir: Path


@dataclass(slots=True, frozen=True)
class AlignRunPaths:
    root_dir: Path
    pairs_csv: Path
    candidates_csv: Path


@dataclass(slots=True, frozen=True)
class ScorePaths:
    root_dir: Path
    pair_scores_csv: Path
    model_scores_csv: Path
    scenario_scores_csv: Path
    scenario_matrix_scores_csv: Path


class AlignmentTimeoutError(RuntimeError):
    pass


ALIGNMENT_CANDIDATE_COLUMNS = [
    "pair_kind",
    "pair_id",
    "source_document_id",
    "source_model_id",
    "target_model_id",
    "source_condition",
    "target_condition",
    "source_producer_id",
    "target_producer_id",
    "source_run",
    "target_run",
    "projection_layer",
    "projection_mode",
    "backend",
    "method",
    "source_column",
    "target_column",
    "source_element_id",
    "target_element_id",
    "score",
    "rank",
]


def available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    detected = os.cpu_count()
    return max(1, detected or 1)


def prepare_paths(repo_root: Path) -> AlignPreparePaths:
    root_dir = repo_root / "artifacts" / "project"
    extracted_dir = root_dir / "extracted"
    projections_dir = root_dir / "projections"
    return AlignPreparePaths(
        root_dir=root_dir,
        extracted_dir=extracted_dir,
        projections_dir=projections_dir,
        model_index_csv=extracted_dir / "model_index.csv",
        elements_csv=extracted_dir / "elements.csv",
        column_maps_dir=extracted_dir / "column_maps",
    )


def project_paths(repo_root: Path) -> AlignPreparePaths:
    return prepare_paths(repo_root)


def align_run_paths(repo_root: Path) -> AlignRunPaths:
    root_dir = repo_root / "artifacts" / "align"
    return AlignRunPaths(
        root_dir=root_dir,
        pairs_csv=root_dir / "pairs.csv",
        candidates_csv=root_dir / "alignment-candidates.csv",
    )


def score_paths(repo_root: Path) -> ScorePaths:
    root_dir = repo_root / "artifacts" / "score"
    return ScorePaths(
        root_dir=root_dir,
        pair_scores_csv=root_dir / "pair-scores.csv",
        model_scores_csv=root_dir / "model-scores.csv",
        scenario_scores_csv=root_dir / "scenario-scores.csv",
        scenario_matrix_scores_csv=root_dir / "scenario-matrix-scores.csv",
    )


def extractor_project_path(repo_root: Path) -> Path:
    return repo_root / "src" / "lib" / "csharp_model_extractor" / "CSharpModelExtractor.csproj"


def extractor_dll_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "src"
        / "lib"
        / "csharp_model_extractor"
        / "bin"
        / "Debug"
        / "net8.0"
        / "CSharpModelExtractor.dll"
    )


def build_model_id(document_stem: str, scenario: str, model: str, run: int) -> str:
    model_slug = infer_pipeline.slugify(model)
    return f"{document_stem}__{scenario}__{model_slug}__run{run}"


def build_model_index(repo_root: Path, artifact_df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = (
        artifact_df.copy()
        if artifact_df is not None
        else analyze_pipeline.discover_infer_artifacts(repo_root)
    )
    if frame.empty:
        return pd.DataFrame()

    frame["model_id"] = [
        build_model_id(document_stem, scenario, model, int(run))
        for document_stem, scenario, model, run in zip(
            cast(list[str], frame["document_stem"].tolist()),
            cast(list[str], frame["scenario"].tolist()),
            cast(list[str], frame["model"].tolist()),
            cast(list[Any], frame["run"].tolist()),
            strict=True,
        )
    ]
    frame["source_document_id"] = frame["document_stem"]
    frame["condition"] = frame["scenario"]
    frame["producer_id"] = frame["model"]
    frame["path"] = frame["final_cs_path"]

    selected = cast(
        pd.DataFrame,
        frame[
            [
                "model_id",
                "source_document_id",
                "condition",
                "producer_id",
                "document",
                "document_stem",
                "scenario",
                "model",
                "run",
                "path",
                "final_cs_path",
                "metadata_path",
                "source_matrix_line",
            ]
        ],
    )
    return cast(
        pd.DataFrame,
        selected.sort_values(
            ["source_document_id", "condition", "producer_id", "run"],
            ignore_index=True,
        ),
    )


def run_csharp_model_extractor(
    repo_root: Path,
    final_cs_paths: list[Path],
    *,
    dotnet_command: str = "dotnet",
) -> list[dict[str, Any]]:
    if shutil.which(dotnet_command) is None:
        raise RuntimeError(
            f"'{dotnet_command}' is not available. Re-enter the project via `nix develop` "
            "so the C# alignment extractor can run."
        )

    if not final_cs_paths:
        return []

    dll_path = extractor_dll_path(repo_root)
    if not dll_path.exists():
        subprocess.run(
            [dotnet_command, "build", str(extractor_project_path(repo_root))],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )

    command = [
        dotnet_command,
        str(dll_path),
        *[str(path) for path in final_cs_paths],
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = completed.stdout.strip()
    if not payload:
        raise RuntimeError("C# model extractor returned an empty response")
    return cast(list[dict[str, Any]], json.loads(payload))


def extract_elements(
    repo_root: Path,
    model_index: pd.DataFrame,
    *,
    dotnet_command: str = "dotnet",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model_index.empty:
        return pd.DataFrame(), model_index.copy()

    final_paths = [
        repo_root / path for path in cast(list[str], model_index["final_cs_path"].tolist())
    ]
    extraction_by_path: dict[str, dict[str, Any]] = {}
    for final_path in final_paths:
        extraction_rows = run_csharp_model_extractor(
            repo_root,
            [final_path],
            dotnet_command=dotnet_command,
        )
        extraction = cast(dict[str, Any], extraction_rows[0])
        extraction_path = Path(cast(str, extraction["path"])).resolve().as_posix()
        extraction_by_path[extraction_path] = extraction

    model_index_rows: list[dict[str, Any]] = []
    element_rows: list[dict[str, Any]] = []

    for model_row in model_index.to_dict(orient="records"):
        final_path = (repo_root / cast(str, model_row["final_cs_path"])).resolve().as_posix()
        extraction = extraction_by_path[final_path]
        model_id = cast(str, model_row["model_id"])

        model_index_rows.append(
            {
                **model_row,
                "parse_error_count": int(extraction["parseErrorCount"]),
                "parse_errors": json.dumps(extraction["parseErrors"], ensure_ascii=False),
                "analysis_error": extraction.get("analysisError"),
            }
        )

        extracted_elements = cast(list[dict[str, Any]], extraction["elements"])
        for ordinal, element in enumerate(extracted_elements, start=1):
            element_rows.append(
                {
                    "model_id": model_id,
                    "source_document_id": model_row["source_document_id"],
                    "condition": model_row["condition"],
                    "producer_id": model_row["producer_id"],
                    "element_id": f"{model_id}:e{ordinal:04d}",
                    "element_kind": element["elementKind"],
                    "symbol_path": element["symbolPath"],
                    "parent_symbol_path": element.get("parentSymbolPath"),
                    "name": element["name"],
                    "csharp_type": element.get("csharpType"),
                    "normalized_type": element.get("normalizedType"),
                    "is_nullable": bool(element.get("isNullable", False)),
                    "is_collection": bool(element.get("isCollection", False)),
                    "collection_item_type": element.get("collectionItemType"),
                    "is_user_defined_type": bool(element.get("isUserDefinedType", False)),
                    "relation_target_type": element.get("relationTargetType"),
                    "comment_text": element.get("commentText"),
                    "source_file": model_row["final_cs_path"],
                }
            )

    elements = pd.DataFrame(element_rows).sort_values(
        ["model_id", "element_kind", "symbol_path"],
        ignore_index=True,
    )
    model_index_with_parse = pd.DataFrame(model_index_rows).sort_values(
        ["source_document_id", "condition", "producer_id", "run"],
        ignore_index=True,
    )
    return elements, model_index_with_parse


def split_identifier(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ").replace(".", " ")
    return " ".join(text.lower().split())


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _harmonic_mean(left: float, right: float) -> float:
    if left <= 0.0 or right <= 0.0:
        return 0.0
    return float((2.0 * left * right) / (left + right))


def _first_int(frame: pd.DataFrame, column: str) -> int:
    values = cast(list[Any], frame[column].tolist())
    return int(values[0]) if values else 0


def _finite_float_values(series: pd.Series) -> list[float]:
    values: list[float] = []
    for raw_value in cast(list[Any], series.tolist()):
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value):
            values.append(numeric_value)
    return values


def _run_with_timeout(timeout_seconds: int | None, operation: Any) -> Any:
    if timeout_seconds is None or timeout_seconds <= 0:
        return operation()

    def _handle_timeout(signum: int, frame: Any) -> None:
        raise AlignmentTimeoutError(f"Alignment method timed out after {timeout_seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        return operation()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _write_column_map(path: Path, column_map: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(column_map, indent=2, ensure_ascii=False), encoding="utf-8")


def _dedupe_column_name(column_name: str, element_id: str, column_map: dict[str, str]) -> str:
    if column_name not in column_map:
        return column_name
    suffix = element_id.split(":")[-1]
    return f"{column_name}__dup_{suffix}"


def build_property_projection_for_model(
    elements: pd.DataFrame,
    model_id: str,
    out_csv: Path,
    out_column_map_json: Path,
    *,
    mode: str,
) -> None:
    model_elements = elements[
        (elements["model_id"] == model_id) & (elements["element_kind"] == "property")
    ].copy()

    column_map: dict[str, str] = {}
    data: dict[str, list[str]] = {}

    if mode == "path_only":
        for _, element in model_elements.iterrows():
            column_name = _dedupe_column_name(
                str(element["symbol_path"]),
                str(element["element_id"]),
                column_map,
            )
            column_map[column_name] = str(element["element_id"])
            data[column_name] = [""]
    elif mode == "path_plus_metadata_values":
        synthetic_rows_by_column: dict[str, list[str]] = {}
        for _, element in model_elements.iterrows():
            column_name = _dedupe_column_name(
                str(element["symbol_path"]),
                str(element["element_id"]),
                column_map,
            )
            column_map[column_name] = str(element["element_id"])

            values = [
                f"name {split_identifier(str(element['name']))}",
                f"parent {split_identifier(str(element.get('parent_symbol_path') or ''))}",
                f"csharp type {split_identifier(str(element.get('csharp_type') or ''))}",
                f"normalized type {split_identifier(str(element.get('normalized_type') or ''))}",
                f"kind {split_identifier(str(element.get('element_kind') or ''))}",
                f"cardinality {'many' if bool(element.get('is_collection', False)) else 'single'}",
            ]

            comment = str(element.get("comment_text") or "").strip()
            if comment:
                values.append(f"comment {split_identifier(comment)}")

            relation_target = str(element.get("relation_target_type") or "").strip()
            if relation_target:
                values.append(f"relation target {split_identifier(relation_target)}")

            synthetic_rows_by_column[column_name] = values

        max_len = max((len(values) for values in synthetic_rows_by_column.values()), default=0)
        for column_name, values in synthetic_rows_by_column.items():
            data[column_name] = values + [""] * (max_len - len(values))
    else:
        raise ValueError(f"Unsupported property projection mode: {mode}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(out_csv, index=False)
    _write_column_map(out_column_map_json, column_map)


def build_type_projection_for_model(
    elements: pd.DataFrame,
    model_id: str,
    out_csv: Path,
    out_column_map_json: Path,
) -> None:
    type_elements = elements[
        (elements["model_id"] == model_id) & (elements["element_kind"] == "type")
    ].copy()
    model_elements = elements[elements["model_id"] == model_id].copy()

    column_map: dict[str, str] = {}
    members_by_column: dict[str, list[str]] = {}

    for _, type_element in type_elements.iterrows():
        type_path = str(type_element["symbol_path"])
        column_name = _dedupe_column_name(type_path, str(type_element["element_id"]), column_map)
        column_map[column_name] = str(type_element["element_id"])

        parent_mask = cast(pd.Series, model_elements["parent_symbol_path"]) == type_path
        kind_mask = cast(pd.Series, model_elements["element_kind"]).isin(
            ["property", "field", "enum_member"]
        )
        members = (
            model_elements[parent_mask & kind_mask]["name"].astype(str).tolist()
        )

        values = [split_identifier(member) for member in members]
        if not values:
            values = [f"type {split_identifier(str(type_element['name']))}"]
        members_by_column[column_name] = values

    max_len = max((len(values) for values in members_by_column.values()), default=0)
    data = {
        column_name: values + [""] * (max_len - len(values))
        for column_name, values in members_by_column.items()
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(out_csv, index=False)
    _write_column_map(out_column_map_json, column_map)


def build_relation_projection_for_model(
    elements: pd.DataFrame,
    model_id: str,
    out_csv: Path,
    out_column_map_json: Path,
) -> None:
    relation_elements = elements[
        (elements["model_id"] == model_id) & (elements["element_kind"] == "relation")
    ].copy()

    column_map: dict[str, str] = {}
    values_by_column: dict[str, list[str]] = {}

    for _, relation in relation_elements.iterrows():
        column_name = _dedupe_column_name(
            str(relation["symbol_path"]),
            str(relation["element_id"]),
            column_map,
        )
        column_map[column_name] = str(relation["element_id"])
        values_by_column[column_name] = [
            f"source {split_identifier(str(relation.get('parent_symbol_path') or ''))}",
            f"property {split_identifier(str(relation['name']))}",
            f"target {split_identifier(str(relation.get('relation_target_type') or ''))}",
            f"cardinality {'many' if bool(relation.get('is_collection', False)) else 'single'}",
        ]

    max_len = max((len(values) for values in values_by_column.values()), default=0)
    data = {
        column_name: values + [""] * (max_len - len(values))
        for column_name, values in values_by_column.items()
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(out_csv, index=False)
    _write_column_map(out_column_map_json, column_map)


def write_prepare_outputs(
    repo_root: Path,
    model_index: pd.DataFrame,
    elements: pd.DataFrame,
) -> dict[str, Any]:
    paths = prepare_paths(repo_root)
    paths.extracted_dir.mkdir(parents=True, exist_ok=True)
    paths.projections_dir.mkdir(parents=True, exist_ok=True)

    model_index.to_csv(paths.model_index_csv, index=False)
    elements.to_csv(paths.elements_csv, index=False)

    for model_id in cast(list[str], model_index["model_id"].tolist()):
        build_property_projection_for_model(
            elements,
            model_id,
            paths.projections_dir / "property" / "path_only" / f"{model_id}.csv",
            paths.column_maps_dir / "property" / f"{model_id}.json",
            mode="path_only",
        )
        build_property_projection_for_model(
            elements,
            model_id,
            paths.projections_dir / "property" / "path_plus_metadata_values" / f"{model_id}.csv",
            paths.column_maps_dir / "property" / f"{model_id}.json",
            mode="path_plus_metadata_values",
        )
        build_type_projection_for_model(
            elements,
            model_id,
            paths.projections_dir / "type" / "members_as_values" / f"{model_id}.csv",
            paths.column_maps_dir / "type" / f"{model_id}.json",
        )
        build_relation_projection_for_model(
            elements,
            model_id,
            paths.projections_dir / "relation" / "path_plus_metadata_values" / f"{model_id}.csv",
            paths.column_maps_dir / "relation" / f"{model_id}.json",
        )

    return {
        "project_root": paths.root_dir.relative_to(repo_root).as_posix(),
        "model_index_csv": paths.model_index_csv.relative_to(repo_root).as_posix(),
        "elements_csv": paths.elements_csv.relative_to(repo_root).as_posix(),
        "property_path_only_dir": (
            paths.projections_dir / "property" / "path_only"
        ).relative_to(repo_root).as_posix(),
        "property_path_plus_metadata_dir": (
            paths.projections_dir / "property" / "path_plus_metadata_values"
        ).relative_to(repo_root).as_posix(),
        "type_members_as_values_dir": (
            paths.projections_dir / "type" / "members_as_values"
        ).relative_to(repo_root).as_posix(),
        "relation_path_plus_metadata_dir": (
            paths.projections_dir / "relation" / "path_plus_metadata_values"
        ).relative_to(repo_root).as_posix(),
        "column_maps_dir": paths.column_maps_dir.relative_to(repo_root).as_posix(),
        "model_count": int(len(model_index)),
        "element_count": int(len(elements)),
        "property_count": (
            int((elements["element_kind"] == "property").sum()) if not elements.empty else 0
        ),
        "type_count": int((elements["element_kind"] == "type").sum()) if not elements.empty else 0,
        "relation_count": (
            int((elements["element_kind"] == "relation").sum()) if not elements.empty else 0
        ),
        "enum_member_count": int((elements["element_kind"] == "enum_member").sum())
        if not elements.empty
        else 0,
    }


def prepare_alignment_artifacts(
    repo_root: Path,
    *,
    dotnet_command: str = "dotnet",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    model_index = build_model_index(repo_root)
    elements, model_index_with_parse = extract_elements(
        repo_root,
        model_index,
        dotnet_command=dotnet_command,
    )
    summary = write_prepare_outputs(repo_root, model_index_with_parse, elements)
    return summary, model_index_with_parse, elements


def prepare_project_artifacts(
    repo_root: Path,
    *,
    dotnet_command: str = "dotnet",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    return prepare_alignment_artifacts(repo_root, dotnet_command=dotnet_command)


def load_projection_csv(repo_root: Path, layer: str, mode: str, model_id: str) -> pd.DataFrame:
    paths = prepare_paths(repo_root)
    return pd.read_csv(paths.projections_dir / layer / mode / f"{model_id}.csv")


def load_column_map(repo_root: Path, layer: str, model_id: str) -> dict[str, str]:
    paths = prepare_paths(repo_root)
    column_map_path = paths.column_maps_dir / layer / f"{model_id}.json"
    return cast(
        dict[str, str],
        json.loads(column_map_path.read_text(encoding="utf-8")),
    )


def load_project_artifacts(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = prepare_paths(repo_root)
    model_index = pd.read_csv(paths.model_index_csv)
    elements = pd.read_csv(paths.elements_csv)
    return model_index, elements


def build_positive_pairs(model_index: pd.DataFrame) -> pd.DataFrame:
    pair_rows: list[dict[str, Any]] = []

    for source_document_id, group in model_index.groupby("source_document_id"):
        records = cast(list[dict[str, Any]], group.to_dict(orient="records"))
        for source_row, target_row in itertools.permutations(records, 2):
            if source_row["model_id"] == target_row["model_id"]:
                continue

            # Exclude same-scenario replicas of the same producer across multiple runs.
            if (
                source_row["producer_id"] == target_row["producer_id"]
                and source_row["condition"] == target_row["condition"]
            ):
                continue

            pair_rows.append(
                {
                    "pair_kind": "positive",
                    "pair_id": (
                        f"{source_row['model_id']}__TO__{target_row['model_id']}"
                    ),
                    "source_document_id": source_document_id,
                    "source_model_id": source_row["model_id"],
                    "target_model_id": target_row["model_id"],
                    "source_condition": source_row["condition"],
                    "target_condition": target_row["condition"],
                    "source_producer_id": source_row["producer_id"],
                    "target_producer_id": target_row["producer_id"],
                    "source_run": int(source_row["run"]),
                    "target_run": int(target_row["run"]),
                }
            )

    return pd.DataFrame(pair_rows).sort_values(
        [
            "source_document_id",
            "source_condition",
            "target_condition",
            "source_producer_id",
            "target_producer_id",
        ],
        ignore_index=True,
    )


def write_pairs_csv(repo_root: Path, pairs: pd.DataFrame) -> Path:
    paths = align_run_paths(repo_root)
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(paths.pairs_csv, index=False)
    return paths.pairs_csv


def load_pairs_csv(repo_root: Path) -> pd.DataFrame:
    return pd.read_csv(align_run_paths(repo_root).pairs_csv)


def append_alignment_candidates(
    repo_root: Path,
    rows: list[dict[str, Any]],
    *,
    append: bool,
) -> Path:
    paths = align_run_paths(repo_root)
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).reindex(columns=ALIGNMENT_CANDIDATE_COLUMNS)
    if frame.empty:
        if not paths.candidates_csv.exists():
            frame.to_csv(paths.candidates_csv, index=False)
        return paths.candidates_csv

    frame.to_csv(
        paths.candidates_csv,
        mode="a" if append and paths.candidates_csv.exists() else "w",
        header=not (append and paths.candidates_csv.exists()),
        index=False,
    )
    return paths.candidates_csv


def load_alignment_candidates(repo_root: Path) -> pd.DataFrame:
    return pd.read_csv(align_run_paths(repo_root).candidates_csv)


def projection_exists(repo_root: Path, layer: str, mode: str, model_id: str) -> bool:
    paths = prepare_paths(repo_root)
    return (paths.projections_dir / layer / mode / f"{model_id}.csv").exists()


def projection_columns_count(repo_root: Path, layer: str, mode: str, model_id: str) -> int:
    projection = load_projection_csv(repo_root, layer, mode, model_id)
    return int(len(projection.columns))


def _normalize_valentine_rows(
    results: Any,
    *,
    pair_row: dict[str, Any],
    layer: str,
    mode: str,
    backend: str,
    method: str,
    source_column_map: dict[str, str],
    target_column_map: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair, score in results.items():
        if hasattr(pair, "source_column"):
            source_column = cast(str, pair.source_column)
            target_column = cast(str, pair.target_column)
        else:
            source_column = cast(str, pair[0][1])
            target_column = cast(str, pair[1][1])

        rows.append(
            {
                **pair_row,
                "projection_layer": layer,
                "projection_mode": mode,
                "backend": backend,
                "method": method,
                "source_column": source_column,
                "target_column": target_column,
                "source_element_id": source_column_map.get(source_column),
                "target_element_id": target_column_map.get(target_column),
                "score": float(score),
            }
        )
    return rows


def _normalize_generic_match_rows(
    matches: Any,
    *,
    pair_row: dict[str, Any],
    layer: str,
    mode: str,
    backend: str,
    method: str,
    source_column_map: dict[str, str],
    target_column_map: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if isinstance(matches, pd.DataFrame):
        iterable: list[Any] = cast(list[Any], matches.to_dict(orient="records"))
    else:
        iterable = list(matches)

    for match in iterable:
        if hasattr(match, "source_column"):
            source_column = cast(str, match.source_column)
            target_column = cast(str, match.target_column)
            score = float(match.similarity)
        elif isinstance(match, dict):
            source_column = cast(
                str,
                match.get("source_column")
                or match.get("source_attribute")
                or match.get("source"),
            )
            target_column = cast(
                str,
                match.get("target_column")
                or match.get("target_attribute")
                or match.get("target"),
            )
            score = float(match.get("similarity") or match.get("score") or 1.0)
        else:
            source_column = cast(str, match[0])
            target_column = cast(str, match[1])
            score = float(match[2] if len(match) > 2 else 1.0)

        rows.append(
            {
                **pair_row,
                "projection_layer": layer,
                "projection_mode": mode,
                "backend": backend,
                "method": method,
                "source_column": source_column,
                "target_column": target_column,
                "source_element_id": source_column_map.get(source_column),
                "target_element_id": target_column_map.get(target_column),
                "score": score,
            }
        )

    return rows


def _rank_alignment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    frame = pd.DataFrame(rows).sort_values(
        ["source_element_id", "score", "target_element_id"],
        ascending=[True, False, True],
        ignore_index=True,
    )
    frame["rank"] = (
        frame.groupby("source_element_id", dropna=False).cumcount() + 1
    )
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def _build_valentine_matcher_factory(method: str) -> Any:
    from valentine.algorithms import (
        ComaPy,
        Cupid,
        DistributionBased,
        JaccardDistanceMatcher,
        SimilarityFlooding,
    )

    factories: dict[str, Any] = {
        "coma_py": lambda: ComaPy(use_instances=False),
        "cupid": lambda: Cupid(),
        "distribution_based": lambda: DistributionBased(),
        "similarity_flooding": lambda: SimilarityFlooding(),
        "jaccard_distance": lambda: JaccardDistanceMatcher(),
    }
    if method not in factories:
        raise ValueError(f"Unsupported Valentine method: {method}")
    return factories[method]


def _build_bdikit_matcher_factory(method: str) -> Any:
    from bdikit.schema_matching.valentine import Coma as BDIComa
    from bdikit.schema_matching.valentine import Cupid as BDICupid
    from bdikit.schema_matching.valentine import DistributionBased as BDIDistributionBased
    from bdikit.schema_matching.valentine import Jaccard as BDIJaccard
    from bdikit.schema_matching.valentine import SimFlood as BDISimFlood

    factories: dict[str, Any] = {
        "coma": lambda: BDIComa(),
        "cupid": lambda: BDICupid(),
        "distribution_based": lambda: BDIDistributionBased(),
        "jaccard_distance": lambda: BDIJaccard(),
        "similarity_flooding": lambda: BDISimFlood(),
    }
    if method not in factories:
        raise ValueError(f"Unsupported BDI-kit method: {method}")
    return factories[method]


def _build_magneto_matcher_factory(method: str) -> Any:
    from magneto import Magneto

    factories: dict[str, Any] = {
        "native_zero_download": lambda: Magneto(
            include_embedding_matches=False,
            include_strsim_matches=True,
            include_equal_matches=True,
            use_bp_reranker=False,
            use_gpt_reranker=False,
            topk=10,
        ),
    }
    if method not in factories:
        raise ValueError(f"Unsupported Magneto method: {method}")
    return factories[method]


def valentine_method_names() -> list[str]:
    return [
        "coma_py",
        "cupid",
        "distribution_based",
        "similarity_flooding",
        "jaccard_distance",
    ]


def bdikit_method_names() -> list[str]:
    return [
        "coma",
        "cupid",
        "distribution_based",
        "jaccard_distance",
        "similarity_flooding",
    ]


def magneto_method_names() -> list[str]:
    return ["native_zero_download"]


def run_valentine_pair(
    repo_root: Path,
    pair_row: dict[str, Any],
    *,
    layer: str,
    mode: str,
    method: str,
    matcher_factory: Any,
    valentine_match_fn: Any,
    timeout_seconds: int | None = None,
) -> list[dict[str, Any]]:
    source_model_id = cast(str, pair_row["source_model_id"])
    target_model_id = cast(str, pair_row["target_model_id"])
    source_projection = load_projection_csv(repo_root, layer, mode, source_model_id)
    target_projection = load_projection_csv(repo_root, layer, mode, target_model_id)
    if len(source_projection.columns) == 0 or len(target_projection.columns) == 0:
        return []

    results = _run_with_timeout(
        timeout_seconds,
        lambda: valentine_match_fn(
            source_projection,
            target_projection,
            matcher_factory(),
        ).one_to_one(),
    )
    rows = _normalize_valentine_rows(
        results,
        pair_row=pair_row,
        layer=layer,
        mode=mode,
        backend="valentine",
        method=method,
        source_column_map=load_column_map(repo_root, layer, source_model_id),
        target_column_map=load_column_map(repo_root, layer, target_model_id),
    )
    return _rank_alignment_rows(rows)


def run_bdikit_pair(
    repo_root: Path,
    pair_row: dict[str, Any],
    *,
    layer: str,
    mode: str,
    method: str,
    matcher_factory: Any,
    timeout_seconds: int | None = None,
) -> list[dict[str, Any]]:
    source_model_id = cast(str, pair_row["source_model_id"])
    target_model_id = cast(str, pair_row["target_model_id"])
    source_projection = load_projection_csv(repo_root, layer, mode, source_model_id)
    target_projection = load_projection_csv(repo_root, layer, mode, target_model_id)
    if len(source_projection.columns) == 0 or len(target_projection.columns) == 0:
        return []

    matches = _run_with_timeout(
        timeout_seconds,
        lambda: matcher_factory().match_schema(source_projection, target_projection),
    )
    rows = _normalize_generic_match_rows(
        matches,
        pair_row=pair_row,
        layer=layer,
        mode=mode,
        backend="bdikit",
        method=method,
        source_column_map=load_column_map(repo_root, layer, source_model_id),
        target_column_map=load_column_map(repo_root, layer, target_model_id),
    )
    return _rank_alignment_rows(rows)


def run_magneto_pair(
    repo_root: Path,
    pair_row: dict[str, Any],
    *,
    layer: str,
    mode: str,
    method: str,
    matcher_factory: Any,
    timeout_seconds: int | None = None,
) -> list[dict[str, Any]]:
    source_model_id = cast(str, pair_row["source_model_id"])
    target_model_id = cast(str, pair_row["target_model_id"])
    source_projection = load_projection_csv(repo_root, layer, mode, source_model_id)
    target_projection = load_projection_csv(repo_root, layer, mode, target_model_id)
    if len(source_projection.columns) == 0 or len(target_projection.columns) == 0:
        return []

    matches = _run_with_timeout(
        timeout_seconds,
        lambda: matcher_factory().get_matches(source_projection, target_projection),
    )
    rows = _normalize_generic_match_rows(
        matches,
        pair_row=pair_row,
        layer=layer,
        mode=mode,
        backend="magneto",
        method=method,
        source_column_map=load_column_map(repo_root, layer, source_model_id),
        target_column_map=load_column_map(repo_root, layer, target_model_id),
    )
    return _rank_alignment_rows(rows)


def build_alignment_tasks(
    repo_root: Path,
    pairs: pd.DataFrame,
    *,
    projection_specs: list[tuple[str, str]],
) -> pd.DataFrame:
    pair_records = cast(list[dict[str, Any]], pairs.to_dict(orient="records"))
    task_rows: list[dict[str, Any]] = []

    backend_methods: list[tuple[str, list[str]]] = [
        ("valentine", valentine_method_names()),
        ("bdikit", bdikit_method_names()),
        ("magneto", magneto_method_names()),
    ]

    for pair_index, pair_row in enumerate(pair_records, start=1):
        source_model_id = cast(str, pair_row["source_model_id"])
        target_model_id = cast(str, pair_row["target_model_id"])
        for layer, mode in projection_specs:
            if not projection_exists(repo_root, layer, mode, source_model_id):
                continue
            if not projection_exists(repo_root, layer, mode, target_model_id):
                continue

            for backend, methods in backend_methods:
                for method in methods:
                    task_rows.append(
                        {
                            **pair_row,
                            "pair_index": pair_index,
                            "projection_layer": layer,
                            "projection_mode": mode,
                            "backend": backend,
                            "method": method,
                            "task_id": (
                                f"{pair_row['pair_id']}::{layer}/{mode}::{backend}::{method}"
                            ),
                        }
                    )

    if not task_rows:
        return pd.DataFrame()

    return pd.DataFrame(task_rows)


def _process_alignment_task_worker(
    repo_root_str: str,
    task_row: dict[str, Any],
    timeout_seconds: int | None,
) -> dict[str, Any]:
    from valentine import valentine_match

    repo_root = Path(repo_root_str)
    pair_row = {
        key: value
        for key, value in task_row.items()
        if key
        not in {
            "projection_layer",
            "projection_mode",
            "backend",
            "method",
            "task_id",
            "pair_index",
        }
    }
    layer = cast(str, task_row["projection_layer"])
    mode = cast(str, task_row["projection_mode"])
    backend = cast(str, task_row["backend"])
    method = cast(str, task_row["method"])

    try:
        if backend == "valentine":
            rows = run_valentine_pair(
                repo_root,
                pair_row,
                layer=layer,
                mode=mode,
                method=method,
                matcher_factory=_build_valentine_matcher_factory(method),
                valentine_match_fn=valentine_match,
                timeout_seconds=timeout_seconds,
            )
        elif backend == "bdikit":
            rows = run_bdikit_pair(
                repo_root,
                pair_row,
                layer=layer,
                mode=mode,
                method=method,
                matcher_factory=_build_bdikit_matcher_factory(method),
                timeout_seconds=timeout_seconds,
            )
        elif backend == "magneto":
            rows = run_magneto_pair(
                repo_root,
                pair_row,
                layer=layer,
                mode=mode,
                method=method,
                matcher_factory=_build_magneto_matcher_factory(method),
                timeout_seconds=timeout_seconds,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        failures: list[dict[str, str]] = []
    except Exception as exc:
        rows = []
        failures = [
            {
                "backend": backend,
                "method": method,
                "layer": layer,
                "mode": mode,
                "pair_id": str(task_row["pair_id"]),
                "error": str(exc),
            }
        ]

    return {
        "task_id": str(task_row["task_id"]),
        "pair_id": str(task_row["pair_id"]),
        "pair_index": int(task_row["pair_index"]),
        "projection_layer": layer,
        "projection_mode": mode,
        "backend": backend,
        "method": method,
        "rows": rows,
        "failures": failures,
    }


def iter_alignment_task_results_parallel(
    repo_root: Path,
    tasks: pd.DataFrame,
    *,
    timeout_seconds: int | None,
    max_workers: int | None = None,
) -> Iterator[dict[str, Any]]:
    task_records = cast(list[dict[str, Any]], tasks.to_dict(orient="records"))
    if not task_records:
        return

    worker_count = max_workers or available_cpu_count()
    total_tasks = len(task_records)

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_pair: dict[
            concurrent.futures.Future[dict[str, Any]],
            tuple[int, dict[str, Any]],
        ] = {}
        for task_index, task_row in enumerate(task_records, start=1):
            future = executor.submit(
                _process_alignment_task_worker,
                repo_root.as_posix(),
                task_row,
                timeout_seconds,
            )
            future_to_pair[future] = (task_index, task_row)

        for future in concurrent.futures.as_completed(future_to_pair):
            task_index, task_row = future_to_pair[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "task_id": str(task_row["task_id"]),
                    "pair_id": str(task_row["pair_id"]),
                    "pair_index": int(task_row["pair_index"]),
                    "projection_layer": str(task_row["projection_layer"]),
                    "projection_mode": str(task_row["projection_mode"]),
                    "backend": "worker",
                    "method": "task_worker",
                    "rows": [],
                    "failures": [
                        {
                            "backend": "worker",
                            "method": "task_worker",
                            "layer": str(task_row["projection_layer"]),
                            "mode": str(task_row["projection_mode"]),
                            "pair_id": str(task_row["pair_id"]),
                            "error": str(exc),
                        }
                    ],
                }
            yield {
                "task_index": task_index,
                "total_tasks": total_tasks,
                "worker_count": worker_count,
                **result,
            }


def run_alignment_pairs_parallel(
    repo_root: Path,
    pairs: pd.DataFrame,
    *,
    projection_specs: list[tuple[str, str]],
    timeout_seconds: int | None,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, Any]] = []
    tasks = build_alignment_tasks(
        repo_root,
        pairs,
        projection_specs=projection_specs,
    )
    for result in iter_alignment_task_results_parallel(
        repo_root,
        tasks,
        timeout_seconds=timeout_seconds,
        max_workers=max_workers,
    ):
        results.append(result)

    if not results:
        return pd.DataFrame(columns=ALIGNMENT_CANDIDATE_COLUMNS), pd.DataFrame()

    results.sort(key=lambda item: str(item["task_id"]))
    rows = [
        row
        for result in results
        for row in cast(list[dict[str, Any]], result["rows"])
    ]
    failures = [
        failure
        for result in results
        for failure in cast(list[dict[str, Any]], result["failures"])
    ]
    return (
        pd.DataFrame(rows).reindex(columns=ALIGNMENT_CANDIDATE_COLUMNS),
        pd.DataFrame(failures),
    )


def projection_element_kind(layer: str) -> str:
    mapping = {
        "property": "property",
        "type": "type",
        "relation": "relation",
    }
    if layer not in mapping:
        raise ValueError(f"Unsupported projection layer: {layer}")
    return mapping[layer]


def _element_counts_by_model_and_kind(elements: pd.DataFrame) -> pd.DataFrame:
    counts_by_key: dict[tuple[str, str], int] = {}
    for model_id, element_kind in zip(
        cast(list[Any], elements["model_id"].tolist()),
        cast(list[Any], elements["element_kind"].tolist()),
        strict=True,
    ):
        key = (str(model_id), str(element_kind))
        counts_by_key[key] = counts_by_key.get(key, 0) + 1

    return pd.DataFrame(
        [
            {
                "model_id": model_id,
                "element_kind": element_kind,
                "element_count": element_count,
            }
            for (model_id, element_kind), element_count in sorted(counts_by_key.items())
        ]
    )


def score_alignment_candidates(
    model_index: pd.DataFrame,
    elements: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    counts = _element_counts_by_model_and_kind(elements)
    pair_groups = candidates.groupby(
        [
            "source_document_id",
            "source_model_id",
            "target_model_id",
            "source_condition",
            "target_condition",
            "source_producer_id",
            "target_producer_id",
            "projection_layer",
            "projection_mode",
            "backend",
            "method",
        ],
        dropna=False,
    )

    pair_rows: list[dict[str, Any]] = []
    for group_key, group in pair_groups:
        group_tuple = cast(tuple[Any, ...], group_key)
        group_frame = cast(pd.DataFrame, group)
        (
            source_document_id,
            source_model_id,
            target_model_id,
            source_condition,
            target_condition,
            source_producer_id,
            target_producer_id,
            projection_layer,
            projection_mode,
            backend,
            method,
        ) = group_tuple
        element_kind = projection_element_kind(cast(str, projection_layer))

        source_count_row = cast(
            pd.DataFrame,
            counts[
            (counts["model_id"] == source_model_id) & (counts["element_kind"] == element_kind)
            ],
        )
        target_count_row = cast(
            pd.DataFrame,
            counts[
            (counts["model_id"] == target_model_id) & (counts["element_kind"] == element_kind)
            ],
        )
        source_element_count = _first_int(source_count_row, "element_count")
        target_element_count = _first_int(target_count_row, "element_count")

        top1 = cast(pd.DataFrame, group_frame[group_frame["rank"] == 1].copy())
        matched_source_count = int(
            cast(pd.Series, group_frame["source_element_id"]).dropna().nunique()
        )
        matched_target_count = int(
            cast(pd.Series, group_frame["target_element_id"]).dropna().nunique()
        )
        source_coverage = (
            matched_source_count / source_element_count if source_element_count else 0.0
        )
        target_coverage = (
            matched_target_count / target_element_count if target_element_count else 0.0
        )
        coverage_f1 = _harmonic_mean(source_coverage, target_coverage)
        top1_scores = _finite_float_values(cast(pd.Series, top1["score"]))
        all_scores = _finite_float_values(cast(pd.Series, group_frame["score"]))
        top1_mean_score = _mean(top1_scores)
        pair_alignment_score = source_coverage * top1_mean_score
        pair_alignment_f1 = coverage_f1 * top1_mean_score

        pair_rows.append(
            {
                "source_document_id": source_document_id,
                "source_model_id": source_model_id,
                "target_model_id": target_model_id,
                "source_condition": source_condition,
                "target_condition": target_condition,
                "source_producer_id": source_producer_id,
                "target_producer_id": target_producer_id,
                "projection_layer": projection_layer,
                "projection_mode": projection_mode,
                "backend": backend,
                "method": method,
                "source_element_count": source_element_count,
                "target_element_count": target_element_count,
                "candidate_count": int(len(group_frame)),
                "matched_source_element_count": matched_source_count,
                "matched_target_element_count": matched_target_count,
                "source_coverage": source_coverage,
                "target_coverage": target_coverage,
                "coverage_f1": coverage_f1,
                "top1_mean_score": top1_mean_score,
                "all_mean_score": _mean(all_scores),
                "pair_alignment_score": pair_alignment_score,
                "pair_alignment_f1": pair_alignment_f1,
            }
        )

    pair_scores = pd.DataFrame(pair_rows).sort_values(
        [
            "source_document_id",
            "source_condition",
            "target_condition",
            "source_producer_id",
            "target_producer_id",
            "backend",
            "method",
            "projection_layer",
            "projection_mode",
        ],
        ignore_index=True,
    )

    model_scores = (
        pair_scores.groupby(
            [
                "source_document_id",
                "source_model_id",
                "source_condition",
                "source_producer_id",
                "projection_layer",
                "projection_mode",
                "backend",
                "method",
            ],
            dropna=False,
        )
        .agg(
            peer_count=("target_model_id", "nunique"),
            mean_source_coverage=("source_coverage", "mean"),
            mean_target_coverage=("target_coverage", "mean"),
            mean_coverage_f1=("coverage_f1", "mean"),
            mean_top1_score=("top1_mean_score", "mean"),
            mean_pair_alignment_score=("pair_alignment_score", "mean"),
            mean_pair_alignment_f1=("pair_alignment_f1", "mean"),
        )
        .reset_index()
        .sort_values(
            [
                "source_document_id",
                "source_condition",
                "source_producer_id",
                "backend",
                "method",
            ],
            ignore_index=True,
        )
    )

    scenario_scores = (
        model_scores.groupby(
            [
                "source_document_id",
                "source_condition",
                "projection_layer",
                "projection_mode",
                "backend",
                "method",
            ],
            dropna=False,
        )
        .agg(
            model_count=("source_model_id", "nunique"),
            mean_peer_count=("peer_count", "mean"),
            mean_source_coverage=("mean_source_coverage", "mean"),
            mean_target_coverage=("mean_target_coverage", "mean"),
            mean_coverage_f1=("mean_coverage_f1", "mean"),
            mean_top1_score=("mean_top1_score", "mean"),
            mean_pair_alignment_score=("mean_pair_alignment_score", "mean"),
            mean_pair_alignment_f1=("mean_pair_alignment_f1", "mean"),
        )
        .reset_index()
        .sort_values(
            [
                "source_document_id",
                "source_condition",
                "backend",
                "method",
            ],
            ignore_index=True,
        )
    )

    scenario_matrix_scores = (
        pair_scores.groupby(
            [
                "source_document_id",
                "source_condition",
                "target_condition",
                "projection_layer",
                "projection_mode",
                "backend",
                "method",
            ],
            dropna=False,
        )
        .agg(
            pair_count=("target_model_id", "count"),
            mean_source_coverage=("source_coverage", "mean"),
            mean_target_coverage=("target_coverage", "mean"),
            mean_coverage_f1=("coverage_f1", "mean"),
            mean_top1_score=("top1_mean_score", "mean"),
            mean_pair_alignment_score=("pair_alignment_score", "mean"),
            mean_pair_alignment_f1=("pair_alignment_f1", "mean"),
        )
        .reset_index()
        .sort_values(
            [
                "source_document_id",
                "source_condition",
                "target_condition",
                "backend",
                "method",
            ],
            ignore_index=True,
        )
    )

    return (
        cast(pd.DataFrame, pair_scores),
        cast(pd.DataFrame, model_scores),
        cast(pd.DataFrame, scenario_scores),
        cast(pd.DataFrame, scenario_matrix_scores),
    )


def write_score_outputs(
    repo_root: Path,
    pair_scores: pd.DataFrame,
    model_scores: pd.DataFrame,
    scenario_scores: pd.DataFrame,
    scenario_matrix_scores: pd.DataFrame,
) -> dict[str, str]:
    paths = score_paths(repo_root)
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    pair_scores.to_csv(paths.pair_scores_csv, index=False)
    model_scores.to_csv(paths.model_scores_csv, index=False)
    scenario_scores.to_csv(paths.scenario_scores_csv, index=False)
    scenario_matrix_scores.to_csv(paths.scenario_matrix_scores_csv, index=False)
    return {
        "pair_scores_csv": paths.pair_scores_csv.relative_to(repo_root).as_posix(),
        "model_scores_csv": paths.model_scores_csv.relative_to(repo_root).as_posix(),
        "scenario_scores_csv": paths.scenario_scores_csv.relative_to(repo_root).as_posix(),
        "scenario_matrix_scores_csv": (
            paths.scenario_matrix_scores_csv.relative_to(repo_root).as_posix()
        ),
    }


def load_score_outputs(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = score_paths(repo_root)
    return (
        pd.read_csv(paths.pair_scores_csv),
        pd.read_csv(paths.model_scores_csv),
        pd.read_csv(paths.scenario_scores_csv),
        pd.read_csv(paths.scenario_matrix_scores_csv),
    )


def build_feedback_delta_frame(
    model_scores: pd.DataFrame,
    *,
    metric_column: str = "mean_pair_alignment_f1",
) -> pd.DataFrame:
    if model_scores.empty:
        return pd.DataFrame()

    pivot = (
        model_scores.pivot_table(
            index=[
                "source_document_id",
                "source_producer_id",
                "projection_layer",
                "projection_mode",
                "backend",
                "method",
            ],
            columns="source_condition",
            values=metric_column,
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    for condition in ("control", "guided", "feedback"):
        if condition not in pivot.columns:
            pivot[condition] = math.nan

    pivot["feedback_minus_control"] = pivot["feedback"] - pivot["control"]
    pivot["feedback_minus_guided"] = pivot["feedback"] - pivot["guided"]
    pivot["guided_minus_control"] = pivot["guided"] - pivot["control"]

    condition_frame = pivot[["control", "guided", "feedback"]]
    pivot["best_condition"] = cast(pd.Series, condition_frame.idxmax(axis=1))
    pivot["feedback_is_best"] = cast(pd.Series, pivot["best_condition"] == "feedback")

    return cast(
        pd.DataFrame,
        pivot.sort_values(
            [
                "feedback_minus_guided",
                "feedback_minus_control",
                "source_document_id",
                "source_producer_id",
                "backend",
                "method",
            ],
            ascending=[False, False, True, True, True, True],
            ignore_index=True,
        ),
    )


def build_feedback_target_delta_frame(
    scenario_matrix_scores: pd.DataFrame,
    *,
    metric_column: str = "mean_pair_alignment_f1",
) -> pd.DataFrame:
    if scenario_matrix_scores.empty:
        return pd.DataFrame()

    pivot = (
        scenario_matrix_scores.pivot_table(
            index=[
                "source_document_id",
                "target_condition",
                "projection_layer",
                "projection_mode",
                "backend",
                "method",
            ],
            columns="source_condition",
            values=metric_column,
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    for condition in ("control", "guided", "feedback"):
        if condition not in pivot.columns:
            pivot[condition] = math.nan

    pivot["feedback_minus_control"] = pivot["feedback"] - pivot["control"]
    pivot["feedback_minus_guided"] = pivot["feedback"] - pivot["guided"]
    pivot["guided_minus_control"] = pivot["guided"] - pivot["control"]

    return cast(
        pd.DataFrame,
        pivot.sort_values(
            [
                "feedback_minus_guided",
                "feedback_minus_control",
                "source_document_id",
                "target_condition",
                "backend",
                "method",
            ],
            ascending=[False, False, True, True, True, True],
            ignore_index=True,
        ),
    )
