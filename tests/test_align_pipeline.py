from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from src.lib import align_pipeline


def test_build_model_id_stays_stable_and_slugged() -> None:
    model_id = align_pipeline.build_model_id(
        "stripe-payment-intent",
        "guided",
        "google/gemini-2.5-flash-lite",
        2,
    )
    assert model_id == "stripe-payment-intent__guided__google-gemini-2-5-flash-lite__run2"


def test_append_alignment_candidates_writes_header_for_empty_result_set(
    tmp_path: Path,
) -> None:
    csv_path = align_pipeline.append_alignment_candidates(
        tmp_path,
        [],
        append=False,
    )

    frame = pd.read_csv(csv_path)

    assert csv_path == tmp_path / "artifacts" / "align" / "alignment-candidates.csv"
    assert frame.empty
    assert list(frame.columns) == align_pipeline.ALIGNMENT_CANDIDATE_COLUMNS


def test_split_identifier_breaks_camel_case_and_separators() -> None:
    assert align_pipeline.split_identifier("PaymentIntent.AmountMinorUnits") == (
        "payment intent amount minor units"
    )


def test_build_property_projection_for_model_writes_metadata_rows_and_column_map(
    tmp_path: Path,
) -> None:
    elements = pd.DataFrame(
        [
            {
                "model_id": "model-1",
                "element_id": "model-1:e0001",
                "element_kind": "property",
                "symbol_path": "PaymentIntent.Amount",
                "parent_symbol_path": "PaymentIntent",
                "name": "Amount",
                "csharp_type": "long",
                "normalized_type": "integer",
                "is_collection": False,
                "relation_target_type": None,
                "comment_text": "Amount in minor units",
            },
            {
                "model_id": "model-1",
                "element_id": "model-1:e0002",
                "element_kind": "property",
                "symbol_path": "PaymentIntent.Status",
                "parent_symbol_path": "PaymentIntent",
                "name": "Status",
                "csharp_type": "PaymentIntentStatus",
                "normalized_type": "enum",
                "is_collection": False,
                "relation_target_type": None,
                "comment_text": "",
            },
        ]
    )

    projection_path = tmp_path / "property.csv"
    column_map_path = tmp_path / "column_map.json"
    align_pipeline.build_property_projection_for_model(
        elements,
        "model-1",
        projection_path,
        column_map_path,
        mode="path_plus_metadata_values",
    )

    projection = pd.read_csv(projection_path)
    column_map = json.loads(column_map_path.read_text(encoding="utf-8"))

    assert list(projection.columns) == ["PaymentIntent.Amount", "PaymentIntent.Status"]
    assert projection.iloc[0, 0] == "name amount"
    assert "comment amount in minor units" in projection["PaymentIntent.Amount"].tolist()
    assert column_map["PaymentIntent.Amount"] == "model-1:e0001"


def test_build_relation_projection_for_model_writes_relation_values(tmp_path: Path) -> None:
    elements = pd.DataFrame(
        [
            {
                "model_id": "model-1",
                "element_id": "model-1:e0003",
                "element_kind": "relation",
                "symbol_path": "relation.Issue.Assignees.User",
                "parent_symbol_path": "Issue",
                "name": "Assignees",
                "is_collection": True,
                "relation_target_type": "User",
            }
        ]
    )

    projection_path = tmp_path / "relation.csv"
    column_map_path = tmp_path / "relation_map.json"
    align_pipeline.build_relation_projection_for_model(
        elements,
        "model-1",
        projection_path,
        column_map_path,
    )

    projection = pd.read_csv(projection_path)
    assert list(projection.columns) == ["relation.Issue.Assignees.User"]
    assert projection.iloc[0, 0] == "source issue"
    assert projection.iloc[1, 0] == "property assignees"
    assert projection.iloc[2, 0] == "target user"
    assert projection.iloc[3, 0] == "cardinality many"


def test_build_positive_pairs_excludes_same_scenario_replicas_but_keeps_cross_scenario() -> None:
    model_index = pd.DataFrame(
        [
            {
                "model_id": "doc__control__m1__run1",
                "source_document_id": "doc",
                "condition": "control",
                "producer_id": "m1",
                "run": 1,
            },
            {
                "model_id": "doc__control__m1__run2",
                "source_document_id": "doc",
                "condition": "control",
                "producer_id": "m1",
                "run": 2,
            },
            {
                "model_id": "doc__guided__m1__run1",
                "source_document_id": "doc",
                "condition": "guided",
                "producer_id": "m1",
                "run": 1,
            },
            {
                "model_id": "doc__control__m2__run1",
                "source_document_id": "doc",
                "condition": "control",
                "producer_id": "m2",
                "run": 1,
            },
        ]
    )

    pairs = align_pipeline.build_positive_pairs(model_index)
    pair_ids = set(pairs["pair_id"].tolist())

    assert "doc__control__m1__run1__TO__doc__control__m1__run2" not in pair_ids
    assert "doc__control__m1__run1__TO__doc__guided__m1__run1" in pair_ids
    assert "doc__guided__m1__run1__TO__doc__control__m2__run1" in pair_ids


def test_build_alignment_tasks_expands_pairs_projection_and_methods(tmp_path: Path) -> None:
    repo_root = tmp_path
    projections_dir = align_pipeline.prepare_paths(repo_root).projections_dir
    projection_specs = [
        ("property", "path_only"),
        ("type", "members_as_values"),
    ]
    for layer, mode in projection_specs:
        projection_dir = projections_dir / layer / mode
        projection_dir.mkdir(parents=True, exist_ok=True)
        for model_id in ("m1", "m2"):
            (projection_dir / f"{model_id}.csv").write_text("col_a\nvalue\n", encoding="utf-8")

    pairs = pd.DataFrame(
        [
            {
                "pair_id": "m1__TO__m2",
                "pair_kind": "positive",
                "source_document_id": "doc",
                "source_model_id": "m1",
                "target_model_id": "m2",
                "source_condition": "control",
                "target_condition": "guided",
                "source_producer_id": "p1",
                "target_producer_id": "p2",
                "source_run": 1,
                "target_run": 1,
            }
        ]
    )

    tasks = align_pipeline.build_alignment_tasks(
        repo_root,
        pairs,
        projection_specs=projection_specs,
    )

    expected_methods = (
        len(align_pipeline.valentine_method_names())
        + len(align_pipeline.bdikit_method_names())
        + len(align_pipeline.magneto_method_names())
    )
    assert len(tasks) == len(projection_specs) * expected_methods
    assert tasks["task_id"].is_unique
    assert set(tasks["backend"]) == {"valentine", "bdikit", "magneto"}


def test_score_alignment_candidates_builds_pair_and_scenario_stats() -> None:
    model_index = pd.DataFrame(
        [
            {
                "model_id": "doc__control__m1__run1",
                "source_document_id": "doc",
                "condition": "control",
                "producer_id": "m1",
                "run": 1,
            },
            {
                "model_id": "doc__guided__m2__run1",
                "source_document_id": "doc",
                "condition": "guided",
                "producer_id": "m2",
                "run": 1,
            },
        ]
    )
    elements = pd.DataFrame(
        [
            {
                "model_id": "doc__control__m1__run1",
                "element_kind": "property",
            },
            {
                "model_id": "doc__control__m1__run1",
                "element_kind": "property",
            },
            {
                "model_id": "doc__guided__m2__run1",
                "element_kind": "property",
            },
            {
                "model_id": "doc__guided__m2__run1",
                "element_kind": "property",
            },
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "source_document_id": "doc",
                "source_model_id": "doc__control__m1__run1",
                "target_model_id": "doc__guided__m2__run1",
                "source_condition": "control",
                "target_condition": "guided",
                "source_producer_id": "m1",
                "target_producer_id": "m2",
                "projection_layer": "property",
                "projection_mode": "path_plus_metadata_values",
                "backend": "valentine",
                "method": "coma_py",
                "source_element_id": "a:e1",
                "target_element_id": "b:e1",
                "score": 0.8,
                "rank": 1,
            },
            {
                "source_document_id": "doc",
                "source_model_id": "doc__control__m1__run1",
                "target_model_id": "doc__guided__m2__run1",
                "source_condition": "control",
                "target_condition": "guided",
                "source_producer_id": "m1",
                "target_producer_id": "m2",
                "projection_layer": "property",
                "projection_mode": "path_plus_metadata_values",
                "backend": "valentine",
                "method": "coma_py",
                "source_element_id": "a:e2",
                "target_element_id": "b:e2",
                "score": 0.6,
                "rank": 1,
            },
        ]
    )

    pair_scores, model_scores, scenario_scores, scenario_matrix_scores = (
        align_pipeline.score_alignment_candidates(model_index, elements, candidates)
    )

    assert len(pair_scores) == 1
    assert pair_scores.iloc[0]["source_coverage"] == 1.0
    assert pair_scores.iloc[0]["target_coverage"] == 1.0
    assert pair_scores.iloc[0]["coverage_f1"] == 1.0
    assert round(float(pair_scores.iloc[0]["top1_mean_score"]), 3) == 0.7
    assert round(float(pair_scores.iloc[0]["pair_alignment_f1"]), 3) == 0.7
    assert len(model_scores) == 1
    assert float(model_scores.iloc[0]["mean_pair_alignment_f1"]) == 0.7
    assert len(scenario_scores) == 1
    assert float(scenario_scores.iloc[0]["mean_coverage_f1"]) == 1.0
    assert scenario_matrix_scores.iloc[0]["source_condition"] == "control"
    assert scenario_matrix_scores.iloc[0]["target_condition"] == "guided"
    assert float(scenario_matrix_scores.iloc[0]["mean_pair_alignment_f1"]) == 0.7


def test_score_alignment_candidates_ignores_nan_scores() -> None:
    model_index = pd.DataFrame(
        [
            {
                "model_id": "doc__control__m1__run1",
                "source_document_id": "doc",
                "condition": "control",
                "producer_id": "m1",
                "run": 1,
            },
            {
                "model_id": "doc__guided__m2__run1",
                "source_document_id": "doc",
                "condition": "guided",
                "producer_id": "m2",
                "run": 1,
            },
        ]
    )
    elements = pd.DataFrame(
        [
            {"model_id": "doc__control__m1__run1", "element_kind": "relation"},
            {"model_id": "doc__guided__m2__run1", "element_kind": "relation"},
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "source_document_id": "doc",
                "source_model_id": "doc__control__m1__run1",
                "target_model_id": "doc__guided__m2__run1",
                "source_condition": "control",
                "target_condition": "guided",
                "source_producer_id": "m1",
                "target_producer_id": "m2",
                "projection_layer": "relation",
                "projection_mode": "path_plus_metadata_values",
                "backend": "bdikit",
                "method": "coma",
                "source_element_id": "a:e1",
                "target_element_id": "b:e1",
                "score": math.nan,
                "rank": 1,
            }
        ]
    )

    pair_scores, _, _, _ = align_pipeline.score_alignment_candidates(
        model_index,
        elements,
        candidates,
    )

    assert float(pair_scores.iloc[0]["top1_mean_score"]) == 0.0
    assert float(pair_scores.iloc[0]["all_mean_score"]) == 0.0
    assert float(pair_scores.iloc[0]["pair_alignment_score"]) == 0.0
    assert float(pair_scores.iloc[0]["coverage_f1"]) == 1.0
    assert float(pair_scores.iloc[0]["pair_alignment_f1"]) == 0.0


def test_build_feedback_delta_frames_compute_condition_deltas() -> None:
    model_scores = pd.DataFrame(
        [
            {
                "source_document_id": "doc",
                "source_model_id": "doc__control__m1__run1",
                "source_condition": "control",
                "source_producer_id": "m1",
                "projection_layer": "property",
                "projection_mode": "path_only",
                "backend": "valentine",
                "method": "coma_py",
                "mean_pair_alignment_f1": 0.40,
            },
            {
                "source_document_id": "doc",
                "source_model_id": "doc__guided__m1__run1",
                "source_condition": "guided",
                "source_producer_id": "m1",
                "projection_layer": "property",
                "projection_mode": "path_only",
                "backend": "valentine",
                "method": "coma_py",
                "mean_pair_alignment_f1": 0.45,
            },
            {
                "source_document_id": "doc",
                "source_model_id": "doc__feedback__m1__run1",
                "source_condition": "feedback",
                "source_producer_id": "m1",
                "projection_layer": "property",
                "projection_mode": "path_only",
                "backend": "valentine",
                "method": "coma_py",
                "mean_pair_alignment_f1": 0.60,
            },
        ]
    )
    scenario_matrix_scores = pd.DataFrame(
        [
            {
                "source_document_id": "doc",
                "source_condition": "control",
                "target_condition": "guided",
                "projection_layer": "property",
                "projection_mode": "path_only",
                "backend": "valentine",
                "method": "coma_py",
                "mean_pair_alignment_f1": 0.41,
            },
            {
                "source_document_id": "doc",
                "source_condition": "guided",
                "target_condition": "guided",
                "projection_layer": "property",
                "projection_mode": "path_only",
                "backend": "valentine",
                "method": "coma_py",
                "mean_pair_alignment_f1": 0.47,
            },
            {
                "source_document_id": "doc",
                "source_condition": "feedback",
                "target_condition": "guided",
                "projection_layer": "property",
                "projection_mode": "path_only",
                "backend": "valentine",
                "method": "coma_py",
                "mean_pair_alignment_f1": 0.62,
            },
        ]
    )

    feedback_delta = align_pipeline.build_feedback_delta_frame(model_scores)
    feedback_target_delta = align_pipeline.build_feedback_target_delta_frame(
        scenario_matrix_scores
    )

    assert round(float(feedback_delta.iloc[0]["feedback_minus_control"]), 3) == 0.2
    assert round(float(feedback_delta.iloc[0]["feedback_minus_guided"]), 3) == 0.15
    assert bool(feedback_delta.iloc[0]["feedback_is_best"]) is True
    assert round(float(feedback_target_delta.iloc[0]["feedback_minus_guided"]), 3) == 0.15
