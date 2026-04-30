from __future__ import annotations

import html
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.lib.infer_pipeline import read_text


def metrics_project_path(repo_root: Path) -> Path:
    return repo_root / "src" / "lib" / "csharp_metrics" / "CSharpMetrics.csproj"


def load_record(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(read_text(path)))


def parse_run_name(name: str) -> int:
    if name.startswith("run"):
        return int(name.removeprefix("run"))
    raise ValueError(f"Unsupported run directory name: {name}")


def discover_infer_artifacts(repo_root: Path) -> pd.DataFrame:
    infer_root = repo_root / "artifacts" / "infer"
    records: list[dict[str, Any]] = []

    for final_cs_path in sorted(infer_root.rglob("final.cs")):
        metadata_path = final_cs_path.with_name("record.json")
        relative_final_cs_path = final_cs_path.relative_to(repo_root).as_posix()
        relative_metadata_path = (
            metadata_path.relative_to(repo_root).as_posix() if metadata_path.exists() else None
        )

        if metadata_path.exists():
            record = load_record(metadata_path)
            model = cast(str, record["model"])
            document = cast(str, record["document"])
            document_stem = cast(str, record["document_stem"])
            scenario = cast(str, record["scenario"])
            run = int(record["run"])
            source_matrix_line = record.get("source_matrix_line")
        else:
            parts = final_cs_path.relative_to(infer_root).parts
            if len(parts) < 5:
                raise ValueError(f"Unexpected infer artifact path: {final_cs_path}")
            document_stem, scenario, model, run_name = parts[:4]
            document = f"{document_stem}.md"
            run = parse_run_name(run_name)
            source_matrix_line = None

        records.append(
            {
                "source_matrix_line": source_matrix_line,
                "document": document,
                "document_stem": document_stem,
                "scenario": scenario,
                "model": model,
                "run": run,
                "final_cs_path": relative_final_cs_path,
                "metadata_path": relative_metadata_path,
            }
        )

    return pd.DataFrame(records)


def run_roslyn_metrics_cli(
    repo_root: Path,
    final_cs_paths: list[Path],
    *,
    dotnet_command: str = "dotnet",
) -> list[dict[str, Any]]:
    if shutil.which(dotnet_command) is None:
        raise RuntimeError(
            f"'{dotnet_command}' is not available. Re-enter the project via `nix develop` "
            "so the Roslyn snapshot can run."
        )

    if not final_cs_paths:
        return []

    command = [
        dotnet_command,
        "run",
        "--project",
        str(metrics_project_path(repo_root)),
        "--",
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
        raise RuntimeError("Roslyn metrics CLI returned an empty response")
    return cast(list[dict[str, Any]], json.loads(payload))


def collect_infer_snapshot_metrics(
    repo_root: Path,
    artifact_df: pd.DataFrame | None = None,
    *,
    dotnet_command: str = "dotnet",
) -> pd.DataFrame:
    snapshot_df = (
        artifact_df.copy() if artifact_df is not None else discover_infer_artifacts(repo_root)
    )
    if snapshot_df.empty:
        return pd.DataFrame()

    final_cs_paths = [
        repo_root / path for path in cast(list[str], snapshot_df["final_cs_path"].tolist())
    ]
    metrics_rows = run_roslyn_metrics_cli(repo_root, final_cs_paths, dotnet_command=dotnet_command)
    metrics_by_path = {
        Path(cast(str, row["path"])).resolve().as_posix(): row for row in metrics_rows
    }

    merged_rows: list[dict[str, Any]] = []
    for artifact in snapshot_df.to_dict(orient="records"):
        final_path = (repo_root / cast(str, artifact["final_cs_path"])).resolve().as_posix()
        metrics = metrics_by_path[final_path]
        merged_rows.append(
            {
                **artifact,
                "byte_count": int(metrics["byteCount"]),
                "line_count": int(metrics["lineCount"]),
                "defined_type_count": int(metrics["definedTypeCount"]),
                "parse_error_count": int(metrics["parseErrorCount"]),
                "parse_errors": metrics["parseErrors"],
                "analysis_error": metrics.get("analysisError"),
            }
        )

    return pd.DataFrame(merged_rows).sort_values(
        ["model", "document_stem", "scenario", "run"],
        ignore_index=True,
    )


def _scale(value: float, min_value: float, max_value: float, start: float, end: float) -> float:
    if math.isclose(min_value, max_value):
        return (start + end) / 2
    ratio = (value - min_value) / (max_value - min_value)
    return start + (end - start) * ratio


def _svg_shape(shape: str, x: float, y: float, size: float, fill: str) -> str:
    stroke = "#111827"
    stroke_width = 1.5

    if shape == "square":
        half = size
        return (
            f'<rect x="{x - half:.2f}" y="{y - half:.2f}" width="{half * 2:.2f}" '
            f'height="{half * 2:.2f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"/>'
        )
    if shape == "diamond":
        points = [
            (x, y - size),
            (x + size, y),
            (x, y + size),
            (x - size, y),
        ]
        points_attr = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        return (
            f'<polygon points="{points_attr}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"/>'
        )
    if shape == "triangle":
        points = [
            (x, y - size),
            (x + size, y + size),
            (x - size, y + size),
        ]
        points_attr = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        return (
            f'<polygon points="{points_attr}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"/>'
        )

    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def render_model_snapshot_svg(
    snapshot_df: pd.DataFrame,
    *,
    title: str = "C# snapshot by model",
    x_column: str = "defined_type_count",
    y_column: str = "line_count",
) -> str:
    if snapshot_df.empty:
        raise ValueError("Snapshot dataframe is empty")

    frame = snapshot_df.copy()
    frame[x_column] = frame[x_column].astype(float)
    frame[y_column] = frame[y_column].astype(float)

    documents = sorted(cast(list[str], frame["document_stem"].unique().tolist()))
    models = sorted(cast(list[str], frame["model"].unique().tolist()))
    scenarios = sorted(cast(list[str], frame["scenario"].unique().tolist()))

    colors = [
        "#2563eb",
        "#dc2626",
        "#059669",
        "#d97706",
        "#7c3aed",
        "#0f766e",
        "#db2777",
        "#4f46e5",
        "#65a30d",
        "#c2410c",
    ]
    shapes = ["circle", "diamond", "square", "triangle"]
    color_by_document = {
        document: colors[index % len(colors)] for index, document in enumerate(documents)
    }
    shape_by_scenario = {
        scenario: shapes[index % len(shapes)] for index, scenario in enumerate(scenarios)
    }

    x_values = [float(value) for value in cast(list[Any], frame[x_column].tolist())]
    y_values = [float(value) for value in cast(list[Any], frame[y_column].tolist())]
    x_floor = math.floor(min(x_values))
    x_ceil = math.ceil(max(x_values))
    y_floor = math.floor(min(y_values))
    y_ceil = math.ceil(max(y_values))

    cols = min(2, max(1, len(models)))
    rows = math.ceil(len(models) / cols)
    panel_width = 420
    panel_height = 300
    panel_gap_x = 32
    panel_gap_y = 34
    panel_left = 22
    panel_top = 72
    plot_left = 58
    plot_top = 44
    plot_width = 270
    plot_height = 180

    legend_top = panel_top + rows * panel_height + (rows - 1) * panel_gap_y + 34
    width = panel_left * 2 + cols * panel_width + (cols - 1) * panel_gap_x
    height = legend_top + 110 + max(len(documents) * 22, len(scenarios) * 22)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{panel_left}" y="34" font-size="24" font-weight="700" fill="#111827">'
        f"{html.escape(title)}</text>",
        f'<text x="{panel_left}" y="56" font-size="13" fill="#475569">'
        "Each panel is one model. Color = source document, shape = infer scenario.</text>",
    ]

    for model_index, model in enumerate(models):
        row_index = model_index // cols
        col_index = model_index % cols
        x0 = panel_left + col_index * (panel_width + panel_gap_x)
        y0 = panel_top + row_index * (panel_height + panel_gap_y)
        y1 = y0 + panel_height
        plot_x0 = x0 + plot_left
        plot_y0 = y0 + plot_top
        plot_x1 = plot_x0 + plot_width
        plot_y1 = plot_y0 + plot_height

        parts.extend(
            [
                f'<rect x="{x0}" y="{y0}" width="{panel_width}" height="{panel_height}" '
                'rx="18" fill="#ffffff" stroke="#dbe4f0" stroke-width="1.5"/>',
                (
                    f'<text x="{x0 + 20}" y="{y0 + 28}" font-size="15" '
                    f'font-weight="700" fill="#111827">{html.escape(model)}</text>'
                ),
                (
                    f'<line x1="{plot_x0}" y1="{plot_y1}" x2="{plot_x1}" y2="{plot_y1}" '
                    'stroke="#111827" stroke-width="2"/>'
                ),
                (
                    f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y1}" '
                    'stroke="#111827" stroke-width="2"/>'
                ),
            ]
        )

        for tick_index in range(5):
            x_value = x_floor + ((x_ceil - x_floor) * tick_index / 4 if x_ceil != x_floor else 0)
            x_pos = _scale(x_value, x_floor, x_ceil or 1, plot_x0, plot_x1)
            parts.append(
                f'<line x1="{x_pos:.2f}" y1="{plot_y1}" x2="{x_pos:.2f}" y2="{plot_y1 + 7}" '
                'stroke="#111827" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x_pos:.2f}" y="{plot_y1 + 24}" text-anchor="middle" '
                'font-size="11" fill="#475569">'
                f"{x_value:.1f}</text>"
            )

        for tick_index in range(5):
            y_value = y_floor + ((y_ceil - y_floor) * tick_index / 4 if y_ceil != y_floor else 0)
            y_pos = _scale(y_value, y_floor, y_ceil or 1, plot_y1, plot_y0)
            parts.append(
                f'<line x1="{plot_x0 - 7}" y1="{y_pos:.2f}" x2="{plot_x0}" y2="{y_pos:.2f}" '
                'stroke="#111827" stroke-width="1"/>'
            )
            parts.append(
                f'<line x1="{plot_x0}" y1="{y_pos:.2f}" x2="{plot_x1}" y2="{y_pos:.2f}" '
                'stroke="#e5e7eb" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{plot_x0 - 12}" y="{y_pos + 4:.2f}" text-anchor="end" '
                'font-size="11" fill="#475569">'
                f"{y_value:.1f}</text>"
            )

        parts.extend(
            [
                (
                    f'<text x="{plot_x0}" y="{y1 - 18}" font-size="12" fill="#475569">'
                    "Defined types</text>"
                ),
                (
                    f'<text x="{x0 + 18}" y="{plot_y0 + plot_height / 2:.2f}" '
                    'font-size="12" fill="#475569" transform="rotate(-90 '
                    f'{x0 + 18} {plot_y0 + plot_height / 2:.2f})">'
                    "Source size in lines</text>"
                ),
            ]
        )

        model_frame = cast(pd.DataFrame, frame[frame["model"] == model].copy())
        model_rows = cast(list[dict[str, Any]], model_frame.to_dict(orient="records"))
        for snapshot in model_rows:
            x_value = float(snapshot[x_column])
            y_value = float(snapshot[y_column])
            x_pos = _scale(x_value, x_floor, x_ceil or 1, plot_x0, plot_x1)
            y_pos = _scale(y_value, y_floor, y_ceil or 1, plot_y1, plot_y0)
            document_stem = cast(str, snapshot["document_stem"])
            scenario = cast(str, snapshot["scenario"])
            tooltip = html.escape(
                " | ".join(
                    [
                        document_stem,
                        scenario,
                        cast(str, snapshot["model"]),
                        f"run{int(snapshot['run'])}",
                        f"types={int(snapshot[x_column])}",
                        f"lines={int(snapshot[y_column])}",
                        f"bytes={int(snapshot['byte_count'])}",
                        f"parse_errors={int(snapshot['parse_error_count'])}",
                    ]
                )
            )
            parts.append(
                "<g>"
                f"<title>{tooltip}</title>"
                f"{_svg_shape(
                    shape_by_scenario[scenario],
                    x_pos,
                    y_pos,
                    7,
                    color_by_document[document_stem],
                )}"
                "</g>"
            )

    parts.append(
        f'<text x="{panel_left}" y="{legend_top}" font-size="18" font-weight="700" fill="#111827">'
        "Documents</text>"
    )
    legend_y = legend_top + 26
    for document in documents:
        parts.append(
            f'<rect x="{panel_left}" y="{legend_y - 12}" width="18" height="18" rx="4" '
            f'fill="{color_by_document[document]}" stroke="#111827" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{panel_left + 28}" y="{legend_y + 2}" font-size="13" fill="#374151">'
            f"{html.escape(document)}</text>"
        )
        legend_y += 22

    scenario_legend_x = width // 2 + 24
    parts.append(
        (
            f'<text x="{scenario_legend_x}" y="{legend_top}" font-size="18" '
            'font-weight="700" fill="#111827">Scenarios</text>'
        ),
    )
    legend_y = legend_top + 26
    for scenario in scenarios:
        parts.append(
            _svg_shape(
                shape_by_scenario[scenario],
                scenario_legend_x + 9,
                legend_y - 4,
                7,
                "#94a3b8",
            )
        )
        parts.append(
            f'<text x="{scenario_legend_x + 28}" y="{legend_y + 1}" font-size="13" fill="#374151">'
            f"{html.escape(scenario)}</text>"
        )
        legend_y += 22

    parts.append("</svg>")
    return "\n".join(parts)
