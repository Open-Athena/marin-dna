"""Aggregate separately selected forward and RC scores for issue 420.

Feature IDs are never averaged. Each orientation has its own discovery/validation-
selected feature or raw dimension. Scores are centered within match groups without
using labels, scaled by their discovery+validation standard deviation, and combined
with a fixed equal-weight mean before the test metrics are calculated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl

from analysis import ORIENTATIONS, SPACES, _baseline_metrics, _seed

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMPONENTS = ("forward", "reverse_complement", "equal_weight_mean")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def group_center(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Center scores within each ten-variant match group without labels."""

    assert values.ndim == 1
    assert groups.shape == values.shape
    output = np.empty_like(values, dtype=np.float64)
    for group in dict.fromkeys(groups.tolist()):
        selected = groups == group
        assert selected.sum() == 10
        output[selected] = values[selected] - values[selected].mean()
        assert abs(float(output[selected].mean())) < 1e-12
    assert np.isfinite(output).all()
    return output


def standardized_orientation_mean(
    forward: np.ndarray,
    reverse_complement: np.ndarray,
    groups: np.ndarray,
    non_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return separately standardized scores and their fixed equal-weight mean."""

    assert forward.shape == reverse_complement.shape == groups.shape == non_test.shape
    assert non_test.dtype == np.bool_ and non_test.any() and (~non_test).any()
    forward_centered = group_center(forward, groups)
    reverse_centered = group_center(reverse_complement, groups)
    forward_scale = float(forward_centered[non_test].std(ddof=1))
    reverse_scale = float(reverse_centered[non_test].std(ddof=1))
    assert np.isfinite(forward_scale) and forward_scale > 0
    assert np.isfinite(reverse_scale) and reverse_scale > 0
    forward_z = forward_centered / forward_scale
    reverse_z = reverse_centered / reverse_scale
    aggregate = (forward_z + reverse_z) / 2
    assert np.isfinite(forward_z).all()
    assert np.isfinite(reverse_z).all()
    assert np.isfinite(aggregate).all()
    return forward_z, reverse_z, aggregate, forward_scale, reverse_scale


def _plot(summary: pl.DataFrame, output_dir: Path) -> None:
    subsets = summary["subset"].unique(maintain_order=True).to_list()
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    colors = {
        "forward": "#0072B2",
        "reverse_complement": "#D55E00",
        "equal_weight_mean": "#009E73",
    }
    x = np.arange(len(subsets))
    for axis, space in zip(axes, SPACES, strict=True):
        for component in COMPONENTS:
            values = summary.filter(
                (pl.col("space") == space) & (pl.col("component") == component)
            ).sort(pl.col("subset").replace_strict(subsets, list(range(len(subsets)))))
            assert values.height == len(subsets)
            axis.plot(
                x,
                values["test_average_precision"],
                marker="o",
                linewidth=1.5,
                color=colors[component],
                label=component,
            )
        axis.axhline(0.1, color="grey", linewidth=0.8)
        axis.set_xticks(x, subsets, rotation=55, ha="right")
        axis.set_ylim(0, 0.22)
        axis.set_ylabel("held-out row-level AUPRC")
        axis.set_title(space.upper())
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "exp420: group-centered, non-test-standardized forward/RC aggregation"
    )
    figure.savefig(output_dir / "aggregate.png", dpi=180)
    figure.savefig(output_dir / "aggregate.svg")
    plt.close(figure)


def _markdown(summary: pl.DataFrame) -> str:
    lines = [
        "# exp420 forward/RC score aggregation",
        "",
        "Feature IDs remain separate. Scores were centered within match groups, scaled by discovery+validation standard deviations, and averaged with fixed equal weights before the chr11 + chrX test was inspected.",
        "",
        "| Subset | Space | Forward AP | RC AP | Equal-mean AP | Equal-mean matched effect (95% CI) | permutation p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for subset in summary["subset"].unique(maintain_order=True):
        for space in SPACES:
            selected = summary.filter(
                (pl.col("subset") == subset) & (pl.col("space") == space)
            )
            by_component = {row["component"]: row for row in selected.to_dicts()}
            aggregate = by_component["equal_weight_mean"]
            forward_ap = by_component["forward"]["test_average_precision"]
            reverse_ap = by_component["reverse_complement"]["test_average_precision"]
            aggregate_ap = aggregate["test_average_precision"]
            aggregate_mean = aggregate["test_matched_mean"]
            aggregate_low = aggregate["test_matched_ci95_low"]
            aggregate_high = aggregate["test_matched_ci95_high"]
            aggregate_p = aggregate["test_permutation_pvalue"]
            lines.append(
                f"| {subset} | {space} | "
                f"{forward_ap:.4f} | {reverse_ap:.4f} | "
                f"{aggregate_ap:.4f} | {aggregate_mean:.4f} "
                f"[{aggregate_low:.4f}, {aggregate_high:.4f}] | "
                f"{aggregate_p:.4f} |"
            )
    lines.extend(
        [
            "",
            "Chance AUPRC is 0.1. Group centering uses scores only, never labels. Scaling parameters use discovery+validation rows only.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate_selected_scores(
    *,
    selected_scores_path: Path,
    results_json_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert selected_scores_path.exists() and results_json_path.exists()
    assert not output_dir.exists()
    source_results = json.loads(results_json_path.read_text())
    scores = pl.read_parquet(selected_scores_path)
    required = {
        "row_index",
        "subset",
        "orientation",
        "space",
        "dimension",
        "direction",
        "validation_direction_consistent",
        "split",
        "label",
        "match_group",
        "chrom",
        "pos",
        "ref",
        "alt",
        "score",
    }
    assert required <= set(scores.columns), required - set(scores.columns)
    assert scores.null_count().sum_horizontal().sum() == 0
    assert set(scores["orientation"]) == set(ORIENTATIONS)
    assert set(scores["space"]) == set(SPACES)
    assert scores["validation_direction_consistent"].all()
    assert (
        scores.select(pl.struct("row_index", "orientation", "space").n_unique()).item()
        == scores.height
    )

    summary_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    metadata_columns = [
        "row_index",
        "subset",
        "split",
        "label",
        "match_group",
        "chrom",
        "pos",
        "ref",
        "alt",
    ]
    subsets = sorted(scores["subset"].unique())
    for subset in subsets:
        for space in SPACES:
            selected = scores.filter(
                (pl.col("subset") == subset) & (pl.col("space") == space)
            )
            orientation_frames = {
                orientation: selected.filter(pl.col("orientation") == orientation).sort(
                    "row_index"
                )
                for orientation in ORIENTATIONS
            }
            forward = orientation_frames["forward"]
            reverse = orientation_frames["reverse_complement"]
            assert forward.select(metadata_columns).equals(
                reverse.select(metadata_columns)
            )
            assert (
                forward["dimension"].n_unique() == reverse["dimension"].n_unique() == 1
            )
            assert (
                forward["direction"].n_unique() == reverse["direction"].n_unique() == 1
            )
            metadata = forward.select(metadata_columns)
            groups = metadata["match_group"].to_numpy()
            non_test = metadata["split"].to_numpy() != "test"
            forward_z, reverse_z, aggregate, forward_scale, reverse_scale = (
                standardized_orientation_mean(
                    forward["score"].to_numpy(),
                    reverse["score"].to_numpy(),
                    groups,
                    non_test,
                )
            )
            component_scores = {
                "forward": forward_z,
                "reverse_complement": reverse_z,
                "equal_weight_mean": aggregate,
            }
            test_indices = np.flatnonzero(~non_test)
            for component, values in component_scores.items():
                summary_rows.append(
                    {
                        "subset": subset,
                        "space": space,
                        "component": component,
                        "forward_dimension": forward["dimension"][0],
                        "forward_direction": forward["direction"][0],
                        "reverse_complement_dimension": reverse["dimension"][0],
                        "reverse_complement_direction": reverse["direction"][0],
                        "forward_non_test_scale": forward_scale,
                        "reverse_complement_non_test_scale": reverse_scale,
                        **_baseline_metrics(
                            values,
                            metadata,
                            test_indices,
                            seed=_seed("aggregate", subset, space, component),
                        ),
                    }
                )
            for index, row in enumerate(metadata.iter_rows(named=True)):
                score_rows.append(
                    {
                        **row,
                        "space": space,
                        "forward_dimension": forward["dimension"][0],
                        "reverse_complement_dimension": reverse["dimension"][0],
                        "forward_z": float(forward_z[index]),
                        "reverse_complement_z": float(reverse_z[index]),
                        "equal_weight_mean": float(aggregate[index]),
                    }
                )

    summary = pl.DataFrame(summary_rows).sort(["subset", "space", "component"])
    aggregate_scores = pl.DataFrame(score_rows).sort(["subset", "space", "row_index"])
    assert summary.height == len(subsets) * len(SPACES) * len(COMPONENTS)
    assert aggregate_scores.height * len(ORIENTATIONS) == scores.height
    assert summary.null_count().sum_horizontal().sum() == 0
    assert aggregate_scores.null_count().sum_horizontal().sum() == 0

    output_dir.mkdir(parents=True, exist_ok=False)
    summary.write_parquet(output_dir / "aggregate_summary.parquet", compression="zstd")
    aggregate_scores.write_parquet(
        output_dir / "aggregate_scores.parquet", compression="zstd"
    )
    _plot(summary, output_dir)
    (output_dir / "AGGREGATE.md").write_text(_markdown(summary))
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_commit": source_results["experiment_commit"],
        "source_results_sha256": _sha256(results_json_path),
        "selected_scores_sha256": _sha256(selected_scores_path),
        "method": {
            "feature_id_aggregation": "none",
            "within_group_centering": "unlabeled mean of each ten-variant match group",
            "orientation_scaling": "sample SD on discovery+validation group-centered rows",
            "weights": {"forward": 0.5, "reverse_complement": 0.5},
            "test_chromosomes": ["11", "X"],
        },
    }
    _write_json(output_dir / "aggregate_results.json", result)
    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        **result,
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        },
    }
    _write_json(output_dir / "aggregate_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-scores", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate_selected_scores(
        selected_scores_path=args.selected_scores,
        results_json_path=args.results_json,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
