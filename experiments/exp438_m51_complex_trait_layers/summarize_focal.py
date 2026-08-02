"""Summarize and plot experiment 438's verified focal association inventory."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from analyze_focal import FDR_THRESHOLD, PRIMARY_RESPONSE, RESPONSES
from common import ISSUE, TRAINING_TOKENS, assert_commit, sha256_file, write_json

TARGET_ORDER = (
    "overall",
    "missense_variant",
    "synonymous_variant",
    "splicing",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "tss_proximal",
    "non_coding_transcript_exon_variant",
    "distal",
)
TARGET_LABELS = {
    "overall": "Overall",
    "missense_variant": "Missense",
    "synonymous_variant": "Synonymous",
    "splicing": "Splicing",
    "5_prime_UTR_variant": "5′ UTR",
    "3_prime_UTR_variant": "3′ UTR",
    "tss_proximal": "TSS-proximal",
    "non_coding_transcript_exon_variant": "ncRNA exon",
    "distal": "Distal",
}
ORIENTATION_LABELS = {"forward": "FWD", "reverse_complement": "RC"}
RESPONSE_LABELS = {"abs_delta": "|Δ activation|", "delta": "signed Δ activation"}


def verify_input_artifacts(root: Path, manifest: dict[str, Any]) -> None:
    for relative, expected in manifest["artifacts"].items():
        path = root / relative
        assert path.is_file(), path
        assert path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]


def read_families(root: Path) -> pl.DataFrame:
    paths = sorted(root.glob("families/**/*.parquet"))
    assert len(paths) == 12, len(paths)
    frame = pl.read_parquet(paths)
    assert set(frame["response"].unique().to_list()) == set(RESPONSES)
    assert set(frame["block"].unique().to_list()) == {1, 10, 19}
    assert set(frame["orientation"].unique().to_list()) == {
        "forward",
        "reverse_complement",
    }
    assert (
        frame.select(
            pl.struct(
                "arm", "orientation", "response", "target", "feature_id"
            ).n_unique()
        ).item()
        == frame.height
    )
    inferential = frame.filter(pl.col("inferential"))
    descriptive = frame.filter(~pl.col("inferential"))
    assert inferential.filter(~pl.col("minimum_q").is_finite()).is_empty()
    assert descriptive.filter(pl.col("minimum_q").is_finite()).is_empty()
    return frame


def target_summary(frame: pl.DataFrame) -> pl.DataFrame:
    keys = [
        "arm",
        "block",
        "orientation",
        "response",
        "target_kind",
        "target",
        "inferential",
    ]
    return (
        frame.group_by(keys)
        .agg(
            pl.len().alias("eligible_features"),
            (pl.col("welch_q") <= FDR_THRESHOLD).sum().alias("welch_discoveries"),
            (pl.col("mann_whitney_q") <= FDR_THRESHOLD)
            .sum()
            .alias("mann_whitney_discoveries"),
            (
                (pl.col("welch_q") <= FDR_THRESHOLD)
                & (pl.col("mann_whitney_q") <= FDR_THRESHOLD)
            )
            .sum()
            .alias("both_test_discoveries"),
            pl.col("best_auprc").max().alias("best_auprc"),
            pl.col("prevalence").first().alias("prevalence"),
        )
        .with_columns(
            (pl.col("best_auprc") / pl.col("prevalence")).alias("best_auprc_lift"),
            (pl.col("both_test_discoveries") / pl.col("eligible_features")).alias(
                "both_test_discovery_fraction"
            ),
        )
        .sort(keys)
    )


def top_features(frame: pl.DataFrame, *, per_target: int = 5) -> pl.DataFrame:
    keys = ["arm", "orientation", "response", "target"]
    return (
        frame.sort(
            [*keys, "best_auprc", "minimum_q", "feature_id"],
            descending=[False, False, False, False, True, False, False],
            nulls_last=True,
        )
        .group_by(keys, maintain_order=True)
        .head(per_target)
        .select(
            *keys,
            "block",
            "target_kind",
            "inferential",
            "feature_id",
            "best_auprc",
            "best_auprc_direction",
            "prevalence",
            "minimum_q",
            "welch_q",
            "mann_whitney_q",
            "standardized_mean_difference",
            "rank_biserial",
            "nonzero_support",
            "n",
            "n_positive",
        )
    )


def layer_winners(summary: pl.DataFrame) -> pl.DataFrame:
    keys = ["orientation", "response", "target"]
    return (
        summary.sort(
            [*keys, "best_auprc", "block"],
            descending=[False, False, False, True, False],
        )
        .group_by(keys, maintain_order=True)
        .head(1)
        .rename({"block": "winning_block", "arm": "winning_arm"})
        .sort(keys)
    )


def strand_overlap(frame: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["arm", "block", "response", "target_kind", "target"]
    for key, group in frame.filter(pl.col("inferential")).group_by(
        keys, maintain_order=True
    ):
        forward = group.filter(pl.col("orientation") == "forward")
        reverse = group.filter(pl.col("orientation") == "reverse_complement")
        forward_sig = set(
            forward.filter(
                (pl.col("welch_q") <= FDR_THRESHOLD)
                & (pl.col("mann_whitney_q") <= FDR_THRESHOLD)
            )["feature_id"].to_list()
        )
        reverse_sig = set(
            reverse.filter(
                (pl.col("welch_q") <= FDR_THRESHOLD)
                & (pl.col("mann_whitney_q") <= FDR_THRESHOLD)
            )["feature_id"].to_list()
        )
        union = forward_sig | reverse_sig
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "forward_significant": len(forward_sig),
                "reverse_complement_significant": len(reverse_sig),
                "significant_overlap": len(forward_sig & reverse_sig),
                "significant_union": len(union),
                "significant_jaccard": len(forward_sig & reverse_sig) / len(union)
                if union
                else None,
            }
        )
    return pl.DataFrame(rows).sort(keys)


def _heatmap_matrix(
    summary: pl.DataFrame, *, orientation: str, response: str, value: str
) -> np.ndarray:
    selected = summary.filter(
        (pl.col("orientation") == orientation) & (pl.col("response") == response)
    )
    rows: list[list[float]] = []
    for target in TARGET_ORDER:
        target_rows = selected.filter(pl.col("target") == target).sort("block")
        assert target_rows["block"].to_list() == [1, 10, 19]
        rows.append([float(item) for item in target_rows[value].to_list()])
    return np.asarray(rows)


def plot_heatmaps(summary: pl.DataFrame, output_dir: Path) -> list[Path]:
    sns.set_theme(context="notebook", style="white", font_scale=0.9)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    for row, response in enumerate(RESPONSES):
        for column, orientation in enumerate(("forward", "reverse_complement")):
            matrix = _heatmap_matrix(
                summary,
                orientation=orientation,
                response=response,
                value="best_auprc_lift",
            )
            axis = axes[row, column]
            sns.heatmap(
                matrix,
                ax=axis,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                vmin=1.0,
                cbar_kws={"label": "best single-feature AUPRC / prevalence"},
                xticklabels=["Block 1", "Block 10", "Block 19"],
                yticklabels=[TARGET_LABELS[target] for target in TARGET_ORDER],
            )
            axis.set_title(
                f"{RESPONSE_LABELS[response]} · {ORIENTATION_LABELS[orientation]}"
            )
            axis.set_xlabel("")
            axis.set_ylabel("")
    fig.suptitle("Complex-trait label association by SAE layer", fontsize=15)
    paths = [
        output_dir / "layer_target_auprc_lift.png",
        output_dir / "layer_target_auprc_lift.svg",
    ]
    fig.savefig(paths[0], dpi=180, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def results_markdown(
    summary: pl.DataFrame, winners: pl.DataFrame, manifest: dict[str, Any]
) -> str:
    primary = winners.filter(pl.col("response") == PRIMARY_RESPONSE)
    win_counts = (
        primary.group_by("winning_block")
        .agg(pl.len().alias("wins"))
        .sort("winning_block")
    )
    count_text = ", ".join(
        f"block {row['winning_block']}: {row['wins']}"
        for row in win_counts.iter_rows(named=True)
    )
    overall = summary.filter(
        (pl.col("target") == "overall") & (pl.col("response") == PRIMARY_RESPONSE)
    ).sort(["orientation", "block"])
    lines = [
        "# Experiment 438 focal SAE layer comparison",
        "",
        f"Run: `{manifest['run_id']}`  ",
        f"Experiment commit: `{manifest['experiment_commit']}`  ",
        "",
        "## Primary descriptive readout",
        "",
        (
            "The table reports the best single-feature AUPRC in each preregistered layer. "
            "All targets have 10% label prevalence, so random-baseline AUPRC is 0.10."
        ),
        "",
        "| orientation | block | best AUPRC | lift over prevalence | both-test discoveries | eligible features |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall.iter_rows(named=True):
        lines.append(
            f"| {ORIENTATION_LABELS[row['orientation']]} | {row['block']} | "
            f"{row['best_auprc']:.4f} | {row['best_auprc_lift']:.2f}× | "
            f"{row['both_test_discoveries']} | {row['eligible_features']} |"
        )
    lines.extend(
        [
            "",
            (
                "Across all target × orientation cells for the primary |Δ activation| response, "
                f"the winning-layer counts are: {count_text}."
            ),
            "",
            (
                "These maxima are descriptive feature-discovery summaries, not held-out predictive estimates. "
                "Inferential rows use Welch and Mann–Whitney tests with BH correction separately within "
                "each layer × orientation × response family; splicing and synonymous subsets are descriptive only."
            ),
            "",
            "FWD and RC remain separate. Signed Δ is a sensitivity analysis; |Δ activation| is primary.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize(*, association_root: Path, output_dir: Path) -> dict[str, Any]:
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    started = time.monotonic()
    manifest = json.loads((association_root / "manifest.json").read_text())
    assert manifest["issue"] == ISSUE
    assert manifest["protocol"]["training_tokens_per_sae"] == TRAINING_TOKENS
    verify_input_artifacts(association_root, manifest)
    frame = read_families(association_root)
    output_dir.mkdir(parents=True)

    summary = target_summary(frame)
    top = top_features(frame)
    winners = layer_winners(summary)
    overlap = strand_overlap(frame)
    tables = {
        "target_summary.parquet": summary,
        "top_features.parquet": top,
        "layer_winners.parquet": winners,
        "strand_overlap.parquet": overlap,
    }
    for name, table in tables.items():
        table.write_parquet(output_dir / name, compression="zstd")
    plot_paths = plot_heatmaps(summary, output_dir)
    markdown = results_markdown(summary, winners, manifest)
    (output_dir / "RESULTS.md").write_text(markdown)

    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": manifest["run_id"],
        "experiment_commit": experiment_commit,
        "elapsed_seconds": time.monotonic() - started,
        "platform": platform.platform(),
        "input": {
            "association_manifest_sha256": sha256_file(
                association_root / "manifest.json"
            ),
            "association_rows": frame.height,
        },
        "protocol": {
            "layer_comparison": "blocks 1, 10, and 19 reported separately",
            "primary_response": PRIMARY_RESPONSE,
            "orientations_aggregated": False,
            "layer_maxima_are_descriptive": True,
        },
        "summary": {
            "target_rows": summary.height,
            "top_feature_rows": top.height,
            "winner_rows": winners.height,
            "strand_overlap_rows": overlap.height,
        },
        "artifacts": {},
    }
    artifact_paths = [
        *[output_dir / name for name in tables],
        *plot_paths,
        output_dir / "RESULTS.md",
    ]
    for path in artifact_paths:
        result["artifacts"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json(output_dir / "results.json", result)
    result["artifacts"]["results.json"] = {
        "bytes": (output_dir / "results.json").stat().st_size,
        "sha256": sha256_file(output_dir / "results.json"),
    }
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--association-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        association_root=args.association_root, output_dir=args.output_dir
    )
    print(json.dumps(result["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
