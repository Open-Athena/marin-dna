"""Summarize and plot issue 436's verified focal association inventory."""

from __future__ import annotations

import argparse
import json
import math
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
import scipy
from scipy.stats import spearmanr

from analyze_focal import (
    FDR_THRESHOLD,
    PRIMARY_RESPONSES,
    RESPONSES,
)
from extract_focal import BUDGETS, ISSUE, sha256_file, write_json
from train import assert_commit

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
RESPONSE_LABELS = {
    "abs_delta": "|Δ activation|",
    "delta": "signed Δ activation",
}
ORIENTATION_LABELS = {
    "forward": "FWD",
    "reverse_complement": "RC",
}


def verify_input_artifacts(root: Path, manifest: dict[str, Any]) -> None:
    for relative, expected in manifest["artifacts"].items():
        path = root / relative
        assert path.is_file(), path
        assert path.stat().st_size == expected["bytes"], path
        assert sha256_file(path) == expected["sha256"], path


def read_primary_families(root: Path) -> pl.DataFrame:
    paths = sorted(
        [
            *root.glob("families/**/abs_delta.parquet"),
            *root.glob("families/**/delta.parquet"),
        ]
    )
    assert len(paths) == 24, len(paths)
    frame = pl.read_parquet(paths)
    assert set(frame["response"].unique().to_list()) == PRIMARY_RESPONSES
    assert (
        frame.select(
            pl.struct(
                "arm",
                "orientation",
                "response",
                "target",
                "feature_id",
            ).n_unique()
        ).item()
        == frame.height
    )
    assert frame.filter(~pl.col("minimum_q").is_finite()).is_empty()
    assert frame.filter(
        (pl.col("minimum_q") < 0) | (pl.col("minimum_q") > 1)
    ).is_empty()
    return frame


def target_summary(frame: pl.DataFrame) -> pl.DataFrame:
    keys = [
        "arm",
        "block",
        "budget",
        "orientation",
        "response",
        "target_kind",
        "target",
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
            (pl.col("minimum_q") <= FDR_THRESHOLD).sum().alias("union_discoveries"),
            pl.col("best_auprc").max().alias("best_auprc"),
            pl.col("minimum_q").min().alias("minimum_q"),
            pl.col("prevalence").first().alias("prevalence"),
        )
        .sort(keys)
    )


def top_features(frame: pl.DataFrame, *, per_family_target: int = 3) -> pl.DataFrame:
    keys = ["arm", "orientation", "response", "target"]
    columns = [
        *keys,
        "block",
        "budget",
        "target_kind",
        "feature_id",
        "best_auprc",
        "best_auprc_direction",
        "minimum_q",
        "welch_q",
        "mann_whitney_q",
        "standardized_mean_difference",
        "rank_biserial",
        "nonzero_support",
        "n",
        "n_positive",
    ]
    return (
        frame.filter(pl.col("minimum_q") <= FDR_THRESHOLD)
        .sort(
            [*keys, "best_auprc", "minimum_q", "feature_id"],
            descending=[False, False, False, False, True, False, False],
        )
        .group_by(keys, maintain_order=True)
        .head(per_family_target)
        .select(columns)
    )


def _sets_and_join(
    group: pl.DataFrame,
    *,
    left_filter: pl.Expr,
    right_filter: pl.Expr,
    left_name: str,
    right_name: str,
) -> tuple[set[int], set[int], pl.DataFrame]:
    left = group.filter(left_filter)
    right = group.filter(right_filter)
    assert left.height > 0 and right.height > 0
    left_sig = set(
        left.filter(pl.col("minimum_q") <= FDR_THRESHOLD)["feature_id"].to_list()
    )
    right_sig = set(
        right.filter(pl.col("minimum_q") <= FDR_THRESHOLD)["feature_id"].to_list()
    )
    joined = left.select(
        "feature_id",
        pl.col("rank_biserial").alias(left_name),
    ).join(
        right.select(
            "feature_id",
            pl.col("rank_biserial").alias(right_name),
        ),
        on="feature_id",
        how="inner",
    )
    return left_sig, right_sig, joined


def _overlap_metrics(
    left_sig: set[int],
    right_sig: set[int],
    joined: pl.DataFrame,
    *,
    left_effect: str,
    right_effect: str,
) -> dict[str, Any]:
    intersection = left_sig & right_sig
    union = left_sig | right_sig
    overlap = joined.filter(pl.col("feature_id").is_in(sorted(intersection)))
    if joined.height >= 2:
        effect_spearman = float(
            spearmanr(
                joined[left_effect].to_numpy(),
                joined[right_effect].to_numpy(),
            ).statistic
        )
    else:
        effect_spearman = math.nan
    if overlap.height:
        sign_concordance = float(
            np.mean(
                np.sign(overlap[left_effect].to_numpy())
                == np.sign(overlap[right_effect].to_numpy())
            )
        )
    else:
        sign_concordance = math.nan
    return {
        "left_significant": len(left_sig),
        "right_significant": len(right_sig),
        "significant_overlap": len(intersection),
        "significant_union": len(union),
        "significant_jaccard": len(intersection) / len(union) if union else math.nan,
        "shared_eligible_features": joined.height,
        "effect_spearman_shared_eligible": effect_spearman,
        "effect_sign_concordance_overlap": sign_concordance,
    }


def strand_overlap(frame: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["arm", "block", "budget", "response", "target_kind", "target"]
    for key, group in frame.group_by(keys, maintain_order=True):
        forward_sig, reverse_sig, joined = _sets_and_join(
            group,
            left_filter=pl.col("orientation") == "forward",
            right_filter=pl.col("orientation") == "reverse_complement",
            left_name="forward_effect",
            right_name="reverse_complement_effect",
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                **_overlap_metrics(
                    forward_sig,
                    reverse_sig,
                    joined,
                    left_effect="forward_effect",
                    right_effect="reverse_complement_effect",
                ),
            }
        )
    return pl.DataFrame(rows).sort(keys)


def budget_overlap(frame: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["block", "orientation", "response", "target_kind", "target"]
    for key, group in frame.group_by(keys, maintain_order=True):
        five_sig, twenty_five_sig, joined = _sets_and_join(
            group,
            left_filter=pl.col("budget") == BUDGETS[0],
            right_filter=pl.col("budget") == BUDGETS[1],
            left_name="five_m_effect",
            right_name="twenty_five_m_effect",
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "comparison_boundary": (
                    "same initialized SAE feature slot along one training trajectory; "
                    "slot identity does not prove semantic identity"
                ),
                **_overlap_metrics(
                    five_sig,
                    twenty_five_sig,
                    joined,
                    left_effect="five_m_effect",
                    right_effect="twenty_five_m_effect",
                ),
            }
        )
    return pl.DataFrame(rows).sort(keys)


def final_layer_feature_recurrence(frame: pl.DataFrame) -> pl.DataFrame:
    selected = frame.filter(
        (pl.col("block") == 19) & (pl.col("target") == "overall")
    ).with_columns(
        pl.concat_str(
            [
                pl.col("budget").cast(pl.String),
                pl.col("orientation"),
                pl.col("response"),
            ],
            separator=":",
        ).alias("family"),
        (
            (pl.col("welch_q") <= FDR_THRESHOLD)
            & (pl.col("mann_whitney_q") <= FDR_THRESHOLD)
        ).alias("passes_both"),
        (pl.col("minimum_q") <= FDR_THRESHOLD).alias("passes_either"),
    )
    assert selected.select(pl.col("family").n_unique()).item() == 8
    return (
        selected.group_by("feature_id")
        .agg(
            pl.len().alias("eligible_families"),
            pl.col("passes_either").sum().alias("significant_families"),
            pl.col("passes_both").sum().alias("both_test_families"),
            pl.col("best_auprc").max().alias("maximum_best_auprc"),
            pl.col("minimum_q").min().alias("minimum_q"),
            pl.col("family")
            .filter(pl.col("passes_either"))
            .sort()
            .alias("significant_family_labels"),
        )
        .sort(
            [
                "significant_families",
                "both_test_families",
                "maximum_best_auprc",
                "feature_id",
            ],
            descending=[True, True, True, False],
        )
    )


def strongest_target_per_family(summary: pl.DataFrame) -> pl.DataFrame:
    keys = ["arm", "orientation", "response"]
    return (
        summary.sort(
            [*keys, "best_auprc", "minimum_q", "target"],
            descending=[False, False, False, True, False, False],
        )
        .group_by(keys, maintain_order=True)
        .head(1)
    )


def _line_style(budget: int) -> tuple[str, str]:
    if budget == BUDGETS[0]:
        return "o", "--"
    assert budget == BUDGETS[1]
    return "s", "-"


def plot_overall(summary: pl.DataFrame, output_dir: Path) -> list[Path]:
    overall = summary.filter(pl.col("target") == "overall")
    assert overall.height == 24
    colors = {
        "forward": "#2563eb",
        "reverse_complement": "#dc2626",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    for axis, response in zip(axes, ("abs_delta", "delta"), strict=True):
        response_frame = overall.filter(pl.col("response") == response)
        for orientation in ("forward", "reverse_complement"):
            for budget in BUDGETS:
                group = response_frame.filter(
                    (pl.col("orientation") == orientation)
                    & (pl.col("budget") == budget)
                ).sort("block")
                assert group["block"].to_list() == [1, 10, 19]
                marker, linestyle = _line_style(budget)
                axis.plot(
                    group["block"],
                    group["best_auprc"],
                    color=colors[orientation],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2,
                    markersize=6,
                    label=(
                        f"{ORIENTATION_LABELS[orientation]}, {budget // 1_000_000}M"
                    ),
                )
        axis.axhline(
            0.10,
            color="#6b7280",
            linewidth=1.2,
            linestyle=":",
            label="prevalence = 0.10",
        )
        axis.set_title(RESPONSE_LABELS[response])
        axis.set_xlabel("Reported transformer block")
        axis.set_xticks([1, 10, 19])
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Best individual-feature AUPRC (descriptive)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Mendelian label signal in paired SAE features rises sharply at the final layer",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.015,
        "FWD and RC are reported separately; BH correction is within each declared family.",
        ha="center",
        fontsize=8.5,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    paths = [
        output_dir / "overall_best_auprc.svg",
        output_dir / "overall_best_auprc.png",
    ]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_final_layer_targets(
    summary: pl.DataFrame,
    output_dir: Path,
) -> list[Path]:
    selected = summary.filter(pl.col("block") == 19)
    assert selected.height == 2 * 2 * 2 * len(TARGET_ORDER)
    colors = {
        "forward": "#2563eb",
        "reverse_complement": "#dc2626",
    }
    x = np.arange(len(TARGET_ORDER))
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.3), sharex=True, sharey=True)
    for axis, response in zip(axes, ("abs_delta", "delta"), strict=True):
        response_frame = selected.filter(pl.col("response") == response)
        for orientation in ("forward", "reverse_complement"):
            for budget in BUDGETS:
                group = (
                    response_frame.filter(
                        (pl.col("orientation") == orientation)
                        & (pl.col("budget") == budget)
                    )
                    .with_columns(
                        pl.col("target")
                        .replace_strict(
                            {
                                target: index
                                for index, target in enumerate(TARGET_ORDER)
                            },
                            return_dtype=pl.Int8,
                        )
                        .alias("target_order")
                    )
                    .sort("target_order")
                )
                assert group["target"].to_list() == list(TARGET_ORDER)
                marker, linestyle = _line_style(budget)
                axis.plot(
                    x,
                    group["best_auprc"],
                    color=colors[orientation],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.8,
                    markersize=5,
                    label=(
                        f"{ORIENTATION_LABELS[orientation]}, {budget // 1_000_000}M"
                    ),
                )
        axis.axhline(0.10, color="#6b7280", linewidth=1.1, linestyle=":")
        axis.set_title(RESPONSE_LABELS[response])
        axis.set_ylabel("Best feature AUPRC")
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    axes[-1].set_xticks(
        x,
        [TARGET_LABELS[target] for target in TARGET_ORDER],
        rotation=28,
        ha="right",
    )
    fig.suptitle(
        "Final-layer paired features associate with label across coding and non-coding domains",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.01,
        "AUPRC is descriptive; inferential claims use complete-family BH-adjusted tests.",
        ha="center",
        fontsize=8.5,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    paths = [
        output_dir / "final_layer_target_auprc.svg",
        output_dir / "final_layer_target_auprc.png",
    ]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        if value == 0:
            return "0"
        if abs(value) < 0.001:
            return f"{value:.2e}"
        return f"{value:.3f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def markdown_table(frame: pl.DataFrame, columns: list[str]) -> str:
    labels = [column.replace("_", " ") for column in columns]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.select(columns).to_dicts():
        lines.append(
            "| " + " | ".join(_format_number(row[column]) for column in columns) + " |"
        )
    return "\n".join(lines)


def render_results_markdown(
    summary: pl.DataFrame,
    strand: pl.DataFrame,
    recurrence: pl.DataFrame,
    strongest: pl.DataFrame,
) -> str:
    overall = summary.filter(pl.col("target") == "overall")
    early_max = overall.filter(pl.col("block") < 19)["best_auprc"].max()
    final_min = overall.filter(pl.col("block") == 19)["best_auprc"].min()
    final_max = overall.filter(pl.col("block") == 19)["best_auprc"].max()
    final_strand = strand.filter(
        (pl.col("block") == 19) & (pl.col("target") == "overall")
    )
    robust = recurrence.head(10)
    final_overall = overall.filter(pl.col("block") == 19).sort(
        ["budget", "orientation", "response"]
    )
    final_strongest = strongest.filter(pl.col("block") == 19).sort(
        ["budget", "orientation", "response"]
    )
    return f"""# Issue 436 focal Mendelian association results

## Takeaway

- Overall-label single-feature AUPRC is at most {early_max:.3f} in blocks 1/10,
  but ranges from {final_min:.3f} to {final_max:.3f} in block 19.
- Every final-layer primary family contains complete-family BH discoveries
  passing both Welch and Mann–Whitney tests.
- Final-layer FWD/RC significant-feature Jaccard ranges from
  {final_strand["significant_jaccard"].min():.3f} to
  {final_strand["significant_jaccard"].max():.3f}; shared-feature effect
  Spearman ranges from
  {final_strand["effect_spearman_shared_eligible"].min():.3f} to
  {final_strand["effect_spearman_shared_eligible"].max():.3f}.
- The strongest subset-specific results are non-coding-transcript exon variants
  in the final layer, while early/middle-layer maxima are often UTR,
  synonymous, or distal subsets.

## Final-layer overall label

{
        markdown_table(
            final_overall,
            [
                "budget",
                "orientation",
                "response",
                "eligible_features",
                "welch_discoveries",
                "mann_whitney_discoveries",
                "both_test_discoveries",
                "union_discoveries",
                "best_auprc",
            ],
        )
    }

## Final-layer FWD/RC agreement

{
        markdown_table(
            final_strand.sort(["budget", "response"]),
            [
                "budget",
                "response",
                "left_significant",
                "right_significant",
                "significant_overlap",
                "significant_jaccard",
                "effect_spearman_shared_eligible",
                "effect_sign_concordance_overlap",
            ],
        )
    }

## Strongest target in each final-layer family

{
        markdown_table(
            final_strongest,
            [
                "budget",
                "orientation",
                "response",
                "target",
                "union_discoveries",
                "best_auprc",
                "minimum_q",
            ],
        )
    }

## Recurrent final-layer feature slots for overall label

{
        markdown_table(
            robust,
            [
                "feature_id",
                "eligible_families",
                "significant_families",
                "both_test_families",
                "maximum_best_auprc",
            ],
        )
    }

Feature IDs are comparable across 5M/25M only as slots along the same initialized
training trajectory; slot identity alone does not prove semantic identity.
FWD and RC are never pooled in these results.
"""


def summarize(
    *,
    associations_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert associations_root.is_dir() and not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    started = time.monotonic()
    manifest = json.loads((associations_root / "manifest.json").read_text())
    verify_input_artifacts(associations_root, manifest)
    assert set(manifest["protocol"]["primary_responses"]) == PRIMARY_RESPONSES
    assert set(manifest["protocol"]["responses_in_execution_order"]) == set(RESPONSES)

    frame = read_primary_families(associations_root)
    summary = target_summary(frame)
    tops = top_features(frame)
    strand = strand_overlap(frame)
    budget = budget_overlap(frame)
    recurrence = final_layer_feature_recurrence(frame)
    strongest = strongest_target_per_family(summary)

    output_dir.mkdir(parents=True)
    outputs = {
        "target_summary.parquet": summary,
        "top_features.parquet": tops,
        "strand_overlap.parquet": strand,
        "budget_overlap.parquet": budget,
        "final_layer_feature_recurrence.parquet": recurrence,
        "strongest_target_per_family.parquet": strongest,
    }
    for filename, table in outputs.items():
        table.write_parquet(output_dir / filename, compression="zstd")

    plot_paths = [
        *plot_overall(summary, output_dir),
        *plot_final_layer_targets(summary, output_dir),
    ]
    results_markdown = render_results_markdown(summary, strand, recurrence, strongest)
    (output_dir / "RESULTS.md").write_text(results_markdown)

    artifacts: dict[str, Any] = {}
    artifact_paths = [
        *(output_dir / filename for filename in outputs),
        *plot_paths,
        output_dir / "RESULTS.md",
    ]
    for path in artifact_paths:
        artifacts[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", ""),
        "experiment_commit": experiment_commit,
        "elapsed_seconds": time.monotonic() - started,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "scipy": scipy.__version__,
        "input": {
            "association_run_id": manifest["run_id"],
            "association_experiment_commit": manifest["experiment_commit"],
            "association_manifest_sha256": sha256_file(
                associations_root / "manifest.json"
            ),
            "verified_artifacts": len(manifest["artifacts"]),
            "primary_rows": frame.height,
        },
        "protocol": {
            "primary_responses": sorted(PRIMARY_RESPONSES),
            "fdr_threshold": FDR_THRESHOLD,
            "orientations_never_pooled": True,
            "strand_overlap_scope": "within same SAE arm, response, and target",
            "budget_overlap_boundary": (
                "same feature slot along one initialized training trajectory"
            ),
            "plot_metric": "best individual-feature AUPRC, descriptive",
        },
        "artifacts": artifacts,
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
    parser.add_argument("--associations-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        associations_root=args.associations_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
