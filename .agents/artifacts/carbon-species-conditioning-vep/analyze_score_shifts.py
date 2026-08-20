"""Render per-variant Carbon prompt-conditioning score-shift artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from marin_dna_carbon_conditioning_vep.score_shifts import (
    assemble_score_shifts,
    bootstrap_matched_score_shifts,
    summarize_score_shifts,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "snakemake/analysis/carbon_conditioning_vep"
SCORE_ROOT = PROJECT_ROOT / "results/full_development/scores/Carbon-3B"
OUTPUT_ROOT = Path(__file__).resolve().parent

CONDITIONS = ("correct", "far_wrong")
CONDITION_LABELS = {
    "correct": "Correct mammalian",
    "far_wrong": "Far-wrong fungal",
}
CONDITION_COLORS = {"correct": "#0072B2", "far_wrong": "#D55E00"}
CONDITION_MARKERS = {"correct": "o", "far_wrong": "s"}
SUBSETS = (
    "_all_development_",
    "tss_proximal",
    "distal",
    "3_prime_UTR_variant",
    "5_prime_UTR_variant",
    "non_coding_transcript_exon_variant",
    "splicing",
    "missense_variant",
    "synonymous_variant",
    "mature_miRNA_variant",
)
SUBSET_LABELS = {
    "_all_development_": "All development",
    "tss_proximal": "Promoter / TSS",
    "distal": "Distal regulatory",
    "3_prime_UTR_variant": "3′ UTR",
    "5_prime_UTR_variant": "5′ UTR",
    "non_coding_transcript_exon_variant": "Noncoding exon",
    "splicing": "Splicing",
    "missense_variant": "Missense",
    "synonymous_variant": "Synonymous",
    "mature_miRNA_variant": "Mature miRNA",
}


def _write_summary(matched: pd.DataFrame, output: Path) -> None:
    overall = matched.loc[matched["subset"].eq("_all_development_")]
    lines = [
        "# Carbon prompt-conditioning score shifts",
        "",
        "`delta_score` is the conditioned score minus the same variant's untagged score.",
        "The label-separation shift is the mean positive delta minus the mean delta across the nine matched negatives.",
        "Positive label-separation shifts move pathogenic positives upward relative to their matched negatives.",
        "Spread ratios compare the standard deviation of positive deltas with negative deltas.",
        "Intervals are 95% match-group bootstrap intervals from 1,000 seeded draws.",
        "",
        "## All development variants",
        "",
        "| condition | positive mean delta | negative mean delta | label-separation shift | 95% CI | positive SD / negative SD | 95% CI (log2 ratio) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        row = overall.loc[overall["condition"].eq(condition)].iloc[0]
        lines.append(
            f"| {CONDITION_LABELS[condition]} | {row.mean_delta_positive:.6f} | "
            f"{row.mean_delta_negative:.6f} | {row.label_separation_shift:.6f} | "
            f"[{row.label_separation_ci_low:.6f}, {row.label_separation_ci_high:.6f}] | "
            f"{row.std_ratio_positive_negative:.3f} | "
            f"[{row.log2_std_ratio_ci_low:.3f}, {row.log2_std_ratio_ci_high:.3f}] |"
        )
    output.write_text("\n".join(lines) + "\n")


def _render_figure(matched: pd.DataFrame, output_root: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 7.4), sharey=True)
    y_positions = {subset: index for index, subset in enumerate(reversed(SUBSETS))}
    offsets = {"correct": -0.12, "far_wrong": 0.12}

    for axis in axes:
        low_sample_y = y_positions["mature_miRNA_variant"]
        axis.axhspan(low_sample_y - 0.42, low_sample_y + 0.42, color="#eeeeee")
        axis.axvline(0.0, color="#666666", linewidth=1.0, linestyle="--")
        axis.grid(axis="x", color="#dddddd", linewidth=0.7)
        axis.set_axisbelow(True)

    for condition in CONDITIONS:
        condition_rows = matched.loc[matched["condition"].eq(condition)].set_index(
            "subset"
        )
        y = [y_positions[subset] + offsets[condition] for subset in SUBSETS]
        separation = condition_rows.loc[list(SUBSETS), "label_separation_shift"] * 1e3
        separation_low = (
            condition_rows.loc[list(SUBSETS), "label_separation_ci_low"] * 1e3
        )
        separation_high = (
            condition_rows.loc[list(SUBSETS), "label_separation_ci_high"] * 1e3
        )
        axes[0].errorbar(
            separation,
            y,
            xerr=[separation - separation_low, separation_high - separation],
            fmt=CONDITION_MARKERS[condition],
            color=CONDITION_COLORS[condition],
            capsize=2.5,
            markersize=5.5,
            linewidth=1.2,
            label=CONDITION_LABELS[condition],
        )

        spread = condition_rows.loc[list(SUBSETS), "log2_std_ratio_positive_negative"]
        spread_low = condition_rows.loc[list(SUBSETS), "log2_std_ratio_ci_low"]
        spread_high = condition_rows.loc[list(SUBSETS), "log2_std_ratio_ci_high"]
        axes[1].errorbar(
            spread,
            y,
            xerr=[spread - spread_low, spread_high - spread],
            fmt=CONDITION_MARKERS[condition],
            color=CONDITION_COLORS[condition],
            capsize=2.5,
            markersize=5.5,
            linewidth=1.2,
        )

    axes[0].set_yticks(
        [y_positions[subset] for subset in SUBSETS],
        [SUBSET_LABELS[subset] for subset in SUBSETS],
    )
    axes[0].set_xlabel("Positive shift − matched-negative shift\n(×10⁻³ nats/token)")
    axes[0].set_title("Change in label separation")
    axes[1].set_xlabel("log₂(SD positive shift / SD negative shift)")
    axes[1].set_title("Relative variability of score shifts")
    figure.suptitle(
        "Carbon prompt tags alter variant scores unevenly across consequence subsets",
        fontsize=14,
        y=0.985,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.5,
        0.015,
        "Conditioned minus untagged score; points are observed estimates and bars are 95% match-group bootstrap intervals. Gray row has 4 groups.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.88))
    svg_output = output_root / "score_shift_by_subset.svg"
    figure.savefig(svg_output)
    svg_lines = (line.rstrip() for line in svg_output.read_text().splitlines())
    svg_output.write_text("\n".join(svg_lines) + "\n")
    figure.savefig(output_root / "score_shift_by_subset.png", dpi=180)
    plt.close(figure)


def main() -> None:
    untagged = pd.read_parquet(SCORE_ROOT / "untagged.parquet")
    tagged = {
        condition: pd.read_parquet(SCORE_ROOT / f"{condition}.parquet")
        for condition in CONDITIONS
    }
    shifts = assemble_score_shifts(untagged, tagged)
    by_label = summarize_score_shifts(shifts)
    matched = bootstrap_matched_score_shifts(
        shifts,
        n_bootstrap=1000,
        bootstrap_seed=486,
        min_groups=30,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    by_label.to_parquet(
        OUTPUT_ROOT / "score_shift_by_subset_label.parquet", index=False
    )
    by_label.to_csv(
        OUTPUT_ROOT / "score_shift_by_subset_label.tsv", sep="\t", index=False
    )
    matched.to_parquet(
        OUTPUT_ROOT / "score_shift_matched_bootstrap.parquet", index=False
    )
    matched.to_csv(
        OUTPUT_ROOT / "score_shift_matched_bootstrap.tsv", sep="\t", index=False
    )
    _write_summary(matched, OUTPUT_ROOT / "score_shift_summary.md")
    _render_figure(matched, OUTPUT_ROOT)


if __name__ == "__main__":
    main()
