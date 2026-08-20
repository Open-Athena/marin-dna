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
    normalize_prompt_mean_scores,
    summarize_score_shifts,
)
from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "snakemake/analysis/carbon_conditioning_vep"
SCORE_ROOT = PROJECT_ROOT / "results/full_development/scores/Carbon-3B"
METRIC_ROOT = PROJECT_ROOT / "results/full_development/metrics/Carbon-3B"
OUTPUT_ROOT = Path(__file__).resolve().parent

APPROACHES = ("untagged", "correct", "far_wrong")
DNA_TARGET_TOKENS = 8_190 // 6
PREFIX_TOKEN_COUNTS = {"untagged": 1, "correct": 9, "far_wrong": 6}
CONDITIONS = ("correct", "far_wrong")
APPROACH_LABELS = {
    "untagged": "Untagged",
    "correct": "Correct mammalian",
    "far_wrong": "Far-wrong fungal",
}
CONDITION_LABELS = {
    "correct": "Correct mammalian − untagged",
    "far_wrong": "Far-wrong fungal − untagged",
}
CONDITION_COLORS = {"correct": "#0072B2", "far_wrong": "#D55E00"}
CONDITION_MARKERS = {"correct": "o", "far_wrong": "s"}
PAIRWISE_COMPARISONS = (
    ("untagged", "correct"),
    ("untagged", "far_wrong"),
    ("correct", "far_wrong"),
)
EXCLUDED_SUBSETS = frozenset({"mature_miRNA_variant"})
SUBSETS = (
    "_all_analyzed_",
    "tss_proximal",
    "distal",
    "3_prime_UTR_variant",
    "5_prime_UTR_variant",
    "non_coding_transcript_exon_variant",
    "splicing",
    "missense_variant",
    "synonymous_variant",
)
SUBSET_LABELS = {
    "_all_analyzed_": "All analyzed",
    "tss_proximal": "Promoter / TSS",
    "distal": "Distal regulatory",
    "3_prime_UTR_variant": "3′ UTR",
    "5_prime_UTR_variant": "5′ UTR",
    "non_coding_transcript_exon_variant": "Noncoding exon",
    "splicing": "Splicing",
    "missense_variant": "Missense",
    "synonymous_variant": "Synonymous",
}


def _assemble_score_matrix(
    untagged: pd.DataFrame, shifts: pd.DataFrame
) -> pd.DataFrame:
    """Build an aligned three-approach score matrix without mature-miRNA rows."""
    columns = ["variant_id", "subset", "match_group", "label", "score"]
    matrix = untagged.loc[:, columns].rename(columns={"score": "untagged"})
    matrix = matrix.loc[~matrix["subset"].isin(EXCLUDED_SUBSETS)].reset_index(drop=True)
    assert len(matrix) == 16_100, "expected 16,100 non-miRNA variants"
    assert matrix["match_group"].nunique() == 1_610, (
        "expected 1,610 non-miRNA match groups"
    )
    for condition in CONDITIONS:
        condition_shifts = shifts.loc[shifts["condition"].eq(condition)].reset_index(
            drop=True
        )
        pd.testing.assert_series_equal(
            condition_shifts["variant_id"],
            matrix["variant_id"],
            check_names=False,
        )
        matrix[condition] = matrix["untagged"] + condition_shifts["delta_score"]
    return matrix


def _summarize_approaches(score_matrix: pd.DataFrame) -> pd.DataFrame:
    """Summarize all three retained approaches on the non-miRNA population."""
    rows: list[dict[str, object]] = []
    for approach in APPROACHES:
        metrics = pd.read_parquet(METRIC_ROOT / f"{approach}.parquet")
        macro = metrics.loc[metrics["subset"].eq("_macro_avg_")].iloc[0]
        positive = score_matrix.loc[score_matrix["label"], approach]
        negative = score_matrix.loc[~score_matrix["label"], approach]
        rows.append(
            {
                "approach": approach,
                "n_variants": len(score_matrix),
                "n_groups": score_matrix["match_group"].nunique(),
                "prefix_tokens": PREFIX_TOKEN_COUNTS[approach],
                "raw_target_tokens": DNA_TARGET_TOKENS
                + PREFIX_TOKEN_COUNTS[approach]
                - 1,
                "score_scale_to_dna_mean": (
                    DNA_TARGET_TOKENS + PREFIX_TOKEN_COUNTS[approach] - 1
                )
                / DNA_TARGET_TOKENS,
                "macro_auprc": float(macro["auprc"]),
                "macro_ci_low": float(macro["ci_low"]),
                "macro_ci_high": float(macro["ci_high"]),
                "mean_score_positive": float(positive.mean()),
                "mean_score_negative": float(negative.mean()),
                "mean_score_separation": float(positive.mean() - negative.mean()),
            }
        )
    return pd.DataFrame(rows)


def _pairwise_correlations(score_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlations for every retained subset and label."""
    rows: list[dict[str, object]] = []
    label_groups = (("all", None), ("positive", True), ("negative", False))
    for x_approach, y_approach in PAIRWISE_COMPARISONS:
        for subset in SUBSETS:
            subset_rows = (
                score_matrix
                if subset == "_all_analyzed_"
                else score_matrix.loc[score_matrix["subset"].eq(subset)]
            )
            for label_group, label in label_groups:
                frame = (
                    subset_rows
                    if label is None
                    else subset_rows.loc[subset_rows["label"].eq(label)]
                )
                pearson_r = float(frame[x_approach].corr(frame[y_approach]))
                assert pd.notna(pearson_r), "Pearson correlation must be finite"
                rows.append(
                    {
                        "x_approach": x_approach,
                        "y_approach": y_approach,
                        "subset": subset,
                        "label_group": label_group,
                        "n_variants": len(frame),
                        "pearson_r": pearson_r,
                    }
                )
    return pd.DataFrame(rows)


def _write_summary(
    approaches: pd.DataFrame, matched: pd.DataFrame, output: Path
) -> None:
    overall = matched.loc[matched["subset"].eq("_all_analyzed_")]
    lines = [
        "# Carbon prompt-conditioning comparison",
        "",
        "All summaries exclude the 40 mature-miRNA variants and use 16,100 variants in 1,610 complete match groups.",
        "",
        "Score-level summaries rescale each raw prompt-mean LLR to a common 1,365 DNA-target-token denominator.",
        "The multipliers are 1365/1365 for untagged, 1373/1365 for correct mammalian, and 1370/1365 for far-wrong fungal.",
        "This removes the deterministic prompt-length scale difference; within-arm AUPRC and pairwise Pearson correlations are unchanged.",
        "",
        "## Three retained approaches",
        "",
        "Macro AUPRC averages the eight retained consequence subsets.",
        "",
        "| approach | macro AUPRC | 95% CI | positive mean score | negative mean score | positive − negative |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for approach in APPROACHES:
        row = approaches.loc[approaches["approach"].eq(approach)].iloc[0]
        lines.append(
            f"| {APPROACH_LABELS[approach]} | {row.macro_auprc:.6f} | "
            f"[{row.macro_ci_low:.6f}, {row.macro_ci_high:.6f}] | "
            f"{row.mean_score_positive:.6f} | {row.mean_score_negative:.6f} | "
            f"{row.mean_score_separation:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Conditioned-minus-untagged score shifts",
            "",
            "`delta_score` is the DNA-token-normalized conditioned score minus the same variant's normalized untagged score.",
            "The label-separation shift is the mean positive delta minus the mean delta across the nine matched negatives.",
            "Positive label-separation shifts move pathogenic positives upward relative to their matched negatives.",
            "Spread ratios compare the standard deviation of positive deltas with negative deltas.",
            "Intervals are 95% match-group bootstrap intervals from 1,000 seeded draws.",
            "",
            "| comparison | positive mean delta | negative mean delta | label-separation shift | 95% CI | positive SD / negative SD | 95% CI (log2 ratio) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
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


def _save_figure(figure: Figure, output_root: Path, stem: str) -> None:
    svg_output = output_root / f"{stem}.svg"
    figure.savefig(svg_output)
    svg_lines = (line.rstrip() for line in svg_output.read_text().splitlines())
    svg_output.write_text("\n".join(svg_lines) + "\n")
    figure.savefig(output_root / f"{stem}.png", dpi=160)
    plt.close(figure)


def _render_figure(matched: pd.DataFrame, output_root: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 7.4), sharey=True)
    y_positions = {subset: index for index, subset in enumerate(reversed(SUBSETS))}
    offsets = {"correct": -0.12, "far_wrong": 0.12}

    for axis in axes:
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
    axes[0].set_xlabel(
        "Positive shift − matched-negative shift\n(×10⁻³ nats/DNA target token)"
    )
    axes[0].set_title("Change in label separation")
    axes[1].set_xlabel("log₂(SD positive shift / SD negative shift)")
    axes[1].set_title("Relative variability of score shifts")
    figure.suptitle(
        "Conditioned-minus-untagged score shifts vary across consequence subsets",
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
        "Scores use a common DNA-token denominator; points are observed estimates and bars are 95% match-group bootstrap intervals. Mature-miRNA is excluded.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.88))
    _save_figure(figure, output_root, "score_shift_by_subset")


def _render_pairwise_scatter(
    score_matrix: pd.DataFrame,
    correlations: pd.DataFrame,
    output_root: Path,
) -> None:
    """Render one point per variant for every approach pair and subset."""
    for x_approach, y_approach in PAIRWISE_COMPARISONS:
        figure, axes = plt.subplots(3, 3, figsize=(13.2, 12.6))
        for axis, subset in zip(axes.flat, SUBSETS, strict=True):
            frame = (
                score_matrix
                if subset == "_all_analyzed_"
                else score_matrix.loc[score_matrix["subset"].eq(subset)]
            )
            negative = frame.loc[~frame["label"]]
            positive = frame.loc[frame["label"]]
            axis.scatter(
                negative[x_approach],
                negative[y_approach],
                s=4,
                alpha=0.14,
                color="#777777",
                edgecolors="none",
                rasterized=True,
                label="Negative",
            )
            axis.scatter(
                positive[x_approach],
                positive[y_approach],
                s=7,
                alpha=0.38,
                color="#CC3311",
                edgecolors="none",
                rasterized=True,
                label="Positive",
            )
            lower = float(frame[[x_approach, y_approach]].min().min())
            upper = float(frame[[x_approach, y_approach]].max().max())
            padding = (upper - lower) * 0.04 or 1.0e-6
            bounds = (lower - padding, upper + padding)
            axis.plot(bounds, bounds, color="#555555", linestyle="--", linewidth=0.8)
            axis.set_xlim(bounds)
            axis.set_ylim(bounds)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(color="#e6e6e6", linewidth=0.6)
            axis.set_axisbelow(True)
            stats = correlations.loc[
                correlations["x_approach"].eq(x_approach)
                & correlations["y_approach"].eq(y_approach)
                & correlations["subset"].eq(subset)
            ].set_index("label_group")
            axis.text(
                0.04,
                0.96,
                f"Pearson r\nall {stats.loc['all', 'pearson_r']:.3f}\n"
                f"positive {stats.loc['positive', 'pearson_r']:.3f}\n"
                f"negative {stats.loc['negative', 'pearson_r']:.3f}",
                transform=axis.transAxes,
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )
            axis.set_title(f"{SUBSET_LABELS[subset]} (n={len(frame):,})", fontsize=10.5)
            axis.set_xlabel(APPROACH_LABELS[x_approach], fontsize=9)
            axis.set_ylabel(APPROACH_LABELS[y_approach], fontsize=9)
            axis.tick_params(labelsize=8)

        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=2,
            frameon=False,
        )
        figure.suptitle(
            f"{APPROACH_LABELS[y_approach]} versus {APPROACH_LABELS[x_approach]} scores",
            fontsize=14,
            y=0.992,
        )
        figure.text(
            0.5,
            0.01,
            "Each non-miRNA variant is one point after DNA-token normalization; dashed lines mark equal scores. Pearson r is reported overall and by label.",
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.035, 1, 0.93))
        _save_figure(
            figure,
            output_root,
            f"score_scatter_{x_approach}_vs_{y_approach}",
        )


def main() -> None:
    scores = {
        approach: normalize_prompt_mean_scores(
            pd.read_parquet(SCORE_ROOT / f"{approach}.parquet"),
            dna_target_tokens=DNA_TARGET_TOKENS,
            prefix_tokens=PREFIX_TOKEN_COUNTS[approach],
        )
        for approach in APPROACHES
    }
    untagged = scores["untagged"]
    tagged = {condition: scores[condition] for condition in CONDITIONS}
    all_shifts = assemble_score_shifts(untagged, tagged)
    shifts = all_shifts.loc[~all_shifts["subset"].isin(EXCLUDED_SUBSETS)].reset_index(
        drop=True
    )
    score_matrix = _assemble_score_matrix(untagged, shifts)
    approaches = _summarize_approaches(score_matrix)
    correlations = _pairwise_correlations(score_matrix)
    by_label = summarize_score_shifts(shifts)
    matched = bootstrap_matched_score_shifts(
        shifts,
        n_bootstrap=1000,
        bootstrap_seed=486,
        min_groups=30,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    approaches.to_parquet(OUTPUT_ROOT / "three_approach_summary.parquet", index=False)
    approaches.to_csv(OUTPUT_ROOT / "three_approach_summary.tsv", sep="\t", index=False)
    correlations.to_parquet(
        OUTPUT_ROOT / "score_pairwise_correlations.parquet", index=False
    )
    correlations.to_csv(
        OUTPUT_ROOT / "score_pairwise_correlations.tsv", sep="\t", index=False
    )
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
    _write_summary(approaches, matched, OUTPUT_ROOT / "score_shift_summary.md")
    _render_figure(matched, OUTPUT_ROOT)
    _render_pairwise_scatter(score_matrix, correlations, OUTPUT_ROOT)


if __name__ == "__main__":
    main()
