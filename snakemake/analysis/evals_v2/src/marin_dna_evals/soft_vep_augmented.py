"""Augment the exp232 specialist assessment with exp351-centered distal.

The exp351-centered enhancer arm replaces exp232's original contaminated cCRE
arm. The other five exp232 arms are unchanged. Only synchronized checkpoints
with durable HF exports and offline score bundles are admitted.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.lines import Line2D
from scipy.special import ndtr

from marin_dna_evals.soft_vep_analysis import (
    ARMS,
    ARM_COLORS,
    ARM_LINESTYLES,
    DETECTABILITY_METRICS,
    METRIC_AXIS_LABELS,
    NON_DISTAL_SUBSETS,
    SPECIALIST_ARM,
    S3_SCORE_ROOT,
    TAU_REFERENCE_ARM,
    TAU_REFERENCE_STEP,
    UNGROUPED_DETECTABILITY_METRICS,
    add_llr_scores,
    compare_metric_detection_timing,
    compute_rank_agreement,
    confidence_filtered_rank_reversals,
    exp232_score_uri,
    pairwise_bootstrap_summary,
    persistent_specialist_detectability,
    plot_metric_detectability_summary,
    read_score_bundle,
    specialist_detectability_summary,
    validate_aligned_bundles,
)
from marin_dna_evals.soft_vep_metrics import (
    AUPRC,
    NORMAL_95_Z,
    VARIANT_POOLED_SMD,
    cohen_d_closed_form_table,
    joint_cluster_bootstrap_soft_metrics,
    joint_stratified_row_bootstrap_ungrouped_metrics,
    reference_soft_win_temperature,
    summarize_joint_bootstrap,
)

DATASET = "mendelian_traits"
SPLIT = "train"
DISTAL_ARM = "distal_centered"
EXP351_MODEL_PREFIX = "exp351-centered-step"
AUGMENTED_STEPS = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999)
AUGMENTED_ARMS = (*ARMS, DISTAL_ARM)
AUGMENTED_SUBSETS = (*NON_DISTAL_SUBSETS, "distal")
AUGMENTED_SPECIALIST_ARM = {**SPECIALIST_ARM, "distal": DISTAL_ARM}

AUGMENTED_ARM_COLORS = {**ARM_COLORS, DISTAL_ARM: "#000000"}
AUGMENTED_ARM_LINESTYLES = {**ARM_LINESTYLES, DISTAL_ARM: "-"}
AUGMENTED_ARM_LABELS = {
    "bg": "exp232 background",
    "cds": "exp232 CDS",
    "utr3": "exp232 3′ UTR",
    "ncrna_exon": "exp232 ncRNA exon",
    "tss_region_and_utr5": "exp232 TSS/5′ UTR",
    DISTAL_ARM: "exp351 centered distal",
}
AUGMENTED_NON_HOME_COLORS = {**AUGMENTED_ARM_COLORS, DISTAL_ARM: "#9467BD"}
AUGMENTED_SUBSET_LABELS = {
    "missense_variant": "Missense",
    "synonymous_variant": "Synonymous",
    "splicing": "Splicing",
    "3_prime_UTR_variant": "3′ UTR",
    "non_coding_transcript_exon_variant": "Noncoding exon",
    "5_prime_UTR_variant": "5′ UTR",
    "tss_proximal": "TSS proximal",
    "distal": "Distal",
}
WIN_DEFINITION_LABELS = {
    "point_rank1": "Point rank 1",
    "rank1_probability_50": "P(rank 1) ≥ 50%",
    "rank1_probability_80": "P(rank 1) ≥ 80%",
    "rank1_probability_95": "P(rank 1) ≥ 95%",
    "margin_ci_95": "95% margin CI > 0",
}
ALL_METRIC_COMPARISON_ROWS = (
    ("matched_groups", "mean_gap_global", "Matched · Global mean gap"),
    ("matched_groups", "mean_gap_group", "Matched · Group mean gap"),
    ("matched_groups", "group_smd", "Matched · Group SMD"),
    ("matched_groups", "group_median_mad", "Matched · Median / MAD"),
    ("matched_groups", "soft_win", "Matched · SoftWin"),
    ("matched_groups", "calibrated_log_loss", "Matched · Calibrated log loss"),
    ("matched_groups", "calibrated_brier", "Matched · Calibrated Brier"),
    (
        "no_groups_bootstrap",
        VARIANT_POOLED_SMD,
        "No group · Cohen's d (bootstrap)",
    ),
    (
        "no_groups_bootstrap",
        "variant_total_sd_gap",
        "No group · Mean gap / all-variant SD",
    ),
    ("no_groups_bootstrap", "student_t", "No group · Student t (bootstrap)"),
    ("no_groups_bootstrap", "welch_t", "No group · Welch t (bootstrap)"),
    (
        "no_groups_closed_form",
        VARIANT_POOLED_SMD,
        "No group · Cohen's d (closed form)",
    ),
)
PRIMARY_WIN_DEFINITIONS = ("point_rank1", "rank1_probability_95", "margin_ci_95")


def augmented_model_name(arm: str, step: int) -> str:
    """Return the evals_v2 model key for one augmented matrix cell."""
    assert arm in AUGMENTED_ARMS, f"unknown augmented arm {arm!r}"
    assert step in AUGMENTED_STEPS, f"step {step} is not synchronized"
    if arm == DISTAL_ARM:
        return f"{EXP351_MODEL_PREFIX}-{step}"
    return f"exp232-v4_{arm}-step-{step}"


def augmented_score_uri(arm: str, step: int) -> str:
    """S3 score URI for one augmented matrix cell."""
    if arm == DISTAL_ARM:
        return f"{S3_SCORE_ROOT}/{augmented_model_name(arm, step)}/{DATASET}.parquet"
    return exp232_score_uri(arm, step)


def augmented_metric_uri(arm: str, step: int) -> str:
    """S3 metric URI paired with one augmented score bundle."""
    return augmented_score_uri(arm, step).replace(
        "/results/scores/",
        "/results/metrics/",
    )


def augmented_manifest() -> pd.DataFrame:
    """Explicit six-arm by ten-checkpoint inventory."""
    rows = [
        {
            "arm": arm,
            "step": step,
            "model": augmented_model_name(arm, step),
            "uri": augmented_score_uri(arm, step),
            "dataset": DATASET,
            "split": SPLIT,
            "role": "replacement_home" if arm == DISTAL_ARM else "exp232_arm",
        }
        for step in AUGMENTED_STEPS
        for arm in AUGMENTED_ARMS
    ]
    manifest = pd.DataFrame(rows).sort_values(["step", "arm"]).reset_index(drop=True)
    assert len(manifest) == len(AUGMENTED_STEPS) * len(AUGMENTED_ARMS)
    assert manifest["uri"].is_unique
    return manifest


def read_stored_auprc(arm: str, step: int) -> pd.DataFrame:
    """Read stored offline AUPRC for exact parity checking."""
    metrics = pl.read_parquet(
        augmented_metric_uri(arm, step),
        columns=["score_type", "subset", "value", "split"],
        storage_options={"aws_region": "us-east-2"},
    ).filter((pl.col("score_type") == "minus_llr_avg") & (pl.col("split") == SPLIT))
    result = metrics.select(
        ["subset", pl.col("value").alias("stored_value")]
    ).to_pandas()
    result["arm"] = arm
    result["step"] = step
    return result


def plot_augmented_distal_trajectories(
    auprc_metrics: pd.DataFrame,
    cohen_d_metrics: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Show the replacement distal home arm against all five exp232 arms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = (
        (auprc_metrics, AUPRC, "AUPRC", True),
        (cohen_d_metrics, VARIANT_POOLED_SMD, "Cohen's d", True),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharex=True)
    for axis, (table, metric, title, show_interval) in zip(axes, panels):
        metric_data = table[
            (table["subset"] == "distal") & (table["metric"] == metric)
        ]
        for arm in AUGMENTED_ARMS:
            arm_data = metric_data[metric_data["arm"] == arm].sort_values("step")
            assert arm_data["step"].tolist() == list(AUGMENTED_STEPS)
            is_home = arm == DISTAL_ARM
            axis.plot(
                arm_data["step"],
                arm_data["value"],
                color=AUGMENTED_ARM_COLORS[arm],
                linestyle=AUGMENTED_ARM_LINESTYLES[arm],
                marker="D" if is_home else "o",
                markersize=5.5 if is_home else 3.5,
                linewidth=3.0 if is_home else 1.5,
                alpha=1.0 if is_home else 0.82,
                label=AUGMENTED_ARM_LABELS[arm],
                zorder=4 if is_home else 2,
            )
            if is_home and show_interval:
                axis.fill_between(
                    arm_data["step"],
                    arm_data["ci_low"],
                    arm_data["ci_high"],
                    color=AUGMENTED_ARM_COLORS[arm],
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )
        axis.set_title(title)
        axis.set_xlabel("Training step")
        axis.set_ylabel(METRIC_AXIS_LABELS[metric])
        axis.set_xticks(AUGMENTED_STEPS)
        axis.tick_params(axis="x", labelrotation=45)
        axis.grid(alpha=0.22, linewidth=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        frameon=False,
        ncol=3,
    )
    fig.suptitle(
        "Distal specialist trajectory after replacing exp232 cCRE with exp351 centered",
        fontsize=14,
        y=0.98,
    )
    fig.text(
        0.5,
        0.012,
        "Development split. Black diamonds are the replacement home arm. "
        "AUPRC ribbon: class-stratified variant-bootstrap 95% interval; Cohen's "
        "d ribbon: conventional IID closed-form 95% interval. Higher is better.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.91))
    svg_path = output_dir / "augmented_distal_metric_trajectories.svg"
    png_path = output_dir / "augmented_distal_metric_trajectories.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {
        "plot_augmented_distal_metric_trajectories_svg": svg_path,
        "plot_augmented_distal_metric_trajectories_png": png_path,
    }


def plot_augmented_specialist_trajectories(
    auprc_metrics: pd.DataFrame,
    cohen_d_metrics: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot every mapped home arm against all non-home arms for both metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = (
        (auprc_metrics, AUPRC, "AUPRC", True),
        (cohen_d_metrics, VARIANT_POOLED_SMD, "Cohen's d", True),
    )
    fig, axes = plt.subplots(
        len(AUGMENTED_SUBSETS),
        len(panels),
        figsize=(14.5, 22),
        sharex=True,
        squeeze=False,
    )
    for row_index, subset in enumerate(AUGMENTED_SUBSETS):
        home_arm = AUGMENTED_SPECIALIST_ARM[subset]
        for column_index, (table, metric, title, show_interval) in enumerate(panels):
            axis = axes[row_index, column_index]
            metric_data = table[
                (table["subset"] == subset) & (table["metric"] == metric)
            ]
            for arm in AUGMENTED_ARMS:
                arm_data = metric_data[metric_data["arm"] == arm].sort_values("step")
                assert arm_data["step"].tolist() == list(AUGMENTED_STEPS)
                is_home = arm == home_arm
                color = "#000000" if is_home else AUGMENTED_NON_HOME_COLORS[arm]
                axis.plot(
                    arm_data["step"],
                    arm_data["value"],
                    color=color,
                    linestyle="-" if is_home else AUGMENTED_ARM_LINESTYLES[arm],
                    marker="D" if is_home else "o",
                    markersize=5.0 if is_home else 3.0,
                    linewidth=2.8 if is_home else 1.2,
                    alpha=1.0 if is_home else 0.72,
                    zorder=4 if is_home else 2,
                )
                if is_home and show_interval:
                    axis.fill_between(
                        arm_data["step"],
                        arm_data["ci_low"],
                        arm_data["ci_high"],
                        color="#000000",
                        alpha=0.11,
                        linewidth=0,
                        zorder=1,
                    )
            if row_index == 0:
                axis.set_title(title, fontsize=12, fontweight="bold")
            axis.set_ylabel(METRIC_AXIS_LABELS[metric])
            axis.set_xticks(AUGMENTED_STEPS)
            axis.grid(alpha=0.22, linewidth=0.7)
            if metric == VARIANT_POOLED_SMD:
                axis.axhline(0, color="#666666", linewidth=0.7, alpha=0.5)
            if row_index == len(AUGMENTED_SUBSETS) - 1:
                axis.set_xlabel("Training step")
                axis.tick_params(axis="x", labelrotation=45, labelbottom=True)

    fig.suptitle(
        "Specialist trajectories: mapped home arm versus all non-home arms",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0.18, 0.09, 1, 0.97), h_pad=1.25, w_pad=2.0)
    for row_index, subset in enumerate(AUGMENTED_SUBSETS):
        position = axes[row_index, 0].get_position()
        fig.text(
            0.012,
            (position.y0 + position.y1) / 2,
            f"{AUGMENTED_SUBSET_LABELS[subset]}\n"
            f"home: {AUGMENTED_ARM_LABELS[AUGMENTED_SPECIALIST_ARM[subset]]}",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#000000",
            marker="D",
            linewidth=2.8,
            label="mapped home arm",
        )
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            color=AUGMENTED_NON_HOME_COLORS[arm],
            linestyle=AUGMENTED_ARM_LINESTYLES[arm],
            marker="o",
            linewidth=1.3,
            label=f"non-home: {AUGMENTED_ARM_LABELS[arm]}",
        )
        for arm in AUGMENTED_ARMS
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        frameon=False,
        ncol=4,
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.008,
        "Development split. Each row maps its specialist to the black diamond "
        "line. AUPRC ribbon: class-stratified variant-bootstrap 95% interval; "
        "Cohen's d ribbon: conventional IID closed-form 95% interval. Higher is "
        "better.",
        ha="center",
        fontsize=8.5,
    )
    svg_path = output_dir / "augmented_specialist_auprc_vs_cohen_d.svg"
    png_path = output_dir / "augmented_specialist_auprc_vs_cohen_d.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {
        "plot_augmented_specialist_auprc_vs_cohen_d_svg": svg_path,
        "plot_augmented_specialist_auprc_vs_cohen_d_png": png_path,
    }


def compute_closed_form_cohen_d_win_table(
    cohen_d_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Pairwise home-win probabilities under independent normal d estimates."""
    required = {"subset", "step", "arm", "value", "se"}
    assert required.issubset(cohen_d_metrics.columns), (
        f"Cohen's d table missing columns: "
        f"{sorted(required - set(cohen_d_metrics.columns))}"
    )
    rows: list[dict[str, str | float | int]] = []
    for subset in AUGMENTED_SUBSETS:
        home_arm = AUGMENTED_SPECIALIST_ARM[subset]
        for step in AUGMENTED_STEPS:
            cell = cohen_d_metrics[
                (cohen_d_metrics["subset"] == subset)
                & (cohen_d_metrics["step"] == step)
            ]
            assert set(cell["arm"]) == set(AUGMENTED_ARMS)
            home = cell[cell["arm"] == home_arm]
            assert len(home) == 1
            home_value = float(home.iloc[0]["value"])
            home_se = float(home.iloc[0]["se"])
            for competitor_arm in AUGMENTED_ARMS:
                if competitor_arm == home_arm:
                    continue
                competitor = cell[cell["arm"] == competitor_arm]
                assert len(competitor) == 1
                competitor_value = float(competitor.iloc[0]["value"])
                competitor_se = float(competitor.iloc[0]["se"])
                difference_se = float(np.hypot(home_se, competitor_se))
                z_score = (home_value - competitor_value) / difference_se
                rows.append(
                    {
                        "subset": subset,
                        "step": step,
                        "home_arm": home_arm,
                        "competitor_arm": competitor_arm,
                        "home_d": home_value,
                        "competitor_d": competitor_value,
                        "difference_se": difference_se,
                        "z_score": z_score,
                        "win_probability": float(ndtr(z_score)),
                        "uncertainty_method": (
                            "independent_normal_closed_form_cohen_d"
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    expected_rows = (
        len(AUGMENTED_SUBSETS) * len(AUGMENTED_STEPS) * (len(AUGMENTED_ARMS) - 1)
    )
    assert len(result) == expected_rows
    return result


def compute_closed_form_cohen_d_rank1_probabilities(
    cohen_d_metrics: pd.DataFrame,
    *,
    quadrature_nodes: int = 64,
) -> pd.DataFrame:
    """Home-rank-first probability under independent normal d estimates.

    The one-dimensional integral conditions on the home-arm estimate and uses
    Gauss-Hermite quadrature, so this remains deterministic and uses only the
    conventional closed-form Cohen's d standard errors.
    """
    required = {"subset", "step", "arm", "value", "se"}
    assert required.issubset(cohen_d_metrics.columns), (
        f"Cohen's d table missing columns: "
        f"{sorted(required - set(cohen_d_metrics.columns))}"
    )
    assert quadrature_nodes >= 16
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_nodes)
    rows: list[dict[str, str | float | int]] = []
    for subset in AUGMENTED_SUBSETS:
        home_arm = AUGMENTED_SPECIALIST_ARM[subset]
        for step in AUGMENTED_STEPS:
            cell = cohen_d_metrics[
                (cohen_d_metrics["subset"] == subset)
                & (cohen_d_metrics["step"] == step)
            ].set_index("arm")
            assert set(cell.index) == set(AUGMENTED_ARMS)
            assert (cell["se"] > 0).all()
            home_mean = float(cell.loc[home_arm, "value"])
            home_se = float(cell.loc[home_arm, "se"])
            home_draws = home_mean + np.sqrt(2) * home_se * nodes
            conditional_probability = np.ones_like(home_draws)
            for competitor_arm in AUGMENTED_ARMS:
                if competitor_arm == home_arm:
                    continue
                competitor = cell.loc[competitor_arm]
                conditional_probability *= ndtr(
                    (home_draws - float(competitor["value"]))
                    / float(competitor["se"])
                )
            rank1_probability = float(
                np.dot(weights, conditional_probability) / np.sqrt(np.pi)
            )
            rows.append(
                {
                    "subset": subset,
                    "step": step,
                    "home_arm": home_arm,
                    "metric": VARIANT_POOLED_SMD,
                    "rank1_probability": rank1_probability,
                    "uncertainty_method": "independent_normal_gauss_hermite",
                }
            )
    result = pd.DataFrame(rows)
    assert len(result) == len(AUGMENTED_SUBSETS) * len(AUGMENTED_STEPS)
    assert result["rank1_probability"].between(0, 1).all()
    return result


def plot_augmented_specialist_closed_form_win_percentage(
    win_probabilities: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot closed-form Cohen's d home-win probability against every other arm."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
    for axis, subset in zip(axes.flat, AUGMENTED_SUBSETS):
        subset_data = win_probabilities[win_probabilities["subset"] == subset]
        home_arm = AUGMENTED_SPECIALIST_ARM[subset]
        for competitor_arm in AUGMENTED_ARMS:
            if competitor_arm == home_arm:
                continue
            competitor_data = subset_data[
                subset_data["competitor_arm"] == competitor_arm
            ].sort_values("step")
            assert competitor_data["step"].tolist() == list(AUGMENTED_STEPS)
            axis.plot(
                competitor_data["step"],
                100 * competitor_data["win_probability"],
                color=AUGMENTED_NON_HOME_COLORS[competitor_arm],
                linestyle=AUGMENTED_ARM_LINESTYLES[competitor_arm],
                marker="o",
                markersize=3.5,
                linewidth=1.3,
                alpha=0.78,
            )
        mean_probability = (
            subset_data.groupby("step", sort=False)["win_probability"]
            .mean()
            .reindex(AUGMENTED_STEPS)
        )
        assert mean_probability.notna().all()
        axis.plot(
            AUGMENTED_STEPS,
            100 * mean_probability,
            color="#000000",
            marker="D",
            markersize=5,
            linewidth=2.8,
            zorder=5,
        )
        axis.axhline(50, color="#555555", linestyle=":", linewidth=1)
        axis.set_title(
            f"{AUGMENTED_SUBSET_LABELS[subset]}\n"
            f"home: {AUGMENTED_ARM_LABELS[home_arm]}",
            fontsize=10,
        )
        axis.set_xlabel("Training step")
        axis.set_ylabel("Closed-form home win probability (%)")
        axis.set_xticks(AUGMENTED_STEPS)
        axis.tick_params(axis="x", labelrotation=45, labelbottom=True)
        axis.set_ylim(-2, 103)
        axis.grid(alpha=0.22, linewidth=0.7)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#000000",
            marker="D",
            linewidth=2.8,
            label="mean over non-home arms",
        )
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            color=AUGMENTED_NON_HOME_COLORS[arm],
            linestyle=AUGMENTED_ARM_LINESTYLES[arm],
            marker="o",
            linewidth=1.3,
            label=AUGMENTED_ARM_LABELS[arm],
        )
        for arm in AUGMENTED_ARMS
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=":",
            label="50% reference",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        frameon=False,
        ncol=4,
        fontsize=8.5,
    )
    fig.suptitle(
        "How strongly does Cohen's d favor the mapped home arm?",
        fontsize=14,
        y=0.99,
    )
    fig.text(
        0.5,
        0.008,
        "Each colored line is Phi((d_home - d_other) / "
        "sqrt(SE_home^2 + SE_other^2)); black diamonds average the five "
        "non-home comparisons. Independent normal approximation.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95), h_pad=1.3, w_pad=1.1)
    svg_path = output_dir / "augmented_specialist_closed_form_win_percentage.svg"
    png_path = output_dir / "augmented_specialist_closed_form_win_percentage.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {
        "plot_augmented_specialist_closed_form_win_percentage_svg": svg_path,
        "plot_augmented_specialist_closed_form_win_percentage_png": png_path,
    }


def compute_win_definition_sensitivity(
    auprc_detectability: pd.DataFrame,
    cohen_d_detectability: pd.DataFrame,
    cohen_d_rank1_probabilities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare earliest wins under rank-only and uncertainty-aware rules."""
    auprc = auprc_detectability.copy()
    auprc["rank1_probability"] = auprc["bootstrap_home_rank1_frequency"]
    cohen_d = cohen_d_detectability.merge(
        cohen_d_rank1_probabilities[
            ["subset", "step", "metric", "rank1_probability"]
        ],
        on=["subset", "step", "metric"],
        how="left",
        validate="one_to_one",
    )
    combined = pd.concat([auprc, cohen_d], ignore_index=True)
    assert combined["rank1_probability"].notna().all()
    assert set(combined["metric"]) == {AUPRC, VARIANT_POOLED_SMD}

    definition_flags = {
        "point_rank1": combined["home_rank"].eq(1),
        "rank1_probability_50": combined["rank1_probability"].ge(0.50),
        "rank1_probability_80": combined["rank1_probability"].ge(0.80),
        "rank1_probability_95": combined["rank1_probability"].ge(0.95),
        "margin_ci_95": combined["confidence_supported"].astype(bool),
    }
    timing_rows: list[dict[str, str | float | int | bool | None]] = []
    for definition, flags in definition_flags.items():
        flagged = combined.assign(win=flags)
        for persistence in (1, 2):
            for (subset, metric), frame in flagged.groupby(
                ["subset", "metric"], sort=False
            ):
                ordered = frame.sort_values("step")
                steps = ordered["step"].astype(int).tolist()
                wins = ordered["win"].astype(bool).tolist()
                earliest: int | None = None
                for index in range(len(steps) - persistence + 1):
                    if all(wins[index : index + persistence]):
                        earliest = steps[index]
                        break
                final = ordered[ordered["step"] == 4999]
                assert len(final) == 1
                timing_rows.append(
                    {
                        "definition": definition,
                        "definition_label": WIN_DEFINITION_LABELS[definition],
                        "persistence": persistence,
                        "subset": str(subset),
                        "home_arm": str(ordered.iloc[0]["home_arm"]),
                        "metric": str(metric),
                        "earliest_step": earliest,
                        "final_step_win": bool(final.iloc[0]["win"]),
                    }
                )
    timing = pd.DataFrame(timing_rows)

    comparison_parts: list[pd.DataFrame] = []
    for (definition, persistence), frame in timing.groupby(
        ["definition", "persistence"], sort=False
    ):
        comparable = frame.rename(
            columns={"earliest_step": "earliest_persistent_detected_step"}
        )
        comparison = compare_metric_detection_timing(
            comparable,
            metrics=(AUPRC, VARIANT_POOLED_SMD),
            subsets=AUGMENTED_SUBSETS,
        )
        comparison["definition"] = definition
        comparison["definition_label"] = WIN_DEFINITION_LABELS[definition]
        comparison["persistence"] = persistence
        comparison_parts.append(comparison)
    comparisons = pd.concat(comparison_parts, ignore_index=True)
    assert len(comparisons) == len(WIN_DEFINITION_LABELS) * 2
    count_columns = [
        "candidate_earlier",
        "same_step",
        "auprc_earlier",
        "neither_detected",
    ]
    assert comparisons[count_columns].sum(axis=1).eq(len(AUGMENTED_SUBSETS)).all()
    return timing, comparisons


def compute_bootstrap_metric_win_definition_sensitivity(
    detectability: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare bootstrap metric timing under rank and uncertainty rules."""
    assert metrics[0] == AUPRC
    required = {
        "subset",
        "home_arm",
        "metric",
        "step",
        "home_rank",
        "bootstrap_home_rank1_frequency",
        "confidence_supported",
    }
    assert required.issubset(detectability.columns), (
        f"detectability table missing columns: "
        f"{sorted(required - set(detectability.columns))}"
    )
    selected = detectability[detectability["metric"].isin(metrics)].copy()
    assert set(selected["metric"]) == set(metrics)
    assert selected["bootstrap_home_rank1_frequency"].notna().all()
    definition_flags = {
        "point_rank1": selected["home_rank"].eq(1),
        "rank1_probability_50": selected[
            "bootstrap_home_rank1_frequency"
        ].ge(0.50),
        "rank1_probability_80": selected[
            "bootstrap_home_rank1_frequency"
        ].ge(0.80),
        "rank1_probability_95": selected[
            "bootstrap_home_rank1_frequency"
        ].ge(0.95),
        "margin_ci_95": selected["confidence_supported"].astype(bool),
    }
    timing_rows: list[dict[str, str | float | int | bool | None]] = []
    for definition, flags in definition_flags.items():
        flagged = selected.assign(win=flags)
        for persistence in (1, 2):
            for (subset, metric), frame in flagged.groupby(
                ["subset", "metric"], sort=False
            ):
                ordered = frame.sort_values("step")
                steps = ordered["step"].astype(int).tolist()
                wins = ordered["win"].astype(bool).tolist()
                earliest: int | None = None
                for index in range(len(steps) - persistence + 1):
                    if all(wins[index : index + persistence]):
                        earliest = steps[index]
                        break
                final = ordered[ordered["step"] == 4999]
                assert len(final) == 1
                timing_rows.append(
                    {
                        "definition": definition,
                        "definition_label": WIN_DEFINITION_LABELS[definition],
                        "persistence": persistence,
                        "subset": str(subset),
                        "home_arm": str(ordered.iloc[0]["home_arm"]),
                        "metric": str(metric),
                        "earliest_step": earliest,
                        "final_step_win": bool(final.iloc[0]["win"]),
                    }
                )
    timing = pd.DataFrame(timing_rows)

    comparison_parts: list[pd.DataFrame] = []
    for (definition, persistence), frame in timing.groupby(
        ["definition", "persistence"], sort=False
    ):
        comparable = frame.rename(
            columns={"earliest_step": "earliest_persistent_detected_step"}
        )
        comparison = compare_metric_detection_timing(
            comparable,
            metrics=metrics,
            subsets=AUGMENTED_SUBSETS,
        )
        comparison["definition"] = definition
        comparison["definition_label"] = WIN_DEFINITION_LABELS[definition]
        comparison["persistence"] = persistence
        comparison_parts.append(comparison)
    comparisons = pd.concat(comparison_parts, ignore_index=True)
    assert len(comparisons) == (len(metrics) - 1) * len(WIN_DEFINITION_LABELS) * 2
    count_columns = [
        "candidate_earlier",
        "same_step",
        "auprc_earlier",
        "neither_detected",
    ]
    assert comparisons[count_columns].sum(axis=1).eq(len(AUGMENTED_SUBSETS)).all()
    return timing, comparisons


def combine_all_metric_win_definition_comparisons(
    matched_comparisons: pd.DataFrame,
    ungrouped_comparisons: pd.DataFrame,
    closed_form_comparisons: pd.DataFrame,
) -> pd.DataFrame:
    """Create one labeled table for the complete alternative-metric reranking."""
    matched = matched_comparisons.assign(evaluation="matched_groups")
    ungrouped = ungrouped_comparisons.assign(evaluation="no_groups_bootstrap")
    closed_form = closed_form_comparisons.assign(
        evaluation="no_groups_closed_form"
    )
    combined = pd.concat([matched, ungrouped, closed_form], ignore_index=True)
    label_by_row = {
        (evaluation, metric): label
        for evaluation, metric, label in ALL_METRIC_COMPARISON_ROWS
    }
    order_by_row = {
        (evaluation, metric): index
        for index, (evaluation, metric, _) in enumerate(ALL_METRIC_COMPARISON_ROWS)
    }
    row_keys = list(zip(combined["evaluation"], combined["metric"]))
    assert set(row_keys) == set(label_by_row)
    combined["display_label"] = [label_by_row[key] for key in row_keys]
    combined["row_order"] = [order_by_row[key] for key in row_keys]
    expected_rows = len(ALL_METRIC_COMPARISON_ROWS) * len(WIN_DEFINITION_LABELS) * 2
    assert len(combined) == expected_rows
    return combined.sort_values(
        ["persistence", "row_order", "definition"]
    ).reset_index(drop=True)


def plot_augmented_all_metric_win_sensitivity(
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot every alternative's timing advantage relative to AUPRC."""
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = comparisons[
        (comparisons["persistence"] == 2)
        & comparisons["definition"].isin(PRIMARY_WIN_DEFINITIONS)
    ]
    row_labels = [label for _, _, label in ALL_METRIC_COMPARISON_ROWS]
    definition_labels = [
        WIN_DEFINITION_LABELS[definition] for definition in PRIMARY_WIN_DEFINITIONS
    ]
    advantage = np.zeros(
        (len(ALL_METRIC_COMPARISON_ROWS), len(PRIMARY_WIN_DEFINITIONS))
    )
    annotations: list[list[str]] = [
        ["" for _ in PRIMARY_WIN_DEFINITIONS]
        for _ in ALL_METRIC_COMPARISON_ROWS
    ]
    for row_index, (evaluation, metric, _) in enumerate(ALL_METRIC_COMPARISON_ROWS):
        for column_index, definition in enumerate(PRIMARY_WIN_DEFINITIONS):
            cell = panel[
                (panel["evaluation"] == evaluation)
                & (panel["metric"] == metric)
                & (panel["definition"] == definition)
            ]
            assert len(cell) == 1
            record = cell.iloc[0]
            advantage[row_index, column_index] = (
                record["candidate_earlier"] - record["auprc_earlier"]
            ) / len(AUGMENTED_SUBSETS)
            annotations[row_index][column_index] = (
                f"{record['candidate_earlier']} / {record['same_step']} / "
                f"{record['auprc_earlier']} / {record['neither_detected']}"
            )

    fig, axis = plt.subplots(figsize=(12.8, 8.6))
    image = axis.imshow(
        advantage,
        cmap="RdYlGn",
        vmin=-0.75,
        vmax=0.75,
        aspect="auto",
    )
    axis.set_xticks(np.arange(len(definition_labels)), definition_labels)
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.tick_params(axis="x", labelrotation=18)
    for row_index, row in enumerate(annotations):
        for column_index, annotation in enumerate(row):
            axis.text(
                column_index,
                row_index,
                annotation,
                ha="center",
                va="center",
                fontsize=9,
            )
    axis.axhline(6.5, color="#222222", linewidth=1.5)
    axis.axhline(10.5, color="#222222", linewidth=1.5)
    axis.set_title(
        "Which alternatives identify the mapped home arm earlier than AUPRC?"
    )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    colorbar.set_label("(alternative earlier − AUPRC earlier) / 8")
    fig.text(
        0.5,
        0.015,
        "Cells are alternative earlier / same step / AUPRC earlier / neither. "
        "Every rule requires two consecutive qualifying checkpoints. "
        "Bootstrap rows use joint resamples; the final Cohen's d row uses "
        "independent normals and closed-form SEs.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    svg_path = output_dir / "augmented_all_metric_win_sensitivity.svg"
    png_path = output_dir / "augmented_all_metric_win_sensitivity.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {
        "plot_augmented_all_metric_win_sensitivity_svg": svg_path,
        "plot_augmented_all_metric_win_sensitivity_png": png_path,
    }


def plot_augmented_win_definition_sensitivity(
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot how the metric comparison changes with the definition of a win."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count_columns = [
        "candidate_earlier",
        "same_step",
        "auprc_earlier",
        "neither_detected",
    ]
    colors = ["#2A9D8F", "#B8B8B8", "#E76F51", "#F1E6D2"]
    labels = ["Cohen's d earlier", "Same step", "AUPRC earlier", "Neither"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharey=True)
    for axis, persistence in zip(axes, (1, 2)):
        panel = (
            comparisons[comparisons["persistence"] == persistence]
            .set_index("definition")
            .reindex(WIN_DEFINITION_LABELS)
        )
        x = np.arange(len(panel))
        bottom = np.zeros(len(panel))
        for column, color, label in zip(count_columns, colors, labels):
            values = panel[column].to_numpy(dtype=int)
            axis.bar(x, values, bottom=bottom, color=color, label=label)
            for x_value, value, base in zip(x, values, bottom):
                if value:
                    axis.text(
                        x_value,
                        base + value / 2,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=9,
                    )
            bottom += values
        axis.set_xticks(x, panel["definition_label"], rotation=24, ha="right")
        axis.set_ylim(0, len(AUGMENTED_SUBSETS))
        axis.set_ylabel("Specialist subsets")
        axis.set_title(
            "First qualifying checkpoint"
            if persistence == 1
            else "First of two consecutive qualifying checkpoints"
        )
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Does the conclusion depend on what counts as a specialist win?")
    fig.text(
        0.5,
        0.005,
        "AUPRC probabilities use joint class-stratified variant bootstrap draws; "
        "Cohen's d probabilities use independent normals with closed-form SEs. "
        "Each bar totals eight mapped specialist subsets.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.94), w_pad=2.0)
    svg_path = output_dir / "augmented_win_definition_sensitivity.svg"
    png_path = output_dir / "augmented_win_definition_sensitivity.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {
        "plot_augmented_win_definition_sensitivity_svg": svg_path,
        "plot_augmented_win_definition_sensitivity_png": png_path,
    }


def compute_closed_form_cohen_d_specialist_detectability(
    cohen_d_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Test whether home-arm d exceeds the strongest non-home d."""
    required = {"subset", "step", "arm", "value", "se"}
    assert required.issubset(cohen_d_metrics.columns), (
        "Cohen's d table missing columns: "
        f"{sorted(required - set(cohen_d_metrics.columns))}"
    )
    rows: list[dict[str, str | float | int | bool]] = []
    for subset in AUGMENTED_SUBSETS:
        home_arm = AUGMENTED_SPECIALIST_ARM[subset]
        for step in AUGMENTED_STEPS:
            cell = cohen_d_metrics[
                (cohen_d_metrics["subset"] == subset)
                & (cohen_d_metrics["step"] == step)
            ]
            assert set(cell["arm"]) == set(AUGMENTED_ARMS)
            indexed = cell.set_index("arm")
            home = indexed.loc[home_arm]
            nonhome = (
                cell[cell["arm"] != home_arm]
                .sort_values(["value", "arm"], ascending=[False, True])
                .iloc[0]
            )
            home_value = float(home["value"])
            competitor_value = float(nonhome["value"])
            margin = home_value - competitor_value
            margin_se = float(np.hypot(home["se"], nonhome["se"]))
            margin_ci_low = margin - NORMAL_95_Z * margin_se
            margin_ci_high = margin + NORMAL_95_Z * margin_se
            rows.append(
                {
                    "subset": subset,
                    "step": step,
                    "home_arm": home_arm,
                    "metric": VARIANT_POOLED_SMD,
                    "higher_is_better": True,
                    "home_rank": 1 + int((cell["value"] > home_value).sum()),
                    "strongest_competitor": str(nonhome["arm"]),
                    "home_value": home_value,
                    "best_nonhome_value": competitor_value,
                    "home_minus_best_oriented": margin,
                    "margin_se": margin_se,
                    "margin_ci_low": margin_ci_low,
                    "margin_ci_high": margin_ci_high,
                    "bootstrap_home_rank1_frequency": float("nan"),
                    "confidence_supported": bool(margin_ci_low > 0),
                    "uncertainty_method": (
                        "independent_normal_closed_form_cohen_d"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    expected_rows = len(AUGMENTED_SUBSETS) * len(AUGMENTED_STEPS)
    assert len(result) == expected_rows
    return result


def _write_table_parts(
    output_dir: Path,
    tables: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        outputs[name] = path
    return outputs


def run_augmented_analysis(
    output_dir: Path,
    *,
    n_bootstrap: int = 1000,
    seed: int = 459,
) -> dict[str, Path]:
    """Run the six-arm, eight-subset replacement assessment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = augmented_manifest()
    manifest_path = output_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    reference = add_llr_scores(
        read_score_bundle(exp232_score_uri(TAU_REFERENCE_ARM, TAU_REFERENCE_STEP))
    )
    reference = reference[reference["subset"].isin(NON_DISTAL_SUBSETS)]
    tau = reference_soft_win_temperature(
        reference["label"],
        reference["minus_llr_avg"],
        reference["match_group"],
    )
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "arms": list(AUGMENTED_ARMS),
                "bootstrap_seed": seed,
                "dataset": DATASET,
                "distal_replacement": "exp351 centered (issue #351)",
                "n_bootstrap": n_bootstrap,
                "soft_win_tau": tau,
                "split": SPLIT,
                "steps": list(AUGMENTED_STEPS),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    point_parts: list[pd.DataFrame] = []
    pairwise_parts: list[pd.DataFrame] = []
    detectability_parts: list[pd.DataFrame] = []
    ungrouped_point_parts: list[pd.DataFrame] = []
    ungrouped_pairwise_parts: list[pd.DataFrame] = []
    ungrouped_detectability_parts: list[pd.DataFrame] = []
    stored_auprc_parts: list[pd.DataFrame] = []

    for step in AUGMENTED_STEPS:
        bundles = {
            arm: add_llr_scores(read_score_bundle(augmented_score_uri(arm, step)))
            for arm in AUGMENTED_ARMS
        }
        validate_aligned_bundles(bundles)
        stored_auprc_parts.extend(
            read_stored_auprc(arm, step) for arm in AUGMENTED_ARMS
        )
        row_metadata = next(iter(bundles.values()))[
            ["label", "subset", "match_group"]
        ]
        for subset in AUGMENTED_SUBSETS:
            keep = row_metadata["subset"] == subset
            subset_metadata = row_metadata.loc[keep].reset_index(drop=True)
            assert not subset_metadata.empty, f"no rows for subset {subset!r}"
            scores = pd.DataFrame(
                {
                    arm: frame.loc[keep, "minus_llr_avg"].reset_index(drop=True)
                    for arm, frame in bundles.items()
                }
            )

            point, samples = joint_cluster_bootstrap_soft_metrics(
                subset_metadata["label"],
                scores,
                subset_metadata["match_group"],
                tau=tau,
                n_bootstrap=n_bootstrap,
                rng=seed,
            )
            summary = summarize_joint_bootstrap(point, samples).rename(
                columns={"score_type": "arm"}
            )
            summary["step"] = step
            summary["subset"] = subset
            summary["score_protocol"] = "minus_llr_avg"
            summary["bootstrap_unit"] = "match_group"
            point_parts.append(summary)

            pairwise = pairwise_bootstrap_summary(point, samples)
            pairwise["step"] = step
            pairwise["subset"] = subset
            pairwise_parts.append(pairwise)

            detectability = specialist_detectability_summary(
                point,
                samples,
                subset=subset,
                arms=AUGMENTED_ARMS,
                specialist_by_subset=AUGMENTED_SPECIALIST_ARM,
            )
            detectability["step"] = step
            detectability_parts.append(detectability)

            ungrouped_point, ungrouped_samples = (
                joint_stratified_row_bootstrap_ungrouped_metrics(
                    subset_metadata["label"],
                    scores,
                    n_bootstrap=n_bootstrap,
                    rng=seed,
                )
            )
            ungrouped_summary = summarize_joint_bootstrap(
                ungrouped_point,
                ungrouped_samples,
            ).rename(columns={"score_type": "arm"})
            ungrouped_summary["step"] = step
            ungrouped_summary["subset"] = subset
            ungrouped_summary["score_protocol"] = "minus_llr_avg"
            ungrouped_summary["bootstrap_unit"] = "class_stratified_variant"
            ungrouped_point_parts.append(ungrouped_summary)

            ungrouped_pairwise = pairwise_bootstrap_summary(
                ungrouped_point,
                ungrouped_samples,
            )
            ungrouped_pairwise["step"] = step
            ungrouped_pairwise["subset"] = subset
            ungrouped_pairwise_parts.append(ungrouped_pairwise)

            ungrouped_detectability = specialist_detectability_summary(
                ungrouped_point,
                ungrouped_samples,
                subset=subset,
                metrics=UNGROUPED_DETECTABILITY_METRICS,
                arms=AUGMENTED_ARMS,
                specialist_by_subset=AUGMENTED_SPECIALIST_ARM,
            )
            ungrouped_detectability["step"] = step
            ungrouped_detectability_parts.append(ungrouped_detectability)

    point_metrics = pd.concat(point_parts, ignore_index=True)
    pairwise_deltas = pd.concat(pairwise_parts, ignore_index=True)
    specialist_detectability = pd.concat(detectability_parts, ignore_index=True)
    ungrouped_point_metrics = pd.concat(ungrouped_point_parts, ignore_index=True)
    cohen_d_metrics = cohen_d_closed_form_table(ungrouped_point_metrics)
    cohen_d_win_probabilities = compute_closed_form_cohen_d_win_table(
        cohen_d_metrics
    )
    cohen_d_rank1_probabilities = (
        compute_closed_form_cohen_d_rank1_probabilities(cohen_d_metrics)
    )
    cohen_d_closed_form_detectability = (
        compute_closed_form_cohen_d_specialist_detectability(cohen_d_metrics)
    )
    ungrouped_pairwise_deltas = pd.concat(
        ungrouped_pairwise_parts,
        ignore_index=True,
    )
    ungrouped_specialist_detectability = pd.concat(
        ungrouped_detectability_parts,
        ignore_index=True,
    )
    auprc_bootstrap_detectability = ungrouped_specialist_detectability[
        ungrouped_specialist_detectability["metric"] == AUPRC
    ].copy()
    hybrid_detectability = pd.concat(
        [auprc_bootstrap_detectability, cohen_d_closed_form_detectability],
        ignore_index=True,
    )

    specialist_detection_timing = persistent_specialist_detectability(
        specialist_detectability,
        synchronized_steps=AUGMENTED_STEPS,
    )
    metric_detection_comparison = compare_metric_detection_timing(
        specialist_detection_timing,
        subsets=AUGMENTED_SUBSETS,
    )
    ungrouped_specialist_detection_timing = persistent_specialist_detectability(
        ungrouped_specialist_detectability,
        synchronized_steps=AUGMENTED_STEPS,
    )
    ungrouped_metric_detection_comparison = compare_metric_detection_timing(
        ungrouped_specialist_detection_timing,
        metrics=UNGROUPED_DETECTABILITY_METRICS,
        subsets=AUGMENTED_SUBSETS,
    )
    hybrid_detection_timing = persistent_specialist_detectability(
        hybrid_detectability,
        synchronized_steps=AUGMENTED_STEPS,
    )
    hybrid_detection_comparison = compare_metric_detection_timing(
        hybrid_detection_timing,
        metrics=(AUPRC, VARIANT_POOLED_SMD),
        subsets=AUGMENTED_SUBSETS,
    )
    win_definition_timing, win_definition_comparison = (
        compute_win_definition_sensitivity(
            auprc_bootstrap_detectability,
            cohen_d_closed_form_detectability,
            cohen_d_rank1_probabilities,
        )
    )
    matched_metric_win_definition_timing, matched_metric_win_definition_comparison = (
        compute_bootstrap_metric_win_definition_sensitivity(
            specialist_detectability,
            metrics=DETECTABILITY_METRICS,
        )
    )
    (
        ungrouped_metric_win_definition_timing,
        ungrouped_metric_win_definition_comparison,
    ) = compute_bootstrap_metric_win_definition_sensitivity(
        ungrouped_specialist_detectability,
        metrics=UNGROUPED_DETECTABILITY_METRICS,
    )
    all_metric_win_definition_comparison = (
        combine_all_metric_win_definition_comparisons(
            matched_metric_win_definition_comparison,
            ungrouped_metric_win_definition_comparison,
            win_definition_comparison,
        )
    )


    stored_auprc = pd.concat(stored_auprc_parts, ignore_index=True)
    reproduced_auprc = point_metrics[point_metrics["metric"] == AUPRC][
        ["arm", "step", "subset", "value"]
    ].merge(
        stored_auprc,
        on=["arm", "step", "subset"],
        how="left",
        validate="one_to_one",
    )
    assert reproduced_auprc["stored_value"].notna().all()
    reproduced_auprc["difference"] = (
        reproduced_auprc["value"] - reproduced_auprc["stored_value"]
    )
    assert np.allclose(reproduced_auprc["difference"], 0.0, atol=1e-12)

    rank_agreement, rank_reversals = compute_rank_agreement(point_metrics)
    confident_rank_reversals = confidence_filtered_rank_reversals(
        pairwise_deltas,
        group_columns=["step", "subset"],
        entity_columns=("arm_a", "arm_b"),
    )
    ungrouped_rank_agreement, ungrouped_rank_reversals = compute_rank_agreement(
        ungrouped_point_metrics
    )
    ungrouped_confident_rank_reversals = confidence_filtered_rank_reversals(
        ungrouped_pairwise_deltas,
        group_columns=["step", "subset"],
        entity_columns=("arm_a", "arm_b"),
    )

    outputs = {"manifest": manifest_path, "metadata": metadata_path}
    outputs.update(
        _write_table_parts(
            output_dir,
            {
                "point_metrics": point_metrics,
                "pairwise_deltas": pairwise_deltas,
                "specialist_detectability": specialist_detectability,
                "specialist_detection_timing": specialist_detection_timing,
                "metric_detection_comparison": metric_detection_comparison,
                "ungrouped_point_metrics": ungrouped_point_metrics,
                "cohen_d_closed_form": cohen_d_metrics,
                "cohen_d_closed_form_win_probabilities": cohen_d_win_probabilities,
                "cohen_d_closed_form_rank1_probabilities": (
                    cohen_d_rank1_probabilities
                ),
                "win_definition_timing": win_definition_timing,
                "win_definition_comparison": win_definition_comparison,
                "auprc_bootstrap_cohen_d_closed_form_detectability": (
                    hybrid_detectability
                ),
                "matched_metric_win_definition_timing": (
                    matched_metric_win_definition_timing
                ),
                "matched_metric_win_definition_comparison": (
                    matched_metric_win_definition_comparison
                ),
                "ungrouped_metric_win_definition_timing": (
                    ungrouped_metric_win_definition_timing
                ),
                "ungrouped_metric_win_definition_comparison": (
                    ungrouped_metric_win_definition_comparison
                ),
                "all_metric_win_definition_comparison": (
                    all_metric_win_definition_comparison
                ),
                "auprc_bootstrap_cohen_d_closed_form_detection_timing": (
                    hybrid_detection_timing
                ),
                "auprc_bootstrap_cohen_d_closed_form_detection_comparison": (
                    hybrid_detection_comparison
                ),
                "ungrouped_pairwise_deltas": ungrouped_pairwise_deltas,
                "ungrouped_specialist_detectability": (
                    ungrouped_specialist_detectability
                ),
                "ungrouped_specialist_detection_timing": (
                    ungrouped_specialist_detection_timing
                ),
                "ungrouped_metric_detection_comparison": (
                    ungrouped_metric_detection_comparison
                ),
                "auprc_reproduction": reproduced_auprc,
                "rank_agreement": rank_agreement,
                "rank_reversals": rank_reversals,
                "confident_rank_reversals": confident_rank_reversals,
                "ungrouped_rank_agreement": ungrouped_rank_agreement,
                "ungrouped_rank_reversals": ungrouped_rank_reversals,
                "ungrouped_confident_rank_reversals": (
                    ungrouped_confident_rank_reversals
                ),
            },
        )
    )
    outputs.update(
        plot_augmented_distal_trajectories(
            ungrouped_point_metrics,
            cohen_d_metrics,
            output_dir / "plots",
        )
    )
    outputs.update(
        plot_augmented_specialist_trajectories(
            ungrouped_point_metrics,
            cohen_d_metrics,
            output_dir / "plots",
        )
    )
    outputs.update(
        plot_augmented_specialist_closed_form_win_percentage(
            cohen_d_win_probabilities,
            output_dir / "plots",
        )
    )
    outputs.update(
        plot_augmented_win_definition_sensitivity(
            win_definition_comparison,
            output_dir / "plots",
        )
    )
    outputs.update(
        plot_metric_detectability_summary(
            hybrid_detection_timing,
            hybrid_detection_comparison,
            output_dir / "plots",
            metrics=(AUPRC, VARIANT_POOLED_SMD),
            stem="augmented_auprc_vs_cohen_d_detection_summary",
            metric_note="Counts compare the eight mapped specialist subsets.",
            detection_note=(
                "Detection = first of two consecutive synchronized steps whose "
                "home-minus-best-non-home 95% interval is above zero.\n"
                "AUPRC "
                "uses a class-stratified variant bootstrap; Cohen's d uses the "
                "independent closed-form normal approximation."
            ),
            title="When does the mapped home arm separate?",
            subsets=AUGMENTED_SUBSETS,
        )
    )
    outputs.update(
        plot_augmented_all_metric_win_sensitivity(
            all_metric_win_definition_comparison,
            output_dir / "plots",
        )
    )
    outputs.update(
        plot_metric_detectability_summary(
            specialist_detection_timing,
            metric_detection_comparison,
            output_dir / "plots",
            stem="augmented_specialist_metric_detectability_summary",
            title=(
                "exp232 assessment with exp351-centered replacing the distal arm"
            ),
            subsets=AUGMENTED_SUBSETS,
        )
    )
    outputs.update(
        plot_metric_detectability_summary(
            ungrouped_specialist_detection_timing,
            ungrouped_metric_detection_comparison,
            output_dir / "plots",
            metrics=UNGROUPED_DETECTABILITY_METRICS,
            stem="augmented_ungrouped_metric_detectability_summary",
            bootstrap_unit="class-stratified variant",
            metric_note="No match groups are used in this sensitivity analysis.",
            title=(
                "No-group sensitivity with exp351-centered replacing the distal arm"
            ),
            subsets=AUGMENTED_SUBSETS,
        )
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=459)
    args = parser.parse_args()
    outputs = run_augmented_analysis(
        args.output_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
