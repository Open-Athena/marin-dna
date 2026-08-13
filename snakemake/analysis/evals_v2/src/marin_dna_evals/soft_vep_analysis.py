"""Bounded CPU analysis for issue #459's exp232 specialist trajectories.

This module consumes the 48 existing development-split Mendelian score
parquets. It does not run model inference and does not infer missing
checkpoints. Each step is loaded and released independently so the working set
stays bounded well below the shared-node 500 MiB limit.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import kendalltau, spearmanr

from marin_dna_evals.soft_vep_metrics import (
    AUPRC,
    HIGHER_IS_BETTER,
    MEAN_GAP_GLOBAL,
    MEAN_GAP_GROUP,
    SOFT_WIN,
    compute_mendelian_soft_metric_table,
    group_differences,
    joint_cluster_bootstrap_soft_metrics,
    reference_soft_win_temperature,
    summarize_joint_bootstrap,
)

S3_SCORE_ROOT = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
DATASET = "mendelian_traits"
SPLIT = "train"

EXP232_STEPS: dict[str, tuple[int, ...]] = {
    "bg": (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999),
    "cds": (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999),
    "utr3": (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999),
    "ncrna_exon": (500, 1000, 1500, 2000, 3000, 3500, 4000, 4500, 4999),
    "tss_region_and_utr5": (
        500,
        1000,
        1500,
        2000,
        3000,
        3500,
        4000,
        4500,
        4999,
    ),
}
ARMS = tuple(EXP232_STEPS)
SYNCHRONIZED_STEPS = tuple(
    sorted(set.intersection(*(set(v) for v in EXP232_STEPS.values())))
)

NON_DISTAL_SUBSETS = (
    "missense_variant",
    "synonymous_variant",
    "splicing",
    "3_prime_UTR_variant",
    "non_coding_transcript_exon_variant",
    "5_prime_UTR_variant",
    "tss_proximal",
)
SPECIALIST_ARM = {
    "missense_variant": "cds",
    "synonymous_variant": "cds",
    "splicing": "cds",
    "3_prime_UTR_variant": "utr3",
    "non_coding_transcript_exon_variant": "ncrna_exon",
    "5_prime_UTR_variant": "tss_region_and_utr5",
    "tss_proximal": "tss_region_and_utr5",
}

TAU_REFERENCE_ARM = "bg"
TAU_REFERENCE_STEP = 500
RESCALING_CONSTANTS = {
    "bg": 0.5,
    "cds": 1.0,
    "utr3": 2.0,
    "ncrna_exon": 4.0,
    "tss_region_and_utr5": 8.0,
}

REQUIRED_COLUMNS = ("label", "subset", "match_group", "llr_fwd", "llr_rc")

ARM_COLORS = {
    "bg": "#7f7f7f",
    "cds": "#0072B2",
    "utr3": "#D55E00",
    "ncrna_exon": "#009E73",
    "tss_region_and_utr5": "#CC79A7",
}
ARM_LINESTYLES = {
    "bg": "--",
    "cds": "-",
    "utr3": "-.",
    "ncrna_exon": ":",
    "tss_region_and_utr5": (0, (5, 1)),
}
METRIC_LABELS = {
    "auprc": "AUPRC",
    "mean_gap_global": "Global positive-minus-negative mean score gap",
    "mean_gap_group": "Group-balanced mean score gap",
    "group_smd": "Mean group difference / SD",
    "group_median_mad": "Median group difference / scaled MAD",
    "soft_win": "Fixed-temperature soft pairwise win rate",
    "calibrated_log_loss": "Grouped-CV calibrated log loss (lower is better)",
    "calibrated_brier": "Grouped-CV calibrated Brier score (lower is better)",
}
METRIC_AXIS_LABELS = {
    "auprc": "AUPRC",
    "mean_gap_global": "Global mean gap",
    "mean_gap_group": "Group-balanced mean gap",
    "group_smd": "Mean group SMD",
    "group_median_mad": "Median group difference / MAD",
    "soft_win": "SoftWin",
    "calibrated_log_loss": "Calibrated log loss (lower is better)",
    "calibrated_brier": "Calibrated Brier score (lower is better)",
}


def exp232_score_uri(arm: str, step: int) -> str:
    """S3 URI for one immutable exp232 score bundle."""
    assert arm in EXP232_STEPS, f"unknown exp232 arm {arm!r}"
    assert step in EXP232_STEPS[arm], f"step {step} is not in the {arm!r} manifest"
    return f"{S3_SCORE_ROOT}/exp232-v4_{arm}-step-{step}/{DATASET}.parquet"


def exp232_metric_uri(arm: str, step: int) -> str:
    """S3 URI for the stored evals_v2 AUPRC parquet paired to a score bundle."""
    score_uri = exp232_score_uri(arm, step)
    return score_uri.replace("/results/scores/", "/results/metrics/")


def exp232_manifest() -> pd.DataFrame:
    """The explicit 48-object inventory; missing steps are never interpolated."""
    rows = [
        {
            "arm": arm,
            "step": step,
            "model": f"exp232-v4_{arm}-step-{step}",
            "uri": exp232_score_uri(arm, step),
            "synchronized": step in SYNCHRONIZED_STEPS,
            "dataset": DATASET,
            "split": SPLIT,
        }
        for arm, steps in EXP232_STEPS.items()
        for step in steps
    ]
    manifest = pd.DataFrame(rows).sort_values(["step", "arm"]).reset_index(drop=True)
    assert len(manifest) == 48, (
        f"exp232 manifest drifted: expected 48, got {len(manifest)}"
    )
    return manifest


def read_score_bundle(uri: str) -> pd.DataFrame:
    """Read only the five columns required by the soft-metric analysis."""
    return pl.read_parquet(
        uri,
        columns=list(REQUIRED_COLUMNS),
        storage_options={"aws_region": "us-east-2"},
    ).to_pandas()


def read_stored_auprc(arm: str, step: int) -> pd.DataFrame:
    """Read stored ``minus_llr_avg`` AUPRC rows for parity checking."""
    metrics = pl.read_parquet(
        exp232_metric_uri(arm, step),
        columns=["score_type", "subset", "value", "split"],
        storage_options={"aws_region": "us-east-2"},
    ).filter((pl.col("score_type") == "minus_llr_avg") & (pl.col("split") == SPLIT))
    result = metrics.select(
        ["subset", pl.col("value").alias("stored_value")]
    ).to_pandas()
    result["arm"] = arm
    result["step"] = step
    return result


def add_llr_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Materialize the signed Mendelian FWD and FWD/RC-average score."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    assert not missing, f"score bundle missing required columns: {sorted(missing)}"
    result = frame.copy()
    # Match workflow/rules/metrics.smk exactly: the persisted atoms are float32,
    # and pandas performs this addition/division in float32. Upcasting before
    # averaging can break score ties differently and therefore change AUPRC.
    result["minus_llr_avg"] = -((result["llr_fwd"] + result["llr_rc"]) / 2)
    result["minus_llr_fwd"] = -result["llr_fwd"]
    return result


def validate_aligned_bundles(bundles: dict[str, pd.DataFrame]) -> None:
    """Assert every arm evaluates the same labeled rows in the same order."""
    assert bundles, "at least one bundle is required"
    reference_arm = next(iter(bundles))
    reference = bundles[reference_arm][["label", "subset", "match_group"]].reset_index(
        drop=True
    )
    for arm, frame in bundles.items():
        observed = frame[["label", "subset", "match_group"]].reset_index(drop=True)
        assert observed.equals(reference), (
            f"row metadata for arm {arm!r} differs from {reference_arm!r}; "
            "joint model bootstrap would be invalid"
        )


def _strict_order(values: pd.Series) -> str:
    return ">".join(
        values.sort_values(ascending=False, kind="stable").index.astype(str)
    )


def _top_k(values: pd.Series, k: int = 3) -> set[str]:
    return set(values.nlargest(min(k, len(values))).index.astype(str))


def compute_rank_agreement(
    point_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every metric ordering with same-step and final-step AUPRC."""
    required = {"arm", "step", "subset", "metric", "value"}
    assert required.issubset(point_metrics.columns), (
        f"point metrics missing columns: {sorted(required - set(point_metrics.columns))}"
    )
    final_auprc = point_metrics[
        (point_metrics["step"] == 4999) & (point_metrics["metric"] == AUPRC)
    ].set_index(["subset", "arm"])["value"]

    summary_rows: list[dict[str, str | float | int]] = []
    pair_rows: list[dict[str, str | float | int | bool]] = []
    for (step, subset), cell in point_metrics.groupby(["step", "subset"], sort=True):
        metric_values = cell.pivot(index="arm", columns="metric", values="value")
        if AUPRC not in metric_values:
            continue
        references = {
            "same_step_auprc": metric_values[AUPRC],
            "final_step_auprc": final_auprc.loc[subset].reindex(metric_values.index),
        }
        for metric in metric_values.columns:
            oriented = metric_values[metric] * (
                1.0 if HIGHER_IS_BETTER[metric] else -1.0
            )
            for reference_name, reference in references.items():
                paired = pd.concat(
                    [oriented.rename("metric"), reference.rename("reference")], axis=1
                ).dropna()
                if len(paired) < 2:
                    continue
                rho = float(spearmanr(paired["metric"], paired["reference"]).statistic)
                tau = float(kendalltau(paired["metric"], paired["reference"]).statistic)
                reversals = 0
                for arm_a, arm_b in combinations(paired.index.astype(str), 2):
                    metric_delta = float(
                        paired.loc[arm_a, "metric"] - paired.loc[arm_b, "metric"]
                    )
                    reference_delta = float(
                        paired.loc[arm_a, "reference"] - paired.loc[arm_b, "reference"]
                    )
                    reversal = metric_delta * reference_delta < 0
                    reversals += int(reversal)
                    pair_rows.append(
                        {
                            "step": int(step),
                            "subset": str(subset),
                            "metric": str(metric),
                            "reference": reference_name,
                            "arm_a": arm_a,
                            "arm_b": arm_b,
                            "metric_delta_oriented": metric_delta,
                            "reference_delta": reference_delta,
                            "reversal": reversal,
                        }
                    )
                summary_rows.append(
                    {
                        "step": int(step),
                        "subset": str(subset),
                        "metric": str(metric),
                        "reference": reference_name,
                        "spearman": rho,
                        "kendall": tau,
                        "top3_overlap": len(
                            _top_k(paired["metric"]) & _top_k(paired["reference"])
                        ),
                        "pairwise_reversals": reversals,
                        "n_arms": len(paired),
                        "metric_order": _strict_order(paired["metric"]),
                        "reference_order": _strict_order(paired["reference"]),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def earliest_persistent_specialist_wins(point_metrics: pd.DataFrame) -> pd.DataFrame:
    """Earliest synchronized step with strict specialist wins twice in a row."""
    rows: list[dict[str, str | int | bool | None]] = []
    synchronized = point_metrics[point_metrics["step"].isin(SYNCHRONIZED_STEPS)]
    for subset, specialist in SPECIALIST_ARM.items():
        subset_frame = synchronized[synchronized["subset"] == subset]
        for metric, metric_frame in subset_frame.groupby("metric", sort=False):
            winners: list[tuple[int, bool]] = []
            for step, step_frame in metric_frame.groupby("step", sort=True):
                if set(step_frame["arm"]) != set(ARMS):
                    continue
                sign = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
                values = step_frame.set_index("arm")["value"] * sign
                specialist_value = float(values.loc[specialist])
                strict_win = bool((specialist_value > values.drop(specialist)).all())
                winners.append((int(step), strict_win))
            earliest: int | None = None
            for (step_a, win_a), (_, win_b) in zip(winners, winners[1:]):
                if win_a and win_b:
                    earliest = step_a
                    break
            rows.append(
                {
                    "subset": subset,
                    "specialist_arm": specialist,
                    "metric": str(metric),
                    "earliest_persistent_step": earliest,
                    "final_step_win": dict(winners).get(4999, False),
                }
            )
    return pd.DataFrame(rows)


def pairwise_bootstrap_summary(
    point: pd.DataFrame,
    bootstrap_samples: pd.DataFrame,
) -> pd.DataFrame:
    """Paired arm deltas from the shared bootstrap draws for one cell."""
    point_wide = point.pivot(index="score_type", columns="metric", values="value")
    rows: list[dict[str, str | float]] = []
    for metric, metric_samples in bootstrap_samples.groupby("metric", sort=False):
        sample_wide = metric_samples.pivot(
            index="draw", columns="score_type", values="value"
        )
        sign = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
        for arm_a, arm_b in combinations(sample_wide.columns.astype(str), 2):
            delta = sign * (sample_wide[arm_a] - sample_wide[arm_b])
            rows.append(
                {
                    "metric": str(metric),
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "delta_oriented": float(
                        sign
                        * (
                            point_wide.loc[arm_a, metric]
                            - point_wide.loc[arm_b, metric]
                        )
                    ),
                    "se": float(np.nanstd(delta, ddof=1)),
                    "ci_low": float(np.nanpercentile(delta, 2.5)),
                    "ci_high": float(np.nanpercentile(delta, 97.5)),
                    "probability_a_better": float(np.nanmean(delta > 0)),
                }
            )
    return pd.DataFrame(rows)


def confidence_filtered_rank_reversals(
    pairwise_deltas: pd.DataFrame,
    *,
    group_columns: list[str],
    entity_columns: tuple[str, str],
) -> pd.DataFrame:
    """Rank reversals after dropping pairs unresolved for either metric or AUPRC."""
    entity_a, entity_b = entity_columns
    required = {
        *group_columns,
        "metric",
        entity_a,
        entity_b,
        "ci_low",
        "ci_high",
    }
    assert required.issubset(pairwise_deltas.columns), (
        f"pairwise deltas missing columns: {sorted(required - set(pairwise_deltas.columns))}"
    )

    rows: list[dict[str, str | float | int]] = []
    grouped = pairwise_deltas.groupby(group_columns, sort=False, dropna=False)
    for group_key, group in grouped:
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        context = dict(zip(group_columns, keys))
        auprc = group[group["metric"] == AUPRC].set_index([entity_a, entity_b])
        for metric, metric_rows in group.groupby("metric", sort=False):
            if metric == AUPRC:
                continue
            candidate = metric_rows.set_index([entity_a, entity_b])
            common = candidate.index.intersection(auprc.index)
            reversals = 0
            informative = 0
            for pair in common:
                candidate_row = candidate.loc[pair]
                auprc_row = auprc.loc[pair]
                candidate_direction = (
                    1
                    if candidate_row["ci_low"] > 0
                    else -1
                    if candidate_row["ci_high"] < 0
                    else 0
                )
                auprc_direction = (
                    1
                    if auprc_row["ci_low"] > 0
                    else -1
                    if auprc_row["ci_high"] < 0
                    else 0
                )
                if candidate_direction == 0 or auprc_direction == 0:
                    continue
                informative += 1
                reversals += int(candidate_direction != auprc_direction)
            rows.append(
                {
                    **context,
                    "metric": str(metric),
                    "n_informative_pairs": informative,
                    "pairwise_reversals": reversals,
                    "agreement_fraction": (
                        float((informative - reversals) / informative)
                        if informative
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_supported_specialist_wins(
    pairwise_deltas: pd.DataFrame,
) -> pd.DataFrame:
    """Persistent specialist wins whose paired 95% intervals clear every arm."""
    rows: list[dict[str, str | int | bool | None]] = []
    synchronized = pairwise_deltas[pairwise_deltas["step"].isin(SYNCHRONIZED_STEPS)]
    for subset, specialist in SPECIALIST_ARM.items():
        subset_frame = synchronized[synchronized["subset"] == subset]
        for metric, metric_frame in subset_frame.groupby("metric", sort=False):
            supported_steps: list[tuple[int, bool]] = []
            for step, step_frame in metric_frame.groupby("step", sort=True):
                wins = []
                for competitor in set(ARMS) - {specialist}:
                    direct = step_frame[
                        (step_frame["arm_a"] == specialist)
                        & (step_frame["arm_b"] == competitor)
                    ]
                    reverse = step_frame[
                        (step_frame["arm_a"] == competitor)
                        & (step_frame["arm_b"] == specialist)
                    ]
                    assert len(direct) + len(reverse) == 1
                    wins.append(
                        bool(direct.iloc[0]["ci_low"] > 0)
                        if len(direct)
                        else bool(reverse.iloc[0]["ci_high"] < 0)
                    )
                supported_steps.append((int(step), all(wins)))
            earliest: int | None = None
            for (step_a, win_a), (_, win_b) in zip(
                supported_steps, supported_steps[1:]
            ):
                if win_a and win_b:
                    earliest = step_a
                    break
            rows.append(
                {
                    "subset": subset,
                    "specialist_arm": specialist,
                    "metric": str(metric),
                    "earliest_bootstrap_supported_step": earliest,
                    "final_step_supported": dict(supported_steps).get(4999, False),
                }
            )
    return pd.DataFrame(rows)


def permute_labels_within_groups(
    match_group: pd.Series,
    *,
    rng: np.random.Generator | int | None,
) -> pd.Series:
    """Choose one random positive row per group, retaining the matched design."""
    generator = np.random.default_rng(rng)
    groups = match_group.reset_index(drop=True)
    permuted = np.zeros(len(groups), dtype=int)
    for row_positions in groups.groupby(groups, sort=False).indices.values():
        permuted[generator.choice(row_positions)] = 1
    return pd.Series(permuted, index=match_group.index)


def compute_final_controls(
    bundles: dict[str, pd.DataFrame],
    *,
    tau: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Final-step score-rescaling, sign, permutation, and strand controls."""
    validate_aligned_bundles(bundles)
    metadata = next(iter(bundles.values()))[["label", "subset", "match_group"]]
    parts: list[pd.DataFrame] = []
    for subset in NON_DISTAL_SUBSETS:
        keep = metadata["subset"] == subset
        subset_meta = metadata.loc[keep].reset_index(drop=True)
        average = pd.DataFrame(
            {
                arm: frame.loc[keep, "minus_llr_avg"].reset_index(drop=True)
                for arm, frame in bundles.items()
            }
        )
        variants = {
            "baseline_avg": (subset_meta["label"], average),
            "positive_rescaling": (
                subset_meta["label"],
                average.mul(pd.Series(RESCALING_CONSTANTS)),
            ),
            "sign_reversal": (subset_meta["label"], -average),
            "fwd_only": (
                subset_meta["label"],
                pd.DataFrame(
                    {
                        arm: frame.loc[keep, "minus_llr_fwd"].reset_index(drop=True)
                        for arm, frame in bundles.items()
                    }
                ),
            ),
            "within_group_label_permutation": (
                permute_labels_within_groups(subset_meta["match_group"], rng=seed),
                average,
            ),
        }
        for control, (labels, scores) in variants.items():
            table = compute_mendelian_soft_metric_table(
                labels,
                scores,
                subset_meta["match_group"],
                tau=tau,
            ).rename(columns={"score_type": "arm"})
            table["subset"] = subset
            table["control"] = control
            table["scale_constant"] = table["arm"].map(RESCALING_CONSTANTS)
            parts.append(table)

    controls = pd.concat(parts, ignore_index=True)
    baseline = controls[controls["control"] == "baseline_avg"]
    summary_rows: list[dict[str, str | float | int]] = []
    baseline_auprc = baseline[baseline["metric"] == AUPRC].set_index(["subset", "arm"])[
        "value"
    ]
    for (control, subset, metric), cell in controls.groupby(
        ["control", "subset", "metric"], sort=True
    ):
        values = cell.set_index("arm")["value"]
        baseline_metric = (
            baseline[(baseline["subset"] == subset) & (baseline["metric"] == metric)]
            .set_index("arm")["value"]
            .reindex(values.index)
        )
        reference_auprc = baseline_auprc.loc[subset].reindex(values.index)
        sign = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
        summary_rows.append(
            {
                "control": str(control),
                "subset": str(subset),
                "metric": str(metric),
                "mean_value": float(values.mean()),
                "max_abs_delta_from_baseline": float(
                    np.max(np.abs(values - baseline_metric))
                ),
                "spearman_vs_baseline_metric": float(
                    spearmanr(sign * values, sign * baseline_metric).statistic
                ),
                "spearman_vs_baseline_auprc": float(
                    spearmanr(sign * values, reference_auprc).statistic
                ),
                "n_arms": len(values),
            }
        )

    rescaled = controls[controls["control"] == "positive_rescaling"]
    rescaled_auprc = rescaled[rescaled["metric"] == AUPRC].set_index(["subset", "arm"])[
        "value"
    ]
    assert np.allclose(rescaled_auprc.sort_index(), baseline_auprc.sort_index()), (
        "positive score rescaling changed AUPRC"
    )
    for metric in (MEAN_GAP_GLOBAL, MEAN_GAP_GROUP):
        base_gap = baseline[baseline["metric"] == metric].set_index(["subset", "arm"])[
            "value"
        ]
        scaled_gap = rescaled[rescaled["metric"] == metric].set_index(
            ["subset", "arm"]
        )["value"]
        expected = (
            base_gap
            * pd.Series(RESCALING_CONSTANTS)
            .reindex(base_gap.index.get_level_values("arm"))
            .to_numpy()
        )
        assert np.allclose(scaled_gap.sort_index(), expected.sort_index()), (
            f"{metric} did not scale linearly under the positive-rescaling control"
        )

    reversed_metrics = controls[controls["control"] == "sign_reversal"]
    for metric in (MEAN_GAP_GLOBAL, MEAN_GAP_GROUP):
        base_value = baseline[baseline["metric"] == metric].set_index(
            ["subset", "arm"]
        )["value"]
        reversed_value = reversed_metrics[
            reversed_metrics["metric"] == metric
        ].set_index(["subset", "arm"])["value"]
        assert np.allclose(reversed_value.sort_index(), -base_value.sort_index())
    base_soft_win = baseline[baseline["metric"] == SOFT_WIN].set_index(
        ["subset", "arm"]
    )["value"]
    reversed_soft_win = reversed_metrics[
        reversed_metrics["metric"] == SOFT_WIN
    ].set_index(["subset", "arm"])["value"]
    assert np.allclose(reversed_soft_win.sort_index(), 1.0 - base_soft_win.sort_index())
    return controls, pd.DataFrame(summary_rows)


def plot_exp232_metric_trajectories(
    point_metrics: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """Seven-panel cross-arm trajectory with 95% cluster-bootstrap intervals."""
    data = point_metrics[point_metrics["metric"] == metric]
    assert not data.empty, f"no point metrics found for {metric!r}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(13, 15), sharex=True, squeeze=False)
    flat_axes = axes.flatten()
    for axis, subset in zip(flat_axes, NON_DISTAL_SUBSETS):
        subset_data = data[data["subset"] == subset]
        specialist = SPECIALIST_ARM[subset]
        for arm in ARMS:
            arm_data = subset_data[subset_data["arm"] == arm].sort_values("step")
            if arm_data.empty:
                continue
            linewidth = 2.8 if arm == specialist else 1.35
            zorder = 4 if arm == specialist else 2
            axis.plot(
                arm_data["step"],
                arm_data["value"],
                color=ARM_COLORS[arm],
                linestyle=ARM_LINESTYLES[arm],
                marker="o",
                markersize=3.2,
                linewidth=linewidth,
                zorder=zorder,
            )
            axis.fill_between(
                arm_data["step"],
                arm_data["ci_low"],
                arm_data["ci_high"],
                color=ARM_COLORS[arm],
                alpha=0.09,
                linewidth=0,
                zorder=1,
            )
        axis.set_title(f"{subset}\nspecialist: {specialist}", fontsize=10)
        axis.set_xlabel("Training step")
        axis.set_ylabel(METRIC_AXIS_LABELS[metric])
        axis.grid(alpha=0.25, linewidth=0.7)

    flat_axes[-1].axis("off")
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=ARM_COLORS[arm],
            linestyle=ARM_LINESTYLES[arm],
            marker="o",
            linewidth=1.8,
            label=arm,
        )
        for arm in ARMS
    ]
    flat_axes[-1].legend(
        handles=legend_handles,
        title="Training arm\n(thick line = mapped specialist)",
        loc="center",
        frameon=False,
    )
    qualifier = ""
    if metric in {"calibrated_log_loss", "calibrated_brier"}:
        qualifier = "; calibration uncertainty is conditional on fixed OOF fits"
    fig.suptitle(
        f"exp232 non-distal trajectories — {METRIC_LABELS[metric]}\n"
        f"development split; ribbons are 95% joint match-group bootstrap intervals{qualifier}",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_exp232_trajectories(
    point_metrics: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Render one comparable-scale faceted SVG per metric."""
    outputs: dict[str, Path] = {}
    for metric in METRIC_LABELS:
        path = output_dir / f"{metric}.svg"
        plot_exp232_metric_trajectories(point_metrics, metric, path)
        outputs[f"plot_{metric}"] = path
    return outputs


def _ecdf(values: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=float))
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def plot_exp232_distribution_diagnostics(
    bundles: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, Path]:
    """Final-step POS/NEG score and matched-group difference ECDFs."""
    validate_aligned_bundles(bundles)
    assert set(bundles) == set(ARMS), "distribution diagnostics require all arms"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = next(iter(bundles.values()))[["label", "subset", "match_group"]]
    outputs: dict[str, Path] = {}
    for subset in NON_DISTAL_SUBSETS:
        keep = metadata["subset"] == subset
        subset_meta = metadata.loc[keep].reset_index(drop=True)
        fig, axes = plt.subplots(2, len(ARMS), figsize=(17, 6.5), squeeze=False)
        for column, arm in enumerate(ARMS):
            scores = bundles[arm].loc[keep, "minus_llr_avg"].reset_index(drop=True)
            for label, linestyle, legend_label in (
                (0, "--", "NEG"),
                (1, "-", "POS"),
            ):
                x, y = _ecdf(scores[subset_meta["label"] == label])
                axes[0, column].plot(
                    x,
                    y,
                    color=ARM_COLORS[arm],
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=legend_label,
                )
            differences = group_differences(
                subset_meta["label"], scores, subset_meta["match_group"]
            )
            x, y = _ecdf(differences)
            axes[1, column].plot(x, y, color=ARM_COLORS[arm], linewidth=1.8)
            axes[1, column].axvline(0, color="black", linestyle=":", linewidth=0.8)
            axes[0, column].set_title(
                f"{arm}{' (specialist)' if arm == SPECIALIST_ARM[subset] else ''}",
                fontweight="bold" if arm == SPECIALIST_ARM[subset] else "normal",
                fontsize=10,
            )
            axes[0, column].set_xlabel("Raw −LLR average")
            axes[1, column].set_xlabel("$D_g$: POS − within-group mean NEG")
            for axis in axes[:, column]:
                axis.set_ylabel("Empirical CDF" if column == 0 else "")
                axis.grid(alpha=0.22, linewidth=0.7)
        axes[0, 0].legend(frameon=False, title="Variant label")
        fig.suptitle(
            f"exp232 step 4999 distribution diagnostics — {subset}\n"
            "development split; each arm keeps its raw score scale",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        path = output_dir / f"{subset}.svg"
        fig.savefig(path, format="svg", bbox_inches="tight")
        plt.close(fig)
        outputs[f"distribution_{subset}"] = path
    return outputs


def _write_json(path: Path, payload: dict[str, str | float | int]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_exp232_analysis(
    output_dir: Path,
    *,
    n_bootstrap: int = 1000,
    seed: int = 459,
) -> dict[str, Path]:
    """Run the 48-file exp232 first pass and write compact local artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = exp232_manifest()
    manifest_path = output_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    reference = add_llr_scores(
        read_score_bundle(exp232_score_uri(TAU_REFERENCE_ARM, TAU_REFERENCE_STEP))
    )
    reference = reference[reference["subset"].isin(NON_DISTAL_SUBSETS)]
    tau = reference_soft_win_temperature(
        reference["label"], reference["minus_llr_avg"], reference["match_group"]
    )
    metadata_path = output_dir / "metadata.json"
    _write_json(
        metadata_path,
        {
            "dataset": DATASET,
            "split": SPLIT,
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": seed,
            "soft_win_tau": tau,
            "tau_reference_model": f"exp232-v4_{TAU_REFERENCE_ARM}-step-{TAU_REFERENCE_STEP}",
            "tau_reference_scope": "seven pooled non-distal consequence subsets",
        },
    )

    point_parts: list[pd.DataFrame] = []
    fwd_parts: list[pd.DataFrame] = []
    pairwise_parts: list[pd.DataFrame] = []
    stored_auprc_parts: list[pd.DataFrame] = []
    final_bundles: dict[str, pd.DataFrame] | None = None
    all_steps = sorted({step for steps in EXP232_STEPS.values() for step in steps})
    for step in all_steps:
        active_arms = [arm for arm in ARMS if step in EXP232_STEPS[arm]]
        bundles = {
            arm: add_llr_scores(read_score_bundle(exp232_score_uri(arm, step)))
            for arm in active_arms
        }
        stored_auprc_parts.extend(read_stored_auprc(arm, step) for arm in active_arms)
        validate_aligned_bundles(bundles)
        if step == 4999:
            final_bundles = bundles
        metadata = next(iter(bundles.values()))[["label", "subset", "match_group"]]
        for subset in NON_DISTAL_SUBSETS:
            keep = metadata["subset"] == subset
            subset_meta = metadata.loc[keep].reset_index(drop=True)
            average_scores = pd.DataFrame(
                {
                    arm: frame.loc[keep, "minus_llr_avg"].reset_index(drop=True)
                    for arm, frame in bundles.items()
                }
            )
            point, samples = joint_cluster_bootstrap_soft_metrics(
                subset_meta["label"],
                average_scores,
                subset_meta["match_group"],
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
            point_parts.append(summary)

            pairwise = pairwise_bootstrap_summary(point, samples)
            pairwise["step"] = step
            pairwise["subset"] = subset
            pairwise_parts.append(pairwise)

            fwd_scores = pd.DataFrame(
                {
                    arm: frame.loc[keep, "minus_llr_fwd"].reset_index(drop=True)
                    for arm, frame in bundles.items()
                }
            )
            fwd = compute_mendelian_soft_metric_table(
                subset_meta["label"],
                fwd_scores,
                subset_meta["match_group"],
                tau=tau,
            ).rename(columns={"score_type": "arm"})
            fwd["step"] = step
            fwd["subset"] = subset
            fwd["score_protocol"] = "minus_llr_fwd"
            fwd_parts.append(fwd)

    point_metrics = pd.concat(point_parts, ignore_index=True)
    fwd_metrics = pd.concat(fwd_parts, ignore_index=True)
    pairwise_deltas = pd.concat(pairwise_parts, ignore_index=True)
    stored_auprc = pd.concat(stored_auprc_parts, ignore_index=True)
    reproduced_auprc = point_metrics[point_metrics["metric"] == AUPRC][
        ["arm", "step", "subset", "value"]
    ].merge(
        stored_auprc,
        on=["arm", "step", "subset"],
        how="left",
        validate="one_to_one",
    )
    assert reproduced_auprc["stored_value"].notna().all(), (
        "stored AUPRC parity table is incomplete"
    )
    reproduced_auprc["difference"] = (
        reproduced_auprc["value"] - reproduced_auprc["stored_value"]
    )
    assert np.allclose(reproduced_auprc["difference"], 0.0, atol=1e-12), (
        "computed AUPRC does not reproduce the stored minus_llr_avg values"
    )
    rank_agreement, rank_reversals = compute_rank_agreement(point_metrics)
    confident_rank_reversals = confidence_filtered_rank_reversals(
        pairwise_deltas,
        group_columns=["step", "subset"],
        entity_columns=("arm_a", "arm_b"),
    )
    specialist_wins = earliest_persistent_specialist_wins(point_metrics)
    supported_specialist_wins = bootstrap_supported_specialist_wins(pairwise_deltas)
    assert final_bundles is not None
    controls, control_summary = compute_final_controls(
        final_bundles,
        tau=tau,
        seed=seed,
    )

    outputs = {
        "manifest": manifest_path,
        "metadata": metadata_path,
        "point_metrics": output_dir / "point_metrics.parquet",
        "fwd_metrics": output_dir / "fwd_metrics.parquet",
        "pairwise_deltas": output_dir / "pairwise_deltas.parquet",
        "auprc_reproduction": output_dir / "auprc_reproduction.parquet",
        "rank_agreement": output_dir / "rank_agreement.parquet",
        "rank_reversals": output_dir / "rank_reversals.parquet",
        "confident_rank_reversals": output_dir / "confident_rank_reversals.parquet",
        "specialist_wins": output_dir / "specialist_wins.parquet",
        "supported_specialist_wins": output_dir / "supported_specialist_wins.parquet",
        "controls": output_dir / "controls.parquet",
        "control_summary": output_dir / "control_summary.parquet",
    }
    point_metrics.to_parquet(outputs["point_metrics"], index=False)
    fwd_metrics.to_parquet(outputs["fwd_metrics"], index=False)
    pairwise_deltas.to_parquet(outputs["pairwise_deltas"], index=False)
    reproduced_auprc.to_parquet(outputs["auprc_reproduction"], index=False)
    rank_agreement.to_parquet(outputs["rank_agreement"], index=False)
    rank_reversals.to_parquet(outputs["rank_reversals"], index=False)
    confident_rank_reversals.to_parquet(
        outputs["confident_rank_reversals"], index=False
    )
    specialist_wins.to_parquet(outputs["specialist_wins"], index=False)
    supported_specialist_wins.to_parquet(
        outputs["supported_specialist_wins"], index=False
    )
    controls.to_parquet(outputs["controls"], index=False)
    control_summary.to_parquet(outputs["control_summary"], index=False)
    outputs.update(plot_exp232_trajectories(point_metrics, output_dir / "plots"))
    outputs.update(
        plot_exp232_distribution_diagnostics(
            final_bundles, output_dir / "distributions"
        )
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=459)
    args = parser.parse_args()
    outputs = run_exp232_analysis(
        args.output_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
