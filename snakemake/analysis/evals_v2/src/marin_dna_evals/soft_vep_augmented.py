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
    GROUP_SMD,
    VARIANT_POOLED_SMD,
    joint_cluster_bootstrap_soft_metrics,
    joint_stratified_row_bootstrap_ungrouped_metrics,
    reference_soft_win_temperature,
    summarize_joint_bootstrap,
)

DATASET = "mendelian_traits"
SPLIT = "train"
DISTAL_ARM = "distal_centered"
EXP351_MODEL_PREFIX = "exp351-centered-step"
AUGMENTED_STEPS = (500, 1500, 2000, 3000, 3500, 4000, 4500, 4999)
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
    """Explicit six-arm by eight-checkpoint inventory."""
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
    point_metrics: pd.DataFrame,
    ungrouped_point_metrics: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Show the replacement distal home arm against all five exp232 arms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = (
        (point_metrics, AUPRC, "AUPRC"),
        (point_metrics, GROUP_SMD, "Group SMD"),
        (ungrouped_point_metrics, VARIANT_POOLED_SMD, "Variant pooled SMD"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharex=True)
    for axis, (table, metric, title) in zip(axes, panels):
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
            if is_home:
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
        "Development split. Black diamonds are the replacement home arm; "
        "the ribbon is its 95% joint bootstrap interval. Higher is better.",
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
    ungrouped_pairwise_deltas = pd.concat(
        ungrouped_pairwise_parts,
        ignore_index=True,
    )
    ungrouped_specialist_detectability = pd.concat(
        ungrouped_detectability_parts,
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
            point_metrics,
            ungrouped_point_metrics,
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
