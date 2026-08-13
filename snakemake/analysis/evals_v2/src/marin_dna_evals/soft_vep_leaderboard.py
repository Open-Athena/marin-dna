"""Current MarinDNA leaderboard soft-metric analysis for issue #459."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml
from scipy.stats import kendalltau, spearmanr
from sklearn.isotonic import IsotonicRegression

from marin_dna_evals.soft_vep_analysis import (
    DATASET,
    NON_DISTAL_SUBSETS,
    S3_SCORE_ROOT,
    SPLIT,
    TAU_REFERENCE_ARM,
    TAU_REFERENCE_STEP,
    add_llr_scores,
    confidence_filtered_rank_reversals,
    exp232_score_uri,
    pairwise_bootstrap_summary,
    permute_labels_within_groups,
    read_score_bundle,
    validate_aligned_bundles,
)
from marin_dna_evals.soft_vep_metrics import (
    AUPRC,
    HIGHER_IS_BETTER,
    compute_mendelian_soft_metric_table,
    joint_cluster_bootstrap_soft_metrics,
    reference_soft_win_temperature,
    summarize_joint_bootstrap,
)


def load_marin_dna_models(models_yaml: Path) -> pd.DataFrame:
    """Load every current registry model with Mendelian leaderboard coverage."""
    entries = yaml.safe_load(models_yaml.read_text())
    rows = []
    for registry_index, entry in enumerate(entries):
        if entry.get("family") != "marin_dna":
            continue
        if DATASET not in entry.get("datasets", []):
            continue
        experiment = entry.get("experiment")
        if experiment is not None:
            experiment_group = f"exp{experiment}"
        elif entry["id"].startswith("mix-v0.9"):
            experiment_group = "mix-v0.9"
        else:
            experiment_group = entry["id"].rsplit("-step-", maxsplit=1)[0]
        rows.append(
            {
                "model": entry["id"],
                "display": entry["display"],
                "experiment_group": experiment_group,
                "registry_index": registry_index,
            }
        )
    result = pd.DataFrame(rows)
    assert not result.empty, f"no MarinDNA Mendelian models found in {models_yaml}"
    assert result["model"].is_unique, "duplicate model IDs in dashboard registry"
    return result


def model_score_uri(model: str) -> str:
    return f"{S3_SCORE_ROOT}/{model}/{DATASET}.parquet"


def model_metric_uri(model: str) -> str:
    return model_score_uri(model).replace("/results/scores/", "/results/metrics/")


def read_model_stored_auprc(model: str) -> pd.DataFrame:
    metrics = pl.read_parquet(
        model_metric_uri(model),
        columns=["score_type", "subset", "value", "split"],
        storage_options={"aws_region": "us-east-2"},
    ).filter(
        (pl.col("score_type") == "minus_llr_avg") & (pl.col("split") == SPLIT)
    )
    result = metrics.select(["subset", pl.col("value").alias("stored_value")]).to_pandas()
    result["model"] = model
    return result


def _rank_summary(point_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, str | float | int]] = []
    pair_rows: list[dict[str, str | float | bool]] = []
    for subset, subset_frame in point_metrics.groupby("subset", sort=False):
        values = subset_frame.pivot(index="model", columns="metric", values="value")
        reference = values[AUPRC]
        for metric in values.columns:
            oriented = values[metric] * (1.0 if HIGHER_IS_BETTER[metric] else -1.0)
            paired = pd.concat(
                [oriented.rename("metric"), reference.rename("auprc")], axis=1
            ).dropna()
            reversals = 0
            for model_a, model_b in combinations(paired.index.astype(str), 2):
                metric_delta = float(
                    paired.loc[model_a, "metric"] - paired.loc[model_b, "metric"]
                )
                auprc_delta = float(
                    paired.loc[model_a, "auprc"] - paired.loc[model_b, "auprc"]
                )
                reversal = metric_delta * auprc_delta < 0
                reversals += int(reversal)
                pair_rows.append(
                    {
                        "subset": str(subset),
                        "metric": str(metric),
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric_delta_oriented": metric_delta,
                        "auprc_delta": auprc_delta,
                        "reversal": reversal,
                    }
                )
            top_k = min(3, len(paired))
            rows.append(
                {
                    "subset": str(subset),
                    "metric": str(metric),
                    "spearman": float(
                        spearmanr(paired["metric"], paired["auprc"]).statistic
                    ),
                    "kendall": float(
                        kendalltau(paired["metric"], paired["auprc"]).statistic
                    ),
                    "top3_overlap": len(
                        set(paired["metric"].nlargest(top_k).index)
                        & set(paired["auprc"].nlargest(top_k).index)
                    ),
                    "pairwise_reversals": reversals,
                    "n_models": len(paired),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def add_macro_average(point_metrics: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight macro average over the seven precomputed consequence cells."""
    macro = (
        point_metrics.groupby(["model", "metric", "higher_is_better"], as_index=False)
        .agg(value=("value", "mean"))
        .assign(
            subset="_macro_avg_",
            se=np.nan,
            ci_low=np.nan,
            ci_high=np.nan,
            n_groups=len(NON_DISTAL_SUBSETS),
            n_rows=np.nan,
            n_pos=np.nan,
        )
    )
    return pd.concat([point_metrics, macro], ignore_index=True, sort=False)


def leave_one_experiment_out_projection(
    point_metrics: pd.DataFrame,
    models: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fit a monotone soft-to-AUPRC map, holding out whole experiments."""
    annotated = point_metrics.merge(
        models[["model", "experiment_group"]],
        on="model",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, str | float]] = []
    for (subset, metric), metric_frame in annotated.groupby(
        ["subset", "metric"], sort=False
    ):
        if metric == AUPRC:
            continue
        auprc = annotated[
            (annotated["subset"] == subset) & (annotated["metric"] == AUPRC)
        ].set_index("model")["value"]
        cell = metric_frame.set_index("model").copy()
        cell["auprc"] = auprc.reindex(cell.index)
        sign = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
        for held_out in cell["experiment_group"].unique():
            train = cell[cell["experiment_group"] != held_out]
            test = cell[cell["experiment_group"] == held_out]
            if train["value"].nunique() < 2 or len(train) < 2:
                continue
            mapper = IsotonicRegression(increasing=True, out_of_bounds="clip")
            mapper.fit(sign * train["value"], train["auprc"])
            predictions = mapper.predict(sign * test["value"])
            rows.extend(
                {
                    "subset": str(subset),
                    "metric": str(metric),
                    "experiment_group": str(held_out),
                    "model": model,
                    "auprc": float(test.loc[model, "auprc"]),
                    "predicted_auprc": float(prediction),
                    "absolute_error": float(
                        abs(prediction - test.loc[model, "auprc"])
                    ),
                }
                for model, prediction in zip(test.index, predictions)
            )
    predictions = pd.DataFrame(rows)
    summary_rows = []
    for (subset, metric), cell in predictions.groupby(["subset", "metric"]):
        if cell["predicted_auprc"].nunique() < 2 or cell["auprc"].nunique() < 2:
            rank_correlation = float("nan")
        else:
            rank_correlation = float(
                spearmanr(cell["predicted_auprc"], cell["auprc"]).statistic
            )
        summary_rows.append(
            {
                "subset": subset,
                "metric": metric,
                "mae": float(cell["absolute_error"].mean()),
                "spearman": rank_correlation,
                "n_models": len(cell),
                "n_experiments": cell["experiment_group"].nunique(),
            }
        )
    return predictions, pd.DataFrame(summary_rows)


def _compute_controls(
    bundles: dict[str, pd.DataFrame],
    *,
    tau: float,
    seed: int,
    models: pd.DataFrame,
) -> pd.DataFrame:
    metadata = next(iter(bundles.values()))[["label", "subset", "match_group"]]
    scale_values = np.geomspace(0.25, 4.0, len(models))
    scale = dict(zip(models["model"], scale_values))
    parts = []
    for subset in NON_DISTAL_SUBSETS:
        keep = metadata["subset"] == subset
        subset_meta = metadata.loc[keep].reset_index(drop=True)
        average = pd.DataFrame(
            {
                model: frame.loc[keep, "minus_llr_avg"].reset_index(drop=True)
                for model, frame in bundles.items()
            }
        )
        fwd = pd.DataFrame(
            {
                model: frame.loc[keep, "minus_llr_fwd"].reset_index(drop=True)
                for model, frame in bundles.items()
            }
        )
        variants = {
            "baseline_avg": (subset_meta["label"], average),
            "positive_rescaling": (
                subset_meta["label"],
                average.mul(pd.Series(scale)),
            ),
            "sign_reversal": (subset_meta["label"], -average),
            "fwd_only": (subset_meta["label"], fwd),
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
            ).rename(columns={"score_type": "model"})
            table["subset"] = subset
            table["control"] = control
            table["scale_constant"] = table["model"].map(scale)
            parts.append(table)
    controls = pd.concat(parts, ignore_index=True)
    baseline_auprc = controls[
        (controls["control"] == "baseline_avg") & (controls["metric"] == AUPRC)
    ].set_index(["subset", "model"])["value"]
    rescaled_auprc = controls[
        (controls["control"] == "positive_rescaling")
        & (controls["metric"] == AUPRC)
    ].set_index(["subset", "model"])["value"]
    assert np.allclose(baseline_auprc.sort_index(), rescaled_auprc.sort_index())
    return controls


def run_leaderboard_analysis(
    output_dir: Path,
    *,
    models_yaml: Path,
    n_bootstrap: int = 1000,
    seed: int = 459,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_marin_dna_models(models_yaml)
    bundles = {
        model: add_llr_scores(read_score_bundle(model_score_uri(model)))
        for model in models["model"]
    }
    validate_aligned_bundles(bundles)

    reference = add_llr_scores(
        read_score_bundle(exp232_score_uri(TAU_REFERENCE_ARM, TAU_REFERENCE_STEP))
    )
    reference = reference[reference["subset"].isin(NON_DISTAL_SUBSETS)]
    tau = reference_soft_win_temperature(
        reference["label"], reference["minus_llr_avg"], reference["match_group"]
    )

    metadata = next(iter(bundles.values()))[["label", "subset", "match_group"]]
    point_parts = []
    pairwise_parts = []
    for subset in NON_DISTAL_SUBSETS:
        keep = metadata["subset"] == subset
        subset_meta = metadata.loc[keep].reset_index(drop=True)
        scores = pd.DataFrame(
            {
                model: frame.loc[keep, "minus_llr_avg"].reset_index(drop=True)
                for model, frame in bundles.items()
            }
        )
        point, samples = joint_cluster_bootstrap_soft_metrics(
            subset_meta["label"],
            scores,
            subset_meta["match_group"],
            tau=tau,
            n_bootstrap=n_bootstrap,
            rng=seed,
        )
        summary = summarize_joint_bootstrap(point, samples).rename(
            columns={"score_type": "model"}
        )
        summary["subset"] = subset
        point_parts.append(summary)
        pairwise = pairwise_bootstrap_summary(point, samples).rename(
            columns={"arm_a": "model_a", "arm_b": "model_b"}
        )
        pairwise["subset"] = subset
        pairwise_parts.append(pairwise)

    leaf_points = pd.concat(point_parts, ignore_index=True)
    point_metrics = add_macro_average(leaf_points)
    pairwise_deltas = pd.concat(pairwise_parts, ignore_index=True)
    confident_rank_reversals = confidence_filtered_rank_reversals(
        pairwise_deltas,
        group_columns=["subset"],
        entity_columns=("model_a", "model_b"),
    )
    rank_agreement, rank_reversals = _rank_summary(point_metrics)
    projection, projection_summary = leave_one_experiment_out_projection(
        point_metrics,
        models,
    )
    controls = _compute_controls(bundles, tau=tau, seed=seed, models=models)

    stored = pd.concat(
        [read_model_stored_auprc(model) for model in models["model"]],
        ignore_index=True,
    )
    reproduction = leaf_points[leaf_points["metric"] == AUPRC][
        ["model", "subset", "value"]
    ].merge(stored, on=["model", "subset"], validate="one_to_one")
    reproduction["difference"] = reproduction["value"] - reproduction["stored_value"]
    reproduction["reproduces"] = reproduction["difference"].abs() <= 1e-12
    bad_reproduction = reproduction[~reproduction["reproduces"]]

    outputs = {
        "models": output_dir / "models.parquet",
        "metadata": output_dir / "metadata.json",
        "point_metrics": output_dir / "point_metrics.parquet",
        "pairwise_deltas": output_dir / "pairwise_deltas.parquet",
        "rank_agreement": output_dir / "rank_agreement.parquet",
        "rank_reversals": output_dir / "rank_reversals.parquet",
        "confident_rank_reversals": output_dir / "confident_rank_reversals.parquet",
        "projection": output_dir / "projection.parquet",
        "projection_summary": output_dir / "projection_summary.parquet",
        "controls": output_dir / "controls.parquet",
        "auprc_reproduction": output_dir / "auprc_reproduction.parquet",
    }
    models.to_parquet(outputs["models"], index=False)
    outputs["metadata"].write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "split": SPLIT,
                "n_models": len(models),
                "n_bootstrap": n_bootstrap,
                "bootstrap_seed": seed,
                "soft_win_tau": tau,
                "tau_reference_model": (
                    f"exp232-v4_{TAU_REFERENCE_ARM}-step-{TAU_REFERENCE_STEP}"
                ),
                "n_auprc_mismatches": len(bad_reproduction),
                "mismatched_models": sorted(bad_reproduction["model"].unique()),
                "max_abs_auprc_mismatch": float(
                    bad_reproduction["difference"].abs().max()
                )
                if len(bad_reproduction)
                else 0.0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    point_metrics.to_parquet(outputs["point_metrics"], index=False)
    pairwise_deltas.to_parquet(outputs["pairwise_deltas"], index=False)
    rank_agreement.to_parquet(outputs["rank_agreement"], index=False)
    rank_reversals.to_parquet(outputs["rank_reversals"], index=False)
    confident_rank_reversals.to_parquet(
        outputs["confident_rank_reversals"], index=False
    )
    projection.to_parquet(outputs["projection"], index=False)
    projection_summary.to_parquet(outputs["projection_summary"], index=False)
    controls.to_parquet(outputs["controls"], index=False)
    reproduction.to_parquet(outputs["auprc_reproduction"], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models-yaml", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=459)
    args = parser.parse_args()
    outputs = run_leaderboard_analysis(
        args.output_dir,
        models_yaml=args.models_yaml,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
