"""Reproduce the frozen MarinDNA m5.1 versus Evo 2 40B headline comparison.

The script reads the exact per-variant score artifacts used for the paper,
asserts one-to-one row identity, and estimates uncertainty for the difference
in macro-average Mendelian AUPRC with a paired, subset-stratified cluster
bootstrap over match groups.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score

MARIN_SCORES = (
    "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/"
    "mix-v0.9-p1B-i24-exp135-m5.1-step-59158/mendelian_traits.parquet"
)
EVO2_SCORES = (
    "https://gist.githubusercontent.com/gonzalobenegas/"
    "3649e68fb63ca1f3443e4486078eb4d8/raw/"
    "2b425e759811c201ca806ae4c8733fd7732220a6/evo2_40b_train.parquet"
)
VARIANT_KEY = ["chrom", "pos", "ref", "alt"]


def average_precision(label: np.ndarray, score: np.ndarray) -> float:
    """Compute non-interpolated average precision, including tied scores."""
    assert label.ndim == score.ndim == 1
    assert len(label) == len(score) > 0
    order = np.argsort(score, kind="mergesort")[::-1]
    sorted_label = label[order]
    sorted_score = score[order]
    threshold_indices = np.r_[
        np.flatnonzero(np.diff(sorted_score)),
        len(sorted_score) - 1,
    ]
    true_positives = np.cumsum(sorted_label)[threshold_indices]
    assert true_positives[-1] > 0
    precision = true_positives / (threshold_indices + 1)
    recall = true_positives / true_positives[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def load_shared_scores() -> pd.DataFrame:
    """Load and strictly align the two frozen score bundles."""
    marin = pl.read_parquet(
        MARIN_SCORES,
        storage_options={"aws_region": "us-east-2"},
    ).with_columns((-((pl.col("llr_fwd") + pl.col("llr_rc")) / 2)).alias("marin_score"))
    evo2 = pl.read_parquet(EVO2_SCORES).rename(
        {
            "label": "evo2_label",
            "subset": "evo2_subset",
            "match_group": "evo2_match_group",
            "minus_llr": "evo2_score",
        }
    )

    assert marin.select(VARIANT_KEY).is_duplicated().sum() == 0
    assert evo2.select(VARIANT_KEY).is_duplicated().sum() == 0
    shared = marin.select(
        VARIANT_KEY + ["label", "subset", "match_group", "marin_score"]
    ).join(
        evo2.select(
            VARIANT_KEY
            + ["evo2_label", "evo2_subset", "evo2_match_group", "evo2_score"]
        ),
        on=VARIANT_KEY,
        how="inner",
        validate="1:1",
    )
    assert shared.height == marin.height == evo2.height == 16_140
    assert shared.filter(pl.col("label") != pl.col("evo2_label")).height == 0
    assert shared.filter(pl.col("subset") != pl.col("evo2_subset")).height == 0
    assert (
        shared.filter(pl.col("match_group") != pl.col("evo2_match_group")).height == 0
    )
    assert (
        shared.select(["marin_score", "evo2_score"]).null_count().sum_horizontal()[0]
        == 0
    )
    return shared.select(
        VARIANT_KEY + ["label", "subset", "match_group", "marin_score", "evo2_score"]
    ).to_pandas()


def paired_macro_bootstrap(
    scores: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    min_groups: int = 30,
) -> dict[str, object]:
    """Compare macro AUPRC with a paired, subset-stratified group bootstrap."""
    assert n_bootstrap > 0
    subset_frames = [
        (str(subset), frame.reset_index(drop=True))
        for subset, frame in scores.groupby("subset", sort=True)
        if int(frame["label"].sum()) >= min_groups
    ]
    assert len(subset_frames) == 8

    bootstrap_deltas = np.empty((n_bootstrap, len(subset_frames)), dtype=float)
    child_seeds = np.random.SeedSequence(seed).spawn(len(subset_frames))
    subset_results: list[dict[str, object]] = []

    for column, ((subset, frame), child_seed) in enumerate(
        zip(subset_frames, child_seeds, strict=True)
    ):
        label = frame["label"].astype(int).to_numpy()
        marin = frame["marin_score"].to_numpy(dtype=float)
        evo2 = frame["evo2_score"].to_numpy(dtype=float)
        groups = list(frame.groupby("match_group", sort=False).indices.values())
        assert len(groups) == int(label.sum())
        assert all(len(group) == 10 and label[group].sum() == 1 for group in groups)

        marin_ap = average_precision(label, marin)
        evo2_ap = average_precision(label, evo2)
        assert np.isclose(marin_ap, average_precision_score(label, marin), atol=1e-15)
        assert np.isclose(evo2_ap, average_precision_score(label, evo2), atol=1e-15)
        subset_results.append(
            {
                "subset": subset,
                "n_rows": len(frame),
                "n_groups": len(groups),
                "marin_auprc": marin_ap,
                "evo2_auprc": evo2_ap,
                "delta": marin_ap - evo2_ap,
            }
        )

        rng = np.random.default_rng(child_seed)
        for iteration in range(n_bootstrap):
            sampled_groups = rng.integers(0, len(groups), size=len(groups))
            row_index = np.concatenate([groups[index] for index in sampled_groups])
            bootstrap_deltas[iteration, column] = average_precision(
                label[row_index], marin[row_index]
            ) - average_precision(label[row_index], evo2[row_index])

    macro_deltas = bootstrap_deltas.mean(axis=1)
    point_delta = float(np.mean([float(result["delta"]) for result in subset_results]))
    ci_low, ci_high = (
        float(value) for value in np.percentile(macro_deltas, [2.5, 97.5])
    )
    p_two_sided = min(
        2
        * min(
            float((macro_deltas <= 0).mean()),
            float((macro_deltas >= 0).mean()),
        ),
        1.0,
    )
    p_two_sided = max(p_two_sided, 1 / n_bootstrap)

    return {
        "inputs": {
            "marin_scores": MARIN_SCORES,
            "evo2_scores": EVO2_SCORES,
        },
        "bootstrap": {
            "unit": "match_group",
            "stratification": "consequence subset",
            "n_iterations": n_bootstrap,
            "seed": seed,
            "min_groups_per_subset": min_groups,
        },
        "sample": {
            "n_rows": int(sum(int(result["n_rows"]) for result in subset_results)),
            "n_groups": int(sum(int(result["n_groups"]) for result in subset_results)),
            "n_subsets": len(subset_results),
        },
        "macro": {
            "marin_auprc": float(
                np.mean([float(result["marin_auprc"]) for result in subset_results])
            ),
            "evo2_auprc": float(
                np.mean([float(result["evo2_auprc"]) for result in subset_results])
            ),
            "delta": point_delta,
            "delta_se": float(np.std(macro_deltas, ddof=1)),
            "delta_ci_95": [ci_low, ci_high],
            "p_two_sided": p_two_sided,
        },
        "subsets": subset_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = paired_macro_bootstrap(
        load_shared_scores(),
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
