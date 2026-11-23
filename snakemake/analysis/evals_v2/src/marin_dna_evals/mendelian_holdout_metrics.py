"""Compute the paired, eight-class Marin–Evo 2 test comparison for Figure 2."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from marin_dna_evals.mendelian_holdout import (
    HEADLINE,
    IDENTITY,
    TEST_SHA256,
    VARIANT_KEY,
    check_digest,
)
from marin_dna_evals.metrics import MACRO_AVG_SUBSET, _per_chrom_components

MODELS = ("marin", "evo2_40b")
READOUTS = ("minus_llr_avg", "probe_score")


def align_predictions(
    release: pd.DataFrame, predictions: pd.DataFrame, model: str
) -> pd.DataFrame:
    if set(predictions["model"]) != {model} or set(predictions["split"]) != {"test"}:
        raise ValueError(f"Unexpected model or split in predictions for {model}")
    if (
        predictions.duplicated(VARIANT_KEY).any()
        or release.duplicated(VARIANT_KEY).any()
    ):
        raise ValueError("Duplicate variant key")
    rows = release.merge(
        predictions[[*IDENTITY, *READOUTS]],
        how="outer",
        on=IDENTITY,
        validate="one_to_one",
        indicator=True,
    )
    if not rows["_merge"].eq("both").all():
        raise ValueError(
            f"Prediction keys or subsets differ from the test release for {model}"
        )
    indexed = rows.set_index(VARIANT_KEY)
    values = indexed.loc[indexed["subset"].isin(HEADLINE), list(READOUTS)]
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"Missing or nonfinite headline prediction for {model}")
    return values.rename(columns=lambda col: f"{model}_{col}")


def pooled_results(
    rows: pd.DataFrame, scores: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[np.ndarray, np.ndarray]:
    labels = rows.label.to_numpy()
    codes, groups = pd.factorize(rows["match_group"], sort=True)
    point = np.array([average_precision_score(labels, col) for col in scores.T])
    bootstrap = np.empty((draws, len(MODELS)))
    for i in range(draws):
        group_draw = rng.integers(len(groups), size=len(groups))
        weights = np.bincount(group_draw, minlength=len(groups))[codes]
        bootstrap[i] = [
            average_precision_score(labels, col, sample_weight=weights)
            for col in scores.T
        ]
    return point, bootstrap


def chromosome_results(
    rows: pd.DataFrame, scores: np.ndarray, chromosomes: pd.Index, draws: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Use the same chromosome multiplicities across models and consequence classes."""
    point = np.empty(len(MODELS))
    bootstrap = np.full((len(draws), len(MODELS)), np.nan)
    for i, col in enumerate(scores.T):
        chrom, ap, weight = _per_chrom_components(
            rows.label.to_numpy(), col, rows.chrom.to_numpy()
        )
        if len(chrom) == 0:
            raise ValueError("No chromosome has both labels")
        point[i] = np.average(ap, weights=weight)
        draw_weights = draws[:, chromosomes.get_indexer(chrom)] * weight
        totals = draw_weights.sum(axis=1)
        np.divide(draw_weights @ ap, totals, out=bootstrap[:, i], where=totals > 0)
    return point, bootstrap


def summarize(
    readout: str,
    subset: str,
    rows: pd.DataFrame,
    point: np.ndarray,
    bootstrap: np.ndarray,
) -> dict:
    valid = bootstrap[np.isfinite(bootstrap).all(axis=1)]
    if len(valid) < 2:
        raise ValueError(f"Insufficient valid paired draws for {readout}, {subset}")
    delta = valid[:, 0] - valid[:, 1]
    lo, hi = np.quantile(delta, [0.025, 0.975])
    return {
        "split": "test",
        "readout": readout,
        "subset": subset,
        "n": len(rows),
        "n_positives": int(rows.label.sum()),
        "marin": point[0],
        "evo2_40b": point[1],
        "marin_se": valid[:, 0].std(ddof=1),
        "evo2_40b_se": valid[:, 1].std(ddof=1),
        "delta_pp": 100 * (point[0] - point[1]),
        "delta_lower_95_pp": 100 * lo,
        "delta_upper_95_pp": 100 * hi,
        "valid_draws": len(valid),
        "total_draws": len(bootstrap),
    }


def compare(frame: pd.DataFrame, *, draws: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Input rows already match both models on the same eight-class evaluation frame."""
    if set(frame["subset"]) != set(HEADLINE):
        raise ValueError("All eight prespecified headline classes are required")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Invalid reference labels")
    groups = frame.groupby("match_group")
    if (
        frame["match_group"].isna().any()
        or not groups.size().eq(10).all()
        or not groups.label.sum().eq(1).all()
        or not groups.chrom.nunique().eq(1).all()
        or not groups.subset.nunique().eq(1).all()
    ):
        raise ValueError(
            "Expected intact 1:9 matched groups within each class and chromosome"
        )
    rng = np.random.default_rng(seed)
    chromosomes = pd.Index(sorted(frame.chrom.unique()))
    chrom_rng = np.random.default_rng(seed)
    chrom_draws = np.array(
        [
            np.bincount(
                chrom_rng.integers(len(chromosomes), size=len(chromosomes)),
                minlength=len(chromosomes),
            )
            for _ in range(draws)
        ]
    )
    result = []
    for readout in READOUTS:
        points, resamples = [], []
        for subset in HEADLINE:
            rows = frame[frame["subset"] == subset]
            scores = rows[[f"{model}_{readout}" for model in MODELS]].to_numpy()
            if not np.isfinite(scores).all():
                raise ValueError(f"Nonfinite {readout} for {subset}")
            if readout == "probe_score":
                point, sample = chromosome_results(
                    rows, scores, chromosomes, chrom_draws
                )
            else:
                point, sample = pooled_results(rows, scores, rng, draws)
            result.append(summarize(readout, subset, rows, point, sample))
            points.append(point)
            resamples.append(sample)
        # An undefined class invalidates the entire draw for the fixed eight-class macro.
        result.append(
            summarize(
                readout,
                MACRO_AVG_SUBSET,
                frame,
                np.mean(points, axis=0),
                np.mean(resamples, axis=0),
            )
        )
    return pd.DataFrame(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("test-release", "marin", "evo2-40b", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() in {
        p.resolve() for p in (args.test_release, args.marin, args.evo2_40b)
    }:
        parser.error("choose an output path distinct from the inputs")
    check_digest(args.test_release, TEST_SHA256)
    release = pd.read_parquet(
        args.test_release, columns=[*IDENTITY, "label", "match_group"]
    )
    reference = release[release["subset"].isin(HEADLINE)].set_index(VARIANT_KEY)
    for model, path in zip(MODELS, (args.marin, args.evo2_40b), strict=True):
        values = align_predictions(release, pd.read_parquet(path), model)
        reference = reference.join(values, validate="one_to_one")
    result = compare(reference.reset_index())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.to_csv(index=False))


if __name__ == "__main__":
    main()
