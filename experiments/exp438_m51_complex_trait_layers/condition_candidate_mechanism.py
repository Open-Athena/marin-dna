"""Post-hoc conditioning checks for feature 1662's missense-label association."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import average_precision_score

from analyze_candidate_mechanism import (
    RESPONSES,
    add_mechanism_columns,
    bh_adjust,
    codon_change_position,
    standardized_mean_difference,
)
from common import ISSUE, write_json


def add_codon_position(frame: pl.DataFrame) -> pl.DataFrame:
    positions = [codon_change_position(value) for value in frame["codons"]]
    return frame.with_columns(pl.Series("codon_position", positions, pl.UInt8))


def within_stratum_residual(frame: pl.DataFrame, response: str) -> np.ndarray:
    return frame.with_columns(
        (
            pl.col(response)
            - pl.col(response).mean().over("allele_change", "codon_position")
        ).alias("residual")
    )["residual"].to_numpy()


def full_vep_residual(frame: pl.DataFrame, response: str) -> np.ndarray:
    strata = frame.select(
        pl.concat_str(
            "allele_change", pl.col("codon_position").cast(pl.String), separator=":"
        ).alias("stratum")
    )["stratum"].to_numpy()
    levels = sorted(set(strata))
    one_hot = np.column_stack(
        [(strata == level).astype(np.float64) for level in levels[1:]]
    )
    predictors = frame.select(
        "sift_deleteriousness",
        "polyphen_deleteriousness",
        "blosum_deleteriousness",
    ).to_numpy()
    means, standard_deviations = predictors.mean(axis=0), predictors.std(axis=0)
    assert np.all(standard_deviations > 0)
    predictors = (predictors - means) / standard_deviations
    design = np.column_stack([np.ones(frame.height), one_hot, predictors])
    response_values = frame[response].to_numpy()
    coefficients = np.linalg.lstsq(design, response_values, rcond=None)[0]
    residual = response_values - design @ coefficients
    assert residual.shape == (frame.height,) and np.isfinite(residual).all()
    return residual


def association_row(
    labels: np.ndarray,
    residual: np.ndarray,
    *,
    adjustment: str,
    response: str,
) -> dict[str, float | int | str]:
    positive, negative = residual[labels], residual[~labels]
    welch = stats.ttest_ind(positive, negative, equal_var=False)
    mann = stats.mannwhitneyu(positive, negative, alternative="two-sided")
    return {
        "adjustment": adjustment,
        "response": response,
        "n": labels.size,
        "n_positive": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "residual_auprc": float(average_precision_score(labels, residual)),
        "welch_p": float(welch.pvalue),
        "mann_whitney_p": float(mann.pvalue),
        "standardized_mean_difference": standardized_mean_difference(
            positive, negative
        ),
        "rank_biserial": float(
            2 * mann.statistic / (positive.size * negative.size) - 1
        ),
    }


def conditioning_results(frame: pl.DataFrame) -> pl.DataFrame:
    codon_complete = frame.drop_nulls("codon_position")
    vep_complete = codon_complete.drop_nulls(
        [
            "sift_deleteriousness",
            "polyphen_deleteriousness",
            "blosum_deleteriousness",
        ]
    )
    assert codon_complete.height >= 2_000 and vep_complete.height >= 1_900
    rows: list[dict[str, float | int | str]] = []
    for response in RESPONSES:
        labels = codon_complete["label"].to_numpy()
        rows.append(
            association_row(
                labels,
                within_stratum_residual(codon_complete, response),
                adjustment="allele_change_plus_codon_position",
                response=response,
            )
        )
        labels = vep_complete["label"].to_numpy()
        rows.append(
            association_row(
                labels,
                full_vep_residual(vep_complete, response),
                adjustment="allele_change_plus_codon_position_plus_vep",
                response=response,
            )
        )
    result = pl.DataFrame(rows)
    pieces: list[pl.DataFrame] = []
    for adjustment in result["adjustment"].unique().sort():
        family = result.filter(pl.col("adjustment") == adjustment)
        pieces.append(
            family.with_columns(
                pl.Series("welch_q", bh_adjust(family["welch_p"].to_numpy())),
                pl.Series(
                    "mann_whitney_q",
                    bh_adjust(family["mann_whitney_p"].to_numpy()),
                ),
            )
        )
    return pl.concat(pieces).sort("adjustment", "response")


def codon_position_summary(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.group_by("codon_position")
        .agg(
            pl.len().alias("n"),
            pl.col("label").sum().alias("n_positive"),
            pl.col("label").mean().alias("prevalence"),
            *(pl.col(response).mean() for response in RESPONSES),
        )
        .sort("codon_position", nulls_last=False)
    )


def run(input_path: Path, output_dir: Path) -> None:
    assert input_path.is_file()
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = add_codon_position(add_mechanism_columns(pl.read_parquet(input_path)))
    assert frame.height == 2_500 and int(frame["label"].sum()) == 250
    associations = conditioning_results(frame)
    codons = codon_position_summary(frame)
    associations.write_parquet(
        output_dir / "conditioned_associations.parquet", compression="zstd"
    )
    codons.write_parquet(
        output_dir / "codon_position_summary.parquet", compression="zstd"
    )
    metadata = {
        "issue": ISSUE,
        "analysis_status": "post_hoc_descriptive",
        "rows": frame.height,
        "positive_rows": int(frame["label"].sum()),
        "all_conditioned_families_concordant_at_fdr_0_05": bool(
            associations.select(
                ((pl.col("welch_q") < 0.05) & (pl.col("mann_whitney_q") < 0.05)).all()
            ).item()
        ),
    }
    write_json(output_dir / "results.json", metadata)
    (output_dir / "RESULTS.md").write_text(
        "# Feature 1662 conditioned missense audit\n\n"
        "This is a post-hoc association decomposition, not a causal claim.\n\n"
        "```json\n" + json.dumps(metadata, indent=2, sort_keys=True) + "\n```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
