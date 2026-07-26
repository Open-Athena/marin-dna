#!/usr/bin/env python3
"""Audit final issue #402 forward/reverse-complement VEP aggregation."""

from __future__ import annotations

import argparse

import polars as pl

ROOTS = {
    "46M": (
        "gs://marin-us-east5/evals/dna-exp402-rag-h640-p46m-30k/2026.07.26/step-29999"
    ),
    "104M": (
        "gs://marin-us-east5/evals/dna-exp402-rag-h768-p104m-30k/2026.07.26/step-29999"
    ),
}
BENCHMARKS = {
    "Mendelian": ("mendelian_traits", "minus_llr_avg"),
    "Complex": ("complex_traits", "abs_llr_avg"),
    "SGE": ("sge", "minus_llr_avg"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-46m", default=ROOTS["46M"])
    parser.add_argument("--input-104m", default=ROOTS["104M"])
    return parser.parse_args()


def audit(roots: dict[str, str] = ROOTS) -> pl.DataFrame:
    """Assert exact RC averaging and return strand-consistency diagnostics."""
    assert set(roots) == set(ROOTS)
    rows: list[dict[str, object]] = []
    for model, root in roots.items():
        for benchmark, (directory, score_column) in BENCHMARKS.items():
            variants = pl.read_parquet(
                f"{root.rstrip('/')}/{directory}/variants.parquet"
            )
            assert variants["variant_id"].n_unique() == variants.height
            required = {
                "llr_fwd",
                "llr_rc",
                "llr_avg",
                "ref_loglikelihood_fwd",
                "ref_loglikelihood_rc",
                "ref_loglikelihood_avg",
                "alt_loglikelihood_fwd",
                "alt_loglikelihood_rc",
                "alt_loglikelihood_avg",
                score_column,
            }
            assert required <= set(variants.columns)
            checked = variants.with_columns(
                ((pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("expected_llr_avg"),
                (
                    (pl.col("ref_loglikelihood_fwd") + pl.col("ref_loglikelihood_rc"))
                    / 2
                ).alias("expected_ref_avg"),
                (
                    (pl.col("alt_loglikelihood_fwd") + pl.col("alt_loglikelihood_rc"))
                    / 2
                ).alias("expected_alt_avg"),
            )
            if score_column == "abs_llr_avg":
                checked = checked.with_columns(
                    pl.col("llr_avg").abs().alias("expected_score")
                )
            else:
                checked = checked.with_columns(
                    (-pl.col("llr_avg")).alias("expected_score")
                )
            summary = checked.select(
                pl.len().alias("n_variants"),
                (pl.col("llr_avg") - pl.col("expected_llr_avg"))
                .abs()
                .max()
                .alias("max_llr_average_error"),
                (pl.col("ref_loglikelihood_avg") - pl.col("expected_ref_avg"))
                .abs()
                .max()
                .alias("max_ref_average_error"),
                (pl.col("alt_loglikelihood_avg") - pl.col("expected_alt_avg"))
                .abs()
                .max()
                .alias("max_alt_average_error"),
                (pl.col(score_column) - pl.col("expected_score"))
                .abs()
                .max()
                .alias("max_score_transform_error"),
                pl.corr("llr_fwd", "llr_rc").alias("fwd_rc_pearson"),
                (pl.col("llr_fwd").sign() == pl.col("llr_rc").sign())
                .mean()
                .alias("fwd_rc_sign_agreement"),
            ).row(0, named=True)
            assert summary["max_llr_average_error"] == 0
            assert summary["max_ref_average_error"] == 0
            assert summary["max_alt_average_error"] == 0
            assert summary["max_score_transform_error"] == 0
            assert 0.8 < summary["fwd_rc_pearson"] <= 1
            assert 0.8 < summary["fwd_rc_sign_agreement"] <= 1
            rows.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "score_column": score_column,
                    **summary,
                }
            )
    result = pl.DataFrame(rows).sort("benchmark", "model")
    assert result.height == len(ROOTS) * len(BENCHMARKS)
    return result


if __name__ == "__main__":
    args = parse_args()
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=180):
        print(audit({"46M": args.input_46m, "104M": args.input_104m}))
