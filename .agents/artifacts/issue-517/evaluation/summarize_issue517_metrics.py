#!/usr/bin/env python3
"""Export the matched-scope issue #517 AUPRC rows from evals_v2 metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


S3_PREFIX = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"
SPECS: list[tuple[str, str, str, str, str]] = [
    ("CDS", "cds", "mendelian_traits", "missense_variant", "minus_llr_avg"),
    ("CDS", "cds", "mendelian_traits", "splicing", "minus_llr_avg"),
    ("CDS", "cds", "mendelian_traits", "synonymous_variant", "minus_llr_avg"),
    ("CDS", "cds", "complex_traits", "missense_variant", "abs_llr_avg"),
    ("CDS", "cds", "sge", "missense_variant", "minus_llr_avg"),
    ("CDS", "cds", "sge", "splicing", "minus_llr_avg"),
    ("3-prime UTR", "utr3", "mendelian_traits", "3_prime_UTR_variant", "minus_llr_avg"),
    ("3-prime UTR", "utr3", "complex_traits", "3_prime_UTR_variant", "abs_llr_avg"),
    (
        "ncRNA exon",
        "ncrna-exon",
        "mendelian_traits",
        "non_coding_transcript_exon_variant",
        "minus_llr_avg",
    ),
    (
        "ncRNA exon",
        "ncrna-exon",
        "complex_traits",
        "non_coding_transcript_exon_variant",
        "abs_llr_avg",
    ),
    (
        "TSS / 5-prime UTR",
        "tss-utr5",
        "mendelian_traits",
        "5_prime_UTR_variant",
        "minus_llr_avg",
    ),
    (
        "TSS / 5-prime UTR",
        "tss-utr5",
        "mendelian_traits",
        "tss_proximal",
        "minus_llr_avg",
    ),
    (
        "TSS / 5-prime UTR",
        "tss-utr5",
        "complex_traits",
        "5_prime_UTR_variant",
        "abs_llr_avg",
    ),
    ("TSS / 5-prime UTR", "tss-utr5", "complex_traits", "tss_proximal", "abs_llr_avg"),
    ("Enhancer A", "enhancer-arm-a", "mendelian_traits", "distal", "minus_llr_avg"),
    ("Enhancer A", "enhancer-arm-a", "complex_traits", "distal", "abs_llr_avg"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-prefix", action="append", required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_arms = set(args.arm)
    records: list[dict[str, str | float | int]] = []
    for model_prefix in args.model_prefix:
        for arm_label, arm_slug, dataset, subset, score_type in SPECS:
            if arm_slug not in selected_arms:
                continue
            model = f"{model_prefix}-{arm_slug}-step-4999"
            path = f"{S3_PREFIX}/{model}/{dataset}.parquet"
            metrics = pd.read_parquet(path)
            row = metrics[
                metrics["score_type"].eq(score_type)
                & metrics["subset"].eq(subset)
            ]
            if dataset == "sge":
                row = row[row["accession"].eq("_macro_avg_")]
            assert len(row) == 1, (path, score_type, subset, len(row))
            value = row.iloc[0]
            n_groups_column = "n_groups" if "n_groups" in row else "n"
            n_rows_column = "n_rows" if "n_rows" in row else "n_pos"
            records.append(
                {
                    "model_prefix": model_prefix,
                    "arm": arm_label,
                    "arm_slug": arm_slug,
                    "dataset": dataset,
                    "subset": subset,
                    "score_type": score_type,
                    "auprc": float(value["value"]),
                    "se": float(value["se"]),
                    "n_groups": int(value[n_groups_column]),
                    "n_rows": int(value[n_rows_column]),
                }
            )
    result = pd.DataFrame.from_records(records)
    assert not result.empty
    assert result["auprc"].between(0, 1).all()
    assert result["se"].ge(0).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
