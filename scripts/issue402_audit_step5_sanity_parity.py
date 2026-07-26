#!/usr/bin/env python3
"""Audit exact parity between issue #402's independent step-5k VEP executions."""

from __future__ import annotations

import argparse

import polars as pl

ARTIFACTS = {
    "46m": "dna-exp402-rag-h640-p46m-b2m-30k",
    "104m": "dna-exp402-rag-h768-p104m-b2m-30k",
}
EVAL_ROOT = "gs://marin-us-east5/evals"
ARTIFACT_VERSION = "2026.07.26.5"
SANITY_SUFFIX = "sanity-step5000-7d9a7c9"
BENCHMARKS = ("mendelian_traits", "complex_traits", "sge")
DOCUMENT_SCORES = ("ref_loglikelihood", "alt_loglikelihood", "llr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=ARTIFACTS, default="46m")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = f"{EVAL_ROOT}/{ARTIFACTS[args.model]}/{ARTIFACT_VERSION}"
    sanity_root = f"{artifact_root}/{SANITY_SUFFIX}/vep"
    offline_root = f"{artifact_root}/step-5000"
    for benchmark in BENCHMARKS:
        sanity_documents = pl.read_parquet(
            f"{sanity_root}/{benchmark}/full/documents.parquet"
        ).sort("document_id")
        offline_documents = pl.read_parquet(
            f"{offline_root}/{benchmark}/documents.parquet"
        ).sort("document_id")
        assert sanity_documents.shape == offline_documents.shape
        assert sanity_documents.columns == offline_documents.columns
        assert sanity_documents.drop(DOCUMENT_SCORES).equals(
            offline_documents.drop(DOCUMENT_SCORES), null_equal=True
        )
        document_max_diff = max(
            float((sanity_documents[column] - offline_documents[column]).abs().max())
            for column in DOCUMENT_SCORES
        )
        assert document_max_diff == 0.0

        sanity_variants = pl.read_parquet(
            f"{sanity_root}/{benchmark}/full/variants.parquet"
        ).sort("variant_id")
        offline_variants = pl.read_parquet(
            f"{offline_root}/{benchmark}/variants.parquet"
        ).sort("variant_id")
        assert sanity_variants.shape == offline_variants.shape
        assert sanity_variants.columns == offline_variants.columns
        variant_score_columns = [
            column
            for column, dtype in sanity_variants.schema.items()
            if dtype in (pl.Float32, pl.Float64)
        ]
        assert variant_score_columns
        assert sanity_variants.drop(variant_score_columns).equals(
            offline_variants.drop(variant_score_columns), null_equal=True
        )
        variant_max_diff = max(
            float((sanity_variants[column] - offline_variants[column]).abs().max())
            for column in variant_score_columns
        )
        assert variant_max_diff == 0.0

        sanity_metrics = pl.read_parquet(
            f"{sanity_root}/{benchmark}/full/metrics.parquet"
        ).drop("context_mode")
        offline_metrics = pl.read_parquet(f"{offline_root}/{benchmark}/metrics.parquet")
        sort_columns = [
            column
            for column, dtype in sanity_metrics.schema.items()
            if dtype == pl.String
        ]
        sanity_metrics = sanity_metrics.sort(sort_columns)
        offline_metrics = offline_metrics.sort(sort_columns)
        assert sanity_metrics.columns == offline_metrics.columns
        assert sanity_metrics.equals(offline_metrics, null_equal=True)

        print(
            args.model,
            benchmark,
            {
                "documents": sanity_documents.height,
                "variants": sanity_variants.height,
                "document_score_max_abs_diff": document_max_diff,
                "variant_score_max_abs_diff": variant_max_diff,
                "metrics_exact": True,
            },
        )


if __name__ == "__main__":
    main()
