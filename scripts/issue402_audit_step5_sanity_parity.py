#!/usr/bin/env python3
"""Audit exact parity between issue #402's two 46M step-5k VEP executions."""

from __future__ import annotations

import polars as pl

SANITY = (
    "gs://marin-us-east5/evals/dna-exp402-rag-h640-p46m-b2m-30k/"
    "2026.07.26.5/sanity-step5000-7d9a7c9/vep"
)
OFFLINE = (
    "gs://marin-us-east5/evals/dna-exp402-rag-h640-p46m-b2m-30k/2026.07.26.5/step-5000"
)
BENCHMARKS = ("mendelian_traits", "complex_traits", "sge")
DOCUMENT_SCORES = ("ref_loglikelihood", "alt_loglikelihood", "llr")


def main() -> None:
    for benchmark in BENCHMARKS:
        sanity_documents = pl.read_parquet(
            f"{SANITY}/{benchmark}/full/documents.parquet"
        ).sort("document_id")
        offline_documents = pl.read_parquet(
            f"{OFFLINE}/{benchmark}/documents.parquet"
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
            f"{SANITY}/{benchmark}/full/variants.parquet"
        ).sort("variant_id")
        offline_variants = pl.read_parquet(
            f"{OFFLINE}/{benchmark}/variants.parquet"
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
            f"{SANITY}/{benchmark}/full/metrics.parquet"
        ).drop("context_mode")
        offline_metrics = pl.read_parquet(f"{OFFLINE}/{benchmark}/metrics.parquet")
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
