#!/usr/bin/env python3
"""Fail-fast acceptance audit for one issue #402 large-batch eval bundle."""

from __future__ import annotations

import argparse
import json
import subprocess

import polars as pl

ARTIFACT_VERSION = "2026.07.26.5"
CODE_REVISION = "7d9a7c9f5a2f8040af3daadb8c2be10804c211fc"
EVAL_STEPS = {5_000, 10_000, 15_000, 20_000, 25_000, 29_999}
MODEL_ARTIFACTS = {
    "46m": "dna-exp402-rag-h640-p46m-b2m-30k",
    "104m": "dna-exp402-rag-h768-p104m-b2m-30k",
}
BENCHMARKS = {
    "mendelian_traits": {
        "dataset_repo": "marin-dna/evals_mendelian_traits_rag_harness_255_v1",
        "dataset_revision": "9acedb683463477f34745af30a63a289873008a4",
        "n_documents": 18_980,
        "n_variants": 9_490,
        "score_column": "minus_llr_avg",
    },
    "complex_traits": {
        "dataset_repo": "marin-dna/evals_complex_traits_rag_harness_255_v1",
        "dataset_revision": "0252a883f650819a8e1fa22062027daafe956540",
        "n_documents": 20_000,
        "n_variants": 10_000,
        "score_column": "abs_llr_avg",
    },
    "sge": {
        "dataset_repo": "marin-dna/evals_sge_rag_harness_255_v1",
        "dataset_revision": "c20cc58fceb9bc053a55152a89d160f1b070f75d",
        "n_documents": 29_776,
        "n_variants": 14_888,
        "score_column": "minus_llr_avg",
    },
}
DOCUMENT_SCORE_COLUMNS = {
    "ref_loglikelihood",
    "alt_loglikelihood",
    "llr",
}
VARIANT_SCORE_COLUMNS = {
    "ref_loglikelihood_fwd",
    "alt_loglikelihood_fwd",
    "llr_fwd",
    "ref_loglikelihood_rc",
    "alt_loglikelihood_rc",
    "llr_rc",
    "ref_loglikelihood_avg",
    "alt_loglikelihood_avg",
    "llr_avg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_ARTIFACTS, required=True)
    parser.add_argument("--step", type=int, choices=sorted(EVAL_STEPS), required=True)
    return parser.parse_args()


def read_manifest(uri: str) -> dict[str, object]:
    completed = subprocess.run(
        ["gcloud", "storage", "cat", uri],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(completed.stdout)
    assert isinstance(manifest, dict)
    return manifest


def assert_finite(frame: pl.DataFrame, *, label: str) -> None:
    float_columns = [
        column
        for column, dtype in frame.schema.items()
        if dtype in (pl.Float32, pl.Float64)
    ]
    assert float_columns, label
    assert all(frame[column].null_count() == 0 for column in float_columns), label
    finite = frame.select(
        [pl.col(column).is_finite().all().alias(column) for column in float_columns]
    ).row(0)
    assert all(finite), (label, dict(zip(float_columns, finite, strict=True)))


def max_abs(frame: pl.DataFrame, expression: pl.Expr) -> float:
    value = frame.select(expression.abs().max()).item()
    assert value is not None
    return float(value)


def audit_bundle(*, model: str, step: int) -> pl.DataFrame:
    assert model in MODEL_ARTIFACTS
    assert step in EVAL_STEPS
    artifact = MODEL_ARTIFACTS[model]
    checkpoint_root = (
        f"gs://marin-us-east5/checkpoints/{artifact}/{ARTIFACT_VERSION}/hf/step-{step}"
    )
    eval_root = f"gs://marin-us-east5/evals/{artifact}/{ARTIFACT_VERSION}/step-{step}"
    summaries: list[dict[str, object]] = []

    for benchmark, spec in BENCHMARKS.items():
        root = f"{eval_root}/{benchmark}"
        manifest = read_manifest(f"{root}/manifest.json")
        expected_manifest = {
            "benchmark": benchmark,
            "split": "test",
            "model_source": checkpoint_root,
            "dataset_repo": spec["dataset_repo"],
            "dataset_revision": spec["dataset_revision"],
            "code_revision": CODE_REVISION,
            "batch_size": 16,
            "row_selection": "all",
            "n_document_rows": spec["n_documents"],
            "n_variants": spec["n_variants"],
            "score_column": spec["score_column"],
            "aggregation": "average raw fwd/rc LLR, then apply score transform",
        }
        for key, expected in expected_manifest.items():
            assert manifest.get(key) == expected, (
                benchmark,
                key,
                manifest.get(key),
                expected,
            )

        documents = pl.read_parquet(f"{root}/documents.parquet")
        variants = pl.read_parquet(f"{root}/variants.parquet")
        metrics = pl.read_parquet(f"{root}/metrics.parquet")
        n_documents = int(spec["n_documents"])
        n_variants = int(spec["n_variants"])
        score_column = str(spec["score_column"])

        assert documents.height == n_documents
        assert variants.height == n_variants
        assert documents["document_id"].n_unique() == n_documents
        assert documents["variant_id"].n_unique() == n_variants
        assert variants["variant_id"].n_unique() == n_variants
        assert DOCUMENT_SCORE_COLUMNS <= set(documents.columns)
        assert VARIANT_SCORE_COLUMNS | {score_column} <= set(variants.columns)
        assert documents["strand"].unique().sort().to_list() == ["+", "-"]
        strand_pairs = documents.group_by("variant_id").agg(
            pl.len().alias("n_rows"),
            pl.col("strand").n_unique().alias("n_strands"),
        )
        assert strand_pairs.filter(
            (pl.col("n_rows") != 2) | (pl.col("n_strands") != 2)
        ).is_empty()
        assert_finite(documents, label=f"{benchmark} documents")
        assert_finite(variants, label=f"{benchmark} variants")
        assert_finite(metrics, label=f"{benchmark} metrics")

        document_llr_error = max_abs(
            documents,
            pl.col("llr") - (pl.col("alt_loglikelihood") - pl.col("ref_loglikelihood")),
        )
        assert document_llr_error == 0.0

        fwd = documents.filter(pl.col("strand") == "+").select(
            "variant_id",
            pl.col("ref_loglikelihood").alias("document_ref_fwd"),
            pl.col("alt_loglikelihood").alias("document_alt_fwd"),
            pl.col("llr").alias("document_llr_fwd"),
        )
        rc = documents.filter(pl.col("strand") == "-").select(
            "variant_id",
            pl.col("ref_loglikelihood").alias("document_ref_rc"),
            pl.col("alt_loglikelihood").alias("document_alt_rc"),
            pl.col("llr").alias("document_llr_rc"),
        )
        checked = variants.join(fwd, on="variant_id", how="inner", validate="1:1").join(
            rc, on="variant_id", how="inner", validate="1:1"
        )
        assert checked.height == n_variants
        exact_errors = {
            "document_ref_fwd": max_abs(
                checked,
                pl.col("document_ref_fwd") - pl.col("ref_loglikelihood_fwd"),
            ),
            "document_alt_fwd": max_abs(
                checked,
                pl.col("document_alt_fwd") - pl.col("alt_loglikelihood_fwd"),
            ),
            "document_llr_fwd": max_abs(
                checked, pl.col("document_llr_fwd") - pl.col("llr_fwd")
            ),
            "document_ref_rc": max_abs(
                checked,
                pl.col("document_ref_rc") - pl.col("ref_loglikelihood_rc"),
            ),
            "document_alt_rc": max_abs(
                checked,
                pl.col("document_alt_rc") - pl.col("alt_loglikelihood_rc"),
            ),
            "document_llr_rc": max_abs(
                checked, pl.col("document_llr_rc") - pl.col("llr_rc")
            ),
            "ref_average": max_abs(
                checked,
                pl.col("ref_loglikelihood_avg")
                - (
                    (pl.col("ref_loglikelihood_fwd") + pl.col("ref_loglikelihood_rc"))
                    / 2
                ),
            ),
            "alt_average": max_abs(
                checked,
                pl.col("alt_loglikelihood_avg")
                - (
                    (pl.col("alt_loglikelihood_fwd") + pl.col("alt_loglikelihood_rc"))
                    / 2
                ),
            ),
            "llr_average": max_abs(
                checked,
                pl.col("llr_avg") - ((pl.col("llr_fwd") + pl.col("llr_rc")) / 2),
            ),
        }
        expected_score = (
            pl.col("llr_avg").abs()
            if score_column == "abs_llr_avg"
            else -pl.col("llr_avg")
        )
        exact_errors["score_transform"] = max_abs(
            checked, pl.col(score_column) - expected_score
        )
        assert all(error == 0.0 for error in exact_errors.values()), (
            benchmark,
            exact_errors,
        )

        assert metrics["score_type"].unique().to_list() == [score_column]
        assert metrics.filter(
            (pl.col("value") < 0) | (pl.col("value") > 1) | (pl.col("se") < 0)
        ).is_empty()
        if benchmark == "sge":
            assert metrics["metric"].unique().to_list() == ["AUPRC"]
            headline = metrics.filter(
                (pl.col("subset") == "_macro_avg_")
                & (pl.col("accession") == "_macro_avg_")
                & (pl.col("gene") == "_macro_avg_")
            )
        else:
            headline = metrics.filter(pl.col("subset") == "_global_")
        assert headline.height == 1
        summaries.append(
            {
                "model": model,
                "step": step,
                "benchmark": benchmark,
                "n_documents": n_documents,
                "n_variants": n_variants,
                "score_column": score_column,
                "headline_auprc": headline["value"].item(),
                "headline_se": headline["se"].item(),
                "max_exact_error": max(exact_errors.values()),
            }
        )

    result = pl.DataFrame(summaries).sort("benchmark")
    assert result.height == len(BENCHMARKS)
    return result


def main() -> None:
    args = parse_args()
    result = audit_bundle(model=args.model, step=args.step)
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=180):
        print(result)


if __name__ == "__main__":
    main()
