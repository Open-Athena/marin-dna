"""Tests for issue #402's offline HF evaluation orchestration."""

from __future__ import annotations

import json

import polars as pl
import pytest
import torch

from marin_dna.pipelines.rag_glm.offline_eval import (
    RAG_BENCHMARK_DATASETS,
    aggregate_rag_variant_scores,
    compute_rag_benchmark_metrics,
    encode_rag_batch,
    nucleotide_token_ids,
    load_rag_eval_split,
    write_rag_evaluation_outputs,
)
from marin_dna.pipelines.rag_glm.tokenizer import create_rag_char_tokenizer


def _context() -> str:
    return ("A" * 255 + "[SEQ]") * 7 + "C" * 127


def test_encode_rag_batch_has_exact_fixed_geometry() -> None:
    tokenizer = create_rag_char_tokenizer()
    rows = pl.DataFrame(
        {
            "context": [_context(), _context()],
            "ref_completion": ["A" + "C" * 127, "G" + "T" * 127],
            "alt_completion": ["G" + "C" * 127, "A" + "T" * 127],
        }
    )
    prefix, ref, alt = encode_rag_batch(tokenizer, rows)
    assert prefix.shape == (2, 1_920)
    assert ref.shape == alt.shape == (2, 128)
    assert nucleotide_token_ids(tokenizer).unique().numel() == 4
    assert torch.equal(ref[:, 1:], alt[:, 1:])


def test_dataset_repo_override_requires_revision() -> None:
    with pytest.raises(AssertionError, match="also requires an explicit revision"):
        load_rag_eval_split("mendelian_traits", "test", repo="someone/other")


@pytest.mark.parametrize(
    ("benchmark", "expected_score"),
    [("mendelian_traits", -2.0), ("complex_traits", 2.0), ("sge", -2.0)],
)
def test_aggregate_raw_strands_before_protocol(
    benchmark: str, expected_score: float
) -> None:
    metadata: dict[str, list[object]]
    if benchmark == "mendelian_traits":
        metadata = {
            "target": [True, True],
            "subset": ["coding"] * 2,
            "match_group": [1, 1],
        }
    elif benchmark == "complex_traits":
        metadata = {
            "label": [True, True],
            "subset": ["coding"] * 2,
            "match_group": [1, 1],
        }
    else:
        metadata = {
            "label": [True, True],
            "subset": ["missense_variant"] * 2,
            "gene": ["GENE1"] * 2,
            "mavedb_urn": ["urn:mavedb:1"] * 2,
        }
    rows = pl.DataFrame(
        {
            "chrom": ["1", "1"],
            "pos": [101, 101],
            "ref": ["A", "A"],
            "alt": ["G", "G"],
            "variant_id": ["1:101:A>G"] * 2,
            "strand": ["+", "-"],
            "ref_loglikelihood": [-10.0, -12.0],
            "alt_loglikelihood": [-7.0, -11.0],
            "llr": [3.0, 1.0],
            **metadata,
        }
    )
    variants = aggregate_rag_variant_scores(rows, benchmark)  # type: ignore[arg-type]
    assert variants.height == 1
    assert variants["llr_avg"].item() == 2.0
    score_column = "abs_llr_avg" if benchmark == "complex_traits" else "minus_llr_avg"
    assert variants[score_column].item() == expected_score
    assert variants["label"].item()


def test_mendelian_metrics_and_output_manifest(tmp_path) -> None:
    variants = pl.DataFrame(
        {
            "label": [True, False, True, False],
            "subset": ["coding"] * 4,
            "match_group": [1, 1, 2, 2],
            "minus_llr_avg": [2.0, 1.0, 4.0, 3.0],
        }
    )
    metrics = compute_rag_benchmark_metrics(
        variants, "mendelian_traits", n_bootstrap=0, n_min=1
    )
    assert set(metrics["subset"]) == {"coding", "_global_", "_macro_avg_"}
    assert set(metrics["score_type"]) == {"minus_llr_avg"}

    documents = pl.DataFrame({"document_id": ["one", "two"], "llr": [1.0, 2.0]})
    write_rag_evaluation_outputs(
        document_scores=documents,
        variant_scores=variants,
        metrics=metrics,
        output_dir=tmp_path,
        benchmark="mendelian_traits",
        split="test",
        model="local-model",
        model_revision=None,
        dataset_repo=RAG_BENCHMARK_DATASETS["mendelian_traits"][0],
        dataset_revision=RAG_BENCHMARK_DATASETS["mendelian_traits"][1],
        code_revision="1" * 40,
        batch_size=16,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["n_document_rows"] == 2
    assert manifest["n_variants"] == 4
    assert manifest["aggregation"].startswith("average raw fwd/rc LLR")
    assert (tmp_path / "documents.parquet").is_file()
    assert (tmp_path / "variants.parquet").is_file()
    assert (tmp_path / "metrics.parquet").is_file()
