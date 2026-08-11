"""Tests for issue #402's offline HF evaluation orchestration."""

from __future__ import annotations

import json

import polars as pl
import pytest
import torch
from marin_dna_evals.rag_glm.offline_eval import (
    RAG_BENCHMARK_DATASETS,
    aggregate_rag_variant_scores,
    assert_rag_mendelian_variant_parity,
    compute_rag_benchmark_metrics,
    encode_rag_batch,
    load_rag_eval_split,
    load_rag_model_config_hf,
    load_rag_tokenizer_hf,
    nucleotide_token_ids,
    run_rag_mendelian_probe,
    select_paired_rag_rows,
    write_rag_evaluation_outputs,
    write_rag_probe_outputs,
)
from marin_dna_rag_glm.tokenizer import create_rag_char_tokenizer


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
    boundary_id = tokenizer.convert_tokens_to_ids("[SEQ]")
    for row in prefix:
        assert torch.nonzero(row == boundary_id).flatten().tolist() == [
            256,
            512,
            768,
            1024,
            1280,
            1536,
            1792,
        ]
    assert bool((prefix[:, 0] == tokenizer.bos_token_id).all())


@pytest.mark.parametrize(
    ("mutated_position", "replacement", "message"),
    [
        (0, 4, "BOS"),
        (512, 4, "SEQ"),
        (1_793, 1, "human"),
    ],
)
def test_encode_rag_batch_rejects_special_or_human_token_corruption(
    monkeypatch, mutated_position: int, replacement: int, message: str
) -> None:
    tokenizer = create_rag_char_tokenizer()
    original_call = type(tokenizer).__call__
    calls = 0

    def corrupt(self, *args, **kwargs):
        nonlocal calls
        encoded = original_call(self, *args, **kwargs)
        if calls == 0:
            encoded["input_ids"][:, mutated_position] = replacement
        calls += 1
        return encoded

    # Special methods are looked up on the class, so wrap the class call instead.
    monkeypatch.setattr(type(tokenizer), "__call__", corrupt)
    rows = pl.DataFrame(
        {
            "context": [_context()],
            "ref_completion": ["A" + "C" * 127],
            "alt_completion": ["G" + "C" * 127],
        }
    )
    with pytest.raises(AssertionError, match=message):
        encode_rag_batch(tokenizer, rows)


def test_select_paired_rag_rows_preserves_complete_strand_pairs() -> None:
    rows = pl.DataFrame(
        {
            "chrom": ["1", "1", "2", "2"],
            "pos": [10, 10, 20, 20],
            "ref": ["A", "A", "C", "C"],
            "alt": ["G", "G", "T", "T"],
            "document_id": ["one|+", "one|-", "two|+", "two|-"],
            "strand": ["+", "-", "+", "-"],
        }
    )

    selected = select_paired_rag_rows(rows, 2)
    assert selected["document_id"].to_list() == ["one|+", "one|-"]
    assert select_paired_rag_rows(rows, None).equals(rows)


def test_select_paired_rag_rows_rejects_partial_pair() -> None:
    rows = pl.DataFrame(
        {
            "chrom": ["1", "1", "2", "2"],
            "pos": [10, 20, 10, 20],
            "ref": ["A", "C", "A", "C"],
            "alt": ["G", "T", "G", "T"],
            "document_id": ["one|+", "two|+", "one|-", "two|-"],
            "strand": ["+", "+", "-", "-"],
        }
    )

    with pytest.raises(AssertionError):
        select_paired_rag_rows(rows, 2)


def _rag_parity_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": ["1", "1", "3", "3"],
            "pos": [10, 10, 20, 20],
            "ref": ["A", "A", "C", "C"],
            "alt": ["G", "G", "T", "T"],
            "target": [True, True, False, False],
            "subset": ["coding", "coding", "distal", "distal"],
            "match_group": [1, 1, 2, 2],
            "strand": ["+", "-", "+", "-"],
        }
    )


def test_rag_mendelian_parity_accepts_exact_official_rows() -> None:
    official = pl.DataFrame(
        {
            "chrom": ["3", "1"],
            "pos": [20, 10],
            "ref": ["C", "A"],
            "alt": ["T", "G"],
            "label": [False, True],
            "subset": ["distal", "coding"],
            "match_group": [2, 1],
        }
    )

    assert_rag_mendelian_variant_parity(_rag_parity_rows(), official)


def test_rag_mendelian_parity_rejects_metric_membership_difference() -> None:
    official = pl.DataFrame(
        {
            "chrom": ["1", "3"],
            "pos": [10, 20],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "label": [True, False],
            "subset": ["coding", "distal"],
            "match_group": [1, 999],
        }
    )

    with pytest.raises(AssertionError, match="differ on variant/metric fields"):
        assert_rag_mendelian_variant_parity(_rag_parity_rows(), official)


def test_rag_lm_eval_download_uses_commit_pinned_parquets(monkeypatch) -> None:
    pytest.importorskip("lm_eval")
    from marin_dna_evals.lm_eval.rag_dna_vep_llr_eval import RagDnaVepLlrEvalTask

    observed = {}

    def fake_load_dataset(**kwargs):
        observed.update(kwargs)
        return {"train": [], "test": []}

    monkeypatch.setattr(
        "marin_dna_evals.lm_eval.rag_dna_vep_llr_eval.datasets.load_dataset",
        fake_load_dataset,
    )
    task = object.__new__(RagDnaVepLlrEvalTask)
    task.DATASET_PATH = "owner/frozen-rag"
    task.DATASET_REVISION = "a" * 40

    task.download(cache_dir="/tmp/test-rag-cache")

    base = f"https://huggingface.co/datasets/owner/frozen-rag/resolve/{'a' * 40}"
    assert observed == {
        "path": "parquet",
        "data_files": {
            "train": f"{base}/train.parquet",
            "test": f"{base}/test.parquet",
        },
        "cache_dir": "/tmp/test-rag-cache",
    }


def test_dataset_repo_override_requires_revision() -> None:
    with pytest.raises(AssertionError, match="also requires an explicit revision"):
        load_rag_eval_split("mendelian_traits", "test", repo="someone/other")


def test_load_transformers_5_exported_tokenizer_with_transformers_4(tmp_path) -> None:
    tokenizer = create_rag_char_tokenizer()
    tokenizer.save_pretrained(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = "TokenizersBackend"
    config["extra_special_tokens"] = ["[SEQ]"]
    config_path.write_text(json.dumps(config))

    loaded = load_rag_tokenizer_hf(tmp_path)

    assert loaded("ACGT", add_special_tokens=False)["input_ids"] == [4, 5, 6, 7]
    assert loaded.bos_token_id == 2
    assert loaded.convert_tokens_to_ids("[SEQ]") == 3


def test_load_transformers_5_exported_rope_with_transformers_4(tmp_path) -> None:
    config = {
        "model_type": "qwen3",
        "vocab_size": 8,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "max_position_embeddings": 2_048,
        "rope_theta": 500_000,
        "rope_parameters": {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8_192,
            "rope_type": "llama3",
            "rope_theta": 500_000,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    loaded = load_rag_model_config_hf(tmp_path)

    expected_scaling = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8_192,
        "rope_type": "llama3",
    }
    if getattr(loaded, "rope_theta", None) is None:
        assert loaded.rope_scaling == {**expected_scaling, "rope_theta": 500_000}
    else:
        assert loaded.rope_theta == 500_000
        assert loaded.rope_scaling == expected_scaling


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


def test_aggregate_embeddings_averages_strands_in_float32() -> None:
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
            "emb_ref": [[1.0, 3.0], [5.0, 7.0]],
            "emb_alt": [[2.0, 4.0], [6.0, 8.0]],
            "target": [True, True],
            "subset": ["coding"] * 2,
            "match_group": [1, 1],
        }
    ).with_columns(
        pl.col("emb_ref").cast(pl.Array(pl.Float32, 2)),
        pl.col("emb_alt").cast(pl.Array(pl.Float32, 2)),
    )

    variants = aggregate_rag_variant_scores(rows, "mendelian_traits")

    assert variants["emb_ref"].dtype == pl.Array(pl.Float32, 2)
    assert variants["emb_alt"].dtype == pl.Array(pl.Float32, 2)
    assert variants["emb_ref"].item().to_list() == [3.0, 5.0]
    assert variants["emb_alt"].item().to_list() == [4.0, 6.0]
    assert not any(
        column.endswith(("_fwd", "_rc"))
        for column in variants.columns
        if column.startswith("emb_")
    )


def test_mendelian_probe_scores_identical_rows(monkeypatch) -> None:
    variants = pl.DataFrame(
        {
            "chrom": ["1", "1", "2", "2"],
            "label": [True, False, True, False],
            "subset": ["coding"] * 4,
            "llr_fwd": [-4.0, -2.0, -6.0, -1.0],
            "llr_rc": [-2.0, 0.0, -4.0, -1.0],
            "minus_llr_avg": [3.0, 1.0, 5.0, 1.0],
            "emb_ref": [[1.0, 2.0]] * 4,
            "emb_alt": [[2.0, 3.0]] * 4,
        }
    ).with_columns(
        pl.col("emb_ref").cast(pl.Array(pl.Float32, 2)),
        pl.col("emb_alt").cast(pl.Array(pl.Float32, 2)),
    )

    def fake_run_subset_probes(df, **kwargs):
        assert kwargs["feature_combo"] == "concat_ref_delta"
        assert kwargs["min_variants"] == 300
        assert kwargs["min_chroms"] == 3
        assert kwargs["inner_splits"] == 5
        predictions = df.drop(columns=["emb_ref", "emb_alt"]).copy()
        predictions["probe_score"] = [0.9, 0.1, 0.8, 0.2]
        return predictions, {"coding": {"C": 1.0}}

    monkeypatch.setattr(
        "marin_dna_evals.rag_glm.offline_eval.run_subset_probes",
        fake_run_subset_probes,
    )
    predictions, metrics, classifiers = run_rag_mendelian_probe(
        variants, n_jobs=2, n_bootstrap=0
    )

    assert classifiers == {"coding": {"C": 1.0}}
    assert predictions["minus_llr_avg"].to_list() == [3.0, 1.0, 5.0, 1.0]
    assert set(metrics["score_type"]) == {"probe_score", "minus_llr_avg"}
    assert set(metrics["subset"]) == {"coding", "_macro_avg_"}


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
        model_source="gs://example/checkpoints/model/hf/step-5000",
        model_revision=None,
        dataset_repo=RAG_BENCHMARK_DATASETS["mendelian_traits"][0],
        dataset_revision=RAG_BENCHMARK_DATASETS["mendelian_traits"][1],
        code_revision="1" * 40,
        batch_size=16,
        max_rows=2,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["n_document_rows"] == 2
    assert manifest["n_variants"] == 4
    assert manifest["row_selection"] == "first_2_paired_rows"
    assert manifest["model_source"] == "gs://example/checkpoints/model/hf/step-5000"
    assert manifest["aggregation"].startswith("average raw fwd/rc LLR")
    assert (tmp_path / "documents.parquet").is_file()
    assert (tmp_path / "variants.parquet").is_file()
    assert (tmp_path / "metrics.parquet").is_file()

    probe_predictions = variants.with_columns(pl.lit(0.5).alias("probe_score"))
    write_rag_probe_outputs(
        predictions=probe_predictions,
        metrics=metrics,
        classifiers={"coding": {"C": 1.0}},
        output_dir=tmp_path,
        model="local-model",
        model_source="gs://example/checkpoints/model/hf/step-5000",
        model_revision=None,
        dataset_repo=RAG_BENCHMARK_DATASETS["mendelian_traits"][0],
        dataset_revision=RAG_BENCHMARK_DATASETS["mendelian_traits"][1],
        code_revision="1" * 40,
    )
    probe_manifest = json.loads((tmp_path / "probe_manifest.json").read_text())
    assert (
        probe_manifest["model_source"] == "gs://example/checkpoints/model/hf/step-5000"
    )
    assert probe_manifest["human_pooling"]["start"] == 1_793
    assert probe_manifest["human_pooling"]["stop"] == 2_048
    assert probe_manifest["n_probe_scored"] == 4
    assert (tmp_path / "probe_predictions.parquet").is_file()
    assert (tmp_path / "probe_metrics.parquet").is_file()
    assert (tmp_path / "probe_classifiers.joblib").is_file()
