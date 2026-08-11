"""Offline Hugging Face evaluation for issue #402's RAG VEP harnesses.

The harness parquets already contain exact HAL-projected ortholog windows, so
this module needs no genome or alignment files. It scores every materialized
strand document, averages the raw LLR across strands, and only then applies the
benchmark's deleteriousness transform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import polars as pl
import torch
from huggingface_hub import hf_hub_download
from torch import Tensor
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

from marin_dna_evals.metrics import (
    compute_auprc_metrics,
    compute_sge_metrics,
    per_chrom_ap_table,
)
from marin_dna_evals.rag_glm.hf_scoring import (
    RAG_COMPLETION_TOKENS,
    RAG_HUMAN_POOL_START,
    RAG_HUMAN_POOL_TOKENS,
    RAG_PREFIX_TOKENS,
    score_rag_completions_hf,
)
from marin_dna_evals.rag_glm.model_sanity import assert_rag_token_geometry
from marin_dna_evals.variant_probe import (
    DEFAULT_C_GRID,
    run_subset_probes,
)

RAGBenchmark = Literal["mendelian_traits", "complex_traits", "sge"]
RAG_BENCHMARK_DATASETS: dict[RAGBenchmark, tuple[str, str]] = {
    "mendelian_traits": (
        "marin-dna/evals_mendelian_traits_rag_harness_255_v1",
        "9acedb683463477f34745af30a63a289873008a4",
    ),
    "complex_traits": (
        "marin-dna/evals_complex_traits_rag_harness_255_v1",
        "0252a883f650819a8e1fa22062027daafe956540",
    ),
    "sge": (
        "marin-dna/evals_sge_rag_harness_255_v1",
        "c20cc58fceb9bc053a55152a89d160f1b070f75d",
    ),
}
RAG_SCORE_COLUMNS: dict[RAGBenchmark, str] = {
    "mendelian_traits": "minus_llr_avg",
    "complex_traits": "abs_llr_avg",
    "sge": "minus_llr_avg",
}
VARIANT_KEY_COLUMNS = ["chrom", "pos", "ref", "alt"]
DOCUMENT_SCORE_COLUMNS = ["ref_loglikelihood", "alt_loglikelihood", "llr"]
DOCUMENT_EMBEDDING_COLUMNS = ["emb_ref", "emb_alt"]
_BENCHMARK_METADATA: dict[RAGBenchmark, list[str]] = {
    "mendelian_traits": ["target", "subset", "match_group"],
    "complex_traits": ["label", "subset", "match_group"],
    "sge": ["label", "subset", "gene", "mavedb_urn"],
}


def rag_eval_columns(benchmark: RAGBenchmark) -> list[str]:
    """Return the minimal harness columns needed for scoring and metrics."""
    assert benchmark in RAG_BENCHMARK_DATASETS
    return [
        *VARIANT_KEY_COLUMNS,
        *_BENCHMARK_METADATA[benchmark],
        "variant_id",
        "document_id",
        "strand",
        "context",
        "ref_completion",
        "alt_completion",
    ]


def load_rag_eval_split(
    benchmark: RAGBenchmark,
    split: str,
    *,
    repo: str | None = None,
    revision: str | None = None,
) -> pl.DataFrame:
    """Read one immutable harness split directly from its pinned parquet."""
    assert split in {"train", "test"}
    assert repo is None or revision is not None, (
        "overriding the dataset repository also requires an explicit revision"
    )
    default_repo, default_revision = RAG_BENCHMARK_DATASETS[benchmark]
    resolved_repo = repo or default_repo
    resolved_revision = revision or default_revision
    assert resolved_repo
    assert len(resolved_revision) == 40, "dataset revision must be a full commit SHA"
    url = (
        f"https://huggingface.co/datasets/{resolved_repo}/resolve/"
        f"{resolved_revision}/{split}.parquet"
    )
    rows = pl.read_parquet(url, columns=rag_eval_columns(benchmark))
    assert rows.height > 0
    return rows


def select_paired_rag_rows(rows: pl.DataFrame, max_rows: int | None) -> pl.DataFrame:
    """Select a deterministic prefix while preserving complete strand pairs."""
    if max_rows is None:
        return rows
    assert max_rows > 0
    assert max_rows % 2 == 0
    assert max_rows <= rows.height
    required = {*VARIANT_KEY_COLUMNS, "document_id", "strand"}
    assert required <= set(rows.columns)
    selected = rows.head(max_rows)
    assert selected["document_id"].n_unique() == selected.height
    pairs = selected.group_by(VARIANT_KEY_COLUMNS).agg(
        pl.len().alias("n_rows"),
        pl.col("strand").sort().alias("strands"),
    )
    assert pairs.height * 2 == selected.height
    for pair in pairs.iter_rows(named=True):
        assert pair["n_rows"] == 2
        assert pair["strands"] == ["+", "-"]
    return selected


def nucleotide_token_ids(tokenizer: Any) -> Tensor:
    """Resolve the tokenizer's four unique A/C/G/T IDs without special tokens."""
    encoded = tokenizer(list("ACGT"), add_special_tokens=False)["input_ids"]
    assert len(encoded) == 4
    assert all(len(ids) == 1 for ids in encoded), "A/C/G/T must each be one token"
    token_ids = torch.tensor([ids[0] for ids in encoded], dtype=torch.long)
    assert token_ids.unique().numel() == 4
    return token_ids


def load_rag_tokenizer_hf(
    pretrained_model_name_or_path: str | Path,
    *,
    revision: str | None = None,
) -> Any:
    """Load the fixed RAG tokenizer without version-specific exporter metadata.

    Levanter currently emits a Transformers 5-only class name and special-token
    schema in ``tokenizer_config.json``. The immutable ``tokenizer.json`` is
    version-neutral, so reconstruct the experiment's known special-token
    contract explicitly instead of interpreting incompatible metadata.
    """
    model_path = Path(pretrained_model_name_or_path)
    if model_path.exists():
        assert model_path.is_dir()
        tokenizer_file = model_path / "tokenizer.json"
    else:
        tokenizer_file = Path(
            hf_hub_download(
                repo_id=str(pretrained_model_name_or_path),
                filename="tokenizer.json",
                revision=revision,
            )
        )
    assert tokenizer_file.is_file()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        bos_token="[BOS]",
        cls_token="[BOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        additional_special_tokens=["[SEQ]"],
        clean_up_tokenization_spaces=False,
        model_max_length=2_048,
    )
    assert tokenizer.vocab_size == 8
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id == tokenizer.cls_token_id == 2
    assert tokenizer.eos_token_id is None
    assert tokenizer.convert_tokens_to_ids("[SEQ]") == 3
    assert nucleotide_token_ids(tokenizer).tolist() == [4, 5, 6, 7]
    return tokenizer


def load_rag_model_config_hf(
    pretrained_model_name_or_path: str | Path,
    *,
    revision: str | None = None,
) -> Any:
    """Load a Levanter HF export without silently dropping Llama-3 RoPE.

    Levanter's Transformers 5 exporter writes the new ``rope_parameters``
    field. Transformers 4.57's Qwen3 config instead consumes
    ``rope_scaling`` and otherwise defaults to unscaled RoPE. Translate the
    exported field explicitly so the offline model matches the native model.
    """
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path,
        revision=revision,
        trust_remote_code=True,
    )
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is None:
        return config

    assert isinstance(rope_parameters, dict)
    translated_scaling = dict(rope_parameters)
    exported_theta = translated_scaling.pop("rope_theta", None)
    assert translated_scaling.get("rope_type") == "llama3"
    observed_scaling = getattr(config, "rope_scaling", None)
    if observed_scaling is not None:
        assert observed_scaling in (translated_scaling, rope_parameters), (
            "conflicting rope_parameters and rope_scaling in model export"
        )
        if exported_theta is not None:
            observed_theta = getattr(config, "rope_theta", None)
            if observed_theta is not None:
                assert float(observed_theta) == float(exported_theta)
        return config

    if exported_theta is not None:
        config.rope_theta = float(exported_theta)
    config.rope_scaling = translated_scaling
    return config


def assert_rag_variant_parity(
    rag_rows: pl.DataFrame,
    official_rows: pl.DataFrame,
    benchmark: RAGBenchmark,
) -> None:
    """Require the materialized RAG harness to equal the official variant set.

    The RAG harness has two documents per variant (forward and reverse
    complement), while the official eval dataset has one row. Normalize that
    representation difference and compare every field that determines metric
    membership. This check deliberately runs before model loading so a stale
    projection cannot consume GPU time or emit a misleading standard artifact.
    """
    assert benchmark in RAG_BENCHMARK_DATASETS
    harness_label = "target" if benchmark == "mendelian_traits" else "label"
    extra_metadata = (
        ["subset", "match_group"]
        if benchmark != "sge"
        else ["subset", "gene", "mavedb_urn"]
    )
    comparison_columns = [*VARIANT_KEY_COLUMNS, "label", *extra_metadata]
    rag_required = {
        *VARIANT_KEY_COLUMNS,
        harness_label,
        *extra_metadata,
        "strand",
    }
    assert rag_required <= set(rag_rows.columns)
    assert set(comparison_columns) <= set(official_rows.columns)

    rag_variants = rag_rows.group_by(VARIANT_KEY_COLUMNS).agg(
        pl.len().alias("n_rows"),
        pl.col("strand").sort().alias("strands"),
        pl.col(harness_label).n_unique().alias("n_label"),
        *[pl.col(column).n_unique().alias(f"n_{column}") for column in extra_metadata],
        pl.col(harness_label).first().alias("label"),
        *[pl.col(column).first() for column in extra_metadata],
    )
    malformed_condition = (
        (pl.col("n_rows") != 2)
        | (pl.col("strands").list.join("") != "+-")
        | (pl.col("n_label") != 1)
    )
    for column in extra_metadata:
        malformed_condition |= pl.col(f"n_{column}") != 1
    malformed = rag_variants.filter(malformed_condition)
    assert malformed.is_empty(), (
        f"RAG {benchmark} harness must have one consistent +/- document pair "
        "per variant"
    )
    rag_variants = rag_variants.select(comparison_columns)
    official_variants = official_rows.select(comparison_columns)
    assert official_variants.unique().height == official_variants.height, (
        f"official {benchmark} eval rows contain duplicate comparison keys"
    )

    casts = {
        "chrom": pl.String,
        "pos": pl.Int64,
        "ref": pl.String,
        "alt": pl.String,
        "label": pl.Boolean,
        "subset": pl.String,
    }
    if benchmark == "sge":
        casts |= {"gene": pl.String, "mavedb_urn": pl.String}
    else:
        casts["match_group"] = pl.Int64
    rag_variants = rag_variants.cast(casts)
    official_variants = official_variants.cast(casts)
    assert (
        rag_variants.select(comparison_columns).null_count().sum_horizontal().item()
        == 0
    )
    assert (
        official_variants.select(comparison_columns)
        .null_count()
        .sum_horizontal()
        .item()
        == 0
    )
    assert rag_variants.height == official_variants.height, (
        f"RAG/official {benchmark} row-count mismatch: "
        f"{rag_variants.height} != {official_variants.height}"
    )
    only_rag = rag_variants.join(
        official_variants,
        on=comparison_columns,
        how="anti",
    )
    only_official = official_variants.join(
        rag_variants,
        on=comparison_columns,
        how="anti",
    )
    assert only_rag.is_empty() and only_official.is_empty(), (
        f"RAG harness and official {benchmark} eval differ on variant/metric fields: "
        f"only_rag={only_rag.height}, only_official={only_official.height}"
    )


def assert_rag_mendelian_variant_parity(
    rag_rows: pl.DataFrame,
    official_rows: pl.DataFrame,
) -> None:
    """Backward-compatible wrapper for the original Mendelian-only contract."""
    assert_rag_variant_parity(rag_rows, official_rows, "mendelian_traits")


def encode_rag_batch(
    tokenizer: Any, rows: pl.DataFrame
) -> tuple[Tensor, Tensor, Tensor]:
    """Tokenize one fixed-layout batch and assert its exact geometry."""
    required = {"context", "ref_completion", "alt_completion"}
    assert required <= set(rows.columns)
    assert rows.height > 0
    prefix = tokenizer(
        rows["context"].to_list(),
        add_special_tokens=True,
        padding=False,
        return_tensors="pt",
    )["input_ids"]
    ref = tokenizer(
        rows["ref_completion"].to_list(),
        add_special_tokens=False,
        padding=False,
        return_tensors="pt",
    )["input_ids"]
    alt = tokenizer(
        rows["alt_completion"].to_list(),
        add_special_tokens=False,
        padding=False,
        return_tensors="pt",
    )["input_ids"]
    assert prefix.shape == (rows.height, RAG_PREFIX_TOKENS)
    assert ref.shape == (rows.height, RAG_COMPLETION_TOKENS)
    assert alt.shape == ref.shape
    assert_rag_token_geometry(
        prefix,
        bos_token_id=tokenizer.bos_token_id,
        boundary_token_id=tokenizer.convert_tokens_to_ids("[SEQ]"),
        pad_token_id=tokenizer.pad_token_id,
        unk_token_id=tokenizer.unk_token_id,
        nucleotide_token_ids=nucleotide_token_ids(tokenizer).tolist(),
    )
    nucleotide_ids = nucleotide_token_ids(tokenizer)
    assert bool(torch.isin(ref, nucleotide_ids).all())
    assert bool(torch.isin(alt, nucleotide_ids).all())
    assert bool((ref[:, 1:] == alt[:, 1:]).all())
    assert bool((ref[:, 0] != alt[:, 0]).all())
    return prefix, ref, alt


def _model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    parameters = getattr(model, "parameters", None)
    assert parameters is not None, "model has neither .device nor .parameters()"
    return next(parameters()).device


def score_rag_rows_hf(
    model: Any,
    tokenizer: Any,
    rows: pl.DataFrame,
    *,
    batch_size: int,
    device: str | torch.device | None = None,
    return_embeddings: bool = False,
) -> pl.DataFrame:
    """Add raw paired-completion log-likelihoods to materialized strand rows."""
    assert batch_size > 0
    required = {
        *VARIANT_KEY_COLUMNS,
        "variant_id",
        "document_id",
        "strand",
        "context",
        "ref_completion",
        "alt_completion",
    }
    assert required <= set(rows.columns)
    assert rows["document_id"].n_unique() == rows.height
    assert set(rows["strand"].unique()) == {"+", "-"}
    resolved_device = (
        torch.device(device) if device is not None else _model_device(model)
    )
    nucleotide_ids = nucleotide_token_ids(tokenizer).to(resolved_device)
    chunks: list[Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, rows.height, batch_size):
            batch = rows.slice(start, batch_size)
            prefix, ref, alt = encode_rag_batch(tokenizer, batch)
            chunks.append(
                score_rag_completions_hf(
                    model,
                    prefix.to(resolved_device),
                    ref.to(resolved_device),
                    alt.to(resolved_device),
                    nucleotide_token_ids=nucleotide_ids,
                    return_embeddings=return_embeddings,
                ).cpu()
            )
    scores = torch.cat(chunks, dim=0).float().numpy()
    hidden_size = int(model.config.hidden_size)
    expected_width = 3 + (2 * hidden_size if return_embeddings else 0)
    assert scores.shape == (rows.height, expected_width)
    assert np.isfinite(scores).all()
    scored = rows.with_columns(
        *[
            pl.Series(column, scores[:, index], dtype=pl.Float32)
            for index, column in enumerate(DOCUMENT_SCORE_COLUMNS)
        ]
    )
    if not return_embeddings:
        return scored
    return scored.with_columns(
        pl.Series(
            "emb_ref",
            scores[:, 3 : 3 + hidden_size].tolist(),
            dtype=pl.Array(pl.Float32, hidden_size),
        ),
        pl.Series(
            "emb_alt",
            scores[:, 3 + hidden_size :].tolist(),
            dtype=pl.Array(pl.Float32, hidden_size),
        ),
    )


def score_rag_checkpoint_hf(
    checkpoint_path: str | Path,
    rows: pl.DataFrame,
    *,
    benchmark: RAGBenchmark,
    batch_size: int,
    device: str | torch.device = "cuda",
    return_embeddings: bool = False,
) -> pl.DataFrame:
    """Load one exported RAG checkpoint and return standard variant-level atoms."""
    resolved_device = torch.device(device)
    assert resolved_device.type != "cuda" or torch.cuda.is_available(), (
        "RAG checkpoint scoring requested CUDA but no CUDA device is available"
    )
    tokenizer = load_rag_tokenizer_hf(checkpoint_path)
    model_config = load_rag_model_config_hf(checkpoint_path)
    model_kwargs: dict[str, Any] = {
        "config": model_config,
        "trust_remote_code": True,
    }
    if resolved_device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
    model.to(resolved_device)
    documents = score_rag_rows_hf(
        model,
        tokenizer,
        rows,
        batch_size=batch_size,
        device=resolved_device,
        return_embeddings=return_embeddings,
    )
    variants = aggregate_rag_variant_scores(documents, benchmark)
    del model
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return variants


def aggregate_rag_variant_scores(
    rows: pl.DataFrame, benchmark: RAGBenchmark
) -> pl.DataFrame:
    """Average raw strand atoms, then apply the benchmark score protocol."""
    required = {
        *VARIANT_KEY_COLUMNS,
        *_BENCHMARK_METADATA[benchmark],
        "variant_id",
        "strand",
        *DOCUMENT_SCORE_COLUMNS,
    }
    assert required <= set(rows.columns)
    has_embeddings = [column in rows.columns for column in DOCUMENT_EMBEDDING_COLUMNS]
    assert len(set(has_embeddings)) == 1, "emb_ref and emb_alt must be present together"
    embedding_columns = DOCUMENT_EMBEDDING_COLUMNS if all(has_embeddings) else []
    group_columns = [*VARIANT_KEY_COLUMNS, "variant_id"]
    metadata_columns = _BENCHMARK_METADATA[benchmark]
    counts = rows.group_by(group_columns).agg(
        pl.len().alias("n_rows"),
        pl.col("strand").n_unique().alias("n_strands"),
        *[
            pl.col(column).n_unique().alias(f"n_{column}")
            for column in metadata_columns
        ],
    )
    inconsistency = (pl.col("n_rows") != 2) | (pl.col("n_strands") != 2)
    for column in metadata_columns:
        inconsistency |= pl.col(f"n_{column}") != 1
    assert counts.filter(inconsistency).is_empty(), (
        "each variant must have exactly two strands and consistent metric metadata"
    )

    forward = rows.filter(pl.col("strand") == "+").select(
        *group_columns,
        *metadata_columns,
        *[pl.col(column).alias(f"{column}_fwd") for column in DOCUMENT_SCORE_COLUMNS],
        *[pl.col(column).alias(f"{column}_fwd") for column in embedding_columns],
    )
    reverse = rows.filter(pl.col("strand") == "-").select(
        *group_columns,
        *[pl.col(column).alias(f"{column}_rc") for column in DOCUMENT_SCORE_COLUMNS],
        *[pl.col(column).alias(f"{column}_rc") for column in embedding_columns],
    )
    assert forward.height == reverse.height == counts.height
    variants = forward.join(reverse, on=group_columns, how="inner", validate="1:1")
    variants = variants.with_columns(
        *[
            ((pl.col(f"{column}_fwd") + pl.col(f"{column}_rc")) / 2).alias(
                f"{column}_avg"
            )
            for column in DOCUMENT_SCORE_COLUMNS
        ]
    )
    if embedding_columns:
        averaged_embeddings: list[pl.Series] = []
        for column in embedding_columns:
            forward_embedding = np.stack(variants[f"{column}_fwd"].to_list()).astype(
                np.float32
            )
            reverse_embedding = np.stack(variants[f"{column}_rc"].to_list()).astype(
                np.float32
            )
            assert forward_embedding.shape == reverse_embedding.shape
            assert forward_embedding.ndim == 2
            averaged = (forward_embedding + reverse_embedding) / np.float32(2)
            assert np.isfinite(averaged).all()
            averaged_embeddings.append(
                pl.Series(
                    column,
                    averaged.tolist(),
                    dtype=pl.Array(pl.Float32, averaged.shape[1]),
                )
            )
        variants = variants.drop(
            *[
                f"{column}_{strand}"
                for column in embedding_columns
                for strand in ("fwd", "rc")
            ]
        ).with_columns(*averaged_embeddings)
    assert np.allclose(
        variants["llr_avg"].to_numpy(),
        (
            variants["alt_loglikelihood_avg"] - variants["ref_loglikelihood_avg"]
        ).to_numpy(),
        rtol=1e-5,
        atol=1e-4,
    )
    score_column = RAG_SCORE_COLUMNS[benchmark]
    if score_column == "minus_llr_avg":
        variants = variants.with_columns((-pl.col("llr_avg")).alias(score_column))
    else:
        assert score_column == "abs_llr_avg"
        variants = variants.with_columns(pl.col("llr_avg").abs().alias(score_column))
    if "label" not in variants.columns:
        variants = variants.with_columns(pl.col("target").alias("label"))
    assert variants.filter(~pl.col(score_column).is_finite()).is_empty()
    return variants.sort(VARIANT_KEY_COLUMNS)


def run_rag_mendelian_probe(
    variants: pl.DataFrame,
    *,
    n_jobs: int = 4,
    n_bootstrap: int = 1_000,
    rng: np.random.Generator | int | None = 0,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Run issue #402's frozen human-segment Mendelian probe protocol."""
    assert n_jobs > 0
    required = {
        "chrom",
        "label",
        "subset",
        "llr_fwd",
        "llr_rc",
        "minus_llr_avg",
        *DOCUMENT_EMBEDDING_COLUMNS,
    }
    assert required <= set(variants.columns)
    c_grid = np.logspace(-12, 4, 17)
    assert np.array_equal(c_grid, DEFAULT_C_GRID)
    predictions, classifiers = run_subset_probes(
        variants.to_pandas(),
        feature_combo="concat_ref_delta",
        c_grid=c_grid,
        min_variants=300,
        min_chroms=3,
        inner_splits=5,
        n_jobs=n_jobs,
    )
    paired_baseline = -(
        predictions["llr_fwd"].to_numpy(dtype=np.float32)
        + predictions["llr_rc"].to_numpy(dtype=np.float32)
    ) / np.float32(2)
    assert np.allclose(
        paired_baseline,
        predictions["minus_llr_avg"].to_numpy(dtype=np.float32),
        rtol=1e-5,
        atol=1e-4,
    )
    predictions["minus_llr_avg"] = paired_baseline
    metrics = per_chrom_ap_table(
        predictions,
        ["probe_score", "minus_llr_avg"],
        n_bootstrap=n_bootstrap,
        rng=rng,
        n_min=30,
    )
    return pl.from_pandas(predictions), pl.from_pandas(metrics), classifiers


def compute_rag_benchmark_metrics(
    variants: pl.DataFrame,
    benchmark: RAGBenchmark,
    *,
    n_bootstrap: int = 1_000,
    rng: np.random.Generator | int | None = 0,
    n_min: int = 30,
) -> pl.DataFrame:
    """Compute the existing leaderboard metric on variant-level RAG scores."""
    score_column = RAG_SCORE_COLUMNS[benchmark]
    assert score_column in variants.columns
    scores = variants.select(score_column).to_pandas()
    if benchmark == "sge":
        dataset = variants.select("mavedb_urn", "gene", "subset", "label").to_pandas()
        metrics = compute_sge_metrics(
            dataset,
            scores,
            score_columns=[score_column],
            n_bootstrap=n_bootstrap,
            rng=rng,
            n_min_auprc=n_min,
        )
    else:
        dataset = variants.select("label", "subset", "match_group").to_pandas()
        metrics = compute_auprc_metrics(
            dataset,
            scores,
            score_columns=[score_column],
            n_bootstrap=n_bootstrap,
            rng=rng,
            n_min=n_min,
        )
    return pl.from_pandas(metrics)


def write_rag_evaluation_outputs(
    *,
    document_scores: pl.DataFrame,
    variant_scores: pl.DataFrame,
    metrics: pl.DataFrame,
    output_dir: str | Path,
    benchmark: RAGBenchmark,
    split: str,
    model: str,
    model_source: str,
    model_revision: str | None,
    dataset_repo: str,
    dataset_revision: str,
    code_revision: str,
    batch_size: int,
    max_rows: int | None = None,
) -> None:
    """Write lossless score tables and a complete reproducibility manifest."""
    assert len(dataset_revision) == 40
    assert len(code_revision) == 40
    assert model_source
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    document_scores.write_parquet(output / "documents.parquet", compression="zstd")
    variant_scores.write_parquet(output / "variants.parquet", compression="zstd")
    metrics.write_parquet(output / "metrics.parquet", compression="zstd")
    manifest = {
        "benchmark": benchmark,
        "split": split,
        "model": model,
        "model_source": model_source,
        "model_revision": model_revision,
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "code_revision": code_revision,
        "batch_size": batch_size,
        "row_selection": (
            "all" if max_rows is None else f"first_{max_rows}_paired_rows"
        ),
        "n_document_rows": document_scores.height,
        "n_variants": variant_scores.height,
        "score_column": RAG_SCORE_COLUMNS[benchmark],
        "aggregation": "average raw fwd/rc LLR, then apply score transform",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_rag_probe_outputs(
    *,
    predictions: pl.DataFrame,
    metrics: pl.DataFrame,
    classifiers: dict[str, Any],
    output_dir: str | Path,
    model: str,
    model_source: str,
    model_revision: str | None,
    dataset_repo: str,
    dataset_revision: str,
    code_revision: str,
) -> None:
    """Write probe predictions, metrics, classifiers, and frozen protocol metadata."""
    assert len(dataset_revision) == 40
    assert len(code_revision) == 40
    assert model_source
    assert classifiers
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(output / "probe_predictions.parquet", compression="zstd")
    metrics.write_parquet(output / "probe_metrics.parquet", compression="zstd")
    joblib.dump(classifiers, output / "probe_classifiers.joblib")
    manifest = {
        "benchmark": "mendelian_traits",
        "model": model,
        "model_source": model_source,
        "model_revision": model_revision,
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "code_revision": code_revision,
        "human_pooling": {
            "coordinates": "0-based half-open token indices",
            "start": RAG_HUMAN_POOL_START,
            "stop": RAG_HUMAN_POOL_START + RAG_HUMAN_POOL_TOKENS,
            "n_tokens": RAG_HUMAN_POOL_TOKENS,
            "strand_aggregation": "float32 arithmetic mean of fwd/rc allele embeddings",
        },
        "feature": "concat_ref_delta = [emb_ref, emb_alt - emb_ref]",
        "probe": "StandardScaler + L2 LogisticRegression",
        "outer_cv": "leave-one-chromosome-out",
        "inner_cv": "GroupKFold(n_splits=5), retuned within every outer fold",
        "c_grid": np.logspace(-12, 4, 17).tolist(),
        "min_variants": 300,
        "min_chroms": 3,
        "metric": "per-chromosome-weighted AUPRC",
        "n_min": 30,
        "n_predictions": predictions.height,
        "n_probe_scored": predictions.filter(pl.col("probe_score").is_finite()).height,
        "n_classifiers": len(classifiers),
    }
    (output / "probe_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
