"""Inference utilities for computing variant scores using genomic language models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from marin_dna.data.genome import Genome
from marin_dna_evals.model.runner import run_variant_score_bundle


def _load_checkpoint_config(checkpoint_path: Path) -> PretrainedConfig:
    """Load a checkpoint config, adapting Transformers 5 RoPE serialization.

    Transformers 5 writes Qwen3 rotary settings under ``rope_parameters``.
    Transformers 4.57 (the pinned eval runtime) preserves that unknown field but
    silently constructs the model with its defaults: ``rope_theta=10_000`` and
    no scaling. That changes inference for a model trained with Llama-3 RoPE.

    Translate the new representation into Transformers 4's ``rope_theta`` plus
    ``rope_scaling`` fields in memory. Native Transformers 4 exports continue
    through the unchanged path.
    """
    config_path = checkpoint_path / "config.json"
    if not config_path.is_file():
        return AutoConfig.from_pretrained(checkpoint_path)

    with config_path.open(encoding="utf-8") as config_file:
        raw_config = json.load(config_file)
    assert isinstance(raw_config, dict), f"{config_path} must contain a JSON object"

    rope_parameters = raw_config.get("rope_parameters")
    if rope_parameters is None:
        return AutoConfig.from_pretrained(checkpoint_path)

    assert isinstance(rope_parameters, dict), (
        f"{config_path}: rope_parameters must be a JSON object"
    )
    assert raw_config.get("rope_scaling") is None, (
        f"{config_path}: refusing ambiguous rope_parameters + rope_scaling"
    )
    assert "rope_theta" in rope_parameters, (
        f"{config_path}: rope_parameters is missing rope_theta"
    )
    rope_theta = rope_parameters["rope_theta"]
    assert isinstance(rope_theta, (int, float)) and rope_theta > 0, (
        f"{config_path}: rope_theta must be positive, got {rope_theta!r}"
    )
    rope_scaling = {
        key: value for key, value in rope_parameters.items() if key != "rope_theta"
    }
    assert "rope_type" in rope_scaling, (
        f"{config_path}: rope_parameters is missing rope_type"
    )

    config = AutoConfig.from_pretrained(
        checkpoint_path,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )
    assert config.rope_theta == rope_theta
    assert config.rope_scaling == rope_scaling
    return config


def _load_checkpoint_tokenizer(checkpoint_path: Path) -> PreTrainedTokenizerBase:
    """Load native Transformers 4 and Transformers 5 tokenizer exports.

    Transformers 5 serializes a raw ``tokenizers.Tokenizer`` as
    ``tokenizer_class: TokenizersBackend``. Transformers 4 cannot resolve that
    class through ``AutoTokenizer``, even though its
    ``PreTrainedTokenizerFast`` reads the same ``tokenizer.json`` and special
    token post-processor without changing token IDs or encoding semantics.
    Keep the fallback exact so unrelated tokenizer errors still fail loudly.
    """
    tokenizer_config_path = checkpoint_path / "tokenizer_config.json"
    if tokenizer_config_path.is_file():
        with tokenizer_config_path.open(encoding="utf-8") as config_file:
            tokenizer_config = json.load(config_file)
        assert isinstance(tokenizer_config, dict), (
            f"{tokenizer_config_path} must contain a JSON object"
        )
        if tokenizer_config.get("tokenizer_class") == "TokenizersBackend":
            tokenizer_json_path = checkpoint_path / "tokenizer.json"
            assert tokenizer_json_path.is_file(), (
                "TokenizersBackend export is missing tokenizer.json: "
                f"{tokenizer_json_path}"
            )
            return PreTrainedTokenizerFast.from_pretrained(checkpoint_path)
    return AutoTokenizer.from_pretrained(checkpoint_path)


def fwd_rc_average_f16(strand_embs: list[np.ndarray]) -> np.ndarray:
    """FWD+RC mean of per-strand fp32 pooled embeddings, cast to float16 for storage.

    Each ``strand_embs[i]`` is one strand's pooled vectors ``[N, D]`` (fp32). The
    mean is the #314 protocol's FWD+RC average. Accumulate in **fp32** and cast to
    f16 only at the very end: f16 storage of allele *means* already bounds the
    downstream probe feature ``delta = emb_alt - emb_ref`` (a small difference of
    two near-equal vectors — catastrophic cancellation), so the aggregation itself
    must not add rounding on top of it (issue #318). Returns ``[N, D]`` f16.
    """
    assert strand_embs, "need at least one strand's embeddings to average"
    acc = np.zeros_like(strand_embs[0], dtype=np.float32)
    for e in strand_embs:
        acc += np.asarray(e, dtype=np.float32)
    acc /= len(strand_embs)
    out = acc.astype(np.float16)
    # Fail loud rather than silently ship inf: a pooled mean beyond float16's
    # ±65504 (e.g. a persistent massive-activation channel) overflows on the cast,
    # which would NaN-poison the downstream delta = emb_alt - emb_ref.
    assert np.isfinite(out).all(), (
        "non-finite pooled embedding after f16 cast — the fp32 mean exceeded the "
        "float16 range (±65504) or was non-finite upstream"
    )
    return out


def compute_variant_scores(
    checkpoint_path: str | Path,
    dataset: pd.DataFrame,
    genome_path: str | Path,
    context_size: int = 512,
    batch_size: int = 512,
    num_workers: int = 4,
    data_transform_on_the_fly: bool = True,
    torch_compile: bool = False,
    rc: bool = False,
    return_embeddings: bool = False,
    eval_accumulation_steps: int | None = None,
) -> pd.DataFrame:
    """Compute variant scores from a CLM: per-strand LLR + next-token JSD.

    Takes a dataset of genomic variants and computes the score bundle
    using a causal language model. Returns the raw per-strand atoms
    only; downstream code derives ``_avg`` / ``minus_llr_*`` /
    ``abs_llr_*`` variants.

    Args:
        checkpoint_path: Path to model checkpoint directory.
        dataset: DataFrame with columns [chrom, pos, ref, alt, label] and optionally [subset].
        genome_path: Path to genome reference FASTA. May be a local filesystem
            path or an fsspec URI (e.g. ``s3://bucket/genome.fa.gz``); the
            latter requires the ``genome-s3`` dependency group.
        context_size: Context window size for model inference.
        batch_size: Number of sequences per batch during inference.
        num_workers: Number of workers for data loading.
        data_transform_on_the_fly: Whether to transform data on the fly during inference.
        torch_compile: Whether to use torch.compile for faster inference.
        rc: If True, also score the reverse-complemented window for
            each variant and emit per-strand columns. Doubles inference
            cost.
        return_embeddings: If True, also emit the entire-window-pooled,
            both-allele, FWD+RC-averaged last-layer embeddings as ``emb_ref`` /
            ``emb_alt`` ``list[f16]`` columns (issue #318), from the *same*
            forwards as the scores. Requires ``rc=True`` (the stored vector is the
            FWD+RC average). ``output_hidden_states`` makes the forward ~2× heavier
            — use a smaller ``batch_size`` and/or ``eval_accumulation_steps``.
        eval_accumulation_steps: If set, offload ``Trainer.predict`` predictions
            to CPU every N steps (passed straight to ``TrainingArguments``).
            Execution-only — the heavier ``[N, 2 + 2D]`` predictions of an
            embedding run can otherwise accumulate on-GPU and OOM. ``None``
            (default) leaves behaviour unchanged.

    Returns:
        DataFrame with per-strand score atoms. Rows align with input
        dataset by index.

        - ``rc=False`` → 2 columns: ``llr_fwd``, ``jsd_fwd``.
        - ``rc=True``  → 4 columns: ``llr_fwd``, ``llr_rc``,
          ``jsd_fwd``, ``jsd_rc``.
        - ``return_embeddings=True`` (requires ``rc=True``) adds ``emb_ref`` and
          ``emb_alt`` — each a length-``D`` ``float16`` vector per row, the
          entire-DNA-window mean-pooled, FWD+RC-averaged hidden state for that
          allele.

        ``llr_*`` is the raw log-likelihood ratio; ``jsd_*`` is the mean
        per-position 4-nucleotide softmax JSD over downstream positions
        (called ``down_jsd_mean`` in Open-Athena/marin-dna#175).
        Downstream consumers compute ``_avg``, ``minus_llr_*``, and
        ``abs_llr_*`` as needed.
    """
    assert rc or not return_embeddings, (
        "return_embeddings=True requires rc=True — the stored embedding is the "
        "FWD+RC average; a forward-only embedding would be silently mislabeled"
    )
    checkpoint_path = Path(checkpoint_path)
    # Don't Path()-cast genome_path: would break s3:// URIs (POSIX path
    # normalization collapses // to /). Genome accepts str | Path and
    # detects the remote scheme itself.
    genome = Genome(genome_path)
    # AutoTokenizer / AutoModelForCausalLM satisfy the duck-typed interface
    # marin_dna_evals.model.runner expects — no adapter wrappers needed.
    tokenizer = _load_checkpoint_tokenizer(checkpoint_path)
    model_config = _load_checkpoint_config(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        config=model_config,
        trust_remote_code=True,
    )
    hf_dataset = Dataset.from_pandas(dataset, preserve_index=False)

    inference_kwargs: dict[str, object] = {
        "per_device_eval_batch_size": batch_size,
        "torch_compile": torch_compile,
        "bf16_full_eval": True,
        "dataloader_num_workers": num_workers,
        "remove_unused_columns": False,
    }
    if eval_accumulation_steps is not None:
        inference_kwargs["eval_accumulation_steps"] = eval_accumulation_steps

    results = run_variant_score_bundle(
        model,
        tokenizer,
        hf_dataset,
        genome,
        context_size,
        rc=rc,
        return_embeddings=return_embeddings,
        data_transform_on_the_fly=data_transform_on_the_fly,
        inference_kwargs=inference_kwargs,
    )

    cols: dict[str, object] = {}
    for strand, arr in results.items():
        cols[f"llr_{strand}"] = arr[:, 0]
        cols[f"jsd_{strand}"] = arr[:, 1]

    if return_embeddings:
        # Each per-strand array is [N, 2 + 2D]: cols [2:2+D] = emb_ref,
        # [2+D:2+2D] = emb_alt (fp32, kernel-pooled). Average the strands in fp32
        # and store each as a length-D float16 vector per row (list[f16] column).
        # Bundle layout is [B, 2 + 2D]: 2 score cols (llr, jsd), then emb_ref, emb_alt.
        d = model.config.hidden_size
        width = next(iter(results.values())).shape[1]
        assert width == 2 + 2 * d, (
            f"score-bundle width {width} != 2 + 2*hidden_size ({d}); the 2 score "
            f"cols + 2 embedding blocks are mis-sized — emb slicing would be wrong"
        )
        cols["emb_ref"] = list(
            fwd_rc_average_f16([arr[:, 2 : 2 + d] for arr in results.values()])
        )
        cols["emb_alt"] = list(
            fwd_rc_average_f16([arr[:, 2 + d : 2 + 2 * d] for arr in results.values()])
        )
    return pd.DataFrame(cols)
