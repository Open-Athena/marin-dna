"""Inference utilities for computing variant scores using genomic language models."""

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.data.genome import Genome
from marin_dna.model.runner import run_variant_score_bundle


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
    return acc.astype(np.float16)


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
    # marin_dna.model.runner expects — no adapter wrappers needed.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
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
        width = next(iter(results.values())).shape[1]
        hidden_size = model.config.hidden_size
        assert width == 2 + 2 * hidden_size, (
            f"score-bundle width {width} != 2 + 2*hidden_size "
            f"(hidden_size={hidden_size}); embedding slicing would be wrong"
        )
        d = hidden_size
        cols["emb_ref"] = list(
            fwd_rc_average_f16([arr[:, 2 : 2 + d] for arr in results.values()])
        )
        cols["emb_alt"] = list(
            fwd_rc_average_f16([arr[:, 2 + d : 2 + 2 * d] for arr in results.values()])
        )
    return pd.DataFrame(cols)
