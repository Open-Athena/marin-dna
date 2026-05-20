"""Inference utilities for computing variant scores using genomic language models."""

from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.data.genome import Genome
from marin_dna.model.runner import run_variant_score_bundle


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

    Returns:
        DataFrame with per-strand score atoms. Rows align with input
        dataset by index.

        - ``rc=False`` → 2 columns: ``llr_fwd``, ``jsd_fwd``.
        - ``rc=True``  → 4 columns: ``llr_fwd``, ``llr_rc``,
          ``jsd_fwd``, ``jsd_rc``.

        ``llr_*`` is the raw log-likelihood ratio; ``jsd_*`` is the mean
        per-position 4-nucleotide softmax JSD over downstream positions
        (called ``down_jsd_mean`` in Open-Athena/marin-dna#175).
        Downstream consumers compute ``_avg``, ``minus_llr_*``, and
        ``abs_llr_*`` as needed.
    """
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

    results = run_variant_score_bundle(
        model,
        tokenizer,
        hf_dataset,
        genome,
        context_size,
        rc=rc,
        data_transform_on_the_fly=data_transform_on_the_fly,
        inference_kwargs={
            "per_device_eval_batch_size": batch_size,
            "torch_compile": torch_compile,
            "bf16_full_eval": True,
            "dataloader_num_workers": num_workers,
            "remove_unused_columns": False,
        },
    )

    cols: dict[str, object] = {}
    for strand, arr in results.items():
        cols[f"llr_{strand}"] = arr[:, 0]
        cols[f"jsd_{strand}"] = arr[:, 1]
    return pd.DataFrame(cols)
