"""Cache full last-layer per-token variant embeddings (issue #314).

The GPU side of the frozen-embedding probe: for each variant, store the entire
last-layer hidden state — every DNA position × every hidden dimension — for both
alleles on both strands (``{fwd, rc} × {ref, alt}``). Extraction goes through the
shared HF ``Trainer`` runner (``run_variant_embeddings``), **not** a bespoke loop.

All downstream features (pooling extents, ``innerprod``, ``cov_delta``,
``delta``/``concat``/...; see ``variant_probe``) are then CPU transforms on this
cache, and FWD+RC averaging (or RC ablation) is a downstream choice — nothing
pooling-specific is decided here.

The dataset is processed in chunks so RAM stays bounded (``Trainer.predict``
accumulates one chunk's ``[chunk, 2, L, D]`` per strand). Each chunk writes one
float16 ``emb`` shard ``[chunk, 2, 2, L, D]`` — axes ``(strand{fwd,rc},
allele{ref,alt}, position, dim)`` — plus a ``keys`` parquet carrying the variant
metadata (label / subset / match_group / chrom for CV).
"""

import argparse
import io
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from datasets import Dataset

from marin_dna.data.genome import Genome
from marin_dna.model.runner import run_variant_embeddings

# Canonical GRCh38 reference staged in S3 (memory: canonical_grch38_reference).
DEFAULT_GENOME = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)
# Train chromosomes (odd) — the dev split; test (even) is never loaded here.
# Mirrors SPLIT_CHROMS["train"] in snakemake/evals/workflow/rules/common.smk.
TRAIN_CHROMS: tuple[str, ...] = tuple(str(i) for i in range(1, 23, 2)) + ("X",)

EMB_KEY_COLUMNS = ("chrom", "pos", "ref", "alt")
_STRANDS: tuple[Literal["+", "-"], ...] = ("+", "-")


def _write_shard(out_dir: str, idx: int, emb: np.ndarray, keys: pd.DataFrame) -> None:
    """Write one ``emb`` npz + ``keys`` parquet shard (local path or ``s3://``)."""
    name = f"shard_{idx:04d}"
    if out_dir.startswith("s3://"):
        import s3fs

        fs = s3fs.S3FileSystem()
        buf = io.BytesIO()
        np.savez(buf, emb=emb)
        with fs.open(f"{out_dir}/{name}.npz", "wb") as f:
            f.write(buf.getvalue())
    else:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.savez(f"{out_dir}/{name}.npz", emb=emb)
    pl.from_pandas(keys.reset_index(drop=True)).write_parquet(
        f"{out_dir}/{name}.keys.parquet"
    )


def cache_variant_embeddings(
    checkpoint_path: str | Path,
    variants: pd.DataFrame,
    out_dir: str,
    *,
    window_size: int,
    genome_path: str | Path = DEFAULT_GENOME,
    chunk_size: int = 1024,
    batch_size: int = 32,
    num_workers: int = 4,
    torch_compile: bool = False,
    layer_index: int = -1,
    limit: int | None = None,
) -> int:
    """Extract + cache ``{fwd,rc}×{ref,alt}`` per-token embeddings for ``variants``.

    Variants are sorted by ``(chrom, pos)`` for genome-read locality, then
    processed in ``chunk_size`` chunks; each chunk runs ``run_variant_embeddings``
    on both strands and writes a float16 ``[chunk, 2, 2, L, D]`` shard + keys.
    Returns the number of variants cached. ``checkpoint_path`` must be a base
    ``AutoModel`` checkpoint dir (``last_hidden_state``) for ``layer_index == -1``.
    """
    # Lazy heavy imports so the module (and its CPU orchestration test) load
    # without torch/transformers eagerly resolving.
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    model = AutoModel.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    genome = Genome(str(genome_path))

    variants = variants.sort_values(["chrom", "pos"]).reset_index(drop=True)
    if limit is not None:
        variants = variants.iloc[:limit].reset_index(drop=True)
    n = len(variants)
    assert n > 0, "no variants to cache"

    inference_kwargs = {
        "per_device_eval_batch_size": batch_size,
        "dataloader_num_workers": num_workers,
        "bf16_full_eval": True,  # GPU-only (errors on CPU); matches evals_v2
        "torch_compile": torch_compile,
        "remove_unused_columns": False,
    }

    for shard_idx, start in enumerate(range(0, n, chunk_size)):
        chunk = variants.iloc[start : start + chunk_size].reset_index(drop=True)
        hf_ds = Dataset.from_pandas(
            chunk.loc[:, list(EMB_KEY_COLUMNS)], preserve_index=False
        )
        per_strand = [
            run_variant_embeddings(
                model,
                tokenizer,
                hf_ds,
                genome,
                window_size,
                strand=strand,
                layer_index=layer_index,
                data_transform_on_the_fly=True,
                inference_kwargs=inference_kwargs,
            )
            for strand in _STRANDS
        ]  # each [chunk, 2(allele), L, D]
        emb = np.stack(per_strand, axis=1).astype(np.float16)  # [chunk, 2, 2, L, D]
        assert emb.shape[0] == len(chunk) and emb.shape[1:3] == (2, 2), emb.shape
        _write_shard(out_dir, shard_idx, emb, chunk)
        print(f"shard {shard_idx}: rows {start}..{start + len(chunk)} -> {emb.shape}")
    return n


def _load_train_variants(hf_dataset: str, revision: str | None) -> pd.DataFrame:
    """Load an evals dataset's train split, restricted to the train chromosomes."""
    from datasets import load_dataset

    ds = load_dataset(hf_dataset, revision=revision, split="train")
    df = ds.to_pandas()
    df["chrom"] = df["chrom"].astype(str)
    df = df[df["chrom"].isin(TRAIN_CHROMS)].reset_index(drop=True)
    assert len(df) > 0, f"no train-chrom rows in {hf_dataset}@{revision}"
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="local HF checkpoint dir")
    p.add_argument(
        "--hf_dataset", required=True, help="e.g. bolinas-dna/evals_mendelian_traits"
    )
    p.add_argument("--revision", default=None)
    p.add_argument("--out_dir", required=True, help="local dir or s3:// prefix")
    p.add_argument("--window_size", type=int, default=255)
    p.add_argument("--genome", default=DEFAULT_GENOME)
    p.add_argument("--chunk_size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--torch_compile", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="smoke-test row cap")
    args = p.parse_args()

    variants = _load_train_variants(args.hf_dataset, args.revision)
    n = cache_variant_embeddings(
        args.checkpoint,
        variants,
        args.out_dir,
        window_size=args.window_size,
        genome_path=args.genome,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        torch_compile=args.torch_compile,
        limit=args.limit,
    )
    print(f"cached {n} variants -> {args.out_dir}")


if __name__ == "__main__":
    main()
