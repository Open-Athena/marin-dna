"""issue #302 — driver: extract per-variant ref/alt embeddings for one checkpoint.

Loads an HF checkpoint (from a local dir or a gs:// path it downloads), reads the
Mendelian missense variants (the same HF dataset+revision the evals_v2 pipeline
scored), extracts ref/alt center-pooled FWD+RC-averaged hidden states at a sparse
set of layer depths, and writes the embeddings (npz) + variant keys (parquet).

The embedding probe (the plateau-vs-degrade test) then runs on these caches.

Usage (on a GPU node, e.g. via the evals_v2 sky cluster):
  uv run --group genome-s3 python scripts/issue302/extract_variant_embeddings.py \
    --gcs_path gs://marin-us-east5/checkpoints/dna-bolinas-scaling-v0.5-h896-p128M-43ec40/hf/step-215573 \
    --name scaling-v0.5-128M --out s3://oa-bolinas/analysis/issue302/embeddings
  # add --limit 50 --device cpu --dtype float32 for a smoke.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from marin_dna.data.genome import Genome
from marin_dna.pipelines.evals.variant_embedding import (
    compute_variant_embeddings,
    resolve_layer_indices,
)

DATASET = "bolinas-dna/evals_mendelian_traits"
REVISION = (
    "4aed58e50c5dea0b878a665007af2ef9e5108e9f"  # PR #194 k=9 (config hf_revision)
)
GENOME = "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
KEEP = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--s3_path",
        default=None,
        help="s3:// HF checkpoint dir to download (preferred)",
    )
    ap.add_argument(
        "--gcs_path", default=None, help="gs:// HF checkpoint dir to download"
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help="local HF checkpoint dir (alternative to --s3_path/--gcs_path)",
    )
    ap.add_argument(
        "--name", required=True, help="model short name for output filenames"
    )
    ap.add_argument("--out", required=True, help="output dir (local or s3://)")
    ap.add_argument("--window_size", type=int, default=255)
    ap.add_argument("--layer_fracs", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--n_center_bp", type=int, default=100)
    ap.add_argument(
        "--subset", default="missense_variant", help="comma-sep subsets, or 'all'"
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=0, help="smoke: only first N variants")
    args = ap.parse_args()

    ckpt = args.ckpt
    if args.s3_path:
        ckpt = tempfile.mkdtemp(prefix="ckpt_")
        print(f"downloading {args.s3_path} -> {ckpt}", flush=True)
        subprocess.run(
            [
                "aws",
                "s3",
                "sync",
                args.s3_path.rstrip("/") + "/",
                ckpt,
                "--no-progress",
            ],
            check=True,
        )
    elif args.gcs_path:
        ckpt = tempfile.mkdtemp(prefix="ckpt_")
        print(f"downloading {args.gcs_path} -> {ckpt}", flush=True)
        subprocess.run(
            ["gcloud", "storage", "cp", "-r", f"{args.gcs_path}/*", ckpt], check=True
        )
    assert ckpt, "need --s3_path, --gcs_path or --ckpt"

    print("loading variants...", flush=True)
    df = load_dataset(DATASET, revision=REVISION, split="train").to_pandas()
    df["chrom"] = df["chrom"].astype(str)
    if args.subset != "all":
        df = df[df["subset"].isin(args.subset.split(","))].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    variants = df[["chrom", "pos", "ref", "alt"]].to_dict("records")
    print(f"  {len(variants)} variants (subset={args.subset})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
    n_layers = model.config.num_hidden_layers
    layer_fracs = tuple(float(x) for x in args.layer_fracs.split(","))
    layer_indices = resolve_layer_indices(n_layers, layer_fracs)
    print(
        f"  model {args.name}: {n_layers} layers, hidden={model.config.hidden_size}, "
        f"layer_indices={layer_indices} (fracs {layer_fracs})",
        flush=True,
    )

    genome = Genome(GENOME)
    ref_emb, alt_emb = compute_variant_embeddings(
        model,
        tokenizer,
        genome,
        variants,
        args.window_size,
        layer_indices=layer_indices,
        n_center_bp=args.n_center_bp,
        rc=True,
        batch_size=args.batch_size,
        device=args.device,
        dtype=getattr(torch, args.dtype),
    )
    assert ref_emb.shape[0] == len(variants)
    assert np.isfinite(ref_emb).all() and np.isfinite(alt_emb).all(), (
        "non-finite embeddings"
    )
    print(f"  ref_emb {ref_emb.shape} alt_emb {alt_emb.shape}", flush=True)

    out = args.out.rstrip("/")
    local = Path(tempfile.mkdtemp(prefix="emb_"))
    npz = local / f"{args.name}.npz"
    keys = local / f"{args.name}.keys.parquet"
    np.savez_compressed(
        npz,
        ref=ref_emb.astype(np.float16),
        alt=alt_emb.astype(np.float16),
        layer_indices=np.array(layer_indices),
        layer_fracs=np.array(layer_fracs),
        n_layers=n_layers,
    )
    pl.from_pandas(df[KEEP]).write_parquet(keys)
    if out.startswith("s3://"):
        for f in (npz, keys):
            subprocess.run(["aws", "s3", "cp", str(f), f"{out}/{f.name}"], check=True)
        print(f"  uploaded -> {out}/{npz.name}, {keys.name}", flush=True)
    else:
        Path(out).mkdir(parents=True, exist_ok=True)
        for f in (npz, keys):
            f.rename(Path(out) / f.name)
        print(f"  wrote -> {out}/", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
