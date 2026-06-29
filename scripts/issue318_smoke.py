"""GPU smoke for #318 — pooled embeddings in the variant LLR/JSD score bundle.

Runs the *production* code path (``compute_variant_scores``) with
``return_embeddings=True`` on a small slice of ``exp135 × mendelian_traits``, on a
real A10G with the real checkpoint + the S3 genome — the one thing the CPU tests
can't cover (does ``output_hidden_states`` fit VRAM, do the wide ``[N, 2+2D]``
predictions collate, are the columns right at scale). It does **not** touch any
production S3 parquet.

Checks:
  - ``emb_ref`` / ``emb_alt`` columns are ``[n, hidden_size]`` float16, finite.
  - ``llr_*`` / ``jsd_*`` are unchanged vs a ``return_embeddings=False`` run on the
    same slice (the invariant that must hold on real hardware: identical forwards).
  - reports peak VRAM so we can confirm the heavier forward fits the A10G.

The checkpoint is pulled from the evals_v2 S3 cache (instance-role creds — no
gcloud). ``num_workers=0`` keeps genome S3 reads in the main process, sidestepping
the fork-after-s3fs-init deadlock (see the off-pipeline-scoring gotcha).

Run on a GPU node:
  uv run --group genome-s3 python scripts/issue318_smoke.py --n 128 --batch-size 16
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import s3fs
import torch
from datasets import load_dataset

from marin_dna.pipelines.evals.inference import compute_variant_scores

MODEL = "mix-v0.9-p1B-i24-exp135-m5.1-step-59158"
HIDDEN = 1920  # exp135 Qwen3-1B hidden size
WINDOW = 255  # 255 bp DNA + BOS
CKPT_S3 = "oa-bolinas/snakemake/analysis/evals_v2/results/checkpoints/" + MODEL
GENOME = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)
MEND_REPO = "bolinas-dna/evals_mendelian_traits"
MEND_REV = (
    "4aed58e50c5dea0b878a665007af2ef9e5108e9f"  # PR #194 k=9 rebuild (config pin)
)


def localize_checkpoint() -> str:
    """Download the cached HF checkpoint into a flat local dir.

    Get each file individually rather than ``fs.get(prefix, dst, recursive=True)``:
    the recursive form nests the files under ``dst/<basename>/`` (fsspec copies the
    dir, not its contents), so ``from_pretrained(dst)`` then can't find config.json.
    """
    fs = s3fs.S3FileSystem()
    assert fs.exists(CKPT_S3), f"checkpoint cache missing: s3://{CKPT_S3}"
    dst = tempfile.mkdtemp(prefix="exp135_ckpt_")
    for f in fs.ls(CKPT_S3, detail=False):
        name = f.rstrip("/").split("/")[-1]
        if name.startswith("."):  # skip .snakemake_timestamp etc.
            continue
        fs.get_file(f, os.path.join(dst, name))
    assert os.path.exists(os.path.join(dst, "config.json")), (
        f"config.json not localized into {dst}"
    )
    return dst


def _score(slice_df, ckpt_dir: str, batch_size: int, return_embeddings: bool):
    return compute_variant_scores(
        checkpoint_path=ckpt_dir,
        dataset=slice_df,
        genome_path=GENOME,
        context_size=WINDOW,
        batch_size=batch_size,
        num_workers=0,  # no DataLoader fork → no s3fs fork-deadlock
        data_transform_on_the_fly=True,
        torch_compile=False,
        rc=True,
        return_embeddings=return_embeddings,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA — this smoke must run on a GPU"
    print("GPU:", torch.cuda.get_device_name(0))

    ckpt_dir = localize_checkpoint()
    print("checkpoint localized to", ckpt_dir)
    ds = load_dataset(MEND_REPO, split="train", revision=MEND_REV).to_pandas()
    slice_df = ds.iloc[: args.n].reset_index(drop=True)
    print(f"scoring {len(slice_df)} variants (rc=True, return_embeddings=True)")

    torch.cuda.reset_peak_memory_stats()
    emb = _score(slice_df, ckpt_dir, args.batch_size, return_embeddings=True)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak torch VRAM (embeddings on): {peak_gb:.2f} GB / 24 GB")

    for col in ("emb_ref", "emb_alt"):
        arr = np.stack(emb[col].to_numpy())
        assert arr.shape == (len(slice_df), HIDDEN), (col, arr.shape)
        assert arr.dtype == np.float16, (col, arr.dtype)
        assert np.isfinite(arr.astype(np.float32)).all(), f"{col} non-finite"
        print(
            f"  {col}: shape {arr.shape} dtype {arr.dtype} finite "
            f"|mean|={np.abs(arr.astype(np.float32)).mean():.4f}"
        )

    # delta = alt - ref should be a small, mostly-nonzero signal (the probe feature).
    d = np.stack(emb["emb_alt"].to_numpy()).astype(np.float32) - np.stack(
        emb["emb_ref"].to_numpy()
    ).astype(np.float32)
    print(
        f"  |emb_alt - emb_ref| mean={np.abs(d).mean():.4f} max={np.abs(d).max():.4f}"
    )

    # Invariant on real hardware: identical forwards ⇒ same llr/jsd.
    base = _score(slice_df, ckpt_dir, args.batch_size, return_embeddings=False)
    max_diff = 0.0
    for col in ("llr_fwd", "llr_rc", "jsd_fwd", "jsd_rc"):
        max_diff = max(
            max_diff, float(np.abs(emb[col].to_numpy() - base[col].to_numpy()).max())
        )
    print(f"max |llr/jsd diff| (emb on vs off): {max_diff:.2e}")
    assert max_diff < 1e-3, f"llr/jsd drifted with embeddings on: {max_diff}"
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
