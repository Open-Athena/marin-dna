"""Validate the #269 marginal-LLR path against the evals_v2 bundle LLR.

`compute_marginal_clm` returns the 4-allele marginal `log p(x)`; the LLR read
off it as `marginal[alt] - marginal[ref]` (FWD+RC-averaged via
`rc_average_marginal`) is — by construction — the same quantity
`compute_variant_score_bundle` already emits as `llr_fwd`/`llr_rc` in the
evals_v2 scores parquets. The unit/end-to-end tests prove that algebra on a tiny
random model; this script checks the *real-checkpoint plumbing* (tokenizer/BOS,
`window_size`, the 4-vs-2 suffix batching) against a known-good baseline.

Two comparisons, on the same variant subset + checkpoint:

  1. **marginal vs locally-recomputed bundle**, both fp32 on this box — the clean
     apples-to-apples check. Expect agreement at fp32 noise (~1e-5).
  2. **marginal (fp32) vs the S3 parquet's bundle LLR** (computed in bf16 on the
     GPU eval). Expect ~1% / r≈0.999+ — the residual is the fp32-vs-bf16 gap, not
     a marginal-vs-bundle discrepancy. (1) confirms the local run is faithful;
     (2) ties it to the canonical numbers.

Run (needs `--group genome-s3` so pyfaidx can read the reference from S3):

    uv run --group genome-s3 python scripts/issue269_validate_marginal_llr.py \
        --model exp136-proj_v30-step-9999 --window-size 255 \
        --dataset mendelian_traits --n 64

`--window-size` must match the model's evals_v2 config entry (255 = BOS runs,
256 = no-BOS, 512 = older 512-context). On a GPU box pass `--device cuda
--bf16` to reproduce the eval precision exactly (then comparison (2) collapses to
bf16 noise too).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("AWS_REGION", "us-east-2")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-2")

import numpy as np
import polars as pl
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.data.dna import NUCLEOTIDES
from marin_dna.data.genome import Genome
from marin_dna.model.runner import run_variant_marginal, run_variant_score_bundle
from marin_dna.model.scoring import rc_average_marginal

S3_BASE = "s3://oa-bolinas/snakemake/analysis/evals_v2/results"
GENOME_PATH = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)
# RC marginal column i is the forward allele complement(NUCLEOTIDES[i]); realign
# to forward ACGT order before reading per-strand RC LLRs (cf. rc_average_marginal).
_RC_PERM = [NUCLEOTIDES.index({"A": "T", "C": "G", "G": "C", "T": "A"}[n]) for n in NUCLEOTIDES]


def _nuc_idx(bases: list[str]) -> np.ndarray:
    return np.array([NUCLEOTIDES.index(b) for b in bases])


def _download_s3_dir(s3_uri: str, local: Path) -> None:
    """Recursively download an S3 prefix to ``local`` via fsspec/s3fs.

    Avoids a hard dependency on the ``aws`` CLI (absent on a fresh sky node);
    s3fs is already pulled in by the ``genome-s3`` group.
    """
    import fsspec

    fs = fsspec.filesystem("s3")
    base = s3_uri.replace("s3://", "").rstrip("/")
    for key in fs.find(base):
        dest = local / key[len(base) + 1 :]
        dest.parent.mkdir(parents=True, exist_ok=True)
        fs.get_file(key, str(dest))


def _upload_dir(local: Path, s3_prefix: str, pattern: str) -> None:
    """Upload ``local``'s files matching ``pattern`` under an S3 prefix via fsspec."""
    import fsspec

    fs = fsspec.filesystem("s3")
    base = s3_prefix.replace("s3://", "").rstrip("/")
    for f in sorted(local.glob(pattern)):
        fs.put_file(str(f), f"{base}/{f.name}")
    print(f"uploaded {pattern} → s3://{base}/")


def _summarize(name: str, marginal: np.ndarray, baseline: np.ndarray) -> dict:
    d = marginal - baseline
    r = float(np.corrcoef(marginal, baseline)[0, 1])
    row = {
        "comparison": name,
        "max_abs_diff": float(np.abs(d).max()),
        "mean_abs_diff": float(np.abs(d).mean()),
        "pearson_r": r,
    }
    print(
        f"  {name:24s}  max|Δ|={row['max_abs_diff']:.2e}  "
        f"mean|Δ|={row['mean_abs_diff']:.2e}  r={r:.7f}"
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="exp136-proj_v30-step-9999")
    ap.add_argument("--window-size", type=int, default=255)
    ap.add_argument("--dataset", default="mendelian_traits")
    ap.add_argument("--n", type=int, default=64, help="variant subset size (0 = all)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--bf16", action="store_true", help="bf16_full_eval (GPU only)")
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads (shared box etiquette)")
    ap.add_argument("--outdir", default="scratch/issue269_validation")
    ap.add_argument("--upload-prefix", default="", help="optional s3:// prefix to upload outputs")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Bundle-LLR baseline from S3 + variant subset.
    scores_uri = f"{S3_BASE}/scores/{args.model}/{args.dataset}.parquet"
    df = pl.read_parquet(scores_uri).select(
        ["chrom", "pos", "ref", "alt", "llr_fwd", "llr_rc"]
    )
    if args.n and args.n < df.height:
        df = df.sample(n=args.n, seed=args.seed)
    print(f"{df.height} variants from {scores_uri}")

    # 2. Pull the cached HF checkpoint locally (idempotent).
    ckpt_local = outdir / "ckpt" / args.model
    if not (ckpt_local / "config.json").exists():
        print(f"downloading checkpoint → {ckpt_local}")
        _download_s3_dir(f"{S3_BASE}/checkpoints/{args.model}/", ckpt_local)

    # 3. Model / tokenizer / genome.
    tokenizer = AutoTokenizer.from_pretrained(ckpt_local)
    model = (
        AutoModelForCausalLM.from_pretrained(ckpt_local, trust_remote_code=True)
        .to(args.device)
        .eval()
    )
    genome = Genome(GENOME_PATH)
    ds = Dataset.from_pandas(
        df.select(["chrom", "pos", "ref", "alt"]).to_pandas(), preserve_index=False
    )
    inference_kwargs = dict(
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        bf16_full_eval=args.bf16,
        use_cpu=(args.device == "cpu"),
    )

    # 4. Marginal path → per-strand LLRs + FWD+RC-averaged LLR.
    print("running run_variant_marginal (rc=True) ...")
    marg = run_variant_marginal(
        model, tokenizer, ds, genome, args.window_size, rc=True,
        data_transform_on_the_fly=True, inference_kwargs=inference_kwargs,
    )
    avg = rc_average_marginal(marg["fwd"], marg["rc"])  # [N, 4], forward ACGT order
    rc_aligned = marg["rc"][:, _RC_PERM]
    ref_idx, alt_idx = _nuc_idx(df["ref"].to_list()), _nuc_idx(df["alt"].to_list())
    rows = np.arange(df.height)
    m_llr = {
        "fwd": marg["fwd"][rows, alt_idx] - marg["fwd"][rows, ref_idx],
        "rc": rc_aligned[rows, alt_idx] - rc_aligned[rows, ref_idx],
        "avg": avg[rows, alt_idx] - avg[rows, ref_idx],
    }

    # 5. Recompute the bundle locally (same precision) → clean apples-to-apples.
    print("running run_variant_score_bundle (rc=True) ...")
    bundle = run_variant_score_bundle(
        model, tokenizer, ds, genome, args.window_size, rc=True,
        data_transform_on_the_fly=True, inference_kwargs=inference_kwargs,
    )
    b_local = {
        "fwd": bundle["fwd"][:, 0],
        "rc": bundle["rc"][:, 0],
        "avg": 0.5 * (bundle["fwd"][:, 0] + bundle["rc"][:, 0]),
    }
    # S3 bf16 baseline.
    b_s3 = {
        "fwd": df["llr_fwd"].to_numpy(),
        "rc": df["llr_rc"].to_numpy(),
        "avg": 0.5 * (df["llr_fwd"].to_numpy() + df["llr_rc"].to_numpy()),
    }

    # 6. Compare.
    print(f"\n=== marginal LLR vs locally-recomputed bundle ({args.device} fp{'16' if args.bf16 else '32'}, apples-to-apples) ===")
    summary = [_summarize(f"{s}", m_llr[s], b_local[s]) for s in ("fwd", "rc", "avg")]
    print("\n=== marginal LLR vs S3 bundle LLR (bf16 GPU-eval baseline) ===")
    summary += [_summarize(f"{s}_vs_s3", m_llr[s], b_s3[s]) for s in ("fwd", "rc", "avg")]

    # 7. Persist comparison table + per-variant LLRs + a scatter plot.
    pl.DataFrame(summary).write_parquet(outdir / f"summary_{args.model}_{args.dataset}.parquet")
    per_variant = df.select(["chrom", "pos", "ref", "alt"]).with_columns(
        marginal_llr_avg=pl.Series(m_llr["avg"]),
        bundle_local_llr_avg=pl.Series(b_local["avg"]),
        bundle_s3_llr_avg=pl.Series(b_s3["avg"]),
    )
    per_variant.write_parquet(outdir / f"per_variant_{args.model}_{args.dataset}.parquet")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, b, title in (
            (axes[0], b_local["avg"], "vs local bundle (fp32)"),
            (axes[1], b_s3["avg"], "vs S3 bundle (bf16)"),
        ):
            lo = min(m_llr["avg"].min(), b.min())
            hi = max(m_llr["avg"].max(), b.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
            ax.scatter(b, m_llr["avg"], s=10, alpha=0.6)
            ax.set_xlabel(f"bundle LLR_avg ({title})")
            ax.set_ylabel("marginal LLR_avg")
            ax.set_title(title)
        fig.suptitle(f"{args.model} · {args.dataset} · n={df.height}")
        fig.tight_layout()
        png = outdir / f"scatter_{args.model}_{args.dataset}.png"
        fig.savefig(png, dpi=130)
        print(f"\nwrote {png}")
    except ImportError:
        print("\n(matplotlib unavailable — skipping scatter)")

    if args.upload_prefix:
        _upload_dir(
            outdir,
            f"{args.upload_prefix.rstrip('/')}/{args.model}_{args.dataset}",
            f"*_{args.model}_{args.dataset}.*",
        )


if __name__ == "__main__":
    main()
