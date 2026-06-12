"""issue #314 iter3 — variant-type transfer heatmaps.

One heatmap per model: rows = the subset a probe was *trained* on (plus the zero-shot
``LLR`` baseline row and a ``pooled_all`` row), columns = the subset it is *evaluated* on,
colour = per-chromosome-weighted AUPRC (chromosome-grouped, leak-proof). The diagonal is
within-subset; off-diagonal is cross-consequence transfer (boxed cells = diagonal). Reads
the ``iter3_transfer`` parquets straight from S3; writes SVG (for the GitHub comment) + PNG
(for a local visual sanity-check).

Run:  uv run python plots/issue314_transfer.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Rectangle

MODELS = ["exp135-1B-m5.1", "scaling-v0.5-1B", "exp166-v0.1-p1B", "exp166-v0.1-p4B"]
EVAL_ORDER = ["missense_variant", "distal", "tss_proximal", "splicing",
              "synonymous_variant", "5_prime_UTR_variant",
              "non_coding_transcript_exon_variant", "3_prime_UTR_variant"]
SH = {"missense_variant": "missense", "distal": "distal", "tss_proximal": "tss",
      "splicing": "splicing", "synonymous_variant": "synon",
      "5_prime_UTR_variant": "5′UTR", "non_coding_transcript_exon_variant": "ncRNA",
      "3_prime_UTR_variant": "3′UTR", "LLR": "LLR (zero-shot)", "pooled_all": "pooled-all"}
BASE = "s3://oa-bolinas/analysis/issue314/iter3_transfer"
OUT = Path("plots/output/issue314_transfer")


def matrix(df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    rows = ["LLR"] + EVAL_ORDER + ["pooled_all"]
    lut = {(r["train"], r["eval"]): r["auprc"] for r in df.iter_rows(named=True)}
    M = np.full((len(rows), len(EVAL_ORDER)), np.nan)
    for i, tr in enumerate(rows):
        for j, ev in enumerate(EVAL_ORDER):
            M[i, j] = lut.get((tr, ev), np.nan)
    return M, rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    im = None
    for ax, model in zip(axes.flat, MODELS):
        try:
            df = pl.read_parquet(f"{BASE}/transfer_{model}.parquet")
        except Exception as e:
            ax.set_title(f"{model}\n(missing: {type(e).__name__})")
            ax.axis("off")
            continue
        M, rows = matrix(df)
        im = ax.imshow(M, cmap="viridis", vmin=0.10, vmax=0.55, aspect="auto")
        ax.set_xticks(range(len(EVAL_ORDER)), [SH[s] for s in EVAL_ORDER],
                      rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(rows)), [SH.get(r, r) for r in rows], fontsize=9)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] < 0.36 else "black")
        # box the within-subset diagonal (train-subset rows are offset by 1 = the LLR row)
        for j, ev in enumerate(EVAL_ORDER):
            ax.add_patch(Rectangle((j - 0.5, rows.index(ev) - 0.5), 1, 1,
                                   fill=False, edgecolor="red", lw=1.6))
        # separator lines under the LLR row and above pooled-all
        ax.axhline(0.5, color="white", lw=2)
        ax.axhline(len(rows) - 1.5, color="white", lw=2)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xlabel("evaluated on", fontsize=10)
        ax.set_ylabel("trained on", fontsize=10)
    fig.suptitle("Variant-type transfer — per-chromosome-weighted AUPRC "
                 "(entire_window/abs_delta, C=1e-3; red box = within-subset diagonal)",
                 fontsize=13, y=0.995)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="per-chrom AUPRC")
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"figure.{ext}", bbox_inches="tight", dpi=120)
    print(f"wrote {OUT}/figure.svg + figure.png")


if __name__ == "__main__":
    main()
