"""issue #314 iter3 — variant-type transfer heatmaps (probe − zero-shot LLR, nested-C).

One heatmap per model. Rows = the subset a probe was *trained* on (+ a ``pooled_all`` row);
columns = the subset it is *evaluated* on, plus an **``all``** column = AUPRC over all subsets
bundled together (the global metric). Colour = per-chromosome-weighted AUPRC of the probe
**minus the zero-shot LLR** on that eval target — green = probe beats the baseline, red = loses.
C is nested-tuned per fold (same protocol as iter2; no fixed-C confound). The diagonal (boxed)
is within-subset; off-diagonal is cross-consequence transfer; the per-column LLR baseline is
printed under each column label.

Reads the ``iter3_transfer`` parquets from S3; writes SVG (GitHub) + PNG (local check).
Run:  uv run python plots/issue314_transfer.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Rectangle

MODELS = ["exp135-1B-m5.1", "scaling-v0.5-1B", "exp166-v0.1-p1B", "exp166-v0.1-p4B"]
EVAL_ORDER = ["missense_variant", "distal", "tss_proximal", "splicing", "synonymous_variant",
              "5_prime_UTR_variant", "non_coding_transcript_exon_variant", "3_prime_UTR_variant"]
COLS = EVAL_ORDER + ["all"]
ROWS = EVAL_ORDER + ["pooled_all"]
SH = {"missense_variant": "missense", "distal": "distal", "tss_proximal": "tss",
      "splicing": "splicing", "synonymous_variant": "synon", "5_prime_UTR_variant": "5′UTR",
      "non_coding_transcript_exon_variant": "ncRNA", "3_prime_UTR_variant": "3′UTR",
      "pooled_all": "pooled-all", "all": "ALL\n(bundled)"}
BASE = "s3://oa-bolinas/analysis/issue314/iter3_transfer"
OUT = Path("plots/output/issue314_transfer")
VLIM = 0.25


def delta_matrix(df: pl.DataFrame):
    lut = {(r["train"], r["eval"]): r["auprc"] for r in df.iter_rows(named=True)}
    llr = {ev: lut.get(("LLR", ev), np.nan) for ev in COLS}
    D = np.full((len(ROWS), len(COLS)), np.nan)
    for i, tr in enumerate(ROWS):
        for j, ev in enumerate(COLS):
            v = lut.get((tr, ev), np.nan)
            if np.isfinite(v) and np.isfinite(llr[ev]):
                D[i, j] = v - llr[ev]
    return D, llr


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 15))
    im = None
    for ax, model in zip(axes.flat, MODELS):
        try:
            df = pl.read_parquet(f"{BASE}/transfer_{model}.parquet")
        except Exception as e:
            ax.set_title(f"{model}\n(missing: {type(e).__name__})")
            ax.axis("off")
            continue
        D, llr = delta_matrix(df)
        im = ax.imshow(D, cmap="RdYlGn", vmin=-VLIM, vmax=VLIM, aspect="auto")
        ax.set_xticks(range(len(COLS)),
                      [f"{SH[s]}\nLLR {llr[s]:.2f}" for s in COLS], rotation=45, ha="right",
                      fontsize=8)
        ax.set_yticks(range(len(ROWS)), [SH.get(r, r) for r in ROWS], fontsize=9)
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if np.isfinite(D[i, j]):
                    ax.text(j, i, f"{D[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(D[i, j]) > 0.16 else "black")
        for j, ev in enumerate(EVAL_ORDER):  # within-subset diagonal (8 subset columns only)
            ax.add_patch(Rectangle((j - 0.5, j - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=1.8))
        ax.axvline(len(EVAL_ORDER) - 0.5, color="0.2", lw=2)   # separate the 'all' column
        ax.axhline(len(ROWS) - 1.5, color="0.2", lw=1.5)        # separate pooled-all
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xlabel("evaluated on  (zero-shot LLR AUPRC below)", fontsize=9)
        ax.set_ylabel("probe trained on", fontsize=10)
    fig.suptitle("Variant-type transfer — probe minus zero-shot LLR (per-chrom AUPRC; "
                 "'all' = global over bundled subsets; nested-tuned C)\n"
                 "green = probe beats LLR · red = loses · black box = within-subset diagonal",
                 fontsize=12, y=0.998)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="probe − LLR  (AUPRC)")
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"figure.{ext}", bbox_inches="tight", dpi=120)
    print(f"wrote {OUT}/figure.svg + figure.png")


if __name__ == "__main__":
    main()
