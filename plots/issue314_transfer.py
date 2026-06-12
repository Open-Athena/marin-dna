"""issue #314 iter3 — variant-type transfer heatmaps (probe − zero-shot LLR).

One heatmap per model. Rows = the subset a probe was *trained* on (plus a ``pooled_all`` row);
columns = the subset it is *evaluated* on. Colour = **per-chromosome-weighted AUPRC of the
probe minus the zero-shot LLR AUPRC on that eval subset** — i.e. how much the probe beats
(blue) or loses to (red) the baseline. This is more readable than absolute AUPRC, which is
dominated by subset difficulty. The diagonal (boxed) is within-subset; off-diagonal is
cross-consequence transfer; the per-eval LLR baseline is printed under each column label.

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
EVAL_ORDER = ["missense_variant", "distal", "tss_proximal", "splicing",
              "synonymous_variant", "5_prime_UTR_variant",
              "non_coding_transcript_exon_variant", "3_prime_UTR_variant"]
SH = {"missense_variant": "missense", "distal": "distal", "tss_proximal": "tss",
      "splicing": "splicing", "synonymous_variant": "synon",
      "5_prime_UTR_variant": "5′UTR", "non_coding_transcript_exon_variant": "ncRNA",
      "3_prime_UTR_variant": "3′UTR", "pooled_all": "pooled-all"}
BASE = "s3://oa-bolinas/analysis/issue314/iter3_transfer"
OUT = Path("plots/output/issue314_transfer")
VLIM = 0.25  # symmetric colour limit on the probe−LLR delta


def delta_matrix(df: pl.DataFrame):
    """Probe AUPRC minus the per-eval-subset zero-shot LLR AUPRC; rows = train-on subsets
    (+ pooled_all), cols = eval-on subsets. Also returns the LLR baseline per eval subset."""
    lut = {(r["train"], r["eval"]): r["auprc"] for r in df.iter_rows(named=True)}
    llr = {ev: lut.get(("LLR", ev), np.nan) for ev in EVAL_ORDER}
    rows = EVAL_ORDER + ["pooled_all"]
    D = np.full((len(rows), len(EVAL_ORDER)), np.nan)
    for i, tr in enumerate(rows):
        for j, ev in enumerate(EVAL_ORDER):
            v = lut.get((tr, ev), np.nan)
            if np.isfinite(v) and np.isfinite(llr[ev]):
                D[i, j] = v - llr[ev]
    return D, rows, llr


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
        D, rows, llr = delta_matrix(df)
        im = ax.imshow(D, cmap="RdBu", vmin=-VLIM, vmax=VLIM, aspect="auto")
        ax.set_xticks(range(len(EVAL_ORDER)),
                      [f"{SH[s]}\nLLR {llr[s]:.2f}" for s in EVAL_ORDER],
                      rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(rows)), [SH.get(r, r) for r in rows], fontsize=9)
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if np.isfinite(D[i, j]):
                    ax.text(j, i, f"{D[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(D[i, j]) > 0.14 else "black")
        for j, ev in enumerate(EVAL_ORDER):  # box the within-subset diagonal
            ax.add_patch(Rectangle((j - 0.5, j - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=1.8))
        ax.axhline(len(rows) - 1.5, color="0.3", lw=1.5)  # separate pooled-all
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xlabel("evaluated on  (with zero-shot LLR AUPRC)", fontsize=9)
        ax.set_ylabel("probe trained on", fontsize=10)
    fig.suptitle("Variant-type transfer — probe minus zero-shot LLR (per-chrom AUPRC)\n"
                 "blue = probe beats LLR · red = probe loses · black box = within-subset",
                 fontsize=13, y=0.998)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="probe − LLR  (per-chrom AUPRC)")
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"figure.{ext}", bbox_inches="tight", dpi=120)
    print(f"wrote {OUT}/figure.svg + figure.png")


if __name__ == "__main__":
    main()
