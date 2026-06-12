"""issue #314 — does a single probe trained on ALL subsets (pooled-all) match the per-subset
probe (the diagonal)? Tests the "one global probe suffices" claim quantitatively rather than
by eyeballing the transfer heatmap. Reads the iter3 transfer parquets.

Left: scatter of diagonal vs pooled-all per-chrom AUPRC, one point per (model, subset), with
the y=x line — points on the line ⇒ equal. Right: the pooled−diagonal gap per subset across
the four models — shows which subsets pooling systematically helps (small subsets borrow
strength) or hurts (high-signal subsets get diluted).

Run:  uv run python plots/issue314_diag_vs_pooled.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

MODELS = {"exp135-1B-m5.1": "exp135", "scaling-v0.5-1B": "scaling",
          "exp166-v0.1-p1B": "exp166-1B", "exp166-v0.1-p4B": "exp166-4B"}
MCOLOR = dict(zip(MODELS.values(), ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))
ORD = ["missense_variant", "distal", "tss_proximal", "splicing", "synonymous_variant",
       "5_prime_UTR_variant", "non_coding_transcript_exon_variant", "3_prime_UTR_variant"]
SH = {"missense_variant": "missense", "distal": "distal", "tss_proximal": "tss",
      "splicing": "splicing", "synonymous_variant": "synon", "5_prime_UTR_variant": "5′UTR",
      "non_coding_transcript_exon_variant": "ncRNA", "3_prime_UTR_variant": "3′UTR"}
BASE = "s3://oa-bolinas/analysis/issue314/iter3_transfer"
OUT = Path("plots/output/issue314_diag_vs_pooled")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # (model, subset) -> (diag, pool)
    rows = []
    for mid, ms in MODELS.items():
        lut = {(r["train"], r["eval"]): r["auprc"]
               for r in pl.read_parquet(f"{BASE}/transfer_{mid}.parquet").iter_rows(named=True)}
        for s in ORD:
            rows.append((ms, s, lut[(s, s)], lut[("pooled_all", s)]))

    fig, (axS, axB) = plt.subplots(1, 2, figsize=(15, 6.5))

    # --- scatter: diagonal vs pooled-all ---
    lo, hi = 0.15, 0.70
    axS.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="y = x (equal)")
    for ms, s, d, p in rows:
        axS.scatter(d, p, color=MCOLOR[ms], s=55, alpha=0.85, edgecolor="white", lw=0.5)
        if abs(p - d) > 0.07:  # annotate the meaningful deviations
            axS.annotate(f"{ms}/{SH[s]}", (d, p), fontsize=7, alpha=0.8,
                         xytext=(3, 3), textcoords="offset points")
    axS.set_xlim(lo, hi); axS.set_ylim(lo, hi)
    axS.set_xlabel("per-subset probe (diagonal: train on the subset)  — per-chrom AUPRC")
    axS.set_ylabel("global probe (pooled-all: train on all subsets)")
    gaps = np.array([p - d for _, _, d, p in rows])
    axS.set_title(f"Diagonal vs pooled-all  (mean gap {gaps.mean():+.3f}, "
                  f"mean|gap| {np.abs(gaps).mean():.3f}, max|gap| {np.abs(gaps).max():.3f})")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=m)
               for m, c in MCOLOR.items()]
    axS.legend(handles=handles + [plt.Line2D([], [], ls="--", color="k", label="y=x")],
               fontsize=8, loc="upper left")

    # --- per-subset gap (pool − diag) across models ---
    x = np.arange(len(ORD))
    axB.axhline(0, color="0.4", lw=1)
    for mi, (mid, ms) in enumerate(MODELS.items()):
        lut = {(r["train"], r["eval"]): r["auprc"]
               for r in pl.read_parquet(f"{BASE}/transfer_{mid}.parquet").iter_rows(named=True)}
        g = [lut[("pooled_all", s)] - lut[(s, s)] for s in ORD]
        axB.bar(x + (mi - 1.5) * 0.2, g, width=0.2, color=MCOLOR[ms], label=ms)
    axB.set_xticks(x, [SH[s] for s in ORD], rotation=45, ha="right")
    axB.set_ylabel("pooled-all − diagonal  (per-chrom AUPRC)")
    axB.set_title("Pooling helps small subsets (+), hurts high-signal subsets (−)")
    axB.legend(fontsize=8)

    fig.suptitle("Does one global probe (pooled-all) match the per-subset probes? "
                 "— Δ≈0 on average, but real per-subset structure", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"figure.{ext}", bbox_inches="tight", dpi=120)
    print(f"wrote {OUT}/figure.svg + figure.png")


if __name__ == "__main__":
    main()
