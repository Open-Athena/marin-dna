"""issue #314 — is variant-type transfer actually "broad"? Tests whether a probe trained on
a *different* consequence (off-diagonal) matches the native per-subset probe (diagonal),
quantitatively rather than by eyeballing the delta-vs-LLR heatmap (whose column colour tracks
LLR weakness, not the diagonal-vs-off-diagonal gap). Reads the iter3 transfer parquets.

Left: diagonal vs the *mean* off-diagonal per-chrom AUPRC, one point per (model, subset), with
the y=x line — points below the line ⇒ training on the wrong subset costs. Right: the transfer
gap (diagonal − mean off-diagonal) per subset across the four models.

Run:  uv run python plots/issue314_transfer_gap.py
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
OUT = Path("plots/output/issue314_transfer_gap")


def luts() -> dict:
    return {ms: {(r["train"], r["eval"]): r["auprc"]
                 for r in pl.read_parquet(f"{BASE}/transfer_{mid}.parquet").iter_rows(named=True)}
            for mid, ms in MODELS.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    L = luts()
    fig, (axS, axB) = plt.subplots(1, 2, figsize=(15, 6.5))

    lo, hi = 0.15, 0.70
    axS.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
    allgap = []
    for ms, lut in L.items():
        for s in ORD:
            diag = lut[(s, s)]
            offm = float(np.mean([lut[(a, s)] for a in ORD if a != s]))
            allgap.append(diag - offm)
            axS.scatter(diag, offm, color=MCOLOR[ms], s=55, alpha=0.85, edgecolor="white", lw=0.5)
            if diag - offm > 0.13:
                axS.annotate(f"{ms}/{SH[s]}", (diag, offm), fontsize=7, alpha=0.8,
                             xytext=(3, -8), textcoords="offset points")
    axS.set_xlim(lo, hi); axS.set_ylim(lo, hi)
    axS.set_xlabel("native probe (diagonal: train on the eval subset)  — per-chrom AUPRC")
    axS.set_ylabel("transfer (mean over probes trained on OTHER subsets)")
    g = np.array(allgap)
    axS.set_title(f"Native vs transferred  (mean gap {g.mean():+.3f}, median {np.median(g):+.3f}, "
                  f"90th pct {np.percentile(g, 90):.3f})")
    axS.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=m)
                        for m, c in MCOLOR.items()], fontsize=8, loc="upper left")

    x = np.arange(len(ORD))
    axB.axhline(0, color="0.4", lw=1)
    for mi, (ms, lut) in enumerate(L.items()):
        gaps = [lut[(s, s)] - float(np.mean([lut[(a, s)] for a in ORD if a != s])) for s in ORD]
        axB.bar(x + (mi - 1.5) * 0.2, gaps, width=0.2, color=MCOLOR[ms], label=ms)
    axB.set_xticks(x, [SH[s] for s in ORD], rotation=45, ha="right")
    axB.set_ylabel("diagonal − mean off-diagonal  (per-chrom AUPRC)")
    axB.set_title("Transfer gap: large for high-signal subsets (missense/splicing), ≈0 for small ones")
    axB.legend(fontsize=8)

    fig.suptitle("Is variant-type transfer 'broad'? — No: training on the wrong consequence costs "
                 "~0.09 AUPRC on average, much more for high-signal subsets", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"figure.{ext}", bbox_inches="tight", dpi=120)
    print(f"wrote {OUT}/figure.svg + figure.png")


if __name__ == "__main__":
    main()
