"""issue #314 — does adding the local-context (ref) term to the signed effect help on
mendelian? Per-subset gain of concat(ref, delta) over plain signed delta (per-chrom AUPRC,
nested-LOCO), across the four models. Reads the iter2_nested_refdelta parquets.

Run:  uv run python plots/issue314_refcontext.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

MODELS = {"exp135-1B-m5.1": "exp135", "scaling-v0.5-1B": "scaling",
          "exp166-v0.1-p1B": "exp166-1B", "exp166-v0.1-p4B": "exp166-4B"}
MCOLOR = dict(zip(MODELS.values(), ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))
# ordered by mean gain (strong-effect → weak-effect)
ORD = ["synonymous_variant", "distal", "missense_variant", "splicing",
       "3_prime_UTR_variant", "5_prime_UTR_variant", "non_coding_transcript_exon_variant",
       "tss_proximal"]
SH = {"missense_variant": "missense", "distal": "distal", "tss_proximal": "tss",
      "splicing": "splicing", "synonymous_variant": "synon", "5_prime_UTR_variant": "5′UTR",
      "non_coding_transcript_exon_variant": "ncRNA", "3_prime_UTR_variant": "3′UTR"}
BASE = "s3://oa-bolinas/analysis/issue314/iter2_nested_refdelta"
OUT = Path("plots/output/issue314_refcontext")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for mid, ms in MODELS.items():
        df = pl.read_parquet(f"{BASE}/nested_{mid}.parquet")
        d = {(r["subset"], r["rep"]): r["probe_perchrom"] for r in df.iter_rows(named=True)}
        data[ms] = {s: d[(s, "entire_window/concat_ref_delta")] - d[(s, "entire_window/delta")]
                    for s in ORD}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(ORD))
    ax.axhline(0, color="0.4", lw=1)
    for mi, (ms, gaps) in enumerate(data.items()):
        ax.bar(x + (mi - 1.5) * 0.2, [gaps[s] for s in ORD], width=0.2,
               color=MCOLOR[ms], label=ms)
    allg = np.array([g for gd in data.values() for g in gd.values()])
    ax.set_xticks(x, [SH[s] for s in ORD], fontsize=10)
    ax.set_ylabel("concat(ref, delta) − delta   (per-chrom AUPRC)")
    ax.set_title(f"Adding the ref-context term to signed delta, on mendelian "
                 f"(mean +{allg.mean():.3f}, helps {(allg > 0).mean():.0%} of cells)\n"
                 f"helps most where the effect-direction is weak (synon, distal); "
                 f"only consistent loss is tss")
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"figure.{ext}", bbox_inches="tight", dpi=120)
    print(f"wrote {OUT}/figure.svg + figure.png")


if __name__ == "__main__":
    main()
