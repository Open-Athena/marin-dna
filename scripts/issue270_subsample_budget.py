"""issue #270: total variants to score vs per-5-mer subsampling cap.

A calibration cell is (5-mer, alt); all 3 alts of a 5-mer share its neutral
sites. Capping at `cap` sites per 5-mer therefore scores
`min(n_sites(5mer), cap) * 3` variants for that 5-mer, and the per-checkpoint
total is `3 * sum_over_5mers min(n_sites, cap)`. With no cap that's
`3 * 5.94M = 17.8M`.

This script computes the per-5-mer neutral-site distribution (central 5-mer for
every site, same extraction + centering assert as issue270_select_variants.py)
and tabulates the total-variants budget across a grid of caps, annotated with
the implied per-bin SE = s/sqrt(cap) at representative within-cell SDs (the
pilot found s in ~[0.72, 1.42] nats).

Run:
    uv run python scripts/issue270_subsample_budget.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyfaidx import Fasta

NEUTRAL_PARQUET = "scratch/issue270/neutral_sites.parquet"
PLAIN_FASTA = "scratch/issue270/GRCh38.fa"
COUNTS_CACHE = "scratch/issue270/pentanuc_counts.parquet"
OUT_FIG = "scratch/issue270/subsample_budget"
N_ALTS = 3
OFFSETS = np.arange(-2, 3)
CENTER = 2
CAPS = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000]
# Representative within-cell SDs from the pilot (median-ish and worst-case).
S_REPR = {"s=1.0": 1.0, "s=1.42 (worst)": 1.42}


def pentanuc_counts(neutral_parquet: str, fasta_path: str) -> pd.Series:
    """Per-5-mer neutral-site count (index=5-mer). Asserts center == ref."""
    df = pd.read_parquet(neutral_parquet).reset_index(drop=True)
    fasta = Fasta(fasta_path, as_raw=True)
    kmers = np.empty(len(df), dtype="S5")
    for chrom, sub in df.groupby("chrom", sort=False):
        arr = np.frombuffer(fasta[chrom][:].upper().encode("ascii"), dtype=np.uint8)
        center0 = sub["pos"].to_numpy() - 1
        assert center0.min() - 2 >= 0 and center0.max() + 2 < len(arr), chrom
        km = arr[center0[:, None] + OFFSETS[None, :]]
        ref_u8 = np.frombuffer("".join(sub["ref"]).encode("ascii"), dtype=np.uint8)
        assert (km[:, CENTER] == ref_u8).all(), f"{chrom}: center != ref"
        kmers[sub.index.to_numpy()] = np.ascontiguousarray(km).view("S5").ravel()
    s = pd.Index(kmers).str.decode("ascii")
    s = s[s.str.fullmatch("[ACGT]{5}")]
    return s.value_counts()


def main() -> None:
    import os

    if os.path.exists(COUNTS_CACHE):
        counts = pd.read_parquet(COUNTS_CACHE)["n_sites"]
        print(f"loaded cached per-5-mer counts ({len(counts)} 5-mers)")
    else:
        counts = pentanuc_counts(NEUTRAL_PARQUET, PLAIN_FASTA)
        counts.rename("n_sites").rename_axis("pentanuc").reset_index().to_parquet(
            COUNTS_CACHE, index=False
        )
        print(f"computed per-5-mer counts ({len(counts)} 5-mers) -> {COUNTS_CACHE}")

    n = counts.to_numpy()
    total_sites = int(n.sum())
    print(f"\nneutral sites: {total_sites:,} across {len(n)} distinct 5-mers")
    print(
        f"per-5-mer count: min={n.min():,} median={int(np.median(n)):,} "
        f"max={n.max():,}  ({(n < 1000).sum()} 5-mers have < 1000 sites)"
    )
    full_variants = N_ALTS * total_sites
    print(
        f"\nfull (no cap): {total_sites:,} sites -> {full_variants:,} variants "
        f"(x{N_ALTS} alts) per checkpoint\n"
    )

    rows = []
    for cap in CAPS:
        sites = int(np.minimum(n, cap).sum())
        variants = N_ALTS * sites
        n_capped = int((n > cap).sum())  # 5-mers hitting the cap (subsampled)
        row = {
            "cap (sites/5-mer)": cap,
            "sites scored": sites,
            "variants scored": variants,
            "% of full": 100 * variants / full_variants,
            "5-mers subsampled": f"{n_capped}/{len(n)}",
        }
        for label, s in S_REPR.items():
            row[f"SE@cap ({label})"] = s / np.sqrt(cap)
        rows.append(row)
    rows.append(
        {
            "cap (sites/5-mer)": "none",
            "sites scored": total_sites,
            "variants scored": full_variants,
            "% of full": 100.0,
            "5-mers subsampled": f"0/{len(n)}",
            **{f"SE@cap ({k})": v / np.sqrt(np.median(n)) for k, v in S_REPR.items()},
        }
    )
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 20)

    def fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:,.3f}" if v < 10 else f"{v:,.1f}"
        return f"{v:,}" if isinstance(v, int) else str(v)

    print(table.to_string(index=False, formatters={c: fmt for c in table.columns}))

    # ---- figure: total variants vs cap ----
    caps = np.array(CAPS)
    variants = np.array([N_ALTS * int(np.minimum(n, c).sum()) for c in CAPS])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(caps, variants / 1e6, "o-", color="C0", label="variants to score")
    ax.axhline(
        full_variants / 1e6,
        color="gray",
        ls="--",
        lw=1,
        label=f"full = {full_variants / 1e6:.1f}M",
    )
    for c, v in zip(caps, variants):
        ax.annotate(
            f"{v / 1e6:.1f}M",
            (c, v / 1e6),
            fontsize=7,
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
        )
    ax.set(
        xscale="log",
        xlabel="cap: neutral sites per 5-mer",
        ylabel="total variants to score per checkpoint (millions)",
        title="cLLR calibration budget vs per-5-mer subsampling cap",
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}.svg")
    fig.savefig(f"{OUT_FIG}.png", dpi=150)
    print(f"\nwrote {OUT_FIG}.svg / .png")


if __name__ == "__main__":
    main()
