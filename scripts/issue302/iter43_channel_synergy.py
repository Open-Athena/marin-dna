"""issue #302 — iteration 43: the two scale-growing plausibility channels (position conservation,
iter41/39; amino-acid severity, iter42) interact SUPER-ADDITIVELY — the scale recruitment of benign
missense concentrates exactly in the conserved x radical corner (the FP signature: a radical AA
substitution at a conserved position). Visualize the recruitment Δ = z(mll@4B) − z(mll@128M) as a
2D map over phyloP_241m (conservation) x Grantham (AA severity), and quantify the interaction.

Grantham from cached aa_ref/aa_alt (iter5 formula). Titles filled from data. CPU; reads S3 + cache.
Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter43_channel_synergy.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

SC = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
CB = "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]
GP = {
    "A": (0, 8.1, 31),
    "R": (0.65, 10.5, 124),
    "N": (1.33, 11.6, 56),
    "D": (1.38, 13, 54),
    "C": (2.75, 5.5, 55),
    "Q": (0.89, 10.5, 85),
    "E": (0.92, 12.3, 83),
    "G": (0.74, 9, 3),
    "H": (0.58, 10.4, 96),
    "I": (0, 5.2, 111),
    "L": (0, 4.9, 111),
    "K": (0.33, 11.3, 119),
    "M": (0, 5.7, 105),
    "F": (0, 5, 132),
    "P": (0.39, 8, 32.5),
    "S": (1.42, 9.2, 32),
    "T": (0.71, 8.6, 61),
    "W": (0.13, 5.4, 170),
    "Y": (0.2, 6.2, 136),
    "V": (0, 5.9, 84),
}


def gr(a, b):
    if a not in GP or b not in GP or a == b:
        return 0.0 if a in GP and b in GP else None
    (ca, pa, va), (cb, pb, vb) = GP[a], GP[b]
    return 50.723 * np.sqrt(
        1.833 * (ca - cb) ** 2 + 0.1018 * (pa - pb) ** 2 + 0.000399 * (va - vb) ** 2
    )


def trk(n):
    return (
        pl.read_parquet(f"{CB}/{n}_train.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .unique(subset=KEY)
        .select([*KEY, pl.col("score").alias(n)])
    )


def z(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def load(t):
    return (
        pl.read_parquet(f"{SC}/scaling-v0.5-{t}-step-215573/mendelian_traits.parquet")
        .filter(pl.col("subset") == "missense_variant")
        .with_columns(
            pl.col("chrom").cast(str),
            (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"),
        )
        .select([*KEY, "label", "mll"])
    )


def main() -> None:
    aa = (
        pl.read_parquet("scratch/issue302/myvariant_aa.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .drop_nulls(["aa_ref", "aa_alt"])
    )
    aa = (
        aa.with_columns(
            pl.Series(
                "grantham", [gr(r, a) for r, a in zip(aa["aa_ref"], aa["aa_alt"])]
            )
        )
        .drop_nulls("grantham")
        .select([*KEY, "grantham"])
    )
    a = load("h896-p128M").with_columns(
        ((pl.col("mll") - pl.col("mll").mean()) / pl.col("mll").std()).alias("z128")
    )
    b = (
        load("h2944-p4B")
        .with_columns(
            ((pl.col("mll") - pl.col("mll").mean()) / pl.col("mll").std()).alias("z4b")
        )
        .select([*KEY, "z4b"])
    )
    m = (
        a.join(b, on=KEY, how="inner")
        .join(aa, on=KEY, how="left")
        .join(trk("phyloP_241m"), on=KEY, how="left")
        .filter(pl.col("label") == 0)
        .drop_nulls(["grantham", "phyloP_241m", "z128", "z4b"])
    )
    n = m.height
    d = (m["z4b"] - m["z128"]).to_numpy()
    P = m["phyloP_241m"].to_numpy()
    G = m["grantham"].to_numpy()
    # interaction stats
    Pz, Gz = z(P), z(G)
    X = np.column_stack([np.ones(n), Pz, Gz, Pz * Gz])
    coef, *_ = np.linalg.lstsq(X, z(d), rcond=None)
    res = z(d) - X @ coef
    se = np.sqrt(np.diag(((res @ res) / (n - X.shape[1])) * np.linalg.inv(X.T @ X)))
    ib, iz = coef[3], coef[3] / se[3]
    hi = (P > np.median(P)) & (G > np.median(G))
    print(f"n={n} | interaction phyloP×grantham β={ib:+.3f} z={iz:+.1f}")
    print(
        f"  conserved&radical corner (n={hi.sum()}) recruitment Δ={d[hi].mean():+.3f} vs rest={d[~hi].mean():+.3f}"
    )

    NB = 5
    pe = np.quantile(P, np.linspace(0, 1, NB + 1))
    ge = np.quantile(G, np.linspace(0, 1, NB + 1))
    pi = np.clip(np.digitize(P, pe[1:-1]), 0, NB - 1)
    gi = np.clip(np.digitize(G, ge[1:-1]), 0, NB - 1)
    grid = np.full((NB, NB), np.nan)
    for r in range(NB):
        for cc in range(NB):
            sel = (gi == r) & (pi == cc)
            if sel.sum() >= 10:
                grid[r, cc] = d[sel].mean()

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    vlim = np.nanmax(np.abs(grid))
    im = ax.imshow(
        grid, origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto"
    )
    for r in range(NB):
        for cc in range(NB):
            if not np.isnan(grid[r, cc]):
                ax.text(
                    cc,
                    r,
                    f"{grid[r, cc]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )
    ax.set_xticks(range(NB))
    ax.set_xticklabels([f"Q{i + 1}" for i in range(NB)])
    ax.set_yticks(range(NB))
    ax.set_yticklabels([f"Q{i + 1}" for i in range(NB)])
    ax.set_xlabel("position conservation phyloP_241m  (Q5 = most conserved →)")
    ax.set_ylabel("amino-acid severity Grantham  (Q5 = most radical →)")
    ax.set_title(
        f"Benign missense recruitment Δ (z@4B − z@128M) over the two channels\nover-call concentrates in the conserved×radical corner — interaction β={ib:+.2f} (z={iz:+.1f})\ncorner Δ={d[hi].mean():+.2f} vs rest {d[~hi].mean():+.2f}",
        fontsize=10,
    )
    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        label="mean recruitment Δ (red = over-called more at 4B)",
    )
    fig.tight_layout()
    fig.savefig(OUT / "channel_synergy.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "channel_synergy.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'channel_synergy'}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (str(OUT / "channel_synergy.png"), str(OUT / "channel_synergy.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
