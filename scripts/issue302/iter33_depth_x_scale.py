"""issue #302 — iteration 33: does the missense scale-degradation concentrate on the
VERTEBRATE-TOLERANT benigns — the class GPN-Star's vertebrate MSA distinguishes?

iter30/21: the confident FPs are mammal-conserved but vertebrate-tolerant (phyloP_100v low) —
GPN-Star reads that depth, our single-seq gLM can't. Direct test across the ladder: split the
mammal-conserved (phyloP_241m>=2) missense BENIGNS into vertebrate-tolerant (phyloP_100v<4) vs
vertebrate-conserved (phyloP_100v>=4), and track AUROC(pathogenic vs each benign subset) across
scale. Prediction: the vertebrate-TOLERANT subset is increasingly over-called with scale (AUROC
drops) — the model leans harder on mammal-conservation and misses the deeper-tolerance signal —
while the truly-deep subset stays hard at all scales and the not-conserved subset stays easy.

Score = minus_llr_avg. CPU; reads S3 (scores + phyloP tracks), writes fig to S3.
Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter33_depth_x_scale.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

SCORES = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
CB = "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]
LADDER = [
    ("scaling-v0.5-h640-p46M-step-215573", 46),
    ("scaling-v0.5-h768-p76M-step-215573", 76),
    ("scaling-v0.5-h896-p128M-step-215573", 128),
    ("scaling-v0.5-h1152-p255M-step-215573", 255),
    ("scaling-v0.5-h1408-p476M-step-215573", 476),
    ("scaling-v0.5-h1920-p1B-step-215573", 1120),
    ("scaling-v0.5-h2432-p2B-step-215573", 2270),
    ("scaling-v0.5-h2944-p4B-step-215573", 4020),
]


def _track(name: str) -> pl.DataFrame:
    return (
        pl.read_parquet(f"{CB}/{name}_train.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .unique(subset=KEY)
        .select([*KEY, pl.col("score").alias(name)])
    )


def main() -> None:
    p241 = _track("phyloP_241m")
    p100 = _track("phyloP_100v")
    rows = []
    for sdir, params in LADDER:
        m = (
            pl.read_parquet(f"{SCORES}/{sdir}/mendelian_traits.parquet")
            .filter(pl.col("subset") == "missense_variant")
            .with_columns(
                pl.col("chrom").cast(str),
                (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"),
            )
            .join(p241, on=KEY, how="left")
            .join(p100, on=KEY, how="left")
            .drop_nulls(["phyloP_241m", "phyloP_100v"])
        )
        pos = m.filter(pl.col("label") == 1)["mll"].to_numpy()
        mamcons = (pl.col("label") == 0) & (pl.col("phyloP_241m") >= 2)
        subsets = {
            "vert-tolerant\n(mammal-cons, phyloP_100v<4)": m.filter(
                mamcons & (pl.col("phyloP_100v") < 4)
            )["mll"].to_numpy(),
            "vert-conserved\n(phyloP_100v≥4)": m.filter(
                mamcons & (pl.col("phyloP_100v") >= 4)
            )["mll"].to_numpy(),
            "not-conserved\n(phyloP_241m<2)": m.filter(
                (pl.col("label") == 0) & (pl.col("phyloP_241m") < 2)
            )["mll"].to_numpy(),
        }
        rec = {"params": params}
        for name, neg in subsets.items():
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            rec[name] = float(roc_auc_score(y, np.r_[pos, neg]))
            rec[f"n::{name}"] = len(neg)
        rows.append(rec)
        print(
            f"{params:>5}M | "
            + "  ".join(
                f"{k.splitlines()[0]}={rec[k]:.3f}(n={rec['n::' + k]})" for k in subsets
            )
        )

    res = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/depth_x_scale.parquet")
    c = res.to_pandas()
    sub_names = [k for k in rows[0] if not k.startswith("n::") and k != "params"]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = {
        "vert-tolerant\n(mammal-cons, phyloP_100v<4)": "tab:red",
        "vert-conserved\n(phyloP_100v≥4)": "tab:blue",
        "not-conserved\n(phyloP_241m<2)": "tab:green",
    }
    for name in sub_names:
        n0 = rows[-1][f"n::{name}"]
        ax.plot(
            c["params"],
            c[name],
            "o-",
            lw=2.5,
            color=colors[name],
            label=f"path. vs {name.splitlines()[0]} (n≈{n0})",
        )
    ax.axvline(128, ls=":", color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("params (M, log)")
    ax.set_ylabel("AUROC (pathogenic vs benign subset)")
    ax.set_title(
        "Scale degrades specifically on the VERTEBRATE-TOLERANT benigns\n(mammal-conserved but vertebrate-variable — GPN-Star's catchable class)"
    )
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    drops = "peak→4B ΔAUROC:  " + "  ".join(
        f"{name.splitlines()[0]} {c[name].max() - c[name].iloc[-1]:+.3f}"
        for name in sub_names
    )
    ax.text(
        0.5,
        -0.16,
        drops,
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(OUT / "depth_x_scale.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "depth_x_scale.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'depth_x_scale'}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (str(OUT / "depth_x_scale.png"), str(OUT / "depth_x_scale.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
