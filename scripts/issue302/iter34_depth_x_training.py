"""issue #302 — iteration 34: is the WITHIN-TRAINING missense degradation the SAME phenomenon
as the across-scale one? (Plan analysis G.) iter33 showed the across-scale degradation localizes
entirely to the vertebrate-tolerant benigns. Here: the #232 0.25B v4_cds specialist — the one
arm whose missense AUPRC rises-then-falls *inside a single model's training* (peaks ~step
3500-4000, drops by 5000) — gets the same depth-split AUROC(pathogenic vs subset) across its 10
checkpoints. Finding: the within-training reversal localizes to the DEEPLY-conserved benigns (vert-conserved
AUROC peaks ~step 3000 then declines), the OPPOSITE depth from the across-scale axis (iter33,
vert-tolerant). Both are conservation-over-reliance degradations but at different depths — and the
0.25B CDS-only specialist differs from the full-genome ladder in data+axis, so the depth-localization
need not match.

Score = minus_llr_avg. CPU; reads S3, writes fig to S3.
Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter34_depth_x_training.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

SCORES = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
CB = "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]
STEPS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999]
SUB = {
    "vert-tolerant": ((pl.col("phyloP_241m") >= 2) & (pl.col("phyloP_100v") < 4)),
    "vert-conserved": ((pl.col("phyloP_241m") >= 2) & (pl.col("phyloP_100v") >= 4)),
    "not-conserved": (pl.col("phyloP_241m") < 2),
}
COLORS = {
    "vert-tolerant": "tab:red",
    "vert-conserved": "tab:blue",
    "not-conserved": "tab:green",
}


def _track(name: str) -> pl.DataFrame:
    return (
        pl.read_parquet(f"{CB}/{name}_train.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .unique(subset=KEY)
        .select([*KEY, pl.col("score").alias(name)])
    )


def main() -> None:
    p241, p100 = _track("phyloP_241m"), _track("phyloP_100v")
    rows = []
    for step in STEPS:
        m = (
            pl.read_parquet(
                f"{SCORES}/exp232-v4_cds-step-{step}/mendelian_traits.parquet"
            )
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
        allneg = m.filter(pl.col("label") == 0)["mll"].to_numpy()
        rec = {
            "step": step,
            "overall_auprc": float(
                average_precision_score(
                    np.r_[np.ones(len(pos)), np.zeros(len(allneg))], np.r_[pos, allneg]
                )
            ),
        }
        for name, expr in SUB.items():
            neg = m.filter((pl.col("label") == 0) & expr)["mll"].to_numpy()
            rec[name] = float(
                roc_auc_score(
                    np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]
                )
            )
        rows.append(rec)
        print(
            f"step {step:>4} | AUPRC={rec['overall_auprc']:.3f} | vert-tol={rec['vert-tolerant']:.3f} vert-cons={rec['vert-conserved']:.3f} not-cons={rec['not-conserved']:.3f}"
        )

    res = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/depth_x_training.parquet")
    c = res.to_pandas()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(c["step"], c["overall_auprc"], "o-", color="black", lw=2.5)
    pk = c.loc[c["overall_auprc"].idxmax(), "step"]
    ax[0].axvline(pk, ls=":", color="gray")
    ax[0].set_xlabel("training step (0.25B v4_cds)")
    ax[0].set_ylabel("overall missense AUPRC")
    ax[0].set_title(f"The within-training reversal\n(peak ~step {int(pk)} → falls)")
    ax[0].grid(alpha=0.3)
    for name in SUB:
        ax[1].plot(
            c["step"],
            c[name],
            "o-",
            color=COLORS[name],
            lw=2.5,
            label=f"path. vs {name}",
        )
    ax[1].axvline(pk, ls=":", color="gray")
    ax[1].set_xlabel("training step (0.25B v4_cds)")
    ax[1].set_ylabel("AUROC (pathogenic vs benign subset)")
    ax[1].set_title(
        "DIFFERENT depth from scale: only the deeply-\nCONSERVED subset reverses; vert-tolerant keeps improving"
    )
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    drops = "peak→end Δ:  " + "  ".join(
        f"{n} {c[n].max() - c[n].iloc[-1]:+.3f}" for n in SUB
    )
    ax[1].text(
        0.5,
        -0.16,
        drops,
        transform=ax[1].transAxes,
        ha="center",
        fontsize=8,
        color="dimgray",
    )
    fig.suptitle(
        "Within-training (0.25B CDS specialist): the reversal localizes to the DEEPLY-conserved benigns — the OPPOSITE depth from across-scale (iter33)",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "depth_x_training.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "depth_x_training.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'depth_x_training'}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (str(OUT / "depth_x_training.png"), str(OUT / "depth_x_training.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
