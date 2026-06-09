"""issue #302 — iteration 12: is the across-scale embedding-probe rise just "more dimensions"?

The iter9/10 finding: a linear probe on the frozen last-layer `delta` embedding separates
pathogenic-vs-benign missense *better* with scale (AUPRC 0.31->0.55, 46M->4B, chromosome-CV)
while the zero-shot LLR degrades. Confound (raised by GB): bigger models have wider hidden
states (D: 640->2944). Even though the probe already PCA-256s every model to a fixed 256-dim
input, a *wider* representation's top PCs could be richer purely from width (Cover's theorem:
more raw directions -> easier linear separation), independent of any learning.

CONTROL 1 (this script; CPU; cached embeddings; no GPU): squeeze every model to the SAME
small PCA budget k and sweep k in {8,16,32,64,128,256}. At k=8 every model feeds the logistic
exactly 8 features -- identical classifier capacity -- so if 4B still beats 46M at k=8, the
rise can't be "more dimensions to fit": it's that the few top directions are more *informative*
(a representation-quality statement, not a width artifact). PCA components are nested, so we
fit one PCA(256) per CV fold and slice the top-k -- the k=8 result is exactly PCA(8).
(CONTROL 2, the random-init baseline, is a separate GPU job.)

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter12_dim_control.py
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EMB_S3 = "s3://oa-bolinas/analysis/issue302/embeddings"
SCORES_S3 = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]
K_GRID = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
K_SHOW = [
    8,
    32,
    256,
]  # representative budgets for the vs-params panel (smallest, best, largest)
LADDER = [
    ("scaling-v0.5-46M", "scaling-v0.5-h640-p46M-step-215573", 46),
    ("scaling-v0.5-76M", "scaling-v0.5-h768-p76M-step-215573", 76),
    ("scaling-v0.5-128M", "scaling-v0.5-h896-p128M-step-215573", 128),
    ("scaling-v0.5-255M", "scaling-v0.5-h1152-p255M-step-215573", 255),
    ("scaling-v0.5-476M", "scaling-v0.5-h1408-p476M-step-215573", 476),
    ("scaling-v0.5-1B", "scaling-v0.5-h1920-p1B-step-215573", 1120),
    ("scaling-v0.5-2B", "scaling-v0.5-h2432-p2B-step-215573", 2270),
    ("scaling-v0.5-4B", "scaling-v0.5-h2944-p4B-step-215573", 4020),
]


def _load_npz(name: str):
    import s3fs

    with s3fs.S3FileSystem().open(f"{EMB_S3}/{name}.npz") as f:
        d = np.load(io.BytesIO(f.read()))
        return {k: d[k] for k in d.files}


def cv_auprc_sweep(x, y, groups, ks, n_splits=5) -> dict[int, float]:
    """Grouped-CV AUPRC sweeping PCA budget. Fit PCA(max k) once per fold, slice top-k
    (PCA components are nested, so the top-k of PCA(256) == PCA(k))."""
    kf = min(n_splits, len(np.unique(groups)))
    oof = {k: np.zeros(len(y)) for k in ks}
    for tr, te in GroupKFold(n_splits=kf).split(x, y, groups):
        sc = StandardScaler().fit(x[tr])
        xtr, xte = sc.transform(x[tr]), sc.transform(x[te])
        ncmax = min(max(ks), xtr.shape[1], len(tr) - 1)
        pca = PCA(n_components=ncmax, random_state=0).fit(xtr)
        ztr, zte = pca.transform(xtr), pca.transform(xte)
        for k in ks:
            kk = min(k, ncmax)
            lr = LogisticRegression(max_iter=2000, C=0.5).fit(ztr[:, :kk], y[tr])
            oof[k][te] = lr.predict_proba(zte[:, :kk])[:, 1]
    return {k: float(average_precision_score(y, oof[k])) for k in ks}


def main() -> None:
    rows = []
    for name, scores_dir, params in LADDER:
        emb = _load_npz(name)
        delta_last = (emb["alt"][:, -1] - emb["ref"][:, -1]).astype(np.float32)
        keys = pl.read_parquet(f"{EMB_S3}/{name}.keys.parquet").with_columns(
            pl.col("chrom").cast(str)
        )
        y, chrom = keys["label"].to_numpy(), keys["chrom"].to_numpy()
        d_full = delta_last.shape[1]
        sc = (
            pl.read_parquet(f"{SCORES_S3}/{scores_dir}/mendelian_traits.parquet")
            .with_columns(pl.col("chrom").cast(str))
            .filter(pl.col("subset") == "missense_variant")
            .with_columns((-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("llr"))
            .select([*KEY, "llr"])
        )
        llr = average_precision_score(
            y, keys.join(sc, on=KEY, how="left")["llr"].to_numpy()
        )
        aps = cv_auprc_sweep(delta_last, y, chrom, K_GRID)
        for k in K_GRID:
            rows.append(
                {"params": params, "D": d_full, "k": k, "auprc": aps[k], "llr": llr}
            )
        print(
            f"{params:>5}M D={d_full:>4} LLR={llr:.3f} | "
            + "  ".join(f"k{k}={aps[k]:.3f}" for k in K_GRID)
        )

    res = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/dim_control.parquet")
    c = res.to_pandas()

    # effective dimensionality: PCA budget that maximizes AUPRC, per model.
    best = c.loc[c.groupby("params")["auprc"].idxmax()].sort_values("params")
    print(
        "\noptimal PCA budget per size (effective dimensionality of the missense signal):"
    )
    for _, r in best.iterrows():
        print(f"  {int(r.params):>5}M  best k={int(r.k):>3}  AUPRC={r.auprc:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    # Panel A — AUPRC vs scale at a few representative budgets; the rise survives at all.
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(K_SHOW):
        sub = c[c.k == k].sort_values("params")
        ax[0].plot(
            sub["params"],
            sub["auprc"],
            "o-",
            color=cmap(i / (len(K_SHOW) - 1)),
            lw=2,
            label=f"PCA-{k}",
        )
    llr_line = c.drop_duplicates("params").sort_values("params")
    ax[0].plot(
        llr_line["params"],
        llr_line["llr"],
        "D--",
        color="black",
        lw=2,
        label="zero-shot LLR",
    )
    ax[0].set_xscale("log")
    ax[0].set_xlabel("params (M, log)")
    ax[0].set_ylabel("missense AUPRC (chromosome CV)")
    ax[0].set_title(
        "Probe rises with scale at every fixed PCA budget\n(even PCA-8, identical 8-feature capacity)"
    )
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    # Panel B — AUPRC vs PCA budget, one line per model: the signal is low-dimensional
    # (peaks ~32-64 PCs) at every scale, and bigger models dominate at every budget.
    for _, (name, _sd, params) in enumerate(LADDER):
        sub = c[c.params == params].sort_values("k")
        frac = (np.log10(params) - np.log10(46)) / (np.log10(4020) - np.log10(46))
        ax[1].plot(
            sub["k"],
            sub["auprc"],
            "o-",
            color=cmap(frac),
            lw=1.8,
            ms=4,
            label=f"{params}M",
        )
    ax[1].set_xscale("log")
    ax[1].set_xlabel("PCA budget k (log)")
    ax[1].set_ylabel("missense AUPRC (chromosome CV)")
    ax[1].set_title(
        "The missense signal is low-dimensional (peaks ~32-64 PCs)\nat every scale — width is not what improves"
    )
    ax[1].legend(fontsize=7, ncol=2, title="model")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "dim_control.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "dim_control.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {OUT / 'dim_control'}.{{png,svg}}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/dim_control.parquet",
        str(OUT / "dim_control.png"),
        str(OUT / "dim_control.svg"),
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
