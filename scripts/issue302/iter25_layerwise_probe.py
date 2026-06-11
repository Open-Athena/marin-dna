"""issue #302 — iteration 25: where in the network does the missense pathogenic-vs-benign
signal live (per-layer probe / probing logit-lens), and does its depth-profile change with
scale? GB flagged per-layer / logit-lens early on; embeddings were extracted at 4 depths
(layer fracs 0.25/0.5/0.75/1.0) but only the last layer has been used (iter9-13).

Per model, per layer: signed-`delta` probe (chromosome-CV, PCA-64) missense AUPRC. If the
across-scale degradation is a *final-representation/readout* effect, a mid-network layer may
hold a strong signal that the last layer / zero-shot LLR doesn't express. Reads cached
embeddings + scores from S3. CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter25_layerwise_probe.py
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EMB = "s3://oa-bolinas/analysis/issue302/embeddings"
SCORES = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]
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


def _load(name):
    import s3fs

    with s3fs.S3FileSystem().open(f"{EMB}/{name}.npz") as f:
        d = np.load(io.BytesIO(f.read()))
        return {k: d[k] for k in d.files}


def cv_auprc(x, y, groups, n_pca=64):
    k = min(5, len(np.unique(groups)))
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(x, y, groups):
        nc = min(n_pca, x[tr].shape[1], len(tr) - 1)
        p = make_pipeline(
            StandardScaler(),
            PCA(nc, random_state=0),
            LogisticRegression(max_iter=2000, C=0.5),
        )
        p.fit(x[tr], y[tr])
        oof[te] = p.predict_proba(x[te])[:, 1]
    return average_precision_score(y, oof)


def main() -> None:
    rows = []
    for name, sdir, params in LADDER:
        emb = _load(name)
        fracs = emb["layer_fracs"].tolist()
        keys = pl.read_parquet(f"{EMB}/{name}.keys.parquet").with_columns(
            pl.col("chrom").cast(str)
        )
        y, chrom = keys["label"].to_numpy(), keys["chrom"].to_numpy()
        sc = (
            pl.read_parquet(f"{SCORES}/{sdir}/mendelian_traits.parquet")
            .with_columns(pl.col("chrom").cast(str))
            .filter(pl.col("subset") == "missense_variant")
            .with_columns((-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("llr"))
            .select([*KEY, "llr"])
        )
        llr = average_precision_score(
            y, keys.join(sc, on=KEY, how="left")["llr"].to_numpy()
        )
        for li, fr in enumerate(fracs):
            delta = (emb["alt"][:, li] - emb["ref"][:, li]).astype(np.float32)
            ap = cv_auprc(delta, y, chrom)
            rows.append({"params": params, "frac": float(fr), "auprc": ap, "llr": llr})
        best = max([r for r in rows if r["params"] == params], key=lambda r: r["auprc"])
        print(
            f"{params:>5}M: "
            + " ".join(
                f"L{r['frac']:.2f}={r['auprc']:.3f}"
                for r in rows
                if r["params"] == params
            )
            + f"  | LLR={llr:.3f}  best=L{best['frac']:.2f}"
        )

    res = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/layerwise_probe.parquet")
    c = res.to_pandas()
    fr = sorted(c["frac"].unique())
    pr = sorted(c["params"].unique())

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))
    cmap = plt.get_cmap("viridis")
    for i, fc in enumerate(fr):
        s = c[c.frac == fc].sort_values("params")
        ax[0].plot(
            s["params"],
            s["auprc"],
            "o-",
            color=cmap(i / (len(fr) - 1)),
            lw=2.2,
            label=f"layer {fc:.2f}",
        )
    ll = c.drop_duplicates("params").sort_values("params")
    ax[0].plot(
        ll["params"], ll["llr"], "D--", color="black", lw=2, label="zero-shot LLR"
    )
    ax[0].set_xscale("log")
    ax[0].set_xlabel("params (M, log)")
    ax[0].set_ylabel("missense AUPRC (chromosome CV)")
    ax[0].set_title(
        "Per-layer probe vs scale — does a mid-network layer\nhold signal the last layer / LLR loses?"
    )
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    M = np.array(
        [[c[(c.params == p) & (c.frac == f)]["auprc"].iloc[0] for p in pr] for f in fr]
    )
    im = ax[1].imshow(M, aspect="auto", cmap="viridis", origin="lower")
    ax[1].set_xticks(range(len(pr)))
    ax[1].set_xticklabels([f"{p}M" for p in pr], rotation=45, fontsize=8)
    ax[1].set_yticks(range(len(fr)))
    ax[1].set_yticklabels([f"{f:.2f}" for f in fr])
    ax[1].set_xlabel("params")
    ax[1].set_ylabel("layer frac (depth)")
    for yi in range(len(fr)):
        for xi in range(len(pr)):
            ax[1].text(
                xi,
                yi,
                f"{M[yi, xi]:.2f}",
                ha="center",
                va="center",
                color="white" if M[yi, xi] < 0.45 else "black",
                fontsize=7,
            )
    ax[1].set_title("probe AUPRC by depth × scale")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.suptitle("Layer-wise missense probe across the ladder", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "layerwise_probe.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "layerwise_probe.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'layerwise_probe'}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/layerwise_probe.parquet",
        str(OUT / "layerwise_probe.png"),
        str(OUT / "layerwise_probe.svg"),
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
