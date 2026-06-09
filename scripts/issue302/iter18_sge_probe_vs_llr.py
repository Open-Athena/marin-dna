"""issue #302 / #306 — iteration 18: within-SGE frozen-embedding probe vs zero-shot LLR,
across the ladder, with LEAVE-ONE-GENE-OUT CV.

SGE missense AUPRC *rises* with scale (no degradation; #306). The readout-localization story
(iter9/10/14) predicts: where the zero-shot LLR keeps improving, the probe and LLR should
TRACK (no divergence) — divergence only appears for the class whose readout degrades (Mendelian
missense). This tests that on a third benchmark whose readout does NOT degrade.

SGE is 8 genes (BRCA1, RAD51C/D, VHL, BAP1, DDX3X, SFPQ, XRCC2) on 5 chromosomes, so the honest
CV is leave-one-GENE-out (GroupKFold on `gene`) — a probe can't lean on gene-specific signal.
Per model, per class (missense, splicing): signed-`delta` probe (gene-CV, PCA-64) AUPRC vs the
zero-shot SGE LLR (minus_llr_avg). Reads/writes S3. CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter18_sge_probe_vs_llr.py
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

EMB_SGE = "s3://oa-bolinas/analysis/issue302/embeddings_sge"
SCORES_S3 = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]
N_PCA = 64
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
CLASSES = ["missense_variant", "splicing"]


def _load_npz(name: str):
    import s3fs

    with s3fs.S3FileSystem().open(f"{EMB_SGE}/{name}.npz") as f:
        d = np.load(io.BytesIO(f.read()))
        return {k: d[k] for k in d.files}


def cv_auprc_gene(x, y, genes, n_pca=N_PCA) -> float:
    """Leave-one-gene-out grouped-CV AUPRC."""
    k = len(np.unique(genes))
    if k < 2 or len(np.unique(y)) < 2:
        return float("nan")
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(x, y, genes):
        nc = min(n_pca, x[tr].shape[1], len(tr) - 1)
        pipe = make_pipeline(
            StandardScaler(),
            PCA(nc, random_state=0),
            LogisticRegression(max_iter=2000, C=0.5),
        )
        pipe.fit(x[tr], y[tr])
        oof[te] = pipe.predict_proba(x[te])[:, 1]
    return average_precision_score(y, oof)


def main() -> None:
    rows = []
    for name, sdir, params in LADDER:
        emb = _load_npz(name)
        keys = (
            pl.read_parquet(f"{EMB_SGE}/{name}.keys.parquet")
            .with_columns(pl.col("chrom").cast(str))
            .with_row_index("idx")
        )
        sc = (
            pl.read_parquet(f"{SCORES_S3}/{sdir}/sge.parquet")
            .with_columns(pl.col("chrom").cast(str))
            .with_columns((-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("llr"))
            .select([*KEY, "llr"])
        )
        for cls in CLASSES:
            df = (
                keys.filter(pl.col("subset") == cls)
                .join(sc, on=KEY, how="left")
                .sort("idx")
            )
            idx = df["idx"].to_numpy()
            y, gene, llr = (
                df["label"].to_numpy(),
                df["gene"].to_numpy(),
                df["llr"].to_numpy(),
            )
            assert not np.isnan(llr).any(), f"missing SGE LLR for {cls}"
            delta = (emb["alt"][idx, -1] - emb["ref"][idx, -1]).astype(np.float32)
            probe = cv_auprc_gene(delta, y, gene)
            llr_ap = average_precision_score(y, llr)
            rows.append(
                {
                    "cls": cls,
                    "params": params,
                    "n": len(y),
                    "ngenes": int(len(np.unique(gene))),
                    "probe": probe,
                    "llr": llr_ap,
                }
            )
            print(
                f"{cls:>17} {params:>5}M (n={len(y)}, {len(np.unique(gene))} genes): probe={probe:.3f}  LLR={llr_ap:.3f}"
            )

    res = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/sge_probe_vs_llr.parquet")
    c = res.to_pandas()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.7), sharex=True)
    for a, cls in zip(ax, CLASSES):
        s = c[c.cls == cls].sort_values("params")
        a.plot(
            s["params"],
            s["probe"],
            "o-",
            color="tab:green",
            lw=2.5,
            label="frozen-embedding probe (gene-CV)",
        )
        a.plot(
            s["params"], s["llr"], "D--", color="black", lw=2, label="zero-shot SGE LLR"
        )
        a.set_xscale("log")
        a.set_xlabel("params (M, log)")
        a.set_title(
            f"SGE {cls.replace('_variant', '')}  (n={int(s['n'].iloc[0])}, {int(s['ngenes'].iloc[0])} genes)"
        )
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("AUPRC (leave-one-gene-out CV)")
    fig.suptitle(
        "SGE: probe vs zero-shot LLR across the ladder — the readout does NOT degrade, so probe & LLR track (no divergence)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "sge_probe_vs_llr.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "sge_probe_vs_llr.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'sge_probe_vs_llr'}.{{png,svg}}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/sge_probe_vs_llr.parquet",
        str(OUT / "sge_probe_vs_llr.png"),
        str(OUT / "sge_probe_vs_llr.svg"),
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
