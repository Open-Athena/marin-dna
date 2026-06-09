"""issue #302 / #306 — iteration 19: three-way cross-dataset probe transfer across the
missense-effect benchmarks — Mendelian (ClinVar pathogenic), Complex (GWAS causal), SGE
(saturation genome editing = direct functional assay).

Does the frozen embedding's "deleterious missense" axis generalize across three different
operationalizations of "functional"? SGE is the closest thing to ground-truth missense
effect, so Mendelian->SGE is the sharpest test: does the ClinVar-pathogenic representation
axis predict direct functional readouts?

Common feature = swap-invariant |delta| (works for the recoding-indifferent Complex too).
Per model: within-dataset CV (diagonal ceiling; Mendelian/Complex = chromosome-CV, SGE =
leave-one-gene-out) and all 6 transfers (fit whole source -> predict target, overlap removed).
Reads S3; CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter19_three_way_transfer.py
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

EMB = {
    "MEN": "s3://oa-bolinas/analysis/issue302/embeddings",
    "CPX": "s3://oa-bolinas/analysis/issue302/embeddings_complex_missense",
    "SGE": "s3://oa-bolinas/analysis/issue302/embeddings_sge",
}
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
N_PCA = 64
LADDER = [
    ("scaling-v0.5-46M", 46),
    ("scaling-v0.5-76M", 76),
    ("scaling-v0.5-128M", 128),
    ("scaling-v0.5-255M", 255),
    ("scaling-v0.5-476M", 476),
    ("scaling-v0.5-1B", 1120),
    ("scaling-v0.5-2B", 2270),
    ("scaling-v0.5-4B", 4020),
]


def _load(ds: str, name: str):
    """Return (|delta| [N,D], y, group, key_tuples) — SGE filtered to missense, grouped by gene."""
    import s3fs

    with s3fs.S3FileSystem().open(f"{EMB[ds]}/{name}.npz") as f:
        d = np.load(io.BytesIO(f.read()))
        absd = np.abs(d["alt"][:, -1] - d["ref"][:, -1]).astype(np.float32)
    keys = (
        pl.read_parquet(f"{EMB[ds]}/{name}.keys.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .with_row_index("idx")
    )
    if ds == "SGE":
        keys = keys.filter(pl.col("subset") == "missense_variant")
        group = keys["gene"].to_numpy()
    else:
        group = keys["chrom"].to_numpy()
    idx = keys["idx"].to_numpy()
    y = keys["label"].to_numpy()
    ktup = list(zip(keys["chrom"].cast(str), keys["pos"], keys["ref"], keys["alt"]))
    return absd[idx], y, group, ktup


def _pipe():
    return make_pipeline(
        StandardScaler(),
        PCA(N_PCA, random_state=0),
        LogisticRegression(max_iter=2000, C=0.5),
    )


def cv_auprc(x, y, groups) -> float:
    k = min(8, len(np.unique(groups)))
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(x, y, groups):
        p = _pipe()
        p.fit(x[tr], y[tr])
        oof[te] = p.predict_proba(x[te])[:, 1]
    return average_precision_score(y, oof)


def transfer(xs, ys, xt, yt, kt, ks_set) -> float:
    keep = np.array([k not in ks_set for k in kt])
    p = _pipe()
    p.fit(xs, ys)
    return average_precision_score(yt[keep], p.predict_proba(xt[keep])[:, 1])


def main() -> None:
    names = ["MEN", "CPX", "SGE"]
    rows = []
    mats = {}
    for name, params in LADDER:
        data = {d: _load(d, name) for d in names}
        ksets = {d: set(data[d][3]) for d in names}
        M = np.full((3, 3), np.nan)
        for i, src in enumerate(names):
            xs, ys, gs, _ = data[src]
            for j, tgt in enumerate(names):
                xt, yt, gt, kt = data[tgt]
                M[i, j] = (
                    cv_auprc(xt, yt, gt)
                    if src == tgt
                    else transfer(xs, ys, xt, yt, kt, ksets[src])
                )
        mats[params] = M
        chance = {d: float(data[d][1].mean()) for d in names}
        rows.append(
            {
                "params": params,
                **{
                    f"{s}_to_{t}": M[i, j]
                    for i, s in enumerate(names)
                    for j, t in enumerate(names)
                },
                **{f"chance_{d}": chance[d] for d in names},
            }
        )
        print(
            f"{params:>5}M | within MEN/CPX/SGE={M[0, 0]:.3f}/{M[1, 1]:.3f}/{M[2, 2]:.3f} | MEN->SGE={M[0, 2]:.3f} SGE->MEN={M[2, 0]:.3f} CPX->SGE={M[1, 2]:.3f}"
        )

    res = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/three_way_transfer.parquet")
    c = res.to_pandas()

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0))
    # Panel A — 4B transfer matrix.
    M4 = mats[4020]
    im = ax[0].imshow(M4, cmap="viridis", vmin=0.1, vmax=0.6)
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels([f"test\n{n}" for n in names])
    ax[0].set_yticks(range(3))
    ax[0].set_yticklabels([f"train {n}" for n in names])
    for i in range(3):
        for j in range(3):
            ax[0].text(
                j,
                i,
                f"{M4[i, j]:.3f}" + ("\n(within)" if i == j else ""),
                ha="center",
                va="center",
                color="white" if M4[i, j] < 0.42 else "black",
                fontsize=10,
            )
    ax[0].set_title(
        "4B |delta| transfer matrix (AUPRC)\ndiagonal = within-dataset CV ceiling"
    )
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    # Panel B — into-SGE across scale (SGE = direct functional assay).
    ax[1].plot(
        c["params"],
        c["SGE_to_SGE"],
        "o-",
        color="tab:blue",
        lw=2.5,
        label="within-SGE (gene-CV ceiling)",
    )
    ax[1].plot(
        c["params"],
        c["MEN_to_SGE"],
        "D--",
        color="tab:red",
        lw=2.5,
        label="Mendelian→SGE transfer",
    )
    ax[1].plot(
        c["params"],
        c["CPX_to_SGE"],
        "s--",
        color="tab:green",
        lw=2,
        label="Complex→SGE transfer",
    )
    ax[1].plot(c["params"], c["chance_SGE"], ":", color="black", lw=1, label="chance")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("params (M, log)")
    ax[1].set_ylabel("SGE missense AUPRC")
    ax[1].set_title(
        "Predicting the DIRECT functional assay (SGE):\ndoes the Mendelian-pathogenic axis transfer, and scale?"
    )
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle(
        "Three-way missense-effect transfer: Mendelian (ClinVar) ↔ Complex (GWAS) ↔ SGE (functional assay)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "three_way_transfer.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "three_way_transfer.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'three_way_transfer'}.{{png,svg}}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/three_way_transfer.parquet",
        str(OUT / "three_way_transfer.png"),
        str(OUT / "three_way_transfer.svg"),
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
