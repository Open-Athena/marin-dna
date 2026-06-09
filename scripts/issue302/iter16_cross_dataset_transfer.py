"""issue #302 — iteration 16: cross-dataset probe transfer, Mendelian missense <-> Complex missense.

Does the frozen-embedding "deleterious missense" signal transfer across two *different*
notions of functional? Mendelian label = ClinVar pathogenic; Complex label = fine-mapped
causal (PIP). A probe trained on one, applied to the other, tests whether the representation
encodes a shared missense-effect axis or a dataset-specific one.

Common feature = swap-invariant **|delta|** (element-wise |emb_alt - emb_ref|, last layer),
so the probe is defined identically on both (Complex is recoding-indifferent; iter15). Per
ladder size: within-dataset chromosome-CV AUPRC (the ceiling) and the two transfer directions
(fit whole source -> predict target, overlap variants removed from the target). Reads S3; CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter16_cross_dataset_transfer.py
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

EMB_MEND = "s3://oa-bolinas/analysis/issue302/embeddings"  # mendelian missense (cached)
EMB_COMPLEX = "s3://oa-bolinas/analysis/issue302/embeddings_complex_missense"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]
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


def _load(emb_dir: str, name: str):
    """Return (|delta| last-layer [N,D], y, chrom, key_tuples) for one model/dataset."""
    import s3fs

    with s3fs.S3FileSystem().open(f"{emb_dir}/{name}.npz") as f:
        d = np.load(io.BytesIO(f.read()))
        absdelta = np.abs(d["alt"][:, -1] - d["ref"][:, -1]).astype(np.float32)
    keys = pl.read_parquet(f"{emb_dir}/{name}.keys.parquet").with_columns(
        pl.col("chrom").cast(str)
    )
    y = keys["label"].to_numpy()
    chrom = keys["chrom"].to_numpy()
    ktup = list(zip(keys["chrom"].cast(str), keys["pos"], keys["ref"], keys["alt"]))
    return absdelta, y, chrom, ktup


def _pipe():
    return make_pipeline(
        StandardScaler(),
        PCA(N_PCA, random_state=0),
        LogisticRegression(max_iter=2000, C=0.5),
    )


def cv_auprc(x, y, groups) -> float:
    k = min(5, len(np.unique(groups)))
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(x, y, groups):
        p = _pipe()
        p.fit(x[tr], y[tr])
        oof[te] = p.predict_proba(x[te])[:, 1]
    return average_precision_score(y, oof)


def transfer_auprc(xs, ys, xt, yt, kt, ks_set) -> tuple[float, int]:
    """Fit on whole source, predict target; drop target variants present in source."""
    keep = np.array([k not in ks_set for k in kt])
    p = _pipe()
    p.fit(xs, ys)
    pred = p.predict_proba(xt[keep])[:, 1]
    return average_precision_score(yt[keep], pred), int((~keep).sum())


def main() -> None:
    rows = []
    for name, params in LADDER:
        xm, ym, cm, km = _load(EMB_MEND, name)
        xc, yc, cc, kc = _load(EMB_COMPLEX, name)
        km_set, kc_set = set(km), set(kc)
        within_m = cv_auprc(xm, ym, cm)
        within_c = cv_auprc(xc, yc, cc)
        m2c, ov_c = transfer_auprc(xm, ym, xc, yc, kc, km_set)  # train M -> test C
        c2m, ov_m = transfer_auprc(xc, yc, xm, ym, km, kc_set)  # train C -> test M
        rows.append(
            {
                "params": params,
                "within_M": within_m,
                "within_C": within_c,
                "M_to_C": m2c,
                "C_to_M": c2m,
                "chance_M": float(ym.mean()),
                "chance_C": float(yc.mean()),
                "overlap": ov_c + ov_m,
            }
        )
        print(
            f"{params:>5}M | within_M={within_m:.3f} within_C={within_c:.3f} | "
            f"M->C={m2c:.3f} C->M={c2m:.3f} | overlap(C/M)={ov_c}/{ov_m}"
        )

    res = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/cross_dataset_transfer.parquet")
    c = res.to_pandas()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7), sharex=True)
    # Panel A — predicting COMPLEX: within-C ceiling vs M->C transfer.
    ax[0].plot(
        c["params"],
        c["within_C"],
        "o-",
        color="tab:blue",
        lw=2.5,
        label="within-Complex CV (ceiling)",
    )
    ax[0].plot(
        c["params"],
        c["M_to_C"],
        "D--",
        color="tab:orange",
        lw=2.5,
        label="Mendelian->Complex transfer",
    )
    ax[0].plot(c["params"], c["chance_C"], ":", color="black", lw=1, label="chance")
    ax[0].set_title(
        "Predicting COMPLEX missense\n(does a Mendelian-trained probe transfer?)"
    )
    ax[0].set_ylabel("AUPRC")
    # Panel B — predicting MENDELIAN: within-M ceiling vs C->M transfer.
    ax[1].plot(
        c["params"],
        c["within_M"],
        "o-",
        color="tab:blue",
        lw=2.5,
        label="within-Mendelian CV (ceiling)",
    )
    ax[1].plot(
        c["params"],
        c["C_to_M"],
        "D--",
        color="tab:red",
        lw=2.5,
        label="Complex->Mendelian transfer",
    )
    ax[1].plot(c["params"], c["chance_M"], ":", color="black", lw=1, label="chance")
    ax[1].set_title(
        "Predicting MENDELIAN missense\n(does a Complex-trained probe transfer?)"
    )
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("params (M, log)")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle(
        "Cross-dataset probe transfer (|delta| feature) — does the missense-effect axis generalize across ClinVar-pathogenic and GWAS-causal?",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "cross_dataset_transfer.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "cross_dataset_transfer.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'cross_dataset_transfer'}.{{png,svg}}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/cross_dataset_transfer.parquet",
        str(OUT / "cross_dataset_transfer.png"),
        str(OUT / "cross_dataset_transfer.svg"),
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
