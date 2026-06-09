"""issue #302 — iteration 15: frozen-embedding probe for COMPLEX-traits missense.

Complex traits are indifferent to allele recoding (the typical zero-shot score is abs(LLR)),
so the signed `delta = emb_alt - emb_ref` has an arbitrary sign and is the wrong probe
feature. The embedding analog of abs(LLR) is the swap-invariant **|delta|** (element-wise
|emb_alt - emb_ref|): even under ref<->alt, full-dimensional, linear-probe-friendly.

This iteration: across the ladder, chromosome-grouped-CV PCA-64 logistic AUPRC on the complex
missense set, using |delta| (primary) vs signed delta (does sign-invariance help/hurt here?).
NOTE: evals_v2 never scored complex_traits for the scaling ladder, so there is no zero-shot
abs(LLR) baseline to compare against (flagged as a follow-up). Chance AUPRC = positive rate.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter15_complex_missense_probe.py
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

EMB_COMPLEX = "s3://oa-bolinas/analysis/issue302/embeddings_complex_missense"
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


def _load_npz(emb_dir: str, name: str):
    import s3fs

    with s3fs.S3FileSystem().open(f"{emb_dir}/{name}.npz") as f:
        d = np.load(io.BytesIO(f.read()))
        return {k: d[k] for k in d.files}


def cv_auprc(x, y, groups, n_pca=N_PCA) -> float:
    k = min(5, len(np.unique(groups)))
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(x, y, groups):
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
    for name, params in LADDER:
        emb = _load_npz(EMB_COMPLEX, name)
        keys = pl.read_parquet(f"{EMB_COMPLEX}/{name}.keys.parquet").with_columns(
            pl.col("chrom").cast(str)
        )
        y, chrom = keys["label"].to_numpy(), keys["chrom"].to_numpy()
        d = (emb["alt"][:, -1] - emb["ref"][:, -1]).astype(np.float32)
        ap_abs = cv_auprc(np.abs(d), y, chrom)
        ap_signed = cv_auprc(d, y, chrom)
        rows.append(
            {
                "params": params,
                "n": len(y),
                "npos": int(y.sum()),
                "abs_delta": ap_abs,
                "signed_delta": ap_signed,
            }
        )
        print(
            f"{params:>5}M (n={len(y)}, pos={int(y.sum())}): |delta|={ap_abs:.3f}  signed={ap_signed:.3f}"
        )

    res = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/complex_missense_probe.parquet")
    c = res.to_pandas()
    chance = c["npos"].iloc[0] / c["n"].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(
        c["params"],
        c["abs_delta"],
        "o-",
        color="tab:purple",
        lw=2.5,
        label="|delta| probe (abs-LLR analog)",
    )
    ax.plot(
        c["params"],
        c["signed_delta"],
        "s--",
        color="tab:gray",
        lw=1.8,
        label="signed delta probe (ref)",
    )
    ax.axhline(
        chance, color="black", ls=":", lw=1, label=f"chance (pos rate {chance:.2f})"
    )
    ax.set_xscale("log")
    ax.set_xlabel("params (M, log)")
    ax.set_ylabel("complex-missense AUPRC (chromosome CV)")
    ax.set_title(
        "Complex-traits missense: frozen-embedding probe across scale\n(sign-invariant |delta|; no zero-shot abs-LLR baseline for the ladder)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "complex_missense_probe.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "complex_missense_probe.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'complex_missense_probe'}.{{png,svg}}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/complex_missense_probe.parquet",
        str(OUT / "complex_missense_probe.png"),
        str(OUT / "complex_missense_probe.svg"),
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
