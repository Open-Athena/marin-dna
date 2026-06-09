"""issue #302 — iteration 11: can a probe flag the model's confident FALSE POSITIVES,
and does that ability change with model scale?

The highly-confident errors we characterized are false positives — conserved-looking
benigns the model over-calls as pathogenic (iter1/iter3). Error-detection probe: of
the variants a model *confidently calls pathogenic* (its zero-shot minus_llr_avg >=
its own pathogenic median), can its frozen embedding tell the FPs (truly benign)
from the TPs (truly pathogenic)? Non-circular: within the high-LLR set every variant
has a similar (high) score, so the detector must use the *truth* signal, not the LLR.

Run ACROSS the 8-rung ladder (each model: own embedding, own confident set,
chromosome-grouped CV) — does self-FP-detection improve with scale (like the label
probe, iter9/10)? Also tracks the FP-rate of the confident-pathogenic set vs scale.
REVEL / phyloP kept only as thin reference lines. Reads everything from S3; writes
results to S3. CPU; run thread-capped (OMP_NUM_THREADS=2).

Run:  OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter11_confident_fp_probe.py
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EMB_S3 = "s3://oa-bolinas/analysis/issue302/embeddings"
SCORES_S3 = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
ENRICHED = "s3://oa-bolinas/analysis/issue302/cache/missense_enriched.parquet"
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


def _load_npz(name: str):
    import s3fs

    with s3fs.S3FileSystem().open(f"{EMB_S3}/{name}.npz") as f:
        d = np.load(io.BytesIO(f.read()))
        return {k: d[k] for k in d.files}


def cv_scores(x, y, groups, n_splits=5, n_pca=256):
    k = min(n_splits, len(np.unique(groups)))
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
    return oof


def main() -> None:
    ann = (
        pl.read_parquet(ENRICHED)
        .with_columns(pl.col("chrom").cast(str))
        .select([*KEY, "phyloP", "revel"])
    )
    rows = []
    for name, scores_dir, params in LADDER:
        emb = _load_npz(name)
        delta_last = (emb["alt"][:, -1] - emb["ref"][:, -1]).astype(np.float32)
        keys = pl.read_parquet(f"{EMB_S3}/{name}.keys.parquet").with_columns(
            pl.col("chrom").cast(str)
        )
        sc = (
            pl.read_parquet(f"{SCORES_S3}/{scores_dir}/mendelian_traits.parquet")
            .with_columns(pl.col("chrom").cast(str))
            .filter(pl.col("subset") == "missense_variant")
            .with_columns((-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"))
            .select([*KEY, "mll"])
        )
        df = (
            keys.with_row_index("idx")
            .join(sc, on=KEY, how="left")
            .join(ann, on=KEY, how="left")
        )
        mll, label, chrom, idx = (
            df["mll"].to_numpy(),
            df["label"].to_numpy(),
            df["chrom"].to_numpy(),
            df["idx"].to_numpy(),
        )
        pos_med = np.median(mll[label == 1])
        H = mll >= pos_med
        is_fp = (label[H] == 0).astype(int)
        emb_auc = roc_auc_score(is_fp, cv_scores(delta_last[idx[H]], is_fp, chrom[H]))

        def feat_auc(v):
            m = ~np.isnan(v)
            return (
                roc_auc_score(is_fp[m], -v[m]) if m.sum() > 10 else float("nan")
            )  # FP=benign → low feature

        revel_auc, phylo_auc = (
            feat_auc(df["revel"].to_numpy()[H]),
            feat_auc(df["phyloP"].to_numpy()[H]),
        )
        rows.append(
            {
                "params": params,
                "n_H": int(H.sum()),
                "fp_rate": float(is_fp.mean()),
                "emb_auroc": emb_auc,
                "revel_auroc": revel_auc,
                "phylo_auroc": phylo_auc,
            }
        )
        print(
            f"{params:>5}M  confident-path n={int(H.sum()):>4} FP-rate={is_fp.mean():.2f}  "
            f"emb_AUROC={emb_auc:.3f}  REVEL={revel_auc:.3f}  phyloP={phylo_auc:.3f}"
        )

    res = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/confident_fp_probe.parquet")
    c = res.to_pandas()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(
        c["params"],
        c["emb_auroc"],
        "o-",
        color="tab:green",
        lw=2.5,
        label="frozen-embedding detector",
    )
    ax[0].plot(
        c["params"],
        c["revel_auroc"],
        "--",
        color="tab:blue",
        lw=1,
        alpha=0.7,
        label="REVEL (ref)",
    )
    ax[0].plot(
        c["params"],
        c["phylo_auroc"],
        ":",
        color="tab:gray",
        lw=1,
        alpha=0.7,
        label="phyloP (ref)",
    )
    ax[0].set_xscale("log")
    ax[0].set_ylim(0.5, 1.0)  # start at chance
    ax[0].set_xlabel("params (M, log)")
    ax[0].set_ylabel("AUROC — flag confident FPs (chromosome CV)")
    ax[0].set_title(
        "Can the embedding flag the model's confident FPs?\n(does self-error-detection improve with scale?)"
    )
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(c["params"], c["fp_rate"] * 100, "s-", color="tab:red", lw=2)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("params (M, log)")
    ax[1].set_ylabel("% of confident-pathogenic calls that are FPs")
    ax[1].set_title("The confident-pathogenic calls get MORE\nFP-dominated with scale")
    ax[1].grid(alpha=0.3)
    fig.suptitle("Iter11 — confident-FP detection across the ladder", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "confident_fp_detector.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "confident_fp_detector.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {OUT / 'confident_fp_detector'}.{{png,svg}}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/confident_fp_probe.parquet",
        str(OUT / "confident_fp_detector.png"),
        str(OUT / "confident_fp_detector.svg"),
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
