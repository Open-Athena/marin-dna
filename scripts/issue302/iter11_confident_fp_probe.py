"""issue #302 — iteration 11: can a probe flag the model's confident FALSE POSITIVES?

The highly-confident errors we characterized are false positives — conserved-looking
benigns the 4B over-calls as pathogenic (iter1/iter3). This is an error-detection /
selective-prediction probe: of the variants the 4B *confidently calls pathogenic*
(zero-shot minus_llr_avg >= the pathogenic median), can the frozen embedding tell
the FPs (truly benign) from the TPs (truly pathogenic)? A good detector is an
abstain/route head — flag the likely-FP "pathogenic" calls and defer them.

Non-circular by construction: within the high-LLR set every variant has a similar
(high) zero-shot score, so the detector cannot just re-derive the LLR — it must use
the *truth* signal (which iter9/10 showed lives in the representation).

Compares the embedding detector (4B last-layer delta, chromosome-grouped CV) to
single-feature baselines: REVEL (a peer VEP — should flag them) and phyloP (high
for BOTH FPs and TPs — should NOT separate them). Reads everything from S3
(embeddings + scores + the iter3 enriched cache); writes results to S3. CPU; run
with OMP_NUM_THREADS capped (see header of the run command).

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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EMB_S3 = "s3://oa-bolinas/analysis/issue302/embeddings"
SCORES = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/scaling-v0.5-h2944-p4B-step-215573/mendelian_traits.parquet"
ENRICHED = "s3://oa-bolinas/analysis/issue302/cache/missense_enriched.parquet"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]


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
    emb = _load_npz("scaling-v0.5-4B")
    ref, alt = emb["ref"].astype(np.float32), emb["alt"].astype(np.float32)
    delta_last = alt[:, -1] - ref[:, -1]  # last layer
    keys = pl.read_parquet(f"{EMB_S3}/scaling-v0.5-4B.keys.parquet").with_columns(
        pl.col("chrom").cast(str)
    )
    sc = (
        pl.read_parquet(SCORES)
        .with_columns(pl.col("chrom").cast(str))
        .filter(pl.col("subset") == "missense_variant")
        .with_columns((-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"))
        .select([*KEY, "mll"])
    )
    ann = (
        pl.read_parquet(ENRICHED)
        .with_columns(pl.col("chrom").cast(str))
        .select([*KEY, "phyloP", "revel"])
    )
    df = (
        keys.with_row_index("idx")
        .join(sc, on=KEY, how="left")
        .join(ann, on=KEY, how="left")
    )
    assert df.height == keys.height
    mll = df["mll"].to_numpy()
    label = df["label"].to_numpy()
    chrom = df["chrom"].to_numpy()
    idx = df["idx"].to_numpy()

    pos_med = np.median(mll[label == 1])
    H = mll >= pos_med  # the model's "confident pathogenic" calls
    is_fp = (label[H] == 0).astype(int)  # within H: benign = FP, pathogenic = TP
    print(f"pathogenic-median minus_llr = {pos_med:.2f}")
    print(
        f"confident-pathogenic set H: n={H.sum()}  TP(pathogenic)={int((label[H] == 1).sum())}  FP(benign)={int(is_fp.sum())}  FP-rate={is_fp.mean():.2f}"
    )

    xH = delta_last[idx[H]]
    chH = chrom[H]
    phyloH = df["phyloP"].to_numpy()[H]
    revelH = df["revel"].to_numpy()[H]

    # embedding detector (chrom-grouped CV)
    oof = cv_scores(xH, is_fp, chH)
    emb_auroc, emb_auprc = (
        roc_auc_score(is_fp, oof),
        average_precision_score(is_fp, oof),
    )

    # single-feature baselines (FP = benign → low REVEL, and both FP/TP are high-phyloP)
    def feat_auc(v, sign):
        m = ~np.isnan(v)
        return roc_auc_score(is_fp[m], sign * v[m]) if m.sum() > 10 else float("nan")

    revel_auroc = feat_auc(revelH, -1.0)  # benign FPs have LOW revel
    phylo_auroc = feat_auc(phyloH, -1.0)
    print(
        "\n=== detecting FPs within the confident-pathogenic set (AUROC; chance=0.5) ==="
    )
    print(
        f"  embedding (4B last-layer delta) : AUROC={emb_auroc:.3f}  AUPRC={emb_auprc:.3f}  (prevalence={is_fp.mean():.3f})"
    )
    print(f"  REVEL (peer VEP)                : AUROC={revel_auroc:.3f}")
    print(f"  phyloP                          : AUROC={phylo_auroc:.3f}")

    # what fraction of confident-pathogenic calls can we safely keep at a given abstain budget?
    order = np.argsort(-oof)  # most-FP-suspect first
    print(
        "\n  precision of the model's confident-pathogenic calls after deferring the top-K embedding-flagged:"
    )
    for frac in (0.0, 0.1, 0.2, 0.3):
        keep = order[int(frac * len(order)) :]
        prec = (label[H][keep] == 1).mean()
        print(
            f"    defer top {frac * 100:3.0f}% suspect -> kept precision (TP rate) = {prec:.3f}  (n_kept={len(keep)})"
        )

    res = pl.DataFrame(
        {
            "detector": ["embedding", "REVEL", "phyloP"],
            "auroc": [emb_auroc, revel_auroc, phylo_auroc],
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/confident_fp_probe.parquet")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(res["detector"], res["auroc"], color=["tab:green", "tab:blue", "tab:gray"])
    ax.axhline(0.5, color="k", ls=":", lw=1, label="chance")
    for i, v in enumerate(res["auroc"]):
        ax.text(i, v, f" {v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("AUROC — detect FPs among confident-pathogenic calls")
    ax.set_title(
        f"Flagging the 4B's confident false positives\n(within its high-LLR calls: {int(is_fp.sum())} FP vs {int((label[H] == 1).sum())} TP)"
    )
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
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
