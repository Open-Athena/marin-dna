"""issue #302 — iteration 14: the probe-vs-LLR across-scale comparison for the
within-CDS CONTRAST classes — splicing and synonymous.

The missense readout-localization (iter9/10): the embedding probe rises with scale while
the zero-shot LLR degrades. Splicing and synonymous *improve* with scale (#279/#274) —
they're the contrast. If the readout-localization is real and missense-specific, here the
probe and the LLR should both rise and **track together** (no divergence). This is the
control that says "missense is special", not a generic probe>LLR artifact.

Per class (missense | splicing | synonymous), per ladder size: last-layer signed-`delta`
probe (chromosome-grouped CV, PCA-64) AUPRC vs the zero-shot LLR AUPRC. Missense embeddings
come from the original cache; splicing+synonymous from embeddings_mend_cds/ (one npz holds
both, split by the `subset` column). Reads/writes S3. CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter14_splicing_synonymous_probe.py
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

EMB_MIS = "s3://oa-bolinas/analysis/issue302/embeddings"  # missense (cached)
EMB_CDS = "s3://oa-bolinas/analysis/issue302/embeddings_mend_cds"  # splicing+synonymous
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
# (display, subset value, embedding dir)
CLASSES = [
    ("missense", "missense_variant", EMB_MIS),
    ("splicing", "splicing", EMB_CDS),
    ("synonymous", "synonymous_variant", EMB_CDS),
]


def _load_npz(emb_dir: str, name: str):
    import s3fs

    with s3fs.S3FileSystem().open(f"{emb_dir}/{name}.npz") as f:
        d = np.load(io.BytesIO(f.read()))
        return {k: d[k] for k in d.files}


def cv_auprc(x, y, groups, n_pca=N_PCA) -> float:
    k = min(5, len(np.unique(groups)))
    if k < 2 or len(np.unique(y)) < 2:
        return float("nan")
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


def probe_and_llr(emb_dir: str, name: str, scores_dir: str, subset: str):
    emb = _load_npz(emb_dir, name)
    keys = (
        pl.read_parquet(f"{emb_dir}/{name}.keys.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .with_row_index("idx")
    )
    sc = (
        pl.read_parquet(f"{SCORES_S3}/{scores_dir}/mendelian_traits.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .filter(pl.col("subset") == subset)
        .with_columns((-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("llr"))
        .select([*KEY, "llr"])
    )
    df = (
        keys.filter(pl.col("subset") == subset).join(sc, on=KEY, how="left").sort("idx")
    )
    idx = df["idx"].to_numpy()
    y, chrom, llr = df["label"].to_numpy(), df["chrom"].to_numpy(), df["llr"].to_numpy()
    assert not np.isnan(llr).any(), f"missing LLR for {subset}"
    delta = (emb["alt"][idx, -1] - emb["ref"][idx, -1]).astype(np.float32)
    return cv_auprc(delta, y, chrom), average_precision_score(y, llr), len(y)


def main() -> None:
    rows = []
    for disp, subset, emb_dir in CLASSES:
        for name, scores_dir, params in LADDER:
            probe, llr, n = probe_and_llr(emb_dir, name, scores_dir, subset)
            rows.append(
                {"cls": disp, "params": params, "n": n, "probe": probe, "llr": llr}
            )
            print(f"{disp:>11} {params:>5}M (n={n}): probe={probe:.3f}  LLR={llr:.3f}")
    res = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/splice_syn_probe.parquet")
    c = res.to_pandas()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for a, (disp, _sub, _d) in zip(ax, CLASSES):
        s = c[c.cls == disp].sort_values("params")
        a.plot(
            s["params"],
            s["probe"],
            "o-",
            color="tab:green",
            lw=2.5,
            label="frozen-embedding probe",
        )
        a.plot(s["params"], s["llr"], "D--", color="black", lw=2, label="zero-shot LLR")
        a.set_xscale("log")
        a.set_xlabel("params (M, log)")
        a.set_title(f"{disp}  (n={int(s['n'].iloc[0])})")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax[0].set_ylabel("AUPRC (chromosome CV)")
    fig.suptitle(
        "Probe vs zero-shot LLR across the ladder — missense DIVERGES (readout-localized); splicing/synonymous TRACK",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "splice_syn_probe.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "splice_syn_probe.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'splice_syn_probe'}.{{png,svg}}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/splice_syn_probe.parquet",
        str(OUT / "splice_syn_probe.png"),
        str(OUT / "splice_syn_probe.svg"),
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
