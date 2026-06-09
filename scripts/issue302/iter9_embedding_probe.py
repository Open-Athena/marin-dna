"""issue #302 — iteration 9: frozen-embedding probe (does the model "know"?).

The decisive plateau-vs-degrade test. For each ladder model and each cached layer
depth ({1/4,1/2,3/4,last}), train a grouped-CV logistic probe on the variant
EMBEDDING to separate pathogenic from benign missense, and compare to the
zero-shot LLR (which degrades with scale). Two embedding features per the design:
  concat = [emb_ref, emb_alt]   (expressive)
  delta  =  emb_alt − emb_ref   (the natural variant-effect representation)

Key readouts:
  - embedding-probe AUPRC vs model size, per layer, overlaid on the LLR baseline.
    If the embedding holds / improves while the LLR degrades → the representation
    keeps the benign-vs-pathogenic info and the degradation is in the readout.
  - AUPRC vs layer depth: does a MIDDLE layer separate better than the last
    (readout-adjacent) layer?

Inputs: s3://oa-bolinas/analysis/issue302/embeddings/{name}.npz + .keys.parquet
(from extract_variant_embeddings.py); the iter1 ladder cache for the LLR baseline.
CPU only. Output: scratch/issue302/figs/embedding_probe.{png,svg}.

Run:  uv run --group genome-s3 python scripts/issue302/iter9_embedding_probe.py
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

EMB_S3 = "s3://oa-bolinas/analysis/issue302/embeddings"
SCORES_S3 = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"  # zero-shot LLR baseline
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"  # results uploaded here (node-independent)
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]
MODELS = [
    ("scaling-v0.5-128M", 128),
    ("scaling-v0.5-1B", 1120),
    ("scaling-v0.5-4B", 4020),
]
SCORES_DIR = {
    "scaling-v0.5-128M": "scaling-v0.5-h896-p128M-step-215573",
    "scaling-v0.5-1B": "scaling-v0.5-h1920-p1B-step-215573",
    "scaling-v0.5-4B": "scaling-v0.5-h2944-p4B-step-215573",
}


def _load_npz(name: str):
    import s3fs

    fs = s3fs.S3FileSystem()
    with fs.open(f"{EMB_S3}/{name}.npz") as f:
        d = np.load(io.BytesIO(f.read()))
        out = {k: d[k] for k in d.files}
    keys = pl.read_parquet(f"{EMB_S3}/{name}.keys.parquet").with_columns(
        pl.col("chrom").cast(str)
    )
    return out, keys


def cv_auprc(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    n_pca: int = 256,
) -> float:
    """Grouped-CV AUPRC of a standardize→PCA→logistic probe (PCA caps the
    logistic at n_pca dims: faster + lighter than a raw ~6k-feature fit, and the
    standard embedding-probe form). PCA/scaler fit on train folds only."""
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(x, y, groups):
        nc = min(n_pca, x[tr].shape[1], len(tr) - 1)
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=nc, random_state=0),
            LogisticRegression(max_iter=2000, C=0.5),
        )
        pipe.fit(x[tr], y[tr])
        oof[te] = pipe.predict_proba(x[te])[:, 1]
    return average_precision_score(y, oof)


def main() -> None:
    rows = []
    llr_base = {}
    layer_fracs = None
    for name, params in MODELS:
        emb, keys = _load_npz(name)
        ref, alt = emb["ref"].astype(np.float32), emb["alt"].astype(np.float32)
        fracs = emb["layer_fracs"].tolist()
        layer_fracs = fracs
        y = keys["label"].to_numpy()
        groups = keys["match_group"].to_numpy()
        # zero-shot LLR baseline from S3 scores (minus_llr_avg) on the exact same variants
        sc = (
            pl.read_parquet(f"{SCORES_S3}/{SCORES_DIR[name]}/mendelian_traits.parquet")
            .with_columns(pl.col("chrom").cast(str))
            .filter(pl.col("subset") == "missense_variant")
            .with_columns((-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("llr"))
            .select([*KEY, "llr"])
        )
        j = keys.join(sc, on=KEY, how="left")
        llr_base[params] = average_precision_score(y, j["llr"].to_numpy())
        print(
            f"\n{name} (n={len(y)}, {ref.shape[1]} layers, D={ref.shape[2]}): LLR AUPRC = {llr_base[params]:.3f}"
        )
        for li, frac in enumerate(fracs):
            concat = np.concatenate([ref[:, li], alt[:, li]], axis=1)
            delta = alt[:, li] - ref[:, li]
            ap_c = cv_auprc(concat, y, groups)
            ap_d = cv_auprc(delta, y, groups)
            rows.append(
                {
                    "name": name,
                    "params": params,
                    "layer_frac": float(frac),
                    "concat_auprc": ap_c,
                    "delta_auprc": ap_d,
                }
            )
            print(
                f"  layer {frac:.2f}: concat AUPRC={ap_c:.3f}  delta AUPRC={ap_d:.3f}"
            )
    res = pl.DataFrame(rows)
    res.write_parquet("scratch/issue302/embedding_probe.parquet")

    # Figures
    c = res.to_pandas()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    # (A) AUPRC vs scale, one line per layer (delta), + LLR baseline
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(layer_fracs)))
    for col, ls, lab in [
        ("delta_auprc", "-", "delta"),
        ("concat_auprc", "--", "concat"),
    ]:
        for fi, frac in enumerate(layer_fracs):
            sub = c[c.layer_frac == frac].sort_values("params")
            axes[0].plot(
                sub["params"],
                sub[col],
                ls,
                marker="o",
                color=cmap[fi],
                label=f"{lab} L{frac:.2f}" if col == "delta_auprc" else None,
            )
    bx = sorted(llr_base)
    axes[0].plot(
        bx, [llr_base[p] for p in bx], "s-", color="black", lw=2, label="zero-shot LLR"
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("params (M, log)")
    axes[0].set_ylabel("missense AUPRC (grouped 5-fold CV)")
    axes[0].set_title(
        "Embedding probe vs scale — does the representation hold\nwhile the LLR degrades?"
    )
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)
    # (B) AUPRC vs layer depth, per model (delta)
    for name, params in MODELS:
        sub = c[c.name == name].sort_values("layer_frac")
        axes[1].plot(sub["layer_frac"], sub["delta_auprc"], "o-", label=f"{params}M")
    axes[1].set_xlabel("layer depth (fraction)")
    axes[1].set_ylabel("missense AUPRC (delta probe)")
    axes[1].set_title("Does a MIDDLE layer separate better than the last?")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(
        "Iter9 — frozen-embedding probe: representation vs likelihood readout", y=1.02
    )
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "embedding_probe.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "embedding_probe.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {OUT / 'embedding_probe'}.{{png,svg}}")

    # Upload results to S3 so they're retrievable regardless of which node ran this.
    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/embedding_probe.parquet",
        str(OUT / "embedding_probe.png"),
        str(OUT / "embedding_probe.svg"),
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded results -> {OUT_S3}")


if __name__ == "__main__":
    main()
