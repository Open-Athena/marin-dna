"""issue #302 — iteration 10: embedding probe, thorough version.

Two extensions of iter9, addressing the two follow-up questions:
  (1) CV rigor — report BOTH grouped-CV by `match_group` (as iter9) AND
      leave-one-gene-out CV (group by `exon_closest_pc_gene_id`), so the
      "representation improves with scale" claim is checked against a probe that
      cannot lean on a gene-level prior (match_group is (subset,distance-bin),
      not within-gene — the #203 correction).
  (2) the WITHIN-TRAINING axis — run the same probe across the 10 #232 0.25B CDS
      checkpoints (step 500…4999), to test whether the late-training LLR dip
      (iter2/iter8) is ALSO readout-localized (representation holds/improves while
      the zero-shot LLR declines).

Per model/checkpoint, each layer depth: grouped-CV PCA-256 logistic on
`delta = emb_alt − emb_ref` (and `concat`), vs the zero-shot LLR — for both
groupings. Reads embeddings + scores + gene map from S3; writes the table +
figures back to S3 (node-independent). CPU.

Run:  uv run --group genome-s3 python scripts/issue302/iter10_embedding_probe_general.py
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
SCORES_S3 = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]
GENE_COL = "exon_closest_pc_gene_id"

# (embedding npz name, scores dir, axis value)
LADDER = [
    ("scaling-v0.5-128M", "scaling-v0.5-h896-p128M-step-215573", 128),
    ("scaling-v0.5-1B", "scaling-v0.5-h1920-p1B-step-215573", 1120),
    ("scaling-v0.5-4B", "scaling-v0.5-h2944-p4B-step-215573", 4020),
]
TRAJ = [
    (f"exp232-cds-step-{s}", f"exp232-v4_cds-step-{s}", s)
    for s in (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999)
]
GENE_SRC = f"{SCORES_S3}/scaling-v0.5-h2944-p4B-step-215573/mendelian_traits.parquet"


def _load_npz(name: str):
    import s3fs

    with s3fs.S3FileSystem().open(f"{EMB_S3}/{name}.npz") as f:
        d = np.load(io.BytesIO(f.read()))
        return {k: d[k] for k in d.files}


def cv_auprc(x, y, groups, n_splits=5, n_pca=256) -> float:
    """Grouped-CV AUPRC of standardize→PCA→logistic; n_splits capped at the
    number of distinct groups (leave-one-gene-out has many small groups)."""
    ng = len(np.unique(groups))
    k = min(n_splits, ng)
    if k < 2:
        return float("nan")
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(x, y, groups):
        nc = min(n_pca, x[tr].shape[1], len(tr) - 1)
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=nc, random_state=0),
            LogisticRegression(max_iter=2000, C=0.5),
        )
        pipe.fit(x[tr], y[tr])
        oof[te] = pipe.predict_proba(x[te])[:, 1]
    return average_precision_score(y, oof)


def gene_map() -> pl.DataFrame:
    g = (
        pl.read_parquet(GENE_SRC)
        .with_columns(pl.col("chrom").cast(str))
        .filter(pl.col("subset") == "missense_variant")
        .select([*KEY, pl.col(GENE_COL).alias("gene")])
    )
    return g


def run_axis(manifest, axis_name: str, genes: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for name, scores_dir, axval in manifest:
        emb = _load_npz(name)
        ref, alt = emb["ref"].astype(np.float32), emb["alt"].astype(np.float32)
        fracs = emb["layer_fracs"].tolist()
        keys = (
            pl.read_parquet(f"{EMB_S3}/{name}.keys.parquet")
            .with_columns(pl.col("chrom").cast(str))
            .join(genes, on=KEY, how="left")
        )
        y = keys["label"].to_numpy()
        mg = keys["match_group"].to_numpy()
        # gene groups: fill missing genes with a unique per-row id so they aren't lumped
        gene = keys["gene"].to_numpy()
        gcode = np.array(
            [g if g is not None else f"__none_{i}" for i, g in enumerate(gene)]
        )
        # LLR baseline from this model's scores
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
        print(
            f"\n{name} ({axis_name}={axval}, D={ref.shape[2]}, n_genes={len(np.unique(gcode))}): LLR={llr:.3f}"
        )
        for li, frac in enumerate(fracs):
            delta = alt[:, li] - ref[:, li]
            ap_mg = cv_auprc(delta, y, mg)
            ap_gene = cv_auprc(delta, y, gcode)
            rows.append(
                {
                    "name": name,
                    "axis": axis_name,
                    "axval": axval,
                    "layer_frac": float(frac),
                    "llr": llr,
                    "delta_mg": ap_mg,
                    "delta_gene": ap_gene,
                }
            )
            print(
                f"  layer {frac:.2f}: delta(match_group)={ap_mg:.3f}  delta(leave-gene-out)={ap_gene:.3f}"
            )
    return pl.DataFrame(rows)


def fig_axis(
    df: pl.DataFrame, axis_name: str, xlabel: str, logx: bool, fname: str, title: str
):
    c = df.to_pandas()
    last = c["layer_frac"].max()
    cl = c[c.layer_frac == last].sort_values("axval")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(
        cl["axval"],
        cl["delta_mg"],
        "o-",
        color="tab:green",
        label="probe, last layer (match_group CV)",
    )
    ax.plot(
        cl["axval"],
        cl["delta_gene"],
        "s--",
        color="tab:olive",
        label="probe, last layer (leave-gene-out CV)",
    )
    ax.plot(cl["axval"], cl["llr"], "D-", color="black", lw=2, label="zero-shot LLR")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("missense AUPRC")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, fname)


def _save(fig, name: str):
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / name}")


def main() -> None:
    genes = gene_map()
    res_l = run_axis(LADDER, "params", genes)
    res_t = run_axis(TRAJ, "step", genes)
    res = pl.concat([res_l, res_t])
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/embedding_probe_general.parquet")
    fig_axis(
        res_l,
        "params",
        "params (M, log)",
        True,
        "embprobe_ladder_genecv",
        "Ladder: probe improves with scale under BOTH CVs;\nLLR degrades (robustness of iter9)",
    )
    fig_axis(
        res_t,
        "step",
        "training step (#232 0.25B CDS arm)",
        False,
        "embprobe_traj",
        "Within-training: does the representation hold while the\nlate-step LLR dips? (#232 CDS arm)",
    )
    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/embedding_probe_general.parquet",
        *[
            str(OUT / f"{n}.{e}")
            for n in ("embprobe_ladder_genecv", "embprobe_traj")
            for e in ("png", "svg")
        ],
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
