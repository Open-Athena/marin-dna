"""issue #302 — iteration 10: embedding probe, thorough version.

Two extensions of iter9, addressing the follow-up questions:
  (1) CV rigor — report BOTH grouped-CV by `match_group` (as iter9) AND
      **chromosome-grouped** CV (GroupKFold on `chrom`: a whole chromosome is held
      out, so no gene/region/positional signal can leak train→test). This is the
      clean genomic split; if "representation improves with scale" survives it,
      it isn't a leakage artifact.
  (2) the WITHIN-TRAINING axis — the same probe across the 10 #232 0.25B CDS
      checkpoints (step 500…4999): is the late-training LLR dip (iter2/iter8) also
      readout-localized (representation holds/improves while the zero-shot LLR
      declines)?

Across the FULL 8-rung ladder (46M…4B) for the scale axis. Per model/checkpoint,
each layer depth: grouped-CV PCA-256 logistic on `delta = emb_alt − emb_ref` vs
the zero-shot LLR, for both groupings. Reads embeddings + scores from S3; writes
the table + figures back to S3 (node-independent). CPU.

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

# (embedding npz name, scores dir, axis value) — full 8-rung ladder.
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
TRAJ = [
    (f"exp232-cds-step-{s}", f"exp232-v4_cds-step-{s}", s)
    for s in (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999)
]


def _load_npz(name: str):
    import s3fs

    with s3fs.S3FileSystem().open(f"{EMB_S3}/{name}.npz") as f:
        d = np.load(io.BytesIO(f.read()))
        return {k: d[k] for k in d.files}


def cv_auprc(x, y, groups, n_splits=5, n_pca=256) -> float:
    """Grouped-CV AUPRC of standardize→PCA→logistic; folds = min(n_splits, #groups)."""
    k = min(n_splits, len(np.unique(groups)))
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


def run_axis(manifest, axis_name: str) -> pl.DataFrame:
    rows = []
    for name, scores_dir, axval in manifest:
        emb = _load_npz(name)
        ref, alt = emb["ref"].astype(np.float32), emb["alt"].astype(np.float32)
        fracs = emb["layer_fracs"].tolist()
        keys = pl.read_parquet(f"{EMB_S3}/{name}.keys.parquet").with_columns(
            pl.col("chrom").cast(str)
        )
        y = keys["label"].to_numpy()
        mg = keys["match_group"].to_numpy()
        chrom = keys["chrom"].to_numpy()
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
            f"\n{name} ({axis_name}={axval}, D={ref.shape[2]}, n_chrom={len(np.unique(chrom))}): LLR={llr:.3f}"
        )
        for li, frac in enumerate(fracs):
            delta = alt[:, li] - ref[:, li]
            ap_mg = cv_auprc(delta, y, mg)
            ap_chrom = cv_auprc(delta, y, chrom)
            rows.append(
                {
                    "name": name,
                    "axis": axis_name,
                    "axval": axval,
                    "layer_frac": float(frac),
                    "llr": llr,
                    "delta_mg": ap_mg,
                    "delta_chrom": ap_chrom,
                }
            )
            print(
                f"  layer {frac:.2f}: delta(match_group)={ap_mg:.3f}  delta(chrom)={ap_chrom:.3f}"
            )
    return pl.DataFrame(rows)


def fig_axis(df: pl.DataFrame, xlabel: str, logx: bool, fname: str, title: str):
    c = df.to_pandas()
    cl = c[c.layer_frac == c["layer_frac"].max()].sort_values("axval")
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(
        cl["axval"],
        cl["delta_mg"],
        "o-",
        color="tab:green",
        label="probe last layer (match_group CV)",
    )
    ax.plot(
        cl["axval"],
        cl["delta_chrom"],
        "s--",
        color="tab:olive",
        label="probe last layer (chromosome CV)",
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
    res_l = run_axis(LADDER, "params")
    res_t = run_axis(TRAJ, "step")
    res = pl.concat([res_l, res_t])
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/embedding_probe_general.parquet")
    fig_axis(
        res_l,
        "params (M, log)",
        True,
        "embprobe_ladder_chromcv",
        "Ladder (8 sizes): probe improves with scale under match_group\nAND chromosome CV; zero-shot LLR degrades",
    )
    fig_axis(
        res_t,
        "training step (#232 0.25B CDS arm)",
        False,
        "embprobe_traj",
        "Within-training (#232 CDS): does the representation hold\nwhile the late-step zero-shot LLR dips?",
    )
    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/embedding_probe_general.parquet",
        *[
            str(OUT / f"{n}.{e}")
            for n in ("embprobe_ladder_chromcv", "embprobe_traj")
            for e in ("png", "svg")
        ],
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
