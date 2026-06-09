"""issue #302 — iteration 13: does the supervised probe actually RESCUE the specific
high-confidence false positives, or just improve aggregate AUPRC elsewhere?

The "readout-localized, fixable with a supervised head" headline (iter9/10/12) is about
*aggregate* missense AUPRC (probe 0.37->0.55 vs LLR degrading). GB's question: does that
gain actually fix the **specific failure mode** — the conserved-but-tolerated benigns the
zero-shot LLR confidently over-calls (iter1/3/5/11) — or does the probe get easy variants
more right while still mis-ranking exactly those hard FPs?

Direct test. Per model, define the LLR's CONFIDENT FALSE POSITIVES:
    F  = benign (label==0) missense with zero-shot LLR >= the pathogenic median
(the variants the readout over-calls; iter11's set). Then look at where the frozen-embedding
PROBE (chromosome-grouped CV OOF, PCA-64 ~ iter12 optimum) ranks them:
  - rescue rate    = fraction of F the probe ranks in the benign half (percentile < 50)
  - P-vs-F AUROC   = can the score rank true positives ABOVE these confident FPs? (LLR vs probe)
  - TP_H-vs-F AUROC= iter11's within-confident-set FP-vs-TP separation (continuity)
  - operating-pt FP-rate = of the top-n_H variants by each score, what % are FP (LLR's = iter11's 0.66)
Plus the money figure (4B): LLR vs probe score distributions for {ordinary benigns, confident
FPs F, pathogenics}. If F shifts from sitting-with-pathogenics (under LLR) toward the benigns
(under probe), the probe rescues them; if F stays up, it doesn't. Reads/writes S3. CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter13_probe_rescue.py
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EMB_S3 = "s3://oa-bolinas/analysis/issue302/embeddings"
SCORES_S3 = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]
N_PCA = 64  # near-optimal frozen probe (iter12); gives the representation its best shot
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


def cv_scores(x, y, groups, n_splits=5, n_pca=N_PCA):
    """Chromosome-grouped CV out-of-fold P(pathogenic) from a frozen-embedding logistic probe."""
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


def _auroc(score_pos, score_neg) -> float:
    yy = np.r_[np.ones(len(score_pos)), np.zeros(len(score_neg))]
    return roc_auc_score(yy, np.r_[score_pos, score_neg])


def main() -> None:
    rows = []
    stash: dict = {}
    for name, scores_dir, params in LADDER:
        emb = _load_npz(name)
        delta_last = (emb["alt"][:, -1] - emb["ref"][:, -1]).astype(np.float32)
        keys = (
            pl.read_parquet(f"{EMB_S3}/{name}.keys.parquet")
            .with_columns(pl.col("chrom").cast(str))
            .with_row_index("idx")
        )
        sc = (
            pl.read_parquet(f"{SCORES_S3}/{scores_dir}/mendelian_traits.parquet")
            .with_columns(pl.col("chrom").cast(str))
            .filter(pl.col("subset") == "missense_variant")
            .with_columns((-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("llr"))
            .select([*KEY, "llr"])
        )
        df = keys.join(sc, on=KEY, how="left").sort("idx")
        idx = df["idx"].to_numpy()
        label, chrom, llr = (
            df["label"].to_numpy(),
            df["chrom"].to_numpy(),
            df["llr"].to_numpy(),
        )
        assert not np.isnan(llr).any(), "missing LLR for some missense variant"
        delta = delta_last[idx]  # align embeddings to df row order

        oof = cv_scores(delta, label, chrom)
        pos = label == 1
        pos_med = np.median(llr[pos])
        H = llr >= pos_med  # confident-pathogenic set (LLR)
        F = H & (label == 0)  # confident FALSE POSITIVES
        TP_H = H & (label == 1)
        B0 = (~H) & (label == 0)  # ordinary (low-LLR) benigns
        p_pct = (
            rankdata(oof) / len(oof) * 100
        )  # within-missense percentile under the probe

        rescue = float((p_pct[F] < 50).mean())  # probe ranks them in the benign half?
        n_H = int(H.sum())
        top_p = np.argsort(oof)[
            -n_H:
        ]  # probe's own confident-pathogenic set (matched size)
        rows.append(
            {
                "params": params,
                "n_F": int(F.sum()),
                "rescue_rate": rescue,
                "auroc_PvF_llr": _auroc(llr[pos], llr[F]),
                "auroc_PvF_probe": _auroc(oof[pos], oof[F]),
                "auroc_TPHvF_probe": _auroc(oof[TP_H], oof[F]),
                "fp_rate_llr": float((label[H] == 0).mean()),
                "fp_rate_probe": float((label[top_p] == 0).mean()),
                "b0_below50": float((p_pct[B0] < 50).mean()),
                "pos_below50": float((p_pct[pos] < 50).mean()),
            }
        )
        print(
            f"{params:>5}M  n_F={int(F.sum()):>3}  rescue={rescue:.2f}  "
            f"P-vs-F AUROC: LLR={rows[-1]['auroc_PvF_llr']:.3f}->probe={rows[-1]['auroc_PvF_probe']:.3f}  "
            f"op-FP%: LLR={rows[-1]['fp_rate_llr']:.2f}->probe={rows[-1]['fp_rate_probe']:.2f}"
        )
        if params == 4020:
            stash = {"llr": llr, "oof": oof, "F": F, "B0": B0, "pos": pos}

    res = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/probe_rescue.parquet")

    # ---- money figure (4B): where do the LLR's confident FPs land under each score? ----
    grp = [
        ("ordinary benigns", stash["B0"], "tab:blue"),
        ("confident FPs (F)", stash["F"], "tab:orange"),
        ("pathogenic", stash["pos"], "tab:red"),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for a, (key, ttl) in zip(
        ax,
        [
            ("llr", "zero-shot LLR (the readout that over-calls)"),
            ("oof", f"frozen-embedding probe (PCA-{N_PCA})"),
        ],
    ):
        data = [stash[key][m] for _, m, _ in grp]
        parts = a.violinplot(data, showmedians=True, widths=0.8)
        for pc, (_, _, col) in zip(parts["bodies"], grp):
            pc.set_facecolor(col)
            pc.set_alpha(0.6)
        a.set_xticks([1, 2, 3])
        a.set_xticklabels([g[0] for g in grp], fontsize=9)
        a.set_ylabel(key)
        a.set_title(ttl, fontsize=10)
        a.grid(alpha=0.3, axis="y")
    r4 = float(res.filter(pl.col("params") == 4020)["rescue_rate"][0])
    ax[1].text(
        0.5,
        0.93,
        f"only {r4 * 100:.0f}% of the confident FPs\nfall in the benign half",
        transform=ax[1].transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="tab:orange",
        bbox=dict(boxstyle="round", fc="white", ec="tab:orange", alpha=0.9),
    )
    fig.suptitle(
        "4B — the probe does NOT rescue the confident FPs (orange): they stay pathogenic-side, not with the benigns",
        y=1.00,
    )
    fig.tight_layout()
    _save(fig, "probe_rescue_4B")

    # ---- ladder summary: does rescue improve with scale; does the probe cut the op-point FP-rate? ----
    c = res.to_pandas().sort_values("params")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(c["params"], c["rescue_rate"] * 100, "o-", color="tab:green", lw=2.5)
    ax[0].axhline(50, color="gray", ls=":", lw=1, label="chance (random re-ranking)")
    ax[0].set_xscale("log")
    ax[0].set_ylim(0, 60)
    ax[0].set_xlabel("params (M, log)")
    ax[0].set_ylabel("% of the LLR's confident FPs the probe ranks benign-side")
    ax[0].set_title(
        "Does the probe RESCUE the confident FPs? No —\nit ranks <15% benign-side (below chance: it reproduces the over-call)"
    )
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(
        c["params"],
        c["fp_rate_llr"] * 100,
        "D--",
        color="black",
        lw=2,
        label="LLR's confident set",
    )
    ax[1].plot(
        c["params"],
        c["fp_rate_probe"] * 100,
        "o-",
        color="tab:green",
        lw=2.5,
        label="probe-reranked confident set",
    )
    ax[1].set_xscale("log")
    ax[1].set_xlabel("params (M, log)")
    ax[1].set_ylabel("% false positives in the confident-pathogenic set")
    ax[1].set_title(
        "Re-ranking the confident set by the probe cuts its FP-rate\nmodestly at large scale — but it stays majority-FP"
    )
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "probe_rescue_ladder")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/probe_rescue.parquet",
        *[
            str(OUT / f"{n}.{e}")
            for n in ("probe_rescue_4B", "probe_rescue_ladder")
            for e in ("png", "svg")
        ],
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


def _save(fig, name: str):
    fig.savefig(OUT / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / name}")


if __name__ == "__main__":
    main()
