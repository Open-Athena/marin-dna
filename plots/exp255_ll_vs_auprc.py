"""Train-set LL vs held-out AUPRC, per matched subset, across all checkpoints — exp255 (#255).

Emits TWO 8-panel figures (one panel per matched subset), identical style:
  - llgap_vs_auprc.png   — x = train-set LL gap (func − nonfunc)
  - llfunc_vs_auprc.png  — x = train-set LL functional (= −loss)
Each panel uses ONLY the region-specialist's checkpoints (the cds model for
missense/syn/splice, the utr3 model for 3'UTR, …), family (○) and order (□), across
training (steps ≥1000; the ≤500 transient excluded), colored by step, with a faint path
through each cohort's trajectory; pooled Pearson annotated per panel.

Metric note: our "validation" sequences come from the TRAINING set (val loss falls
monotonically), so the LL **gap** is the meaningful conservation signal; AUPRC is the
genuinely held-out metric. Family AUPRC = offline; order AUPRC = online (post-#266, ≈
offline; family *online* is pre-#266 BOS-bugged). LL from val loss (LL = −loss).

Run:  uv run python plots/exp255_ll_vs_auprc.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt
import wandb
from _exp255_style import save, set_style
from matplotlib.lines import Line2D

S3 = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"
GO, GF = "dna-exp255-v0.1", "dna-exp232-v0.1"
ONLINE_KEY = "lm_eval/mendelian_traits_255/{subset}/avg/auprc"
CAND = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999, 5000]
MIN_STEP = 1000  # drop the ≤500 early-training transient (warmup/relaunch artifact)
STEM = "exp255_ll_vs_auprc"
# region -> (val recipe, order sub, family sub, offline region token, matched subsets)
REGIONS = {
    "cds": (
        "val_cds",
        "v4_cds_order",
        "v4_cds",
        "cds",
        ["missense_variant", "synonymous_variant", "splicing"],
    ),
    "utr3": ("val_utr3", "v4_utr3_order", "v4_utr3", "utr3", ["3_prime_UTR_variant"]),
    "ncrna": (
        "val_ncrna",
        "v4_ncrna_exon_order",
        "v4_ncrna_exon",
        "ncrna_exon",
        ["non_coding_transcript_exon_variant"],
    ),
    "tss": (
        "val_tss_pc",
        "v4_tss_region_and_utr5_order",
        "v4_tss_region_and_utr5",
        "tss_region_and_utr5",
        ["5_prime_UTR_variant", "tss_proximal"],
    ),
    "ccre": (
        "val_enhancer",
        "v4_ccre_non_promoter_order",
        "v4_ccre_non_promoter",
        "ccre_non_promoter",
        ["distal"],
    ),
}
PANELS = [(s, r) for r, (_, _, _, _, subs) in REGIONS.items() for s in subs]
_mcache: dict[str, dict | None] = {}


def _ll_traj(api: wandb.Api, grp: str, sub: str, rec: str) -> pd.DataFrame:
    fk, nk = f"eval/{rec}_functional/loss", f"eval/{rec}_nonfunctional/loss"
    fr = [
        r.history(keys=[fk, nk], samples=10000, pandas=True)
        for r in api.runs("marin", filters={"group": grp})
        if sub in r.name
    ]
    df = (
        pd.concat([f for f in fr if len(f)])
        .dropna(subset=[fk, nk])
        .drop_duplicates("_step")
        .sort_values("_step")
    )
    df["gap"], df["func"] = df[nk] - df[fk], -df[fk]
    return df


def _read(model: str) -> dict | None:
    if model not in _mcache:
        try:
            df = pl.read_parquet(f"{S3}/{model}/mendelian_traits.parquet").filter(
                pl.col("score_type") == "minus_llr_avg"
            )
            _mcache[model] = {r["subset"]: r["value"] for r in df.to_dicts()}
        except Exception:
            _mcache[model] = None
    return _mcache[model]


def _fam_auprc(off: str, subset: str) -> dict[int, float]:
    out = {}
    for n in CAND:
        m = _read(f"exp232-v4_{off}-step-{n}")
        if m and subset in m:
            out[n] = m[subset]
    return out


def _ord_auprc(api: wandb.Api, sub: str, subset: str) -> dict[int, float]:
    key = ONLINE_KEY.format(subset=subset)
    fr = [
        h
        for r in api.runs("marin", filters={"group": GO})
        if sub in r.name
        for h in [r.history(keys=[key], samples=10000, pandas=True)]
        if len(h) and key in h.columns
    ]
    if not fr:
        return {}
    df = (
        pd.concat(fr).dropna(subset=[key]).drop_duplicates("_step").sort_values("_step")
    )
    return dict(zip(df["_step"].astype(int), df[key]))


def _short(s: str) -> str:
    return (
        s.replace("_variant", "")
        .replace("non_coding_transcript_exon", "ncRNA")
        .replace("3_prime_UTR", "3′UTR")
        .replace("5_prime_UTR", "5′UTR")
    )


def _figure(pdata: dict, key: str, xlabel: str, mlabel: str, name: str) -> dict:
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8))
    smin, smax = MIN_STEP, 5000
    sc, rs = None, {}
    for k, panel in enumerate(PANELS):
        subset, region = panel
        ax = axes[k // 4][k % 4]
        allx, ally = [], []
        for coh, mk in (("family", "o"), ("order", "s")):
            rows = pdata[panel][coh]  # list of (step, gap, func, auprc)
            steps = [r[0] for r in rows]
            xs = [r[1] if key == "gap" else r[2] for r in rows]
            ys = [r[3] for r in rows]
            ax.plot(xs, ys, "-", color="#cfcfcf", lw=0.8, zorder=1)
            sc = ax.scatter(
                xs,
                ys,
                c=steps,
                cmap="viridis",
                vmin=smin,
                vmax=smax,
                marker=mk,
                s=44,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
            allx += xs
            ally += ys
        r = np.corrcoef(allx, ally)[0, 1] if len(allx) > 2 else float("nan")
        rs[panel] = r
        ax.set_title(f"{_short(subset)}  ({region})", fontsize=10.5)
        ax.text(
            0.05,
            0.95,
            f"r = {r:+.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="#bbb", alpha=0.85, pad=2),
        )
        if k % 4 == 0:
            ax.set_ylabel("held-out AUPRC")
        if k // 4 == 1:
            ax.set_xlabel(xlabel)
    fig.subplots_adjust(
        left=0.055, right=0.905, top=0.875, bottom=0.075, wspace=0.27, hspace=0.34
    )
    cax = fig.add_axes((0.925, 0.12, 0.014, 0.64))
    fig.colorbar(sc, cax=cax).set_label("training step")
    leg = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#5a5a5a",
            ls="",
            label="family (108 sp.) — offline",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="#5a5a5a",
            ls="",
            label="order (19 sp.) — online",
        ),
    ]
    fig.legend(
        handles=leg,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        fontsize=11,
    )
    meanr = float(np.nanmean(list(rs.values())))
    fig.suptitle(
        f"exp255 (#255) — train-set {mlabel} vs held-out AUPRC, per matched subset "
        f"(region specialist only)  ·  mean per-subset r = {meanr:+.2f}",
        fontsize=12,
        y=0.99,
    )
    print("wrote", save(fig, STEM, name))
    return rs


def main() -> None:
    set_style()
    api = wandb.Api()
    llt = {}
    for region, (rec, so, sf, _off, _subs) in REGIONS.items():
        llt[(region, "family")] = _ll_traj(api, GF, sf, rec)
        llt[(region, "order")] = _ll_traj(api, GO, so, rec)
    pdata: dict = {}
    for subset, region in PANELS:
        _rec, so, _sf, off, _ = REGIONS[region]
        d = {}
        for coh, auf in (
            ("family", _fam_auprc(off, subset)),
            ("order", _ord_auprc(api, so, subset)),
        ):
            tr = llt[(region, coh)]
            d[coh] = [
                (
                    s,
                    float(np.interp(s, tr._step, tr.gap)),
                    float(np.interp(s, tr._step, tr.func)),
                    auf[s],
                )
                for s in sorted(x for x in auf if x >= MIN_STEP)
            ]
        pdata[(subset, region)] = d

    rg = _figure(pdata, "gap", "train-set LL gap (nats)", "LL gap", "llgap_vs_auprc")
    rf = _figure(
        pdata,
        "func",
        "train-set LL functional (nats, = −loss)",
        "LL functional",
        "llfunc_vs_auprc",
    )
    print(f"\n{'subset (region)':30}{'r(LL gap)':>11}{'r(LL func)':>11}")
    for panel in PANELS:
        s, region = panel
        print(
            f"{_short(s) + ' (' + region + ')':30}{rg[panel]:>+11.2f}{rf[panel]:>+11.2f}"
        )
    print(
        f"{'MEAN across 8 subsets':30}{np.nanmean(list(rg.values())):>+11.2f}{np.nanmean(list(rf.values())):>+11.2f}"
    )


if __name__ == "__main__":
    main()
