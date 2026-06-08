"""exp255 (#255) — order-vs-family analysis & figures, all in one file.

The diversity-vs-quantity ablation: five 0.25B region specialists trained on the
**19-order** zoonomia cohort vs exp232's matched **108-family** arms, same region +
compute. This single module generates every figure and table for the #255 Results.

Subcommands:
  uv run python plots/exp255_analysis.py figures   # write the 4 figures to output/exp255_analysis/
  uv run python plots/exp255_analysis.py tables    # print the analysis tables (markdown)
  uv run python plots/exp255_analysis.py all        # both (default)

Figures (output/exp255_analysis/):
  - ll_trajectory.{png,svg}        LL gap + functional vs step, 5 regions
  - auprc_trajectory.{png,svg}     per-step held-out AUPRC, family offline vs order online
  - llgap_vs_auprc.{png,svg}       LL gap vs AUPRC, 8 per-subset panels
  - llfunc_vs_auprc.{png,svg}      LL functional vs AUPRC, 8 per-subset panels

Tables: matched AUPRC (with the paired cluster-bootstrap Δ/CI/p), endpoint LL gap/func,
online-vs-offline BOS sanity, and the per-subset Pearson summary.

Metric notes: "validation" sequences come from the TRAINING set (val loss falls
monotonically) so the LL **gap** is the meaningful within-train signal; AUPRC is the held-
out metric. Family AUPRC = offline (family *online* is pre-#266 BOS-bugged); order AUPRC =
online (post-#266, ≈ offline). The paired AUPRC delta uses the library primitive
``marin_dna.pipelines.evals.metrics.paired_metric_delta_bootstrap``.

Run:  uv run python plots/exp255_analysis.py [figures|tables|all]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb
from marin_dna.pipelines.evals.metrics import paired_metric_delta_bootstrap
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- config
S3_METRICS = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"
S3_SCORES = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
GO, GF = "dna-exp255-v0.1", "dna-exp232-v0.1"  # wandb groups: order, family
ONLINE_KEY = "lm_eval/mendelian_traits_255/{subset}/avg/auprc"
SCORE = "minus_llr_avg"
CAND = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999, 5000]
MIN_STEP = 1000  # drop the ≤500 early-training transient (warmup/relaunch artifact)
KEY = ["chrom", "pos", "ref", "alt"]
BASELINE = 0.10  # 1:9 prevalence
COL_FAMILY = "#8c98a4"  # 108-sp family — muted slate (baseline)
COL_ORDER = "#1f6fb2"  # 19-sp order — confident blue
COL_ACCENT = "#d1495b"  # family-edge highlight
OUT = Path(__file__).parent / "output" / "exp255_analysis"

# region -> (val recipe, order wandb sub, family wandb sub, offline region token, matched subsets)
REGIONS: dict[str, tuple] = {
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
# flattened (subset, region) panels, in display order
PANELS = [(s, r) for r, (_, _, _, _, subs) in REGIONS.items() for s in subs]
ALL_SUBSETS = [s for s, _ in PANELS]  # the 8 distinct matched subsets


def order_model(off: str) -> str:
    return f"exp255-v4_{off}_order-step-4999"


def family_model(off: str) -> str:
    return f"exp232-v4_{off}-step-4999"


# --------------------------------------------------------------------------- style
def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.labelcolor": "#222",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#5a5a5a",
            "axes.linewidth": 1.0,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": "#e2e2e2",
            "grid.linewidth": 0.8,
            "xtick.color": "#444",
            "ytick.color": "#444",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    return OUT / f"{name}.png"


def _short(s: str) -> str:
    return (
        s.replace("_variant", "")
        .replace("non_coding_transcript_exon", "ncRNA")
        .replace("3_prime_UTR", "3′UTR")
        .replace("5_prime_UTR", "5′UTR")
    )


# --------------------------------------------------------------------------- data loaders (cached)
_ll_cache: dict[tuple[str, str], pd.DataFrame] = {}
_metrics_cache: dict[str, dict | None] = {}
_scores_cache: dict[str, pl.DataFrame] = {}
_online_cache: dict[str, dict] = {}


def ll_traj(api: wandb.Api, grp: str, sub: str, rec: str) -> pd.DataFrame:
    """Stitched val-loss trajectory for `rec` across the group's runs (order arms are
    split across relaunch runs). Columns: _step, gap, func, nonfunc (LL = −loss)."""
    if (grp, sub) not in _ll_cache:
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
        df["func"], df["nonfunc"], df["gap"] = -df[fk], -df[nk], df[nk] - df[fk]
        _ll_cache[(grp, sub)] = df
    return _ll_cache[(grp, sub)]


def read_metrics(model: str) -> dict | None:
    """{subset: (auprc, se)} for minus_llr_avg at `model`'s metrics parquet, or None."""
    if model not in _metrics_cache:
        try:
            df = pl.read_parquet(
                f"{S3_METRICS}/{model}/mendelian_traits.parquet"
            ).filter(pl.col("score_type") == SCORE)
            _metrics_cache[model] = {
                r["subset"]: (r["value"], r["se"]) for r in df.to_dicts()
            }
        except Exception:
            _metrics_cache[model] = None
    return _metrics_cache[model]


def fam_auprc(off: str, subset: str) -> dict[int, float]:
    """Family offline AUPRC trajectory {step: auprc} (the evals_v2 checkpoint sweep)."""
    out = {}
    for n in CAND:
        df = read_metrics(f"exp232-v4_{off}-step-{n}")
        if df and subset in df:
            out[n] = df[subset][0]
    return out


def ord_auprc(api: wandb.Api, sub: str, subset: str) -> dict[int, float]:
    """Order online AUPRC trajectory {step: auprc} (in-training lm_eval, post-#266).

    Pulls the arm's full 8-subset history ONCE (cached per arm), since the
    online-vs-offline table queries every subset for every arm.
    """
    if sub not in _online_cache:
        keys = [ONLINE_KEY.format(subset=s) for s in ALL_SUBSETS]
        frames = [
            h
            for r in api.runs("marin", filters={"group": GO})
            if sub in r.name
            for h in [r.history(keys=keys, samples=10000, pandas=True)]
            if len(h)
        ]
        out: dict[str, dict[int, float]] = {s: {} for s in ALL_SUBSETS}
        if frames:
            df = pd.concat(frames).drop_duplicates("_step").sort_values("_step")
            for s in ALL_SUBSETS:
                k = ONLINE_KEY.format(subset=s)
                if k in df.columns:
                    d = df.dropna(subset=[k])
                    out[s] = dict(zip(d["_step"].astype(int), d[k]))
        _online_cache[sub] = out
    return _online_cache[sub].get(subset, {})


def load_scores(model: str) -> pl.DataFrame:
    """Per-variant scores parquet with minus_llr_avg derived from the llr_* atoms."""
    if model not in _scores_cache:
        df = pl.read_parquet(f"{S3_SCORES}/{model}/mendelian_traits.parquet")
        _scores_cache[model] = df.with_columns(
            (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("score")
        )
    return _scores_cache[model]


# --------------------------------------------------------------------------- figures
def fig_ll_trajectory(api: wandb.Api) -> None:
    rows = [("gap", "LL gap (nats)"), ("func", "LL functional (nats, = −loss)")]
    arms = list(REGIONS)
    fig, axes = plt.subplots(
        len(rows), len(arms), figsize=(3.0 * len(arms), 7.0), squeeze=False
    )
    for ri, (key, ylab) in enumerate(rows):
        for ci, arm in enumerate(arms):
            rec, so, sf, _off, _ = REGIONS[arm]
            ax = axes[ri][ci]
            for coh, color, sub, grp in (
                ("family", COL_FAMILY, sf, GF),
                ("order", COL_ORDER, so, GO),
            ):
                d = ll_traj(api, grp, sub, rec)
                d = d[d["_step"] >= MIN_STEP]
                ax.plot(
                    d["_step"],
                    d[key],
                    marker="o",
                    ms=4,
                    lw=1.8,
                    color=color,
                    label=f"{coh} ({'108' if coh == 'family' else '19'} sp.)",
                )
            if ri == 0:
                ax.set_title(f"{arm}  ({rec})", fontsize=11)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=10)
            ax.set_xlabel("training step", fontsize=9)
            ax.grid(alpha=0.3)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(
        h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, fontsize=11
    )
    fig.suptitle(
        "exp255 (#255) — matched-region LL trajectories: family (baseline) vs order\n"
        "LL = −loss; single run per arm (trajectory spread = the uncertainty cue); from step 1000 (≤500 transient excluded)",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    print("wrote", _save(fig, "ll_trajectory"))


def fig_auprc_trajectory(api: wandb.Api) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4), squeeze=False)
    seen = False
    for k, (subset, region) in enumerate(PANELS):
        _rec, so, _sf, off, _ = REGIONS[region]
        ax = axes[k // 4][k % 4]
        fs = fam_auprc(off, subset)
        fsteps = [s for s in sorted(fs) if s >= MIN_STEP]
        fv = [fs[s] for s in fsteps]
        fe = [read_metrics(f"exp232-v4_{off}-step-{s}")[subset][1] for s in fsteps]
        os_ = ord_auprc(api, so, subset)
        osteps = [s for s in sorted(os_) if s >= MIN_STEP]
        h1 = ax.errorbar(
            fsteps,
            fv,
            yerr=fe,
            color=COL_FAMILY,
            marker="o",
            ms=5,
            lw=1.8,
            capsize=0,
            elinewidth=1.2,
            label="family (108 sp.) — offline, ±1 SE",
        )
        (h2,) = ax.plot(
            osteps,
            [os_[s] for s in osteps],
            color=COL_ORDER,
            marker=".",
            ms=7,
            lw=1.8,
            label="order (19 sp.) — online",
        )
        ax.axhline(BASELINE, ls=":", color="gray", lw=0.9)
        ax.set_title(f"{_short(subset)}  ({region})", fontsize=10)
        ax.grid(alpha=0.3)
        if k % 4 == 0:
            ax.set_ylabel("matched AUPRC")
        if k // 4 == 1:
            ax.set_xlabel("training step")
        if not seen:
            fig.legend(
                handles=[h1, h2],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.925),
                fontsize=10.5,
                ncol=2,
                frameon=False,
            )
            seen = True
    fig.suptitle(
        "exp255 (#255) — matched AUPRC vs training step: family (108 sp.) vs order (19 sp.)\n"
        "0.25B; family = offline (±1 SE), order = online; dotted = 0.10 prevalence baseline",
        fontsize=10,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    print("wrote", _save(fig, "auprc_trajectory"))


def _llva_points(api: wandb.Api) -> dict:
    """panel -> cohort -> [(step, gap, func, auprc)] over checkpoints ≥ MIN_STEP."""
    llt = {}
    for region, (rec, so, sf, _off, _subs) in REGIONS.items():
        llt[(region, "family")] = ll_traj(api, GF, sf, rec)
        llt[(region, "order")] = ll_traj(api, GO, so, rec)
    pdata: dict = {}
    for subset, region in PANELS:
        _rec, so, _sf, off, _ = REGIONS[region]
        d = {}
        for coh, auf in (
            ("family", fam_auprc(off, subset)),
            ("order", ord_auprc(api, so, subset)),
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
    return pdata


def _pearson(pdata: dict, idx: int) -> dict:
    """{panel: Pearson r} between the LL column (idx 1=gap, 2=func) and AUPRC, pooled cohorts."""
    rs = {}
    for panel in PANELS:
        xs = [row[idx] for coh in ("family", "order") for row in pdata[panel][coh]]
        ys = [row[3] for coh in ("family", "order") for row in pdata[panel][coh]]
        rs[panel] = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")
    return rs


def _llva_figure(pdata: dict, idx: int, xlabel: str, mlabel: str, name: str) -> None:
    rs = _pearson(pdata, idx)
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8))
    sc = None
    for k, panel in enumerate(PANELS):
        subset, region = panel
        ax = axes[k // 4][k % 4]
        for coh, mk in (("family", "o"), ("order", "s")):
            rows = pdata[panel][coh]  # (step, gap, func, auprc)
            xs = [r[idx] for r in rows]
            ys = [r[3] for r in rows]
            ax.plot(xs, ys, "-", color="#cfcfcf", lw=0.8, zorder=1)
            sc = ax.scatter(
                xs,
                ys,
                c=[r[0] for r in rows],
                cmap="viridis",
                vmin=MIN_STEP,
                vmax=5000,
                marker=mk,
                s=44,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        ax.set_title(f"{_short(subset)}  ({region})", fontsize=10.5)
        ax.text(
            0.05,
            0.95,
            f"r = {rs[panel]:+.2f}",
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
    print("wrote", _save(fig, name))


def fig_ll_vs_auprc(api: wandb.Api) -> None:
    pdata = _llva_points(api)
    _llva_figure(pdata, 1, "train-set LL gap (nats)", "LL gap", "llgap_vs_auprc")
    _llva_figure(
        pdata,
        2,
        "train-set LL functional (nats, = −loss)",
        "LL functional",
        "llfunc_vs_auprc",
    )


# --------------------------------------------------------------------------- tables
def _md(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def table_matched_auprc() -> None:
    """Matched AUPRC: family/order levels (±1 SE) + paired cluster-bootstrap Δ / 95% CI / p."""
    rows = []
    for region, (_rec, _so, _sf, off, subs) in REGIONS.items():
        o = load_scores(order_model(off)).select(
            [*KEY, "label", "subset", "match_group", "score"]
        )
        f = (
            load_scores(family_model(off))
            .select([*KEY, "score"])
            .rename({"score": "score_fam"})
        )
        m = o.join(f, on=KEY, how="inner")
        assert len(m) == len(o) == len(f), f"{region}: join not 1:1"
        fam_m, ord_m = read_metrics(family_model(off)), read_metrics(order_model(off))
        for ss in subs:
            sub = m.filter(pl.col("subset") == ss)
            r = paired_metric_delta_bootstrap(
                label=sub["label"].to_pandas(),
                score_a=sub["score"].to_pandas(),
                score_b=sub["score_fam"].to_pandas(),
                match_group=sub["match_group"].to_pandas(),
            )
            fv, fe = fam_m[ss]
            ov, oe = ord_m[ss]
            sig = " ✓" if (r["ci_low"] > 0) == (r["ci_high"] > 0) else ""
            rows.append(
                [
                    region,
                    _short(ss),
                    f"{fv:.3f} ± {fe:.3f}",
                    f"{ov:.3f} ± {oe:.3f}",
                    f"{r['delta']:+.3f}",
                    f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]",
                    f"{r['p_two_sided']:.3f}{sig}",
                ]
            )
    print("\n### Matched Mendelian AUPRC — paired cluster bootstrap @ step-4999\n")
    print(
        "Levels ±1 SE (per-model cluster bootstrap); Δ / CI / p paired (shared match_groups). Baseline 0.10.\n"
    )
    print(
        _md(
            [
                "arm",
                "subset",
                "family (108 sp.)",
                "order (19 sp.)",
                "Δ (o−f)",
                "95% CI",
                "p",
            ],
            rows,
        )
    )


def table_ll(api: wandb.Api) -> None:
    """Endpoint LL gap + functional, family vs order, per arm (single-run, no SE)."""
    rows = []
    for region, (rec, so, sf, _off, _) in REGIONS.items():
        fd = ll_traj(api, GF, sf, rec).iloc[-1]
        od = ll_traj(api, GO, so, rec).iloc[-1]
        rows.append(
            [
                region,
                rec,
                f"{fd.gap:+.3f}",
                f"{od.gap:+.3f}",
                f"{od.gap - fd.gap:+.3f}",
                f"{fd.func:+.3f}",
                f"{od.func:+.3f}",
                f"{od.func - fd.func:+.3f}",
            ]
        )
    print(
        "\n### Endpoint LL (val=train; single run per arm, no SE — suggestive context)\n"
    )
    print(
        _md(
            [
                "arm",
                "val set",
                "gap fam",
                "gap ord",
                "Δgap",
                "func fam",
                "func ord",
                "Δfunc",
            ],
            rows,
        )
    )


def table_online_offline(api: wandb.Api) -> None:
    """Order arms: online (in-training) vs offline AUPRC across all 8 subsets (post-#266 BOS sanity)."""
    all_subsets = [s for _, (_, _, _, _, subs) in REGIONS.items() for s in subs]
    rows, deltas = [], []
    for region, (_rec, so, _sf, off, _) in REGIONS.items():
        off_m = read_metrics(order_model(off))
        for ss in all_subsets:
            on = ord_auprc(api, so, ss)
            onv = (
                on.get(4999)
                or on.get(5000)
                or (sorted(on.items())[-1][1] if on else None)
            )
            offv = off_m.get(ss, (None,))[0]
            if onv is None or offv is None:
                continue
            deltas.append(abs(offv - onv))
            if region in ("cds", "ccre"):  # representative arms, keep the table compact
                rows.append(
                    [
                        f"{region}_order",
                        _short(ss),
                        f"{onv:.3f}",
                        f"{offv:.3f}",
                        f"{offv - onv:+.3f}",
                    ]
                )
    print("\n### Online vs offline AUPRC — order arms (post-#266 BOS sanity)\n")
    print(
        f"mean |Δ| across all order arm × subset = {np.mean(deltas):.4f} → online reproduces offline.\n"
    )
    print(_md(["arm", "subset", "online", "offline", "Δ (off−on)"], rows))


def table_pearson(api: wandb.Api) -> None:
    pdata = _llva_points(api)
    rg, rf = _pearson(pdata, 1), _pearson(pdata, 2)
    rows = [
        [
            f"{_short(s)} ({region})",
            f"{rg[(s, region)]:+.2f}",
            f"{rf[(s, region)]:+.2f}",
        ]
        for s, region in PANELS
    ]
    rows.append(
        [
            "**MEAN across 8 subsets**",
            f"**{np.nanmean(list(rg.values())):+.2f}**",
            f"**{np.nanmean(list(rf.values())):+.2f}**",
        ]
    )
    print("\n### LL → AUPRC tracking — per-subset Pearson (all checkpoints)\n")
    print(_md(["subset (region)", "r(LL gap)", "r(LL func)"], rows))


# --------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description="exp255 order-vs-family analysis & figures")
    p.add_argument(
        "mode", nargs="?", default="all", choices=["figures", "tables", "all"]
    )
    args = p.parse_args()
    set_style()
    api = wandb.Api()
    if args.mode in ("figures", "all"):
        fig_ll_trajectory(api)
        fig_auprc_trajectory(api)
        fig_ll_vs_auprc(api)
    if args.mode in ("tables", "all"):
        table_matched_auprc()
        table_ll(api)
        table_online_offline(api)
        table_pearson(api)


if __name__ == "__main__":
    main()
