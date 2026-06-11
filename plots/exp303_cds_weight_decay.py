"""exp303 (#303) — CDS weight-decay sweep: AUPRC + LL trajectories (CDS-relevant only).

Compares the ``v4_cds`` 0.25B specialist at ``weight_decay`` ∈ {0.1, 0.3, 0.5} over the
full 10-checkpoint trajectory (steps 500..4999). WD=0.1 is exp232's matched arm (no
re-run); 0.3/0.5 are exp303. All three are on the SAME step grid and all BOS-faithful:

  * WD=0.1 — exp232 **offline** evals_v2 (``minus_llr_avg``, ±1 SE). exp232's *online*
    metric is pre-#266 BOS-bugged, so we use offline — exactly as plots/exp255_analysis.py
    does for the family baseline.
  * WD=0.3/0.5 — exp303 **online** in-training lm_eval. The #266 BOS fix landed before
    exp303 trained, so its online AUPRC prepends BOS = the offline protocol (verified:
    the WD=0.1 online↔offline match is <0.005 on every CDS subset). Same online-new /
    offline-baseline split as exp255.

CDS-relevant subsets only: missense / synonymous / splicing (+ the val_cds LL trajectory).
Single seed per arm, no replicates — present trajectories, not significance claims.
weight_decay is a quantitative (continuous viridis) seaborn hue; seaborn orders it.

Figures (output/exp303_cds_weight_decay/):
  - auprc_trajectory.{png,svg}   missense/synonymous/splicing AUPRC vs step, 3 WD curves
  - ll_trajectory.{png,svg}      val_cds LL gap + functional vs step, 3 WD curves

Run:  uv run python plots/exp303_cds_weight_decay.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import wandb

# --------------------------------------------------------------------------- config
S3_METRICS = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"
ONLINE_KEY = "lm_eval/mendelian_traits_255/{subset}/avg/auprc"
SCORE = "minus_llr_avg"
CAND = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999, 5000]
BASELINE = 0.10  # 1:9 prevalence
CDS_SUBSETS = ["missense_variant", "synonymous_variant", "splicing"]
N_VARIANTS = {"missense_variant": 580, "synonymous_variant": 46, "splicing": 319}
VAL_REC = "val_cds"

# Quantitative color: weight_decay through viridis as a continuous hue. HUE_NORM is
# shared by every seaborn call (so colors are identical across panels/figures) and by
# the offline SE band; vmax padded past 0.5 to keep the top arm off the pale extreme.
WD_CMAP = "viridis"
HUE_NORM = (0.0, 0.65)


def wd_color(wd: float):
    return mpl.colormaps[WD_CMAP](mpl.colors.Normalize(*HUE_NORM)(wd))


# wd -> auprc source. ("offline", model_prefix) or ("online", wandb_group, name_token)
SRC = {
    0.1: ("offline", "exp232-v4_cds"),
    0.3: ("online", "dna-exp303-v0.1", "wd0p3"),
    0.5: ("online", "dna-exp303-v0.1", "wd0p5"),
}
# val_cds LL trajectory lives in each arm's own wandb group/run
LL_SRC = {
    0.1: ("dna-exp232-v0.1", "v4_cds-v0.1"),  # the lone exp232 cds arm
    0.3: ("dna-exp303-v0.1", "wd0p3"),
    0.5: ("dna-exp303-v0.1", "wd0p5"),
}
OUT = Path(__file__).parent / "output" / "exp303_cds_weight_decay"


# --------------------------------------------------------------------------- style
def set_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font="DejaVu Sans",
        font_scale=1.05,
        rc={
            "figure.dpi": 130,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#5a5a5a",
            "grid.color": "#e6e6e6",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        },
    )


def _save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    return OUT / f"{name}.png"


def _short(s: str) -> str:
    return s.replace("_variant", "")


# --------------------------------------------------------------------------- loaders
_metrics_cache: dict[str, dict | None] = {}
_online_cache: dict[str, dict] = {}
_ll_cache: dict[tuple[str, str], pd.DataFrame] = {}


def read_metrics(model: str) -> dict | None:
    """{subset: (auprc, se)} for minus_llr_avg at `model`'s offline parquet, or None."""
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


def offline_auprc(prefix: str, subset: str) -> dict[int, tuple[float, float]]:
    """Offline {step: (auprc, se)} across the evals_v2 checkpoint sweep."""
    out: dict[int, tuple[float, float]] = {}
    for n in CAND:
        m = read_metrics(f"{prefix}-step-{n}")
        if m and subset in m:
            out[n] = m[subset]
    return out


def online_auprc(
    api: wandb.Api, group: str, token: str, subset: str
) -> dict[int, float]:
    """Online {step: auprc} from the in-training lm_eval history (post-#266, BOS-faithful).

    Pulls the run's full CDS-subset history once (cached per token); dedups
    relaunch-overlap steps keeping the last logged value.
    """
    if token not in _online_cache:
        keys = [ONLINE_KEY.format(subset=s) for s in CDS_SUBSETS]
        frames = [
            h
            for r in api.runs("marin", filters={"group": group})
            if token in r.name
            for h in [r.history(keys=keys, samples=10000, pandas=True)]
            if len(h)
        ]
        out: dict[str, dict[int, float]] = {s: {} for s in CDS_SUBSETS}
        if frames:
            df = (
                pd.concat(frames)
                .sort_values("_step")
                .drop_duplicates("_step", keep="last")
            )
            for s in CDS_SUBSETS:
                k = ONLINE_KEY.format(subset=s)
                if k in df.columns:
                    d = df.dropna(subset=[k])
                    out[s] = dict(zip(d["_step"].astype(int), d[k]))
        _online_cache[token] = out
    return _online_cache[token].get(subset, {})


def arm_auprc(
    api: wandb.Api, wd: float, subset: str
) -> tuple[list[int], list[float], list[float]]:
    """(steps, auprc, se) for an arm/subset. SE is 0 for online arms (no per-step SE)."""
    src = SRC[wd]
    if src[0] == "offline":
        d = offline_auprc(src[1], subset)
        steps = sorted(d)
        return steps, [d[s][0] for s in steps], [d[s][1] for s in steps]
    d = online_auprc(api, src[1], src[2], subset)
    steps = sorted(d)
    return steps, [d[s] for s in steps], [0.0] * len(steps)


def ll_traj(api: wandb.Api, wd: float) -> pd.DataFrame:
    """val_cds LL trajectory for the arm: columns _step, gap, func (LL = −loss)."""
    grp, token = LL_SRC[wd]
    if (grp, token) not in _ll_cache:
        fk, nk = f"eval/{VAL_REC}_functional/loss", f"eval/{VAL_REC}_nonfunctional/loss"
        frames = [
            r.history(keys=[fk, nk], samples=10000, pandas=True)
            for r in api.runs("marin", filters={"group": grp})
            if token in r.name
        ]
        df = (
            pd.concat([f for f in frames if len(f)])
            .dropna(subset=[fk, nk])
            .sort_values("_step")
            .drop_duplicates("_step", keep="last")
        )
        df["func"], df["gap"] = -df[fk], df[nk] - df[fk]
        _ll_cache[(grp, token)] = df
    return _ll_cache[(grp, token)]


# --------------------------------------------------------------------------- figures
def fig_auprc_trajectory(api: wandb.Api) -> None:
    rows = []
    for wd in SRC:
        for subset in CDS_SUBSETS:
            steps, vals, ses = arm_auprc(api, wd, subset)
            for st, v, se in zip(steps, vals, ses):
                rows.append(
                    {
                        "step": st,
                        "AUPRC": v,
                        "se": se,
                        "weight_decay": wd,
                        "subset": _short(subset),
                    }
                )
    df = pd.DataFrame(rows)
    order = [_short(s) for s in CDS_SUBSETS]

    g = sns.relplot(
        data=df,
        x="step",
        y="AUPRC",
        hue="weight_decay",
        hue_norm=HUE_NORM,
        palette=WD_CMAP,
        col="subset",
        col_order=order,
        kind="line",
        marker="o",
        markersize=7,
        height=4.0,
        aspect=1.15,
        facet_kws={"sharey": False},
    )
    for subset in CDS_SUBSETS:
        ax = g.axes_dict[_short(subset)]
        # ±1 SE band for the offline baseline only (online arms have no per-step SE)
        b = df[
            (df["subset"] == _short(subset))
            & (df["weight_decay"] == 0.1)
            & (df["se"] > 0)
        ].sort_values("step")
        ax.fill_between(
            b["step"],
            b["AUPRC"] - b["se"],
            b["AUPRC"] + b["se"],
            color=wd_color(0.1),
            alpha=0.18,
            lw=0,
        )
        ax.axhline(BASELINE, ls=":", color="gray", lw=0.9)
        ax.set_title(f"{_short(subset)}  (n = {N_VARIANTS[subset]})")
    g.set_axis_labels("training step", "Mendelian AUPRC (minus_llr_avg)")
    g.tight_layout()
    print("wrote", _save(g.figure, "auprc_trajectory"))


def fig_ll_trajectory(api: wandb.Api) -> None:
    rows = []
    for wd in SRC:
        d = ll_traj(api, wd)
        for _, r in d.iterrows():
            rows.append(
                {
                    "step": int(r["_step"]),
                    "weight_decay": wd,
                    "metric": "LL gap",
                    "value": r["gap"],
                }
            )
            rows.append(
                {
                    "step": int(r["_step"]),
                    "weight_decay": wd,
                    "metric": "LL functional",
                    "value": r["func"],
                }
            )
    df = pd.DataFrame(rows)

    g = sns.relplot(
        data=df,
        x="step",
        y="value",
        hue="weight_decay",
        hue_norm=HUE_NORM,
        palette=WD_CMAP,
        col="metric",
        col_order=["LL gap", "LL functional"],
        kind="line",
        marker="o",
        markersize=6,
        height=4.2,
        aspect=1.2,
        facet_kws={"sharey": False},
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("training step", "val_cds log-likelihood (nats, = −loss)")
    g.tight_layout()
    print("wrote", _save(g.figure, "ll_trajectory"))


def table_endpoints(api: wandb.Api) -> None:
    print(
        "\n### CDS AUPRC — peak / endpoint / late-decline, per WD (same 500..4999 grid; single seed, no replicates)\n"
    )
    print("| WD (source) | subset | n | peak | @step | endpoint | peak−end | end SE |")
    print("|---|---|--:|--:|--:|--:|--:|--:|")
    for wd, src in SRC.items():
        tag = src[0]
        for subset in CDS_SUBSETS:
            steps, vals, ses = arm_auprc(api, wd, subset)
            if not steps:
                continue
            pk_i = int(np.argmax(vals))
            se = f"±{ses[-1]:.3f}" if ses[-1] else "—"
            print(
                f"| {wd} ({tag}) | {_short(subset)} | {N_VARIANTS[subset]} | "
                f"{vals[pk_i]:.3f} | {steps[pk_i]} | {vals[-1]:.3f} | "
                f"{vals[pk_i] - vals[-1]:+.3f} | {se} |"
            )


def main() -> None:
    set_style()
    api = wandb.Api()
    fig_auprc_trajectory(api)
    fig_ll_trajectory(api)
    table_endpoints(api)


if __name__ == "__main__":
    main()
