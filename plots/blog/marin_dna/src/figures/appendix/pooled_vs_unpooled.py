"""Does pooling shift the per-region learning curve at matched tokens?

For each genomic region (cds, upstream, downstream) we have:
  - one pooled run training on all three regions simultaneously, with mixture
    weights TRAIN_WEIGHTS_POOLED below (single epoch)
  - one unpooled run training on that region alone

Step counts are not directly comparable across runs (the pooled run sees only
a fraction of its tokens from any given region, and batch sizes may differ).
We instead align on `tokens seen of region R`:
  pooled:    tokens_R(step) = step * batch * seq_len * TRAIN_WEIGHTS[R]
  unpooled:  tokens_R(step) = step * batch * seq_len

For each region/task we fit PCHIP curves to loss(tokens_R) and auprc(tokens_R)
and overlay the pooled and unpooled traces. At any tokens_R value:
  - if the loss curves agree -> pooling is loss-equivalent per region-token
  - if the auprc curves agree -> pooling is downstream-equivalent per region-token

Outputs:
  plots/output/blog/marin_dna/appendix/pooled_vs_unpooled_raw.{png,pdf}      scatter only (no interpolation) —
                                                         shows where data was actually measured
  plots/output/blog/marin_dna/appendix/pooled_vs_unpooled_matched.{png,pdf}  PCHIP curves on a common grid restricted
                                                         to the data overlap (no extrapolation)

Usage:
  uv run src/figures/appendix/pooled_vs_unpooled.py
  uv run src/figures/appendix/pooled_vs_unpooled.py --unpooled-loss-key eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.pchip_interp import clean as _clean_xy, interp_on_overlap  # noqa: E402
from utils.savefig import save_figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = (
    ROOT.parents[2]
    / "plots"
    / "output"
    / "blog"
    / "marin_dna"
    / "appendix"
)

WANDB_PROJECT = "eric-czech/marin"
TRAITGYM_PREFIX = "lm_eval/traitgym_mendelian_v2_255"

RUN_IDS: dict[str, str] = {
    "pooled": "dna-bolinas-scaling-v0.5-h1920-p1B-0dc6f4",
    "cds": "dna-bolinas-mix-v0.9-p1B-i1-cds_only-ac67f5",
    "upstream": "dna-bolinas-mix-v0.9-p1B-i2-upstream_only-68544e",
    "downstream": "dna-bolinas-mix-v0.9-p1B-i3-downstream_only-56bf9b",
}

REGION_TASKS: dict[str, tuple[str, ...]] = {
    "cds": ("missense_variant", "splicing", "synonymous_variant"),
    "upstream": ("tss_proximal", "5_prime_UTR_variant"),
    "downstream": ("3_prime_UTR_variant",),
}
ALL_REGIONS = ("cds", "upstream", "downstream")
ALL_TASKS = tuple(t for r in ALL_REGIONS for t in REGION_TASKS[r])
TASK_REGION = {t: r for r, ts in REGION_TASKS.items() for t in ts}

# Mixture weights used in the pooled scaling-v0.5 run (single epoch).
TRAIN_WEIGHTS_POOLED: dict[str, float] = {
    "cds": 0.7319,
    "upstream": 0.2062,
    "downstream": 0.0619,
}

COLORS = {"pooled": "#3B6FB6", "unpooled": "#D1603D"}

FIGURE_WIDTH = 12.0


# ---------------------------------------------------------------- wandb fetch


def _resolve_run(api: wandb.Api, run_id: str):
    return api.run(f"{WANDB_PROJECT}/{run_id}")


def _config_get(cfg: dict, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _tag_int(tags: list[str], key: str, run_name: str) -> int:
    prefix = f"{key}="
    matches = [t[len(prefix) :] for t in tags if t.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"{run_name}: expected exactly one {prefix}... tag, got {matches}"
        )
    return int(matches[0])


def _tokens_per_step(run) -> float:
    total_tokens = _tag_int(run.tags, "tokens", run.name)
    n_steps = _config_get(run.config, "trainer.num_train_steps")
    if n_steps is None:
        raise RuntimeError(f"{run.name}: missing trainer.num_train_steps in config")
    return float(total_tokens) / float(n_steps)


def _history(run, keys: list[str]) -> pd.DataFrame:
    df = run.history(keys=keys, samples=10000)
    if df.empty:
        return df
    return (
        df.dropna(how="all", subset=[k for k in keys if k in df.columns])
        .sort_values("_step")
        .reset_index(drop=True)
    )


def fetch_all() -> dict[str, dict]:
    api = wandb.Api(timeout=300)
    auprc_keys = [f"{TRAITGYM_PREFIX}/{t}/auprc" for t in ALL_TASKS]
    out: dict[str, dict] = {}
    for tag, run_id in RUN_IDS.items():
        print(f"fetching {tag}: {run_id}")
        run = _resolve_run(api, run_id)
        tps = _tokens_per_step(run)
        if tag == "pooled":
            loss_keys = [f"eval/val_{r}/loss" for r in ALL_REGIONS]
        else:
            loss_keys = ["eval/loss", f"eval/val_{tag}/loss"]
        loss_df = _history(run, loss_keys)
        auprc_df = _history(run, auprc_keys)
        out[tag] = {
            "run": run,
            "loss": loss_df,
            "auprc": auprc_df,
            "tokens_per_step": tps,
        }
        print(
            f"  tokens_per_step={tps:,.0f}  loss rows={len(loss_df)}, auprc rows={len(auprc_df)}"
        )
    return out


def _tokens_in_region(
    steps: np.ndarray, tag: str, region: str, data: dict
) -> np.ndarray:
    tps = data[tag]["tokens_per_step"]
    if tag == "pooled":
        weight = TRAIN_WEIGHTS_POOLED[region]
    else:
        weight = 1.0 if tag == region else 0.0
    return np.asarray(steps, dtype=float) * tps * weight


# ---------------------------------------------------------------- per-panel data extraction


def _clean(steps: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _clean_xy(steps, y)


def _x_label(region: str) -> str:
    return f"tokens of {region} seen"


def _panel_series(data: dict, task: str, unpooled_loss_key: str) -> dict:
    region = TASK_REGION[task]
    auprc_key = f"{TRAITGYM_PREFIX}/{task}/auprc"
    u_loss_key = (
        "eval/loss" if unpooled_loss_key == "eval" else f"eval/val_{region}/loss"
    )

    df_pa = data["pooled"]["auprc"]
    pa_x, pa_y = _clean(
        _tokens_in_region(df_pa["_step"].values, "pooled", region, data),
        df_pa[auprc_key].values,
    )

    df_ua = data[region]["auprc"]
    ua_x, ua_y = _clean(
        _tokens_in_region(df_ua["_step"].values, region, region, data),
        df_ua[auprc_key].values,
    )

    df_pl = data["pooled"]["loss"]
    pl_x, pl_y = _clean(
        _tokens_in_region(df_pl["_step"].values, "pooled", region, data),
        df_pl[f"eval/val_{region}/loss"].values,
    )

    df_ul = data[region]["loss"]
    if u_loss_key not in df_ul.columns:
        u_loss_key = "eval/loss"
    ul_x, ul_y = _clean(
        _tokens_in_region(df_ul["_step"].values, region, region, data),
        df_ul[u_loss_key].values,
    )

    return {
        "region": region,
        "pooled_auprc": (pa_x, pa_y),
        "unpooled_auprc": (ua_x, ua_y),
        "pooled_loss": (pl_x, pl_y),
        "unpooled_loss": (ul_x, ul_y),
    }


def _setup_panel(ax, ax2, task: str, region: str) -> None:
    ax.set_xscale("log")
    ax.set_xlabel(_x_label(region))
    ax.set_ylabel(f"auprc ({task})")
    ax2.set_ylabel(f"loss ({region})", color="0.35")
    ax2.tick_params(axis="y", labelcolor="0.35")
    ax.set_title(task)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)


def _shared_legend(fig) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["pooled"],
            lw=1.8,
            label="pooled — auprc (solid, left) / loss (dashed, right)",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["unpooled"],
            lw=1.8,
            label="unpooled — auprc (solid, left) / loss (dashed, right)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )


def _save(fig, name: str) -> None:
    save_figure(fig, FIGURES_DIR, name)


# ---------------------------------------------------------------- plot 1: raw scatter


def plot_raw(data: dict, unpooled_loss_key: str) -> None:
    n_cols = 3
    n_rows = (len(ALL_TASKS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIGURE_WIDTH, 4.0 * n_rows))
    flat = np.atleast_1d(axes).flatten()

    for ai, task in enumerate(ALL_TASKS):
        ax = flat[ai]
        ax2 = ax.twinx()
        ser = _panel_series(data, task, unpooled_loss_key)
        for tag, color in (
            ("pooled_auprc", COLORS["pooled"]),
            ("unpooled_auprc", COLORS["unpooled"]),
        ):
            x, y = ser[tag]
            order = np.argsort(x)
            if len(x) >= 2:
                ax.plot(x[order], y[order], color=color, lw=1.4, alpha=0.9, zorder=3)
            ax.scatter(
                x,
                y,
                color=color,
                s=28,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )
        for tag, color in (
            ("pooled_loss", COLORS["pooled"]),
            ("unpooled_loss", COLORS["unpooled"]),
        ):
            x, y = ser[tag]
            order = np.argsort(x)
            if len(x) >= 2:
                ax2.plot(
                    x[order],
                    y[order],
                    color=color,
                    lw=1.0,
                    ls="--",
                    alpha=0.55,
                    zorder=2,
                )
            ax2.scatter(x, y, color=color, s=14, alpha=0.55, marker="x", zorder=2)
        _setup_panel(ax, ax2, task, ser["region"])

    for k in range(len(ALL_TASKS), len(flat)):
        flat[k].set_visible(False)

    suffix = "eval/loss" if unpooled_loss_key == "eval" else "eval/val_<region>/loss"
    fig.suptitle(
        f"Raw measured points (no interpolation) — auprc (circles, left axis) and loss (× marks, right axis)"
        f"   [unpooled loss key: {suffix}]",
        fontsize=12,
    )
    _shared_legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, "pooled_vs_unpooled_raw")


# ---------------------------------------------------------------- plot 2: matched / interpolated


def plot_matched(data: dict, unpooled_loss_key: str) -> None:
    n_cols = 3
    n_rows = (len(ALL_TASKS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIGURE_WIDTH, 4.0 * n_rows))
    flat = np.atleast_1d(axes).flatten()

    for ai, task in enumerate(ALL_TASKS):
        ax = flat[ai]
        ax2 = ax.twinx()
        ser = _panel_series(data, task, unpooled_loss_key)

        gx, y_p, y_u = interp_on_overlap(*ser["pooled_auprc"], *ser["unpooled_auprc"])
        if gx is not None:
            ax.plot(gx, y_p, color=COLORS["pooled"], lw=2.0, zorder=3)
            ax.plot(gx, y_u, color=COLORS["unpooled"], lw=2.0, zorder=3)
        else:
            ax.text(
                0.5,
                0.5,
                "auprc: insufficient overlap",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=COLORS["unpooled"],
                fontsize=9,
                alpha=0.8,
            )

        gxL, y_pL, y_uL = interp_on_overlap(*ser["pooled_loss"], *ser["unpooled_loss"])
        if gxL is not None:
            ax2.plot(
                gxL, y_pL, color=COLORS["pooled"], lw=1.4, ls="--", alpha=0.7, zorder=2
            )
            ax2.plot(
                gxL,
                y_uL,
                color=COLORS["unpooled"],
                lw=1.4,
                ls="--",
                alpha=0.7,
                zorder=2,
            )

        _setup_panel(ax, ax2, task, ser["region"])

    for k in range(len(ALL_TASKS), len(flat)):
        flat[k].set_visible(False)

    suffix = "eval/loss" if unpooled_loss_key == "eval" else "eval/val_<region>/loss"
    fig.suptitle(
        "Matched comparison — PCHIP curves on common grid, restricted to data overlap (no extrapolation)"
        f"   [unpooled loss key: {suffix}]",
        fontsize=12,
    )
    _shared_legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, "pooled_vs_unpooled_matched")


# ---------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--unpooled-loss-key",
        choices=["val_region", "eval"],
        default="val_region",
        help="loss key to use on the unpooled side (default: same eval/val_<region>/loss as pooled)",
    )
    args = p.parse_args()

    data = fetch_all()
    plot_raw(data, args.unpooled_loss_key)
    plot_matched(data, args.unpooled_loss_key)


if __name__ == "__main__":
    main()
