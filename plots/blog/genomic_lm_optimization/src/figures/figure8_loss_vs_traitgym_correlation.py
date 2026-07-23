"""Figure 8: per-model Spearman correlation between loss and VEP AUPRC across training steps.

For each of the 8 parameter-scaling runs (v0.5), we fetch the full training
history of `eval/loss` and `lm_eval/traitgym_mendelian_v2_255/<trait>/auprc`
for six variant types. Series are logged at potentially different cadences,
so we align them with the same PCHIP-on-log10(step) interpolation used by
`pooled_vs_unpooled.py` (see `src/utils/pchip_interp.py`):

  - restrict to the overlap range between the two series (no extrapolation)
  - PCHIP-interp loss at the AUPRC series' native steps within the overlap
  - Spearman ρ on the matched (−loss, AUPRC) pairs — negating loss so that
    "well-correlated" (loss-drop tracks AUPRC-gain) reads as positive ρ.

(Within a single run, "step" and "tokens seen" are linearly related, so the
rank-based ρ is identical whether the x-axis is steps or tokens.)

The figure is a 1x2 layout:
  left:  horizontal bar of mean ρ across the 6 variants per model, sorted
         small → large top → bottom.
  right: heatmap of per-(model, variant) ρ, rows aligned to the bars on the
         left. Row labels appear only on the bar chart.

Output:
  plots/output/blog/genomic_lm_optimization/figure8_loss_vs_traitgym_correlation.{png,pdf,svg}

Usage:
  uv run src/figures/figure8_loss_vs_traitgym_correlation.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.eval_history import DEFAULT_MAX_GAP_FRACTION, dedup_eval_history  # noqa: E402
from utils.figure_style import DIVERGING_CMAP, LARGEST_MODEL_COLOR, figsize  # noqa: E402
from utils.pchip_interp import clean, interp_on_overlap  # noqa: E402
from utils.savefig import save_figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "parameter_scaling_results.csv"
FIGURES_DIR = ROOT.parents[2] / "plots" / "output" / "blog" / "genomic_lm_optimization"

WANDB_PROJECT = "eric-czech/marin"
SCALING_PREFIX = "dna-bolinas-scaling-v0.5-"
TRAITGYM_PREFIX = "lm_eval/traitgym_mendelian_v2_255"

# Same six variant subsets used in figures 5 & 6.
TRAITS: tuple[tuple[str, str], ...] = (
    ("missense_variant", "missense"),
    ("tss_proximal", "promoter"),
    ("5_prime_UTR_variant", "5' UTR"),
    ("3_prime_UTR_variant", "3' UTR"),
    ("splicing", "splicing"),
    ("synonymous_variant", "synonymous"),
)

# Match other figures (figures.py FIGURE_WIDTH=12.0). Height kept compact for a
# single-row figure; comparable to figures 1/2/5.
FIGURE_WIDTH = 12.0
FIGURE_HEIGHT = 4.2

# Padding between x tick labels and the x-axis label — matches figures.py.
_X_LABEL_PAD = 0

LOSS_KEY = "eval/loss"


def _params_label(num_params: int | float) -> str:
    n = int(num_params)
    if n >= 1_000_000_000:
        return f"{round(n / 1e9)}B"
    return f"{round(n / 1e6)}M"


def _resolve(api: wandb.Api, name: str):
    runs = list(
        api.runs(
            WANDB_PROJECT, filters={"display_name": {"$regex": f"^{re.escape(name)}$"}}
        )
    )
    if not runs:
        raise RuntimeError(f"no run matching display_name {name!r}")
    if len(runs) > 1:
        runs.sort(key=lambda r: r.created_at, reverse=True)
        print(f"  WARN: {len(runs)} runs match {name!r}; taking newest")
    return runs[0]


def _num_train_steps(run) -> int:
    """Pull `trainer.num_train_steps` from a (nested) wandb run.config."""
    cfg = run.config
    for part in ("trainer", "num_train_steps"):
        if not isinstance(cfg, dict) or part not in cfg:
            raise RuntimeError(
                f"{run.name}: trainer.num_train_steps missing in run.config"
            )
        cfg = cfg[part]
    return int(cfg)


def fetch_run_series(api: wandb.Api, run_name: str) -> dict:
    """Fetch eval/loss and per-trait auprc histories for one run.

    Returns {"loss": (steps, values), "auprc": {trait: (steps, values)}}.
    Fetched separately (not in one history() call) because wandb's
    multi-key history applies AND-filtering, which would drop steps where
    only one series logged.
    """
    run = _resolve(api, run_name)
    max_gap = DEFAULT_MAX_GAP_FRACTION * _num_train_steps(run)
    auprc_keys = [f"{TRAITGYM_PREFIX}/{t}/auprc" for t, _ in TRAITS]

    loss_df = run.history(keys=[LOSS_KEY], samples=10000)
    auprc_df = run.history(keys=auprc_keys, samples=10000)

    def _xy(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
        if df.empty or col not in df.columns:
            return np.array([]), np.array([])
        sub = df[["_step", col]].dropna().sort_values("_step")
        x, y = clean(sub["_step"].to_numpy(dtype=float), sub[col].to_numpy(dtype=float))
        return dedup_eval_history(x, y, max_gap)

    return {
        "loss": _xy(loss_df, LOSS_KEY),
        "auprc": {t: _xy(auprc_df, f"{TRAITGYM_PREFIX}/{t}/auprc") for t, _ in TRAITS},
    }


def fetch_all(run_names: list[str]) -> dict[str, dict]:
    api = wandb.Api(timeout=300)
    out: dict[str, dict] = {}
    for name in run_names:
        print(f"fetching {name} ...")
        ser = fetch_run_series(api, name)
        n_loss = len(ser["loss"][0])
        n_auprc = {t: len(xy[0]) for t, xy in ser["auprc"].items()}
        print(f"  eval/loss rows={n_loss}, auprc rows={n_auprc}")
        out[name] = ser
    return out


def compute_correlation_matrix(
    runs: pd.DataFrame, series: dict[str, dict]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a (n_runs, n_traits) matrix of Spearman ρ values.

    Rows sorted by `params` ascending. Returns (matrix, row_labels, col_labels).
    """
    ordered = runs.sort_values("params").reset_index(drop=True)
    row_labels = [_params_label(p) for p in ordered["params"]]
    col_labels = [label for _, label in TRAITS]
    M = np.full((len(ordered), len(TRAITS)), np.nan, dtype=float)
    for i, row in ordered.iterrows():
        name = row["run_name"]
        loss_x, loss_y = series[name]["loss"]
        if len(loss_x) < 2:
            print(f"  {name}: <2 eval/loss points, skipping all traits")
            continue
        for j, (trait, _label) in enumerate(TRAITS):
            ax, ay = series[name]["auprc"][trait]
            if len(ax) < 2:
                print(f"  {name}/{trait}: <2 auprc points, skipping")
                continue
            # Align: PCHIP-interp loss at the AUPRC native steps within the
            # overlap range. Sample size = #AUPRC points in overlap.
            grid, loss_at_grid, auprc_at_grid = interp_on_overlap(
                loss_x,
                loss_y,
                ax,
                ay,
                grid=ax,
            )
            if grid is None or len(grid) < 3:
                print(f"  {name}/{trait}: insufficient overlap, skipping")
                continue
            # Spearman ρ(−loss, AUPRC). Negating loss so a positive ρ means
            # "loss reduction tracks AUPRC gain" (the desirable direction);
            # Spearman is rank-based so the negation only flips the sign.
            rho, _ = spearmanr(-loss_at_grid, auprc_at_grid)
            M[i, j] = float(rho)
    return M, row_labels, col_labels


def _save(fig, name: str) -> None:
    save_figure(fig, FIGURES_DIR, name)


def plot(M: np.ndarray, row_labels: list[str], col_labels: list[str]) -> None:
    """Plot bar chart (left) + heatmap (right), rows aligned smallest→largest top→bottom.

    M is indexed row-ascending in params (caller's responsibility). For both
    panels we want row 0 = smallest model at the *top*, which is the default
    for imshow(origin='upper') but requires inverting the y-axis for barh.
    """
    # Bar values: row mean over the (up-to-6) traits with valid ρ.
    bar_vals = np.nanmean(M, axis=1)

    fig, (ax_bar, ax_hm) = plt.subplots(
        1,
        2,
        figsize=figsize(FIGURE_WIDTH, FIGURE_HEIGHT),
        gridspec_kw={"width_ratios": [1.0, 1.6], "wspace": 0.06},
    )

    # --- Left: horizontal bars of row-mean ρ.
    y = np.arange(len(row_labels))
    # Bars in the 4B model's palette color (rust), tying this summary to the
    # largest-model end of the scaling figures.
    ax_bar.barh(
        y,
        bar_vals,
        color=LARGEST_MODEL_COLOR,
        edgecolor=LARGEST_MODEL_COLOR,
        linewidth=0.4,
        alpha=0.9,
    )
    ax_bar.axvline(0, color="0.4", lw=0.6)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(row_labels)
    ax_bar.set_ylabel("model params")
    ax_bar.set_xlabel(r"mean Spearman $\rho$", labelpad=_X_LABEL_PAD)
    ax_bar.set_ylim(-0.5, len(y) - 0.5)
    ax_bar.invert_yaxis()  # smallest model (y=0) → top, to match imshow
    ax_bar.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax_bar.set_axisbelow(True)
    # Close the box: show the top/right spines, matching the bottom/left axis
    # lines' width and color.
    _bottom = ax_bar.spines["bottom"]
    for spine in ("top", "right"):
        ax_bar.spines[spine].set(
            visible=True,
            linewidth=_bottom.get_linewidth(),
            edgecolor=_bottom.get_edgecolor(),
        )
    # X-range: anchor at 0 (so the y-axis spine sits flush with the bar bases)
    # and pad outward only on the side that actually has data, so the value
    # labels don't get clipped.
    finite_vals = bar_vals[np.isfinite(bar_vals)]
    if len(finite_vals):
        lo = min(0.0, float(finite_vals.min()))
        hi = max(0.0, float(finite_vals.max()))
        span = hi - lo if hi > lo else 1.0
        left = lo - 0.15 * span if lo < 0 else 0.0
        # Extra room on the right so the value labels clear the right border.
        right = hi + 0.30 * span if hi > 0 else 0.0
        ax_bar.set_xlim(left, right)
        pad = 0.02 * span
    else:
        pad = 0.02
    # Numeric value next to each bar, on the outside of the bar tip.
    for yi, v in zip(y, bar_vals):
        if not np.isfinite(v):
            continue
        ha = "left" if v >= 0 else "right"
        ax_bar.text(
            v + (pad if v >= 0 else -pad),
            yi,
            f"{v:+.2f}",
            va="center",
            ha=ha,
            fontsize=8,
            color="0.2",
        )

    # --- Right: heatmap of per-(model, trait) ρ.
    # Symmetric color scale around 0 so positive/negative ρ are visually
    # comparable. Bound by the matrix's largest |ρ| (min 0.2 to avoid a flat
    # scale if all values are tiny).
    vmax = max(0.2, float(np.nanmax(np.abs(M))) if np.any(np.isfinite(M)) else 1.0)
    im = ax_hm.imshow(
        M,
        cmap=DIVERGING_CMAP,
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    ax_hm.set_xticks(np.arange(len(col_labels)))
    ax_hm.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax_hm.set_yticks(np.arange(len(row_labels)))
    ax_hm.set_yticklabels([])  # labels live on the bar chart, don't repeat
    # Full box border around the heatmap, matching the bar chart's axis lines.
    for spine in ax_hm.spines.values():
        spine.set(
            visible=True,
            linewidth=_bottom.get_linewidth(),
            edgecolor=_bottom.get_edgecolor(),
        )
    # Cell annotations.
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                ax_hm.text(j, i, "·", ha="center", va="center", color="0.4", fontsize=9)
                continue
            # Pick text color for contrast against cell background.
            color = "white" if abs(v) > 0.6 * vmax else "0.1"
            ax_hm.text(
                j, i, f"{v:+.2f}", ha="center", va="center", color=color, fontsize=8
            )

    cbar = fig.colorbar(im, ax=ax_hm, pad=0.02, fraction=0.04)
    cbar.set_label(r"Spearman $\rho$")

    fig.suptitle(
        "Parameter scaling — loss vs VEP AUPRC correlation across training steps",
        fontsize=11,
        y=0.965,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    _save(fig, "figure8_loss_vs_traitgym_correlation")


def main() -> None:
    results = pd.read_csv(DATA_PATH)
    runs = results[results["run_name"].str.startswith(SCALING_PREFIX)][
        ["run_name", "params"]
    ].copy()
    if len(runs) == 0:
        raise RuntimeError(f"no scaling runs found in {DATA_PATH}")
    print(f"will fetch {len(runs)} runs: {runs['run_name'].tolist()}")

    series = fetch_all(runs["run_name"].tolist())
    M, row_labels, col_labels = compute_correlation_matrix(runs, series)

    print("\nSpearman ρ matrix (rows = params asc, cols = traits):")
    df_print = pd.DataFrame(M, index=row_labels, columns=col_labels)
    print(df_print.round(3).to_string())

    plot(M, row_labels, col_labels)


if __name__ == "__main__":
    main()
