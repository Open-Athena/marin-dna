"""Prototype Figure 16 from existing offline Mendelian checkpoint metrics.

This recipe reads the already-produced evals_v2 metric parquets for existing
HF-exported, on-path checkpoints in the m5.1 / m1.3 / m3.3 lineages. It does not
run inference or fit new probes.

Two nine-panel figures are emitted:

* Mendelian zero-shot LLR
* Mendelian frozen-embedding linear probe

Raw observations are connected by straight segments and shown with the
already-computed ±1 SE error bars. No kernel smoother is applied: the sparse,
irregular checkpoint cadence does not support a smooth trajectory estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from figures import mixture_lineage as ml
from figures.data import load_mixture, save
from marin_dna.blog_figure_typography import FIGURE_NOTE_SIZE_PX
from utils.figure_style import FIGURE_WIDTH, LEGEND_KW, figsize

TOKENS_PER_STEP = 8192 * 256
RESULTS_ROOT = "s3://oa-bolinas/snakemake/analysis/evals_v2/results"
STORAGE_OPTIONS = {"aws_region": "us-east-2"}


@dataclass(frozen=True)
class OfflineRun:
    """One lineage stage and its existing HF-exported checkpoint steps."""

    mix: str
    index: int
    steps: tuple[int, ...]


RUNS: tuple[OfflineRun, ...] = (
    OfflineRun("uniform", 0, (10000, 20000, 25004)),
    OfflineRun("uniform_to_uniform_1", 18, (10000, 20000, 29579)),
    OfflineRun("exp135-zoonomia-m5.1", 24, (30000, 40000, 50000, 59158)),
    OfflineRun("exp135-zoonomia-m1", 20, (10000, 20000, 29579)),
    OfflineRun("exp135-zoonomia-m1.1", 26, (30000, 40000, 50000, 53243)),
    OfflineRun("exp135-zoonomia-m1.2", 28, (50000, 60000, 70000, 70991)),
    OfflineRun("exp135-zoonomia-m1.3", 30, (60000, 70000, 80000, 82823)),
    OfflineRun("exp135-zoonomia-m3", 22, (10000, 20000, 29578)),
    OfflineRun("exp135-zoonomia-m3.1", 25, (30000, 40000, 50000, 53238)),
    OfflineRun("exp135-zoonomia-m3.2", 27, (50000, 60000, 70000, 70983)),
    OfflineRun("exp135-zoonomia-m3.3", 29, (60000, 70000, 80000, 82813)),
)
BY_MIX = {run.mix: run for run in RUNS}

SHORT_NAMES = {
    "exp135-zoonomia-m5.1": "exp135-m5.1",
}

# A method-neutral, colorblind-safe lineage palette. These colors are deliberately
# distinct from the earthy genomic-region palette used elsewhere in the post.
LINEAGES: tuple[tuple[str, str, str, str], ...] = (
    ("exp135-zoonomia-m5.1", "m5.1", "#D55E00", "o"),
    ("exp135-zoonomia-m1.3", "m1.3", "#0072B2", "o"),
    ("exp135-zoonomia-m3.3", "m3.3", "#CC79A7", "o"),
)

PANELS: tuple[tuple[str, str], ...] = (
    ("missense_variant", "Missense"),
    ("splicing", "Splicing"),
    ("synonymous_variant", "Synonymous"),
    ("tss_proximal", "Promoter"),
    ("5_prime_UTR_variant", "5' UTR"),
    ("3_prime_UTR_variant", "3' UTR"),
    ("distal", "Distal"),
    ("non_coding_transcript_exon_variant", "ncRNA"),
)
ALL_SUBSETS = tuple(subset for subset, _label in PANELS)
MACRO_SUBSET = "_macro_avg_"
ALL_METRIC_SUBSETS = (*ALL_SUBSETS, MACRO_SUBSET)

WORLD_CONFIG = {
    "llr": ("metrics", "minus_llr_avg", "pooled AUPRC", "zero-shot LLR"),
    "probe": (
        "probe_metrics",
        "probe_score",
        "chromosome-weighted AUPRC",
        "frozen-embedding probe",
    ),
}

MACRO_ACCENT = "#59636e"
MACRO_FILL = "#f1eee8"


def _model_id(mix: str, step: int) -> str:
    run = BY_MIX[mix]
    short = SHORT_NAMES.get(mix, mix)
    return f"mix-v0.9-p1B-i{run.index}-{short}-step-{step}"


@lru_cache(maxsize=None)
def _read_checkpoint(kind: str, mix: str, step: int) -> dict[str, tuple[float, float]]:
    """Return stored ``(AUPRC, SE)`` values for the eight subsets and their macro."""
    result_kind, score_type, _metric_label, _method_label = WORLD_CONFIG[kind]
    model_id = _model_id(mix, step)
    path = f"{RESULTS_ROOT}/{result_kind}/{model_id}/mendelian_traits.parquet"
    data = (
        pl.read_parquet(path, storage_options=STORAGE_OPTIONS)
        .filter(
            (pl.col("score_type") == score_type)
            & (pl.col("split") == "train")
            & pl.col("subset").is_in(ALL_METRIC_SUBSETS)
        )
        .select("subset", "value", "se")
    )
    assert data.height == len(ALL_METRIC_SUBSETS), (
        f"{model_id}/{kind}: expected {len(ALL_METRIC_SUBSETS)} metric rows, "
        f"got {data.height}"
    )
    assert data["subset"].n_unique() == len(ALL_METRIC_SUBSETS)
    assert data["value"].is_not_null().all()
    assert data["se"].is_not_null().all()
    assert data["value"].is_between(0, 1, closed="both").all()
    assert data["se"].is_between(0, 1, closed="both").all()
    return {subset: (float(value), float(se)) for subset, value, se in data.iter_rows()}


def _chain(leaf: str) -> list[str]:
    """Return the root-to-leaf stage sequence for one lineage."""
    chain: list[str] = []
    mix: str | None = leaf
    while mix is not None:
        chain.append(mix)
        mix = ml.BY_MIX[mix].parent
    return chain[::-1]


def _phase_start_step(mix: str, results: pd.DataFrame) -> float:
    row = results.loc[mix]
    own_steps = round(float(row["tokens"]) / TOKENS_PER_STEP)
    return float(row["num_train_steps"]) - own_steps


def _composed_curve(
    kind: str,
    leaf: str,
    subset: str,
    results: pd.DataFrame,
    own_tokens: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose existing offline checkpoints on the cumulative-token axis."""
    chain = _chain(leaf)
    offsets = [sum(ml.inherited_components(mix, own_tokens).values()) for mix in chain]
    tokens: list[float] = []
    values: list[float] = []
    ses: list[float] = []

    for phase_index, mix in enumerate(chain):
        start = _phase_start_step(mix, results)
        cutoff = (
            offsets[phase_index + 1] + TOKENS_PER_STEP
            if phase_index + 1 < len(chain)
            else np.inf
        )
        for step in BY_MIX[mix].steps:
            # Drop the final pre-shift m5.1 checkpoint: it is only 0.9B tokens
            # before the first post-shift point, which aligns with both controls.
            if (
                leaf == "exp135-zoonomia-m5.1"
                and mix == "uniform_to_uniform_1"
                and step == 29579
            ):
                continue
            cumulative_tokens = offsets[phase_index] + (step - start) * TOKENS_PER_STEP
            if cumulative_tokens > cutoff:
                continue
            value, se = _read_checkpoint(kind, mix, step)[subset]
            tokens.append(cumulative_tokens)
            values.append(value)
            ses.append(se)

    order = np.argsort(tokens)
    result_tokens = np.asarray(tokens)[order]
    result_values = np.asarray(values)[order]
    result_ses = np.asarray(ses)[order]
    expected_checkpoints = 8 if leaf == "exp135-zoonomia-m5.1" else 9
    assert len(result_tokens) == expected_checkpoints, (
        f"{kind}/{leaf}: expected {expected_checkpoints} on-path checkpoints, "
        f"got {len(result_tokens)}"
    )
    assert np.all(np.diff(result_tokens) > 0), (
        f"{kind}/{leaf}: token order is not strict"
    )
    return result_tokens, result_values, result_ses


def _draw_panel(
    ax: plt.Axes,
    *,
    kind: str,
    subset: str,
    results: pd.DataFrame,
    own_tokens: dict[str, float],
    token_cutoff: float,
) -> None:
    """Draw raw offline checkpoints with capless ±1 SE bars."""
    for leaf, _label, color, marker in LINEAGES:
        tokens, values, ses = _composed_curve(kind, leaf, subset, results, own_tokens)
        if leaf != "exp135-zoonomia-m5.1":
            keep = tokens <= token_cutoff
            if keep.any() and not keep.all():
                keep[np.argmax(~keep)] = True
            tokens = tokens[keep]
            values = values[keep]
            ses = ses[keep]
        ax.errorbar(
            tokens / 1e9,
            values * 100.0,
            yerr=ses * 100.0,
            color=color,
            linewidth=1.2,
            marker=marker,
            markersize=4.0,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            ecolor=color,
            elinewidth=0.65,
            capsize=0,
            alpha=0.9,
            zorder=3,
        )


def _highlight_macro(ax: plt.Axes) -> None:
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            facecolor=MACRO_FILL,
            edgecolor="none",
            zorder=-10,
        )
    )
    for spine in ax.spines.values():
        spine.set_color(MACRO_ACCENT)
        spine.set_linewidth(1.4)


def _attach_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=1.2,
            marker=marker,
            markersize=4,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.35,
        )
        for _leaf, _label, color, marker in LINEAGES
    ]
    labels = [label for _leaf, label, _color, _marker in LINEAGES]
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        **LEGEND_KW,
    )


def build(kind: str, results_df: pd.DataFrame | None = None) -> None:
    """Build one nine-panel offline prototype without smoothing."""
    assert kind in WORLD_CONFIG
    if results_df is None:
        results_df = load_mixture()
    results = results_df.set_index("mix")
    own_tokens = {mix: float(results.loc[mix, "tokens"]) for mix in results.index}

    m5_tokens, _m5_values, _m5_ses = _composed_curve(
        kind,
        "exp135-zoonomia-m5.1",
        MACRO_SUBSET,
        results,
        own_tokens,
    )
    token_cutoff = float(m5_tokens.max())
    shift_x = (
        sum(
            ml.inherited_components(
                "exp135-zoonomia-m5.1",
                own_tokens,
            ).values()
        )
        / 1e9
    )

    grid_panels: list[tuple[str, str, bool]] = [(MACRO_SUBSET, "Macro Avg", True)]
    grid_panels.extend((subset, label, False) for subset, label in PANELS)

    fig, axes = plt.subplots(
        3,
        3,
        sharex=True,
        figsize=figsize(FIGURE_WIDTH * 0.70, FIGURE_WIDTH * 0.80),
    )
    fig.subplots_adjust(
        left=0.115,
        right=0.99,
        bottom=0.07,
        top=0.88,
        wspace=0.18,
        hspace=0.16,
    )

    for ax, (subset, title, is_macro) in zip(axes.flat, grid_panels, strict=True):
        _draw_panel(
            ax,
            kind=kind,
            subset=subset,
            results=results,
            own_tokens=own_tokens,
            token_cutoff=token_cutoff,
        )
        ax.grid(False)
        ax.margins(y=0.12)
        ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.set_box_aspect(1)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
        if is_macro:
            _highlight_macro(ax)
        ax.set_title(
            title,
            fontweight="bold" if is_macro else "normal",
            y=1.03,
            pad=0,
        )

        ax.axvline(
            shift_x,
            color="#6b7280",
            linestyle=(0, (4, 2)),
            linewidth=0.9,
            zorder=1,
        )
        if is_macro:
            ax.annotate(
                "m5.1 shift\n3→5 regions",
                xy=(shift_x, 0.02),
                xycoords=ax.get_xaxis_transform(),
                xytext=(0.96, 0.02),
                textcoords=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=FIGURE_NOTE_SIZE_PX,
                color="#59636e",
                zorder=4,
                linespacing=1.35,
                bbox={
                    "boxstyle": "square,pad=0.15",
                    "facecolor": MACRO_FILL,
                    "edgecolor": "none",
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#59636e",
                    "linewidth": 0.8,
                    "shrinkA": 3,
                    "shrinkB": 0,
                },
            )
    axes[-1, 1].set_xlabel("Tokens (B)", labelpad=4)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUPRC (%)")

    _result_kind, _score_type, metric_label, method_label = WORLD_CONFIG[kind]
    fig.suptitle(
        f"Mendelian {metric_label} by mixture strategy · {method_label}",
        y=0.985,
    )
    _attach_legend(fig)
    save(fig, f"figure16_offline_lineage_{kind}_prototype")
    plt.close(fig)


def main() -> None:
    results = load_mixture()
    build("llr", results)
    build("probe", results)


if __name__ == "__main__":
    main()
