"""Figure 8: loss ↔ VEP AUPRC correlation across training, by scale.

Redo of Eric's Fig. 8 (mean-correlation bars + per-variant heatmap) on the new
offline eval, for every metric world (Mendelian/SGE × zero-shot LLR/probe). The
figure restores the original's complete eight-model scaling ladder and all
variant types available in each dataset. Variant columns follow the updated
Fig. 5 order.

For each model and variant type, correlate AUPRC with negated corresponding-run
validation loss, so positive values mean that falling loss tracks improving
AUPRC. Render parallel versions for Spearman rank correlation (rho) and Pearson
product-moment correlation (r); the plots use the conventional symbols.

Cross-scale fairness (issue #365): the offline eval only scores saved HF
checkpoints. All eight ladder sizes have the same seven checkpoints from 160k
through the final step, so every cell uses that fixed shared-step window. Full-
cadence trajectory shapes remain in Fig. 7.

Run:  uv run python -m plots.blog.figure8_loss_vs_traitgym_correlation
Out:  plots/output/blog/figure8_loss_vs_traitgym_correlation__{world}__{method}.{svg,png,pdf}
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from plots.blog._regions import MENDELIAN_VARIANT_ORDER, SGE_VARIANT_ORDER
from plots.blog._scaling import VEP_PANELS
from plots.blog._style.figure_style import DIVERGING_CMAP, figsize
from plots.blog._style.savefig import save_figure
from plots.blog._worlds import WORLDS, World

HISTORY = Path(__file__).resolve().parent / "data" / "parameter_scaling_history.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "blog"

CorrelationMethod = Literal["spearman", "pearson"]

# Fixed intersection of saved HF checkpoints across all eight model sizes. The
# final step is already scored; the other six steps are the issue-365 expansion.
COMMON_STEPS: tuple[int, ...] = (
    160000,
    170000,
    180000,
    190000,
    200000,
    210000,
    215573,
)

# (display label, W&B run name used by the vendored loss history, evals_v2 stem).
# Ordered exactly as the original Fig. 8: smallest to largest.
SCALES: tuple[tuple[str, str, str], ...] = (
    (
        "46M",
        "dna-bolinas-scaling-v0.5-h640-p46M",
        "scaling-v0.5-h640-p46M",
    ),
    (
        "76M",
        "dna-bolinas-scaling-v0.5-h768-p76M",
        "scaling-v0.5-h768-p76M",
    ),
    (
        "128M",
        "dna-bolinas-scaling-v0.5-h896-p128M",
        "scaling-v0.5-h896-p128M",
    ),
    (
        "255M",
        "dna-bolinas-scaling-v0.5-h1152-p255M",
        "scaling-v0.5-h1152-p255M",
    ),
    (
        "476M",
        "dna-bolinas-scaling-v0.5-h1408-p476M",
        "scaling-v0.5-h1408-p476M",
    ),
    (
        "1B",
        "dna-bolinas-scaling-v0.5-h1920-p1B",
        "scaling-v0.5-h1920-p1B",
    ),
    (
        "2B",
        "dna-bolinas-scaling-v0.5-h2432-p2B",
        "scaling-v0.5-h2432-p2B",
    ),
    (
        "4B",
        "dna-bolinas-scaling-v0.5-h2944-p4B",
        "scaling-v0.5-h2944-p4B",
    ),
)


def _loss_at(run_name: str, steps: np.ndarray) -> np.ndarray:
    """Interpolate validation loss at the requested steps on log10(step)."""
    history = pd.read_csv(HISTORY)
    history = history[
        (history["run_name"] == run_name) & (history["metric"] == "eval/loss")
    ].sort_values("step")
    assert not history.empty, f"no eval/loss history for {run_name!r}"
    xs = np.log10(history["step"].clip(lower=1).to_numpy())
    requested = np.log10(np.clip(steps, 1, None))
    assert requested.min() >= xs.min() and requested.max() <= xs.max(), (
        f"requested steps for {run_name!r} exceed the loss-history range"
    )
    return np.interp(requested, xs, history["value"].to_numpy())


def _scored(stem: str, world: World) -> dict[int, dict[str, float]]:
    """Return step to subset-to-AUPRC mappings for available shared checkpoints."""
    out: dict[int, dict[str, float]] = {}
    for step in COMMON_STEPS:
        try:
            df = world.read(f"{stem}-step-{step}")
        except (LookupError, FileNotFoundError, OSError):
            continue
        out[step] = {row["subset"]: row["value"] for row in df.iter_rows(named=True)}
    return out


def _traits_for(world: World) -> tuple[tuple[str, str], ...]:
    """Variant keys and labels in the updated Fig. 5 panel order."""
    label_for = {subset: label for subset, label in VEP_PANELS}
    if world.key.startswith("mendelian_"):
        order = MENDELIAN_VARIANT_ORDER
    elif world.key.startswith("sge_"):
        order = SGE_VARIANT_ORDER
    else:
        raise AssertionError(f"unknown Fig. 8 dataset for world {world.key!r}")
    assert set(order).issubset(label_for), (
        f"missing labels for {set(order) - label_for.keys()}"
    )
    return tuple((subset, label_for[subset]) for subset in order)


def _correlation(x: np.ndarray, y: np.ndarray, method: CorrelationMethod) -> float:
    """Compute the requested correlation, using its conventional definition."""
    assert method in ("spearman", "pearson"), f"unknown method {method!r}"
    assert len(x) == len(y) and len(x) >= 3
    fn = spearmanr if method == "spearman" else pearsonr
    return float(fn(x, y).statistic)


def _complete_scored(world: World) -> dict[str, dict[int, dict[str, float]]]:
    """Load every scale and fail if any shared checkpoint has not been scored."""
    scored = {label: _scored(stem, world) for label, _run, stem in SCALES}
    missing = {
        label: sorted(set(COMMON_STEPS) - set(per_step))
        for label, per_step in scored.items()
        if set(per_step) != set(COMMON_STEPS)
    }
    if missing:
        details = "; ".join(f"{label}: {steps}" for label, steps in missing.items())
        raise RuntimeError(
            f"Fig. 8 requires all shared checkpoints for {world.key}; missing {details}"
        )
    return scored


def correlation_matrix(
    world: World, method: CorrelationMethod
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return the eight-model by all-variant correlation matrix."""
    traits = _traits_for(world)
    scored = _complete_scored(world)
    matrix = np.full((len(SCALES), len(traits)), np.nan)
    steps = np.array(COMMON_STEPS, dtype=float)

    for i, (label, run_name, _stem) in enumerate(SCALES):
        loss = _loss_at(run_name, steps)
        for j, (subset, _trait_label) in enumerate(traits):
            auprc = np.array(
                [scored[label][step].get(subset, np.nan) for step in COMMON_STEPS],
                dtype=float,
            )
            finite = np.isfinite(auprc) & np.isfinite(loss)
            if finite.sum() >= 3:
                matrix[i, j] = _correlation(-loss[finite], auprc[finite], method)

    return (
        matrix,
        [label for label, _run, _stem in SCALES],
        [label for _subset, label in traits],
    )


def correlations(
    world: World, method: CorrelationMethod = "spearman"
) -> tuple[dict[str, float], int]:
    """Mean per-scale correlation across variants; retained for reporting scripts."""
    matrix, labels, _trait_labels = correlation_matrix(world, method)
    return dict(zip(labels, np.nanmean(matrix, axis=1), strict=True)), len(COMMON_STEPS)


def _method_text(method: CorrelationMethod) -> tuple[str, str]:
    """Display name and standard statistical symbol for one method."""
    assert method in ("spearman", "pearson"), f"unknown method {method!r}"
    return ("Spearman", "ρ") if method == "spearman" else ("Pearson", "r")


def build(world: World, method: CorrelationMethod) -> None:
    matrix, row_labels, trait_labels = correlation_matrix(world, method)
    method_name, symbol = _method_text(method)
    mean_correlation = np.nanmean(matrix, axis=1)

    fig, (ax_bar, ax_hm) = plt.subplots(
        1,
        2,
        figsize=figsize(12.0, 5.2),
        gridspec_kw={"width_ratios": [1, 2.3]},
    )
    y = np.arange(len(row_labels))
    ax_bar.barh(y, mean_correlation, color="#7e8a45", edgecolor="k", linewidth=0.5)
    ax_bar.axvline(0, color="0.5", linewidth=0.8)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(row_labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel(f"mean {method_name} {symbol}")
    ax_bar.set_title("mean across variants", fontsize=10)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)

    image = ax_hm.imshow(matrix, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, aspect="auto")
    ax_hm.set_xticks(range(len(trait_labels)))
    ax_hm.set_xticklabels(trait_labels, rotation=30, ha="right", fontsize=9)
    ax_hm.set_yticks(y)
    ax_hm.set_yticklabels([])
    for i in range(len(row_labels)):
        for j in range(len(trait_labels)):
            value = matrix[i, j]
            if np.isfinite(value):
                ax_hm.text(
                    j,
                    i,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > 0.55 else "black",
                )
            else:
                ax_hm.text(j, i, "·", ha="center", va="center", color="0.4")
    ax_hm.set_title(f"per-variant {symbol}", fontsize=10)
    colorbar = fig.colorbar(image, ax=ax_hm, fraction=0.025, pad=0.02)
    colorbar.set_label(f"{method_name} {symbol}", fontsize=9)

    fig.suptitle(
        "Loss ↔ VEP AUPRC correlation · "
        f"{world.label} · {method_name} {symbol} · shared 160k–215573 steps",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(
        fig,
        OUTPUT_DIR,
        f"figure8_loss_vs_traitgym_correlation__{world.key}__{method}",
    )


def build_all() -> None:
    for method in ("spearman", "pearson"):
        for world in WORLDS.values():
            build(world, method)


if __name__ == "__main__":
    build_all()
