"""Figure 10: composed VEP AUPRC trajectories across continuation lineages.

Three model lineages, each three training stages warm-started in sequence, are
*composed* into a single trajectory over cumulative training tokens:

  - m5.1  : 1·L uniform → 1.7·L uniform → 1.7.2·L +zoonomia (warm-started from the
            parent's FINAL, fully-cooled checkpoint — a fresh rewarmup with no
            overlapping tokens).
  - m1.3  : 5·L → 5.1·L → 5.1.1·L → 5.1.1.1·L  (zoonomia uniform ⅕; pre-cooldown chain).
  - m3.3  : 6·L → 6.1·L → 6.1.1·L → 6.1.1.1·L  (zoonomia upstream-tilted; pre-cooldown chain).

Composition rule (the whole point of this figure): a continuation that branched
BEFORE its parent's cooldown shares the parent's W&B step counter, so the parent's
post-branch evals (its own cooldown tail, trained on a path we did NOT take) would
double-count. We therefore keep a parent phase's evals only up to the child's fork,
i.e. up to the cumulative-token level the child inherited. A continuation from a
parent's FINAL checkpoint has no such overlap — the parent's whole run is on-path.
Both cases fall out of one operation: truncate each parent phase at its child's
inherited-token offset (`mixture_lineage.inherited_components`). This keeps the
token axis identical to the cumulative-token accounting in the Appendix mixture
tree (validated at build time for the finished m5.1 lineage).

Token accounting and the per-stage fork fractions live in `mixture_lineage`; the
per-eval step→token mapping uses the constant batch×seq = 2,097,152 tokens/step
shared by every v0.9 1B run. The secondary top axis shows the consolidated step
count (cumulative tokens ÷ tokens-per-step), i.e. total optimizer steps along the
composed path — which is NOT the leaf run's raw W&B `_step` whenever an earlier
phase reset its counter (e.g. the HF-reinit of `uniform_to_uniform_1`).

The terminal m1.3 / m3.3 legs were originally tracked while running on preemptible
VMs; if a future refresh catches a resumed run mid-flight, the leaf stage simply
contributes the eval points available at that time.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from figures import mixture_lineage as ml
from figures.data import save
from utils.eval_history import DEFAULT_MAX_GAP_FRACTION, dedup_eval_history
from utils.figure_style import EARTH_QUAL, FIGURE_WIDTH, LEGEND_KW, figsize

# Constant across every v0.9 1B run (train_batch_size × train_seq_len).
TOKENS_PER_STEP = 8192 * 256

# The 8 metric panels: the 6 composite VEP subsets (Figs 5/6 order) followed by
# the two extra subsets. (subset, title).
METRIC_PANELS: tuple[tuple[str, str], ...] = (
    ("missense_variant", "missense"),
    ("tss_proximal", "promoter"),
    ("5_prime_UTR_variant", "5' UTR"),
    ("3_prime_UTR_variant", "3' UTR"),
    ("splicing", "splicing"),
    ("synonymous_variant", "synonymous"),
    ("distal", "distal"),
    ("non_coding_transcript_exon_variant", "non-coding exon"),
)
ALL_SUBSETS = tuple(s for s, _ in METRIC_PANELS)

# The three lineages, keyed by leaf `mix`; color + display label per lineage.
LINEAGES: tuple[tuple[str, str], ...] = (
    ("exp135-zoonomia-m5.1", "m5.1"),
    ("exp135-zoonomia-m1.3", "m1.3"),
    ("exp135-zoonomia-m3.3", "m3.3"),
)
LINEAGE_COLORS = {leaf: EARTH_QUAL[i] for i, (leaf, _) in enumerate(LINEAGES)}

# Accent + fill for the highlighted macro-average panel.
MACRO_ACCENT = "#5e3418"
MACRO_FILL = "#f1ece0"


def _metric_col(subset: str) -> str:
    return f"lm_eval/traitgym_mendelian_v2_255/{subset}/auprc"


def _chain(leaf: str) -> list[str]:
    """Root→leaf `mix` chain for a lineage (walk parent pointers)."""
    chain: list[str] = []
    mix: str | None = leaf
    while mix is not None:
        chain.append(mix)
        mix = ml.BY_MIX[mix].parent
    return chain[::-1]


def _phase_start_step(mix: str, results: pd.DataFrame) -> float:
    """Step at which `mix`'s OWN portion begins in its W&B counter.

    = configured total steps − own steps. Equals the parent's branch step for a
    continuation that kept the counter, and 0 for a run that reset (root or
    HF-reinit). Robust to unfinished runs (uses the configured span, not the last
    logged step).
    """
    row = results.loc[mix]
    own_steps = round(float(row["tokens"]) / TOKENS_PER_STEP)
    return float(row["num_train_steps"]) - own_steps


def _phase_series(
    mix: str, subsets: tuple[str, ...], history: pd.DataFrame, ntrain: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step values for one run: a single subset's AUPRC, or — for the macro
    panel — the unweighted mean across `subsets` at each eval step (all metrics
    are logged together, so the mean is over the full set). Deduped for resumes.
    """
    cols = [_metric_col(s) for s in subsets]
    sub = history[(history["mix"] == mix) & (history["metric"].isin(cols))]
    if sub.empty:
        return np.array([]), np.array([])
    wide = sub.pivot_table(index="step", columns="metric", values="value")
    series = wide.mean(axis=1).sort_index()  # macro mean (identity for one subset)
    return dedup_eval_history(
        series.index.to_numpy(),
        series.to_numpy(),
        max_gap_steps=DEFAULT_MAX_GAP_FRACTION * ntrain,
    )


def _composed_curve(
    leaf: str,
    subsets: tuple[str, ...],
    results: pd.DataFrame,
    history: pd.DataFrame,
    own_full: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compose one lineage's evals into (tokens, values) sorted by cumulative token.

    Each phase's evals are placed at cumulative tokens = its inherited-token offset
    + own-portion tokens, then truncated at the next phase's offset to drop the
    on-path overlap (the parent's cooldown tail). `subsets` is a single metric, or
    all eight for the macro-average panel.
    """
    chain = _chain(leaf)
    offsets = [sum(ml.inherited_components(m, own_full).values()) for m in chain]
    tokens: list[float] = []
    values: list[float] = []
    for i, mix in enumerate(chain):
        steps, vals = _phase_series(
            mix, subsets, history, float(results.loc[mix, "num_train_steps"])
        )
        if len(steps) == 0:
            continue
        cum = offsets[i] + (steps - _phase_start_step(mix, results)) * TOKENS_PER_STEP
        # Keep only up to where the child forked (one step of float slack); a final
        # leaf has no successor → no truncation. The child's offset IS the fork's
        # cumulative-token level, so this drops exactly the parent's off-path tail.
        cutoff = offsets[i + 1] + TOKENS_PER_STEP if i + 1 < len(chain) else np.inf
        keep = cum <= cutoff
        tokens.extend(cum[keep])
        values.extend(vals[keep])
    order = np.argsort(tokens)
    return np.asarray(tokens)[order], np.asarray(values)[order]


def _validate_consistency(
    results: pd.DataFrame, history: pd.DataFrame, own_full: dict[str, float]
) -> None:
    """End-to-end check: the cumulative token of a finished leaf's LAST composed
    eval (built from W&B steps via the step→token mapping) must land on the
    lineage's cumulative-token total — the same number the Appendix mixture tree
    draws for that node. Catches drift between the figure and the tree accounting.
    """
    for leaf, _ in LINEAGES:
        if results.loc[leaf, "state"] != "finished":
            continue
        tokens, _ = _composed_curve(
            leaf, ("missense_variant",), results, history, own_full
        )
        if len(tokens) == 0:
            continue
        expected = ml.cumulative_total(leaf, own_full)
        # The last eval lands one eval-cadence short of the configured end, so the
        # composed endpoint should be within ~one eval interval of the total.
        eval_gap = float(results.loc[leaf, "num_train_steps"]) / 10 * TOKENS_PER_STEP
        assert tokens[-1] <= expected + TOKENS_PER_STEP, (
            f"{leaf}: {tokens[-1]} > {expected}"
        )
        assert expected - tokens[-1] <= 1.5 * eval_gap, (
            f"{leaf}: composed endpoint {ml.format_tokens(tokens[-1])} far below "
            f"cumulative total {ml.format_tokens(expected)}"
        )
        print(
            f"  consistency OK: {leaf} composed endpoint = {ml.format_tokens(tokens[-1])} "
            f"≈ mixture-tree total {ml.format_tokens(expected)}"
        )


# Gaussian-kernel smoother bandwidth, as a multiple of the median point spacing.
# Larger → smoother. ~2.5× the eval cadence reads as a clean trend without
# erasing real structure.
SMOOTH_BANDWIDTH_MULT = 1.8
SMOOTH_GRID = 200


def _smooth_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nadaraya–Watson Gaussian-kernel regression on a dense grid.

    A standard kernel smoother: each grid point is a Gaussian-weighted mean of the
    observations, with bandwidth set from the median point spacing (so it adapts to
    the uneven token spacing and to either lineage length). Edges are one-sided and
    smooth — no boxcar kinks. Falls back to the raw points when too few to smooth.
    """
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    if len(xs) < 4:
        return xs, ys
    bw = SMOOTH_BANDWIDTH_MULT * float(np.median(np.diff(xs)))
    if bw <= 0:
        return xs, ys
    grid = np.linspace(xs[0], xs[-1], SMOOTH_GRID)
    w = np.exp(-0.5 * ((grid[:, None] - xs[None, :]) / bw) ** 2)
    return grid, (w @ ys) / w.sum(axis=1)


def _draw_panel(
    ax, subsets, results, history_df, own_full, token_cutoff: float
) -> None:
    """Draw the three lineage trajectories on `ax`: raw evals as dots, with a
    Gaussian-kernel-smoothed trend line over them."""
    for leaf, _ in LINEAGES:
        tokens, values = _composed_curve(leaf, subsets, results, history_df, own_full)
        if len(tokens) == 0:
            continue
        if leaf != "exp135-zoonomia-m5.1":
            # Truncate just past m5.1's endpoint, but keep the first eval beyond
            # the cutoff so each trend extends one point past the shared window.
            # (tokens is sorted ascending; argmax(~keep) is that first point, and
            # a no-op flip on index 0 when nothing exceeds the cutoff.)
            keep = tokens <= token_cutoff
            keep[np.argmax(~keep)] = True
            tokens, values = tokens[keep], values[keep]
            if len(tokens) == 0:
                continue
        color = LINEAGE_COLORS[leaf]
        x = tokens / 1e9
        ax.scatter(x, values, s=7, color=color, alpha=0.55, edgecolors="none", zorder=2)
        gx, gy = _smooth_xy(x, values)
        ax.plot(gx, gy, color=color, lw=1.6, zorder=3)


def build(results_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    results = results_df.set_index("mix")
    own_full = {m: float(results.loc[m, "tokens"]) for m in results.index}
    _validate_consistency(results, history_df, own_full)
    m5_tokens, _ = _composed_curve(
        "exp135-zoonomia-m5.1", ALL_SUBSETS, results, history_df, own_full
    )
    if len(m5_tokens) == 0:
        raise ValueError(
            "m5.1 has no composed eval points; cannot set Figure 10 token cutoff"
        )
    token_cutoff = float(np.max(m5_tokens))

    fig, axes = plt.subplots(3, 3, sharex=True, figsize=figsize(FIGURE_WIDTH, 9.6))

    # Secondary top axis: cumulative tokens (B) ↔ consolidated steps (k = ×10³).
    def to_ksteps(b):
        return b * 1e9 / TOKENS_PER_STEP / 1e3

    def to_btokens(k):
        return k * 1e3 * TOKENS_PER_STEP / 1e9

    # Row-major panel layout: row 1 = macro avg + the two distal-type tasks;
    # row 2 = coding tasks; row 3 = the upstream/UTR tasks.
    titles = dict(METRIC_PANELS)
    macro = (ALL_SUBSETS, "macro average", True)

    def sub(key):
        return ((key,), titles[key], False)

    panels = [
        macro,
        sub("distal"),
        sub("non_coding_transcript_exon_variant"),
        sub("missense_variant"),
        sub("splicing"),
        sub("synonymous_variant"),
        sub("tss_proximal"),
        sub("5_prime_UTR_variant"),
        sub("3_prime_UTR_variant"),
    ]

    # Consolidated-token position where the m5.1 lineage's mixture shifts from 3
    # regions (uniform / uniform_to_uniform_1) to 5 (m5.1 adds ncrna_exon +
    # ccre_non_promoter). This is exactly m5.1's inherited-token offset — i.e. where
    # its curve's final phase begins — so the marker aligns with the plotted line.
    shift_x = (
        sum(ml.inherited_components("exp135-zoonomia-m5.1", own_full).values()) / 1e9
    )

    for ax, (subsets, title, is_macro) in zip(axes.flat, panels, strict=True):
        _draw_panel(ax, subsets, results, history_df, own_full, token_cutoff)
        ax.grid(False)
        ax.margins(y=0.10)
        # Consistent tick-label size across both axes (matches the secondary
        # step-axis ticks); the token tick labels otherwise render larger.
        ax.tick_params(axis="both", labelsize=7.5)
        # Thin full border around every facet (the theme drops top/right spines).
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
        if is_macro:
            _highlight_macro(ax, title)
        else:
            ax.set_title(title, fontsize=10, pad=3)
        # Row 1 only: mark the m5.1 3-region → 5-region mixture shift in every
        # panel, but annotate it once (macro panel, bottom-right where there's room).
        if ax in axes[0]:
            ax.axvline(shift_x, color="0.35", ls=(0, (4, 2)), lw=1.0, zorder=1)
            if is_macro:
                ax.text(
                    0.825,
                    0.10,
                    "m5.1 3→5 region\nmixture shift",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="0.35",
                    zorder=4,
                    linespacing=1.2,
                )
            # Secondary step axis (shared x → identical ticks); its label hugs the
            # ticks. Set on every top panel (blank on the sides) so the three top
            # titles stay vertically aligned, with text only on the center.
            sec = ax.secondary_xaxis("top", functions=(to_ksteps, to_btokens))
            sec.tick_params(labelsize=7.5, pad=0.5)
            sec.set_xlabel(
                "training steps (k)" if ax is axes[0, 1] else " ",
                fontsize=8.5,
                labelpad=4,
            )

    # Primary x-axis label once, under the center facet of the bottom row.
    axes[-1, 1].set_xlabel("tokens (B)", labelpad=4, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("VEP AUPRC")

    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.suptitle("VEP AUPRC trajectories by mixture strategy", fontsize=11, y=0.952)
    _attach_legends(fig)
    save(fig, "figure10_lineage_vep_trajectory")


def _highlight_macro(ax, title: str) -> None:
    """Visually set the macro-average panel apart: tinted background + accent title.
    The background is a real patch (not the Axes facecolor) so it survives the
    transparent SVG save. (The uniform thin facet border is applied separately.)"""
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
    ax.set_title(title, fontsize=11, fontweight="bold", pad=3)
    # Thicker accent border on all sides to emphasize the macro panel.
    for spine in ax.spines.values():
        spine.set_color(MACRO_ACCENT)
        spine.set_linewidth(1.4)


def _attach_legends(fig) -> None:
    """Single lineage (color) legend, centered below the panels."""
    lineage_handles = [
        Line2D(
            [0],
            [0],
            color=LINEAGE_COLORS[leaf],
            lw=1.8,
            marker="o",
            markersize=4,
            markerfacecolor=LINEAGE_COLORS[leaf],
            markeredgecolor="none",
        )
        for leaf, _ in LINEAGES
    ]
    lineage_labels = [label for _, label in LINEAGES]
    # Legend title + entry labels at the facet-title size (10) for consistency.
    legend_kw = {**LEGEND_KW, "fontsize": 10, "title_fontsize": 10}
    fig.legend(
        lineage_handles,
        lineage_labels,
        ncol=3,
        title="mixture strategy",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.06),
        **legend_kw,
    )
