"""Figure 10 — composed Mendelian VEP AUPRC trajectories across continuation
lineages.

Redo of the blog's Figure 10 on the offline evals_v2 eval. Three model lineages,
each warm-started in sequence, are *composed* into a single trajectory over
cumulative training tokens (root→leaf, each phase truncated at its child's fork so
the parent's off-path cooldown tail is dropped):

  - m5.1 : uniform → uniform_to_uniform_1 → +zoonomia (warm-start from FINAL ckpt).
  - m1.3 : m1 → m1.1 → m1.2 → m1.3  (zoonomia uniform ⅕; pre-cooldown chain).
  - m3.3 : m3 → m3.1 → m3.2 → m3.3  (zoonomia upstream-tilted; pre-cooldown chain).

The composition (chain-stitch → cumulative-token placement → fork truncation) and its
build-time consistency guard live in ``_mixture`` (ported verbatim from Eric); this
recipe only sources per-checkpoint AUPRC from ``World.read`` and draws. **Caveat vs
Eric's Figure 10:** offline scores only the HF-exported checkpoints (3-4/stage, and
after truncation ~9 land on-path per lineage) versus W&B's dense in-training evals —
so these trajectories are sparser and lean harder on the smoother. Raw points are
drawn prominently for honesty.

One parallel figure per Mendelian scoring method. SGE is intentionally excluded:
this section optimizes the macro average across all Mendelian variant types, whereas
SGE assays only missense and splicing. Renders whichever Mendelian worlds are scored
(skips a world if its leaf has no evals yet).

Run:  uv run python -m plots.blog.figure10_lineage_vep_trajectory
Out:  plots/output/blog/figure10_lineage_vep_trajectory__{world.key}.{svg,png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from plots.blog import _mixture as mx
from plots.blog import _mixture_lineage as ml
from plots.blog._style.figure_style import EARTH_QUAL, FIGURE_WIDTH, LEGEND_KW, figsize
from plots.blog._style.savefig import save_figure
from plots.blog._worlds import WORLDS

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "blog"

# The three lineages, keyed by leaf mix; color + label per lineage (Eric's order).
LINEAGES: tuple[tuple[str, str], ...] = (
    ("exp135-zoonomia-m5.1", "m5.1"),
    ("exp135-zoonomia-m1.3", "m1.3"),
    ("exp135-zoonomia-m3.3", "m3.3"),
)
LINEAGE_COLORS = {leaf: EARTH_QUAL[i] for i, (leaf, _) in enumerate(LINEAGES)}

# Mendelian variant panels in the Figure 6 order, followed by the two additional
# Figure 10 subsets. The macro-average panel is prepended in ``build``.
MENDELIAN_PANELS: tuple[tuple[str, str], ...] = (
    ("missense_variant", "missense"),
    ("synonymous_variant", "synonymous"),
    ("splicing", "splicing"),
    ("tss_proximal", "promoter"),
    ("5_prime_UTR_variant", "5' UTR"),
    ("3_prime_UTR_variant", "3' UTR"),
    ("distal", "distal"),
    ("non_coding_transcript_exon_variant", "ncRNA"),
)

MACRO_ACCENT = "#5e3418"
MACRO_FILL = "#f1ece0"

# Constant across every v0.9 1B run — cumulative tokens ↔ consolidated steps.
TOKENS_PER_STEP = mx.TOKENS_PER_STEP

# Gaussian-kernel smoother bandwidth as a multiple of median point spacing (Eric's).
SMOOTH_BANDWIDTH_MULT = 1.8
SMOOTH_GRID = 200


def _reader(world):
    """Cached ``mix, step -> {subset: value}`` over ``World.read`` (soft-fails to
    ``{}`` for a checkpoint not scored in this world yet)."""
    cache: dict[tuple[str, int], dict[str, float]] = {}

    def read(mix: str, step: int) -> dict[str, float]:
        key = (mix, step)
        if key not in cache:
            try:
                df = world.read(mx.cfg_name(mix, step)).to_pandas()
                cache[key] = dict(zip(df["subset"], df["value"], strict=True))
            except (LookupError, FileNotFoundError, OSError):
                cache[key] = {}
        return cache[key]

    return read


def _smooth_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nadaraya–Watson Gaussian-kernel regression on a dense grid (Eric's smoother).
    Falls back to raw points when too few to smooth."""
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


def _draw_panel(ax, read, subsets: tuple[str, ...], token_cutoff: float) -> None:
    """Draw the three lineage trajectories: raw evals as dots + a smoothed trend."""
    for leaf, _ in LINEAGES:
        tokens, values = mx.composed_curve(read, leaf, subsets)
        if len(tokens) == 0:
            continue
        if leaf != "exp135-zoonomia-m5.1":
            keep = tokens <= token_cutoff
            if keep.any() and not keep.all():
                keep[np.argmax(~keep)] = True  # one point past the shared window
            tokens, values = tokens[keep], values[keep]
            if len(tokens) == 0:
                continue
        color = LINEAGE_COLORS[leaf]
        x = tokens / 1e9
        ax.scatter(x, values, s=12, color=color, alpha=0.6, edgecolors="none", zorder=2)
        gx, gy = _smooth_xy(x, values)
        ax.plot(gx, gy, color=color, lw=1.6, zorder=3)


def _highlight_macro(ax, title: str) -> None:
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
    for spine in ax.spines.values():
        spine.set_color(MACRO_ACCENT)
        spine.set_linewidth(1.4)


def _attach_legend(fig) -> None:
    handles = [
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
    labels = [label for _, label in LINEAGES]
    legend_kw = {**LEGEND_KW, "fontsize": 10, "title_fontsize": 10}
    fig.legend(
        handles,
        labels,
        ncol=3,
        title="mixture strategy",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.055),
        **legend_kw,
    )


def build(world) -> None:
    read = _reader(world)
    assert world.key.startswith("mendelian"), f"Figure 10 excludes {world.key}"
    panels = MENDELIAN_PANELS
    all_subsets = tuple(s for s, _ in panels)

    # Guard: consistency check (crashes on a mis-stitched lineage) + skip an unscored
    # world. Uses the first panel subset as the probe (present in every checkpoint).
    endpoint = mx.validate_consistency(read, "exp135-zoonomia-m5.1", (all_subsets[0],))
    if not np.isfinite(endpoint):
        print(f"figure10: world {world.key} — m5.1 not scored yet, skipping")
        return
    m5_tokens, _ = mx.composed_curve(read, "exp135-zoonomia-m5.1", all_subsets)
    if len(m5_tokens) < 2:
        print(
            f"figure10: world {world.key} — only {len(m5_tokens)} m5.1 point(s), skipping"
        )
        return
    token_cutoff = float(np.max(m5_tokens))

    # Panel grid: macro + eight per-subset panels → 3×3.
    grid_panels: list[tuple[tuple[str, ...], str, bool]] = [
        (all_subsets, "macro average", True)
    ]
    grid_panels += [((s,), title, False) for s, title in panels]
    n = len(grid_panels)
    nrows, ncols = 3, 3
    height = 9.6
    fig, axes = plt.subplots(
        nrows, ncols, sharex=True, figsize=figsize(FIGURE_WIDTH, height)
    )
    axes_flat = np.atleast_1d(axes).flatten()

    def to_ksteps(b):
        return b * 1e9 / TOKENS_PER_STEP / 1e3

    def to_btokens(k):
        return k * 1e3 * TOKENS_PER_STEP / 1e9

    shift_x = (
        sum(ml.inherited_components("exp135-zoonomia-m5.1", mx.own_tokens()).values())
        / 1e9
    )

    for ax, (subsets, title, is_macro) in zip(axes_flat, grid_panels, strict=False):
        _draw_panel(ax, read, subsets, token_cutoff)
        ax.grid(False)
        ax.margins(y=0.10)
        ax.tick_params(axis="both", labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
        if is_macro:
            _highlight_macro(ax, title)
        else:
            ax.set_title(title, fontsize=10, pad=3)
        # Mixture-shift marker + secondary step axis on the top row.
        if ax in axes_flat[:ncols]:
            ax.axvline(shift_x, color="0.35", ls=(0, (4, 2)), lw=1.0, zorder=1)
            sec = ax.secondary_xaxis("top", functions=(to_ksteps, to_btokens))
            sec.tick_params(labelsize=7.5, pad=0.5)
            sec.set_xlabel(
                "training steps (k)" if ax is axes_flat[min(1, ncols - 1)] else " ",
                fontsize=8.5,
                labelpad=4,
            )
    # Defensive in case the panel set changes without the grid being updated.
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    bottom_row = axes_flat[(nrows - 1) * ncols :] if nrows > 1 else axes_flat
    bottom_row[min(1, len(bottom_row) - 1)].set_xlabel(
        "tokens (B)", labelpad=4, fontsize=9
    )
    for r in range(nrows):
        axes_flat[r * ncols].set_ylabel("VEP AUPRC")

    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.suptitle(
        f"VEP AUPRC trajectories by mixture strategy · {world.label}",
        fontsize=11,
        y=0.955,
    )
    _attach_legend(fig)
    save_figure(fig, OUTPUT_DIR, f"figure10_lineage_vep_trajectory__{world.key}")
    print(
        f"figure10: {world.key} rendered (m5.1 endpoint {endpoint / 1e9:.0f}B tokens)"
    )


def build_all() -> None:
    for key in ("mendelian_llr", "mendelian_probe"):
        build(WORLDS[key])


if __name__ == "__main__":
    build_all()
