"""Plot equal-step trajectories for the upstream x CDS mixture experiment.

The four arms are Qwen3 1B models trained on 100% upstream sequence, a
balanced 50/50 upstream/CDS mixture, the dataset-proportional 10/90 mixture,
or 100% CDS. Only checkpoints available in every arm are plotted, so each
vertical slice is an equal-training-step comparison.

The historical run names use "promoter"; public-facing labels use "upstream"
to match the blog's dataset terminology.

Run:
    uv run python plots/upstream_cds_balance.py

Outputs:
    plots/output/upstream_cds_balance/figure.{svg,png}
    blog/marin-dna/static/assets/images/blog/
        marin-dna/upstream_cds_balance.svg
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge

from marin_dna.blog_figure_typography import (
    FIGURE_GLOBAL_RENDER_SCALE,
    matplotlib_typography_rcparams,
    normalize_matplotlib_svg_typography_file,
    sync_article_figure_width,
    validate_svg_typography,
)


class Arm(NamedTuple):
    prefix: str
    label: str
    upstream_fraction: float
    steps: tuple[int, ...]


ARMS: dict[str, Arm] = {
    "upstream_only": Arm(
        prefix="exp21-promoters-yolo",
        label="Upstream only (100 / 0)",
        upstream_fraction=1.0,
        steps=(2000, 6000, 10000, 12000, 14000, 16000, 18000, 20000, 22000),
    ),
    "balanced": Arm(
        prefix="exp13-mixture-equal",
        label="Uniform mix (50 / 50)",
        upstream_fraction=0.5,
        steps=(2000, 6000, 10000, 14000, 18000, 22000, 26000),
    ),
    "proportional": Arm(
        prefix="exp13-mixture-proportional",
        label="Proportional mix (10 / 90)",
        upstream_fraction=0.1,
        steps=(2000, 6000, 10000, 14000, 18000, 22000, 26000),
    ),
    "cds_only": Arm(
        prefix="exp27-cds-yolo",
        label="CDS only (0 / 100)",
        upstream_fraction=0.0,
        steps=(2000, 6000, 10000, 14000, 18000, 22000, 26000, 34000),
    ),
}

S3_BASE = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"
SCORE_TYPE = "minus_llr_avg"
SUBSETS: dict[str, str] = {
    "Promoter": "tss_proximal",
    "Missense": "missense_variant",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "plots" / "output" / Path(__file__).stem
BLOG_SVG = (
    REPO_ROOT
    / "blog"
    / "marin-dna"
    / "static"
    / "assets"
    / "images"
    / "blog"
    / "marin-dna"
    / "upstream_cds_balance.svg"
)
BLOG_ARTICLE = REPO_ROOT / "blog" / "marin-dna" / "content" / "blog" / "marin-dna.md"
FIGURE_ID = "fig-upstream-cds-balance"

# The only figure-size control. All three plotting axes are square; the canvas
# dimensions follow from this height and the fixed one-row, three-panel layout.
SUBPLOT_HEIGHT_PX = 120.0

TEXT_COLOR = "#1f1e1b"
# Canonical issue #370 region palette: Upstream = teal, CDS = rust.
REGION_COLORS = {"upstream": "#2f6f63", "cds": "#9c4f2f"}
TITLE_COLORS = {
    "Promoter": REGION_COLORS["upstream"],
    "Missense": REGION_COLORS["cds"],
    "Mean of both": TEXT_COLOR,
}


def blend_region_colors(upstream_fraction: float) -> tuple[float, float, float]:
    """Blend the issue #370 Upstream and CDS colors by mixture fraction."""
    assert 0.0 <= upstream_fraction <= 1.0
    upstream = mpl.colors.to_rgb(REGION_COLORS["upstream"])
    cds = mpl.colors.to_rgb(REGION_COLORS["cds"])
    return tuple(
        upstream_fraction * upstream_channel + (1.0 - upstream_fraction) * cds_channel
        for upstream_channel, cds_channel in zip(upstream, cds, strict=True)
    )


class HandlerLineMarkerWithPie(HandlerBase):
    """Draw a line marker followed by an upstream/CDS composition pie."""

    def __init__(self, upstream_fraction: float, **kwargs) -> None:
        super().__init__(**kwargs)
        assert 0.0 <= upstream_fraction <= 1.0
        self.upstream_fraction = upstream_fraction

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        pie_diameter = height
        gap = 0.35 * fontsize
        line_y = height / 2 - ydescent
        line_x0 = -xdescent
        line_x1 = -xdescent + width - pie_diameter - gap

        color = orig_handle.get_color()
        line = Line2D(
            [line_x0, line_x1],
            [line_y, line_y],
            color=color,
            linewidth=orig_handle.get_linewidth(),
            solid_capstyle="butt",
        )
        marker = Line2D(
            [(line_x0 + line_x1) / 2],
            [line_y],
            color=color,
            marker=orig_handle.get_marker(),
            markersize=orig_handle.get_markersize(),
            markeredgecolor=TEXT_COLOR,
            markeredgewidth=orig_handle.get_markeredgewidth(),
            linestyle="None",
        )
        line.set_transform(trans)
        marker.set_transform(trans)
        artists = [line, marker]

        center_x = -xdescent + width - pie_diameter / 2
        radius = pie_diameter / 2 * 0.92
        upstream_fraction = self.upstream_fraction
        if upstream_fraction in (0.0, 1.0):
            region = "upstream" if upstream_fraction == 1.0 else "cds"
            circle = Circle(
                (center_x, line_y),
                radius,
                facecolor=REGION_COLORS[region],
                edgecolor=TEXT_COLOR,
            )
            circle.set_transform(trans)
            artists.append(circle)
        else:
            theta_top = 90.0
            theta_start = theta_top - upstream_fraction * 360.0
            upstream_wedge = Wedge(
                (center_x, line_y),
                radius,
                theta_start,
                theta_top,
                facecolor=REGION_COLORS["upstream"],
                edgecolor=TEXT_COLOR,
            )
            cds_wedge = Wedge(
                (center_x, line_y),
                radius,
                theta_top,
                theta_top + (1.0 - upstream_fraction) * 360.0,
                facecolor=REGION_COLORS["cds"],
                edgecolor=TEXT_COLOR,
            )
            upstream_wedge.set_transform(trans)
            cds_wedge.set_transform(trans)
            artists.extend((upstream_wedge, cds_wedge))
        return artists


def apply_style() -> None:
    """Apply the compact Open Athena blog style used by the draft."""
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": Path(__file__).stem,
            "font.family": "sans-serif",
            **matplotlib_typography_rcparams(),
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "legend.frameon": False,
            "savefig.facecolor": "none",
            "savefig.transparent": True,
        }
    )


def load_trajectories() -> pl.DataFrame:
    """Load all available checkpoints for the four experiment arms."""
    parts: list[pl.DataFrame] = []
    missing: list[str] = []
    total = sum(len(arm.steps) for arm in ARMS.values())
    for arm_key, arm in ARMS.items():
        for step in arm.steps:
            uri = f"{S3_BASE}/{arm.prefix}-step-{step}/mendelian_traits.parquet"
            try:
                checkpoint = pl.read_parquet(uri)
            except Exception as exc:
                missing.append(f"  {arm_key} step-{step}: {exc}")
                continue
            parts.append(
                checkpoint.with_columns(
                    pl.lit(arm_key).alias("arm"),
                    pl.lit(step).alias("step"),
                )
            )
    if missing:
        print(
            f"WARNING: {len(missing)} of {total} parquets unreadable:\n"
            + "\n".join(missing),
            file=sys.stderr,
        )
    assert parts, "no metrics parquets loaded"
    return pl.concat(parts)


def shared_step_data(trajectories: pl.DataFrame) -> pl.DataFrame:
    """Select the two focal subsets at steps represented in every arm."""
    common_steps = set.intersection(*(set(arm.steps) for arm in ARMS.values()))
    assert common_steps == {2000, 6000, 10000, 14000, 18000, 22000}

    filtered = trajectories.filter(
        (pl.col("score_type") == SCORE_TYPE)
        & (pl.col("subset").is_in(list(SUBSETS.values())))
        & (pl.col("step").is_in(sorted(common_steps)))
    )
    expected_rows = len(ARMS) * len(SUBSETS) * len(common_steps)
    assert filtered.height == expected_rows, (
        f"expected {expected_rows} arm/subset/step rows, got {filtered.height}"
    )
    assert filtered.select("arm", "subset", "step").unique().height == expected_rows
    assert filtered["value"].is_not_null().all()
    assert filtered["value"].is_between(0.0, 1.0).all()
    return filtered


def plot(data: pl.DataFrame) -> plt.Figure:
    """Create upstream, missense, and two-subset-mean trajectory panels."""
    apply_style()
    axes_height_fraction = 0.70 - 0.09
    figure_height_inches = SUBPLOT_HEIGHT_PX / (
        72.0 * axes_height_fraction * FIGURE_GLOBAL_RENDER_SCALE
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(2.5 * figure_height_inches, figure_height_inches),
        sharex=True,
        sharey=False,
    )

    for axis, (title, subset) in zip(axes[:2], SUBSETS.items(), strict=True):
        for arm_key, arm in ARMS.items():
            arm_data = data.filter(
                (pl.col("arm") == arm_key) & (pl.col("subset") == subset)
            ).sort("step")
            axis.plot(
                arm_data["step"].to_numpy(),
                arm_data["value"].to_numpy() * 100.0,
                marker="o",
                color=blend_region_colors(arm.upstream_fraction),
                label=arm.label,
            )
        axis.set_title(
            title,
            color=TITLE_COLORS[title],
        )

    mean_axis = axes[2]
    for arm_key, arm in ARMS.items():
        arm_data = (
            data.filter(pl.col("arm") == arm_key)
            .group_by("step")
            .agg(pl.col("value").mean().alias("value"))
            .sort("step")
        )
        assert arm_data.height == data["step"].n_unique()
        mean_axis.plot(
            arm_data["step"].to_numpy(),
            arm_data["value"].to_numpy() * 100.0,
            marker="o",
            color=blend_region_colors(arm.upstream_fraction),
            label=arm.label,
        )
    mean_axis.set_title(
        "Mean of both",
        color=TITLE_COLORS["Mean of both"],
    )

    for axis in axes:
        axis.set_xlabel("Training step")
        axis.grid(False)
        axis.set_box_aspect(1)
    axes[0].set_ylabel("AUPRC (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    # Matplotlib fills a two-column legend column-first. This ordering displays
    # the two pure-region arms on the first row and the two mixtures below.
    legend_order = (0, 1, 3, 2)
    handler_map = {
        handle: HandlerLineMarkerWithPie(arm.upstream_fraction)
        for handle, arm in zip(handles, ARMS.values(), strict=True)
    }
    fig.legend(
        [handles[index] for index in legend_order],
        [labels[index] for index in legend_order],
        title="Training data mixture",
        handler_map=handler_map,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        alignment="center",
        columnspacing=0.4,
        handlelength=4.0,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.0,
    )
    fig.subplots_adjust(bottom=0.09, top=0.70, left=0.09, right=0.98, wspace=0.28)
    return fig


def main() -> None:
    trajectories = load_trajectories()
    data = shared_step_data(trajectories)
    figure = plot(data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BLOG_SVG.parent.mkdir(parents=True, exist_ok=True)
    outputs = (OUTPUT_DIR / "figure.svg", OUTPUT_DIR / "figure.png", BLOG_SVG)
    for output in outputs:
        metadata = {"Date": None} if output.suffix == ".svg" else None
        figure.savefig(output, dpi=180, bbox_inches="tight", metadata=metadata)
        if output.suffix == ".svg":
            svg = output.read_text(encoding="utf-8")
            output.write_text(
                "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                encoding="utf-8",
            )
            normalize_matplotlib_svg_typography_file(output)
            validate_svg_typography(output)
        print(f"wrote {output}")
    sync_article_figure_width(BLOG_ARTICLE, FIGURE_ID, BLOG_SVG)
    plt.close(figure)


if __name__ == "__main__":
    main()
