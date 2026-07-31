"""Promoter/CDS specialist comparison for the genomic-LM optimization blog.

The five-panel bar chart follows the CSHL 2026 poster's specialist figure and
the reworked blog Figure 5 small multiples, but uses the current offline
Mendelian eval and keeps only region-matched tasks:

* Promoter YOLO: 5' UTR and TSS-proximal variants.
* CDS YOLO: missense, splicing, and synonymous variants.

Each specialist is compared with Evo 2 40B and GPN-Star (M) under the canonical
zero-shot protocol for that model family. Error bars are capless +/-1
chromosome-cluster bootstrap SE.

Run: uv run python -m plots.blog.promoter_cds_specialists
Out: plots/output/blog/promoter_cds_specialists.{svg,png}
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from marin_dna.blog_figure_typography import (
    FIGURE_GLOBAL_RENDER_SCALE,
    matplotlib_typography_rcparams,
    normalize_matplotlib_svg_typography_file,
    sync_article_figure_width,
    validate_svg_typography,
)
from marin_dna.pipelines.evals.leaderboard import (
    DEFAULT_PROTOCOL,
    fetch_method_metrics,
)
from marin_dna.pipelines.evals.models import models_for_dataset

DATASET = "mendelian_traits"
OUTPUT_DIR = Path("plots/output/blog")
OUTPUT_NAME = "promoter_cds_specialists"
REPO_ROOT = Path(__file__).resolve().parents[2]
BLOG_ARTICLE = (
    REPO_ROOT
    / "blog"
    / "genomic-lm-optimization"
    / "content"
    / "blog"
    / "genomic-lm-optimization.md"
)
BLOG_SVG = (
    REPO_ROOT
    / "blog"
    / "genomic-lm-optimization"
    / "static"
    / "assets"
    / "images"
    / "blog"
    / "genomic-lm-optimization"
    / f"{OUTPUT_NAME}.svg"
)
FIGURE_ID = "fig-upstream-cds-specialists"

# The sole display-size control. Each plotting axes is square, so its displayed
# width follows from this height; the 3x2 grid then determines the SVG and
# article-frame widths automatically. Typography and tick locators remain
# independent global/style concerns.
SUBPLOT_HEIGHT_PX = 100.0

INK = "#1f1e1b"
AUPRC_BASELINE_PCT = 10.0
ROLE_ORDER = ("Specialist", "Evo 2 (40B)", "GPN-Star (M)")
BASELINE_COLORS = {
    "Evo 2 (40B)": "#999999",
    "GPN-Star (M)": "#555555",
}
REGION_COLORS = {"upstream": "#2f6f63", "cds": "#9c4f2f"}

PANEL_ORDER = (
    "tss_proximal",
    "5_prime_UTR_variant",
    "missense_variant",
    "splicing",
    "synonymous_variant",
)
PANEL_SPECS = {
    "5_prime_UTR_variant": {
        "specialist_id": "exp21-promoters-yolo-step-22000",
        "region": "upstream",
    },
    "tss_proximal": {
        "specialist_id": "exp21-promoters-yolo-step-22000",
        "region": "upstream",
    },
    "missense_variant": {
        "specialist_id": "exp27-cds-yolo-step-34000",
        "region": "cds",
    },
    "splicing": {
        "specialist_id": "exp27-cds-yolo-step-34000",
        "region": "cds",
    },
    "synonymous_variant": {
        "specialist_id": "exp27-cds-yolo-step-34000",
        "region": "cds",
    },
}
SUBSET_DISPLAY = {
    "5_prime_UTR_variant": "5′ UTR",
    "tss_proximal": "Promoter",
    "missense_variant": "Missense",
    "splicing": "Splicing",
    "synonymous_variant": "Synonymous",
}
BASELINE_IDS = {
    "Evo 2 (40B)": "evo2_40b",
    "GPN-Star (M)": "GPN-Star-M",
}


def load_plot_data() -> pd.DataFrame:
    """Load the five region-matched comparisons from current evals_v2 metrics."""
    required_ids = {
        *(spec["specialist_id"] for spec in PANEL_SPECS.values()),
        *BASELINE_IDS.values(),
    }
    models = {
        model.id: model
        for model in models_for_dataset(DATASET)
        if model.id in required_ids
    }
    assert set(models) == required_ids, (
        f"missing registered models: {sorted(required_ids - set(models))}"
    )

    records: list[dict[str, Any]] = []
    for subset in PANEL_ORDER:
        spec = PANEL_SPECS[subset]
        role_to_model_id = {
            "Specialist": spec["specialist_id"],
            **BASELINE_IDS,
        }
        for role, model_id in role_to_model_id.items():
            model = models[model_id]
            metrics = fetch_method_metrics(model, DATASET)
            row = metrics.filter(metrics["subset"] == subset)
            assert row.height == 1, (
                f"expected one {subset!r} row for {model_id!r}, got {row.height}"
            )
            values = row.row(0, named=True)
            value = float(values["value"])
            se = float(values["se"])
            n = int(values["n"])
            n_positives = int(values["n_positives"])
            assert 0.0 <= value <= 1.0
            assert 0.0 <= se <= 1.0
            assert 0 < n_positives <= n
            records.append(
                {
                    "subset": subset,
                    "consequence": SUBSET_DISPLAY[subset],
                    "region": spec["region"],
                    "role": role,
                    "model_id": model_id,
                    "protocol": DEFAULT_PROTOCOL[model.family],
                    "auprc_pct": 100.0 * value,
                    "se_pct": 100.0 * se,
                    "n": n,
                    "n_positives": n_positives,
                }
            )

    frame = pd.DataFrame.from_records(records)
    expected_rows = len(PANEL_ORDER) * len(ROLE_ORDER)
    assert len(frame) == expected_rows
    assert not frame.isna().any().any()
    return frame


def _draw_panel(ax: Axes, data: pd.DataFrame, subset: str) -> None:
    """Draw one consequence panel on its own y-axis."""
    assert set(data["subset"]) == {subset}
    assert data["n"].nunique() == 1
    region = str(data["region"].iloc[0])
    assert region in REGION_COLORS
    palette = {
        "Specialist": REGION_COLORS[region],
        **BASELINE_COLORS,
    }
    sns.barplot(
        data=data,
        x="role",
        y="auprc_pct",
        hue="role",
        order=ROLE_ORDER,
        hue_order=ROLE_ORDER,
        palette=palette,
        errorbar=None,
        edgecolor=INK,
        saturation=0.9,
        ax=ax,
        legend=False,
        dodge=False,
    )

    for x, role in enumerate(ROLE_ORDER):
        row = data.loc[data["role"] == role]
        assert len(row) == 1
        value = float(row["auprc_pct"].iloc[0])
        se = float(row["se_pct"].iloc[0])
        ax.errorbar(
            x,
            value,
            yerr=se,
            color=INK,
            capsize=0,
            zorder=5,
        )
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.set_ylim(bottom=AUPRC_BASELINE_PCT)
    ax.set_xlabel("")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        SUBSET_DISPLAY[subset],
        color=REGION_COLORS[region],
        pad=7,
    )
    ax.set_box_aspect(1)


def build_figure(data: pd.DataFrame) -> Figure:
    """Build five independently scaled consequence panels in a 2x3 grid."""
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": OUTPUT_NAME,
            **matplotlib_typography_rcparams(),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.titlecolor": INK,
        }
    )
    layout_scale = SUBPLOT_HEIGHT_PX / 100.0
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(5.0 * layout_scale, 3.2 * layout_scale),
        sharex=False,
        sharey=False,
    )
    panel_axes = (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2])
    for ax, subset in zip(panel_axes, PANEL_ORDER, strict=True):
        _draw_panel(ax, data.loc[data["subset"] == subset], subset)
    axes[0, 2].axis("off")
    axes[0, 0].set_ylabel("AUPRC (%)")
    axes[1, 0].set_ylabel("AUPRC (%)")
    for ax in (axes[0, 1], axes[1, 1], axes[1, 2]):
        ax.set_ylabel("")

    specialist_handles = [
        Patch(
            facecolor=REGION_COLORS["upstream"],
            edgecolor=INK,
            label="Upstream",
        ),
        Patch(
            facecolor=REGION_COLORS["cds"],
            edgecolor=INK,
            label="CDS",
        ),
    ]
    generalist_handles = [
        Patch(
            facecolor=BASELINE_COLORS["Evo 2 (40B)"],
            edgecolor=INK,
            label="Evo 2 (40B)",
        ),
        Patch(
            facecolor=BASELINE_COLORS["GPN-Star (M)"],
            edgecolor=INK,
            label="GPN-Star (M)",
        ),
    ]
    specialist_legend = axes[0, 2].legend(
        handles=specialist_handles,
        title="Specialists",
        loc="upper left",
        bbox_to_anchor=(0.02, 1.22),
        ncol=1,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=0,
    )
    axes[0, 2].add_artist(specialist_legend)
    axes[0, 2].legend(
        handles=generalist_handles,
        title="Generalists",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.68),
        ncol=1,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=0,
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.995,
        top=0.96,
        bottom=0.12,
        wspace=0.25,
        hspace=0.32,
    )
    return figure


def _square_panel_height_points(figure: Figure) -> float:
    """Return a representative square panel height in SVG point units."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    panel_axes = [axis for axis in figure.axes if axis.axison]
    assert len(panel_axes) == len(PANEL_ORDER)
    panel_sizes = [axis.get_window_extent(renderer=renderer) for axis in panel_axes]
    for bounds in panel_sizes:
        assert math.isclose(bounds.width, bounds.height, rel_tol=1e-6), bounds
    heights = [bounds.height * 72.0 / figure.dpi for bounds in panel_sizes]
    assert max(heights) - min(heights) < 1e-6, heights
    return heights[0]


def save_figure(figure: Figure) -> None:
    """Save the web SVG and a high-resolution PNG for visual review."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_height_points = _square_panel_height_points(figure)
    for extension, kwargs in (
        ("svg", {"metadata": {"Date": None}}),
        ("png", {"dpi": 300}),
    ):
        path = OUTPUT_DIR / f"{OUTPUT_NAME}.{extension}"
        figure.savefig(path, bbox_inches="tight", transparent=True, **kwargs)
        if extension == "svg":
            svg = path.read_text(encoding="utf-8")
            path.write_text(
                "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                encoding="utf-8",
            )
        print(f"Wrote {path}")

    svg_path = OUTPUT_DIR / f"{OUTPUT_NAME}.svg"
    displayed_panel_height = panel_height_points * FIGURE_GLOBAL_RENDER_SCALE
    assert math.isclose(displayed_panel_height, SUBPLOT_HEIGHT_PX, abs_tol=0.2), (
        displayed_panel_height,
        SUBPLOT_HEIGHT_PX,
    )
    normalize_matplotlib_svg_typography_file(svg_path)
    validate_svg_typography(svg_path)
    BLOG_SVG.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(svg_path, BLOG_SVG)
    frame_width = sync_article_figure_width(BLOG_ARTICLE, FIGURE_ID, BLOG_SVG)
    print(
        f"Synced {BLOG_SVG} at {SUBPLOT_HEIGHT_PX:g}px square panels "
        f"({frame_width:.1f}px frame)"
    )


def build() -> None:
    data = load_plot_data()
    figure = build_figure(data)
    save_figure(figure)
    plt.close(figure)


if __name__ == "__main__":
    build()
