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

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from marin_dna.pipelines.evals.leaderboard import (
    DEFAULT_PROTOCOL,
    fetch_method_metrics,
)
from marin_dna.pipelines.evals.models import models_for_dataset

DATASET = "mendelian_traits"
OUTPUT_DIR = Path("plots/output/blog")
OUTPUT_NAME = "promoter_cds_specialists"

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
        linewidth=0.7,
        saturation=0.9,
        ax=ax,
        legend=False,
        dodge=False,
    )

    panel_top = 0.0
    for x, role in enumerate(ROLE_ORDER):
        row = data.loc[data["role"] == role]
        assert len(row) == 1
        value = float(row["auprc_pct"].iloc[0])
        se = float(row["se_pct"].iloc[0])
        panel_top = max(panel_top, value + se)
        ax.errorbar(
            x,
            value,
            yerr=se,
            color=INK,
            linewidth=0.9,
            capsize=0,
            zorder=5,
        )
        ax.text(
            x,
            value + se + 0.9,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )

    ax.set_axisbelow(True)
    ax.grid(False)
    top_padding = max(2.0, 0.07 * (panel_top - AUPRC_BASELINE_PCT))
    y_max = panel_top + top_padding
    assert y_max > AUPRC_BASELINE_PCT
    ax.set_ylim(AUPRC_BASELINE_PCT, y_max)
    ax.set_xlabel("")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        SUBSET_DISPLAY[subset],
        fontsize=10,
        color=REGION_COLORS[region],
        pad=7,
    )


def build_figure(data: pd.DataFrame) -> Figure:
    """Build five independently scaled consequence panels in a 2x3 grid."""
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
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
    figure, axes = plt.subplots(2, 3, figsize=(5.0, 3.2), sharex=False, sharey=False)
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
            linewidth=0.7,
            label="Upstream",
        ),
        Patch(
            facecolor=REGION_COLORS["cds"],
            edgecolor=INK,
            linewidth=0.7,
            label="CDS",
        ),
    ]
    generalist_handles = [
        Patch(
            facecolor=BASELINE_COLORS["Evo 2 (40B)"],
            edgecolor=INK,
            linewidth=0.7,
            label="Evo 2 (40B)",
        ),
        Patch(
            facecolor=BASELINE_COLORS["GPN-Star (M)"],
            edgecolor=INK,
            linewidth=0.7,
            label="GPN-Star (M)",
        ),
    ]
    specialist_legend = axes[0, 2].legend(
        handles=specialist_handles,
        title="Specialists",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=1,
        frameon=False,
        fontsize=8,
        title_fontsize=8,
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
        bbox_to_anchor=(0.02, 0.58),
        ncol=1,
        frameon=False,
        fontsize=8,
        title_fontsize=8,
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


def save_figure(figure: Figure) -> None:
    """Save the web SVG and a high-resolution PNG for visual review."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (("svg", {}), ("png", {"dpi": 300})):
        path = OUTPUT_DIR / f"{OUTPUT_NAME}.{extension}"
        figure.savefig(path, bbox_inches="tight", transparent=True, **kwargs)
        if extension == "svg":
            svg = path.read_text(encoding="utf-8")
            path.write_text(
                "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                encoding="utf-8",
            )
        print(f"Wrote {path}")


def build() -> None:
    data = load_plot_data()
    figure = build_figure(data)
    save_figure(figure)
    plt.close(figure)


if __name__ == "__main__":
    build()
