"""Figure 5: parameter scaling — params vs VEP AUPRC by variant type (1x3)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from figures.data import VEP_PANELS, save
from utils.figure_style import EARTH_QUAL, FIGURE_WIDTH, X_LABEL_PAD, figsize

# 1x3 task-group panels for Figure 5. Each tuple is (panel title, list of subset keys).
# Subset order within each panel determines the EARTH_QUAL color slot (matched to VEP_PANELS).
FIGURE5_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CDS", ("missense_variant", "synonymous_variant")),
    ("upstream", ("tss_proximal", "5_prime_UTR_variant")),
    ("other", ("3_prime_UTR_variant", "splicing")),
)


def build(results: pd.DataFrame) -> None:
    """1x3 line+scatter panels by task group: params (log-x) vs AUPRC.

    Each variant type's color matches its slot in VEP_PANELS so the visual mapping
    is consistent across this figure and Figure 6; per-panel legends list only the
    variants drawn there.
    """
    color_for_subset = {
        subset: EARTH_QUAL[i] for i, (subset, _label, _n) in enumerate(VEP_PANELS)
    }
    label_for_subset = {subset: label for subset, label, _n in VEP_PANELS}

    fig, axes = plt.subplots(1, 3, figsize=figsize(FIGURE_WIDTH, 4.2))
    for ax, (group_title, subsets) in zip(axes, FIGURE5_GROUPS, strict=True):
        handles: list = []
        labels: list[str] = []
        for subset in subsets:
            col = f"lm_eval/traitgym_mendelian_v2_255/{subset}/auprc"
            valid = results.dropna(subset=["params", col]).sort_values("params")
            (line,) = ax.plot(
                valid["params"],
                valid[col],
                marker="o",
                linestyle="-",
                color=color_for_subset[subset],
                linewidth=1.3,
                markersize=6,
                markeredgecolor="k",
                markeredgewidth=0.4,
                zorder=3,
            )
            handles.append(line)
            labels.append(label_for_subset[subset])
        ax.set_xscale("log")
        ax.set_title(group_title, fontsize=10)
        ax.grid(False)
        # CDS curves climb into the top-left, so its legend goes bottom-right,
        # pushed flush to the right axis limit; the other facets keep top-left.
        if group_title == "CDS":
            # Bottom-right, flush to the right axis limit but lifted off the
            # x-axis line. markerfirst=False puts the markers on the right so both
            # rows' right edges align at the axis limit (left-aligned entries
            # leave the shorter label short of the edge).
            ax.legend(
                handles,
                labels,
                loc="lower right",
                bbox_to_anchor=(1.0, 0.06),
                borderaxespad=0.0,
                borderpad=0.0,
                markerfirst=False,
                fontsize=8,
                frameon=False,
                handletextpad=0.4,
            )
        else:
            ax.legend(
                handles,
                labels,
                loc="upper left",
                fontsize=8,
                frameon=False,
                handletextpad=0.4,
            )
    axes[0].set_ylabel("AUPRC")
    # x-label only on the middle panel.
    axes[1].set_xlabel("model params", labelpad=X_LABEL_PAD)

    fig.suptitle(
        "Parameter scaling — params vs VEP AUPRC by variant type", fontsize=11, y=0.97
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    save(fig, "figure5_params_vs_vep_auprc")
