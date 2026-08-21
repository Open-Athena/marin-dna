"""Compare issue #489 conservation AUPRC from loss levels and loss slope."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ARTIFACT_ROOT = Path(__file__).resolve().parent
FIGURE_ROOT = ARTIFACT_ROOT / "figures"
SCOPE_ORDER = ("global", "cds", "upstream", "downstream", "ncrna", "enhancer")
SCOPE_LABELS = {
    "global": "Global",
    "cds": "CDS",
    "upstream": "Upstream",
    "downstream": "Downstream",
    "ncrna": "ncRNA",
    "enhancer": "Enhancer",
}
METHOD_ORDER = ("prevalence", "loss_21b", "loss_174b", "loss_slope")
METHOD_LABELS = {
    "prevalence": "Prevalence",
    "loss_21b": "Loss at 21B",
    "loss_174b": "Loss at 174B",
    "loss_slope": "Loss slope",
}
METHOD_COLORS = {
    "prevalence": "0.55",
    "loss_21b": "#4c72b0",
    "loss_174b": "#dd8452",
    "loss_slope": "#55a868",
}


def comparison_frame() -> pd.DataFrame:
    """Combine exact pooled endpoint and slope AUPRC values."""
    endpoints = pd.read_parquet(ARTIFACT_ROOT / "conservation_auprc.parquet")
    endpoints = endpoints[
        (endpoints["statistic"] == "loss")
        & endpoints["checkpoint_order"].isin([0, 4])
    ].copy()
    endpoints["method"] = endpoints["checkpoint_order"].map(
        {0: "loss_21b", 4: "loss_174b"}
    )
    slope = pd.read_csv(ARTIFACT_ROOT / "slope_conservation_auprc.csv")
    slope["method"] = "loss_slope"
    prevalence = (
        endpoints[endpoints["checkpoint_order"] == 0][
            ["scope", "prevalence"]
        ]
        .rename(columns={"prevalence": "auprc"})
        .assign(method="prevalence")
    )
    frame = pd.concat(
        [
            prevalence[["scope", "method", "auprc"]],
            endpoints[["scope", "method", "auprc"]],
            slope[["scope", "method", "auprc"]],
        ],
        ignore_index=True,
    )
    assert len(frame) == len(SCOPE_ORDER) * len(METHOD_ORDER)
    return frame


def plot(frame: pd.DataFrame) -> None:
    """Render a compact exact-pooled AUPRC method comparison."""
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(6, 5))
    base_y = np.arange(len(SCOPE_ORDER), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(METHOD_ORDER))
    for method_index, method in enumerate(METHOD_ORDER):
        values = (
            frame[frame["method"] == method]
            .set_index("scope")
            .loc[list(SCOPE_ORDER), "auprc"]
        )
        axis.scatter(
            values,
            base_y + offsets[method_index],
            color=METHOD_COLORS[method],
            s=48,
            label=METHOD_LABELS[method],
            zorder=3,
        )
    axis.set_yticks(base_y, [SCOPE_LABELS[scope] for scope in SCOPE_ORDER])
    axis.invert_yaxis()
    axis.set_xlim(0.15, 0.72)
    axis.set_xlabel("AUPRC")
    axis.set_ylabel("Validation scope")
    axis.set_title("Conservation AUPRC from loss level and slope")
    axis.grid(axis="y", visible=False)
    axis.set_box_aspect(0.8)
    axis.legend(
        title="Score",
        loc="upper left",
        bbox_to_anchor=(0.0, -0.18),
        ncol=2,
        borderaxespad=0,
    )
    figure.subplots_adjust(bottom=0.28)
    figure.savefig(
        FIGURE_ROOT / "loss_level_vs_slope_auprc.svg",
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    """Write the comparison table and reviewed figure artifacts."""
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    frame = comparison_frame()
    frame.to_csv(
        ARTIFACT_ROOT / "slope_conservation_comparison.csv",
        index=False,
    )
    plot(frame)


if __name__ == "__main__":
    main()
