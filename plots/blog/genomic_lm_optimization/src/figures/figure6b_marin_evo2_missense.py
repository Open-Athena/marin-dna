"""Missense readout scaling across MarinDNA and Evo 2.

Adapts the combined missense result from issue #341 to the blog figure system.
Both families use the same Mendelian variants, the same frozen-embedding probe,
and the same chromosome-weighted AUPRC implementation. Color and line style
encode scoring protocol; marker shape encodes model family.

Run from ``plots/blog/genomic_lm_optimization/src``:

    uv run --project ../../../.. python -m figures.figure6b_marin_evo2_missense

Outputs:

    plots/output/blog/genomic_lm_optimization/
        figure6b_marin_evo2_missense.{png,pdf,svg}
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator
import pandas as pd
import polars as pl

from figures.data import save
from figures.figure5_params_vs_vep_auprc import (
    READOUTS,
    load_parameter_scaling_metrics,
)
from marin_dna.pipelines.evals.leaderboard import (
    EVO2_DATASET_SHORT,
    EVO2_PROBE_METRICS_GIST_BASE,
)
from utils.figure_style import X_LABEL_PAD, figsize

DATASET = "mendelian_traits"
SUBSET = "missense_variant"
MARIN = "MarinDNA"
EVO2 = "Evo 2"
FAMILIES: tuple[tuple[str, str], ...] = (
    (MARIN, "o"),
    (EVO2, "D"),
)
EVO2_MODELS: tuple[tuple[str, float], ...] = (
    ("evo2_1b_base", 1.0e9),
    ("evo2_7b", 7.0e9),
    ("evo2_40b", 40.0e9),
)


def _format_parameter_count(value: float, _position: int) -> str:
    """Compact labels for Matplotlib-selected log-scale parameter ticks."""
    if value >= 1e9:
        return f"{value / 1e9:g}B"
    if value >= 1e6:
        return f"{value / 1e6:g}M"
    return f"{value:g}"


def _evo2_path(model: str) -> str:
    short = EVO2_DATASET_SHORT[DATASET]
    return f"{EVO2_PROBE_METRICS_GIST_BASE}/{short}_{model}_train_probe_metrics.parquet"


def load_missense_comparison(
    marin_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load paired missense AUPRCs for both model families."""
    if marin_metrics is None:
        marin_metrics = load_parameter_scaling_metrics()
    required = {"dataset", "subset", "score_type", "value", "se", "n", "params"}
    assert required.issubset(marin_metrics.columns), (
        f"MarinDNA metrics missing {sorted(required - set(marin_metrics.columns))}"
    )
    marin = marin_metrics[
        (marin_metrics["dataset"] == DATASET) & (marin_metrics["subset"] == SUBSET)
    ][["score_type", "value", "se", "n", "params"]].copy()
    assert len(marin) == 8 * len(READOUTS)
    marin["family"] = MARIN

    evo_frames: list[pd.DataFrame] = []
    for model, params in EVO2_MODELS:
        path = _evo2_path(model)
        print(f"Reading {path}")
        frame = pl.read_parquet(path).filter(
            (pl.col("subset") == SUBSET)
            & pl.col("score_type").is_in([score for score, *_ in READOUTS])
            & (pl.col("split") == "train")
        )
        assert frame.height == len(READOUTS), (
            f"{model}: expected {len(READOUTS)} missense score rows, got {frame.height}"
        )
        evo = frame.select("score_type", "value", "se", "n").to_pandas()
        evo["params"] = params
        evo["family"] = EVO2
        evo_frames.append(evo)

    result = pd.concat([marin, *evo_frames], ignore_index=True)
    assert len(result) == (8 + len(EVO2_MODELS)) * len(READOUTS)
    assert result["value"].between(0, 1).all()
    assert result["se"].notna().all()
    assert (result["se"] >= 0).all()
    assert result["n"].nunique() == 1, (
        f"MarinDNA and Evo 2 missense rows differ: {sorted(result['n'].unique())}"
    )
    expected_cells = {
        (family, score) for family, _marker in FAMILIES for score, *_rest in READOUTS
    }
    assert (
        set(zip(result["family"], result["score_type"], strict=True)) == expected_cells
    )
    return result


def build(marin_metrics: pd.DataFrame | None = None) -> None:
    """Build the combined MarinDNA + Evo 2 missense scaling figure."""
    data = load_missense_comparison(marin_metrics)
    fig, ax = plt.subplots(figsize=figsize(7.2, 7.2))

    for family, marker in FAMILIES:
        for score_type, _label, color, _score_marker, linestyle in READOUTS:
            series = data[
                (data["family"] == family) & (data["score_type"] == score_type)
            ].sort_values("params")
            expected = 8 if family == MARIN else len(EVO2_MODELS)
            assert len(series) == expected
            ax.errorbar(
                series["params"],
                series["value"] * 100.0,
                yerr=series["se"] * 100.0,
                color=color,
                ecolor=color,
                marker=marker,
                linestyle=linestyle,
                capsize=0,
                markeredgecolor="#1f1e1b",
                zorder=3,
            )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_parameter_count))
    ax.set_xlabel("Model parameters", labelpad=X_LABEL_PAD)
    ax.set_ylabel("AUPRC (%)")
    ax.grid(False)
    ax.margins(x=0.04, y=0.12)
    ax.set_box_aspect(1)

    protocol_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle=linestyle,
            label=label,
        )
        for _score, label, color, _marker, linestyle in READOUTS
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            color="#4b4b4b",
            marker=marker,
            linestyle="none",
            markeredgecolor="#1f1e1b",
            label=family,
        )
        for family, marker in FAMILIES
    ]
    protocol_legend = fig.legend(
        handles=protocol_handles,
        title="Scoring protocol",
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.34, 0.99),
        frameon=False,
        handlelength=2.4,
        columnspacing=1.4,
    )
    fig.add_artist(protocol_legend)
    fig.legend(
        handles=family_handles,
        title="Model family",
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.76, 0.99),
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.subplots_adjust(top=0.82)
    save(fig, "figure6b_marin_evo2_missense")


if __name__ == "__main__":
    build()
