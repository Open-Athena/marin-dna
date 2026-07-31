"""Parameter scaling: chromosome-weighted AUPRC by consequence and readout.

Both zero-shot LLR and the frozen-embedding linear probe are recomputed from
the same probe-prediction rows with ``per_chrom_ap_table``. This avoids mixing
the pooled / matched-pair / per-gene AUPRC variants used by older versions of
the blog figures.

Run from ``plots/blog/genomic_lm_optimization/src``:

    uv run --project ../../../.. python -m figures.figure5_params_vs_vep_auprc

Outputs:

    plots/output/blog/genomic_lm_optimization/
        figure5_params_vs_vep_auprc.{png,pdf,svg}
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figures.data import SCALING_RESULTS_PATH, save
from marin_dna.pipelines.evals.metrics import per_chrom_ap_table
from utils.figure_style import X_LABEL_PAD, figsize

FINAL_STEP = 215573
PROBE_RESULTS = (
    "s3://oa-bolinas/snakemake/analysis/evals_v2/results/probe/"
    "{model}/{dataset}.parquet"
)

MENDELIAN_SUBSETS: tuple[tuple[str, str], ...] = (
    ("missense_variant", "Missense"),
    ("splicing", "Splicing"),
    ("synonymous_variant", "Synonymous"),
    ("tss_proximal", "Promoter"),
    ("5_prime_UTR_variant", "5′ UTR"),
    ("3_prime_UTR_variant", "3′ UTR"),
)
SGE_SUBSETS: tuple[tuple[str, str], ...] = (
    ("missense_variant", "Missense"),
    ("splicing", "Splicing"),
)
SUBSETS = {
    "mendelian_traits": MENDELIAN_SUBSETS,
    "sge": SGE_SUBSETS,
}

# The two readouts are directly comparable within every panel: same variants,
# same chromosome-weighted AUPRC implementation, and the same score rows.
READOUTS: tuple[tuple[str, str, str, str, str], ...] = (
    ("minus_llr_avg", "Zero-shot LLR", "#465c6e", "o", "--"),
    ("probe_score", "Linear probe", "#9c4f2f", "s", "-"),
)


def _model_rows() -> pd.DataFrame:
    """Return the eight endpoint model IDs and their parameter counts."""
    metadata = pd.read_csv(SCALING_RESULTS_PATH)[["run_name", "params"]].copy()
    assert len(metadata) == 8, f"expected 8 scaling models, got {len(metadata)}"
    assert metadata["run_name"].is_unique
    assert metadata["params"].is_unique
    metadata["model"] = (
        metadata["run_name"].str.removeprefix("dna-bolinas-") + f"-step-{FINAL_STEP}"
    )
    return metadata.sort_values("params").reset_index(drop=True)


def _read_paired_metrics(model: str, dataset: str) -> pd.DataFrame:
    """Compute both readouts' AUPRC on identical finite probe-score rows."""
    path = PROBE_RESULTS.format(model=model, dataset=dataset)
    print(f"Reading {path}")
    data = pd.read_parquet(path)
    required = {
        "label",
        "subset",
        "chrom",
        "probe_score",
        "llr_fwd",
        "llr_rc",
    }
    assert required.issubset(data.columns), (
        f"{path} missing columns {sorted(required - set(data.columns))}"
    )

    # A probe can be absent for a subset that failed its training-data gate.
    # Restrict both readouts to the exact rows on which the probe is defined.
    finite_probe = np.isfinite(data["probe_score"].to_numpy(dtype=float))
    data = data.loc[finite_probe].copy()
    assert len(data) > 0, f"{path} has no finite probe predictions"
    data["minus_llr_avg"] = (
        -(data["llr_fwd"].to_numpy(dtype=float) + data["llr_rc"].to_numpy(dtype=float))
        / 2
    )
    assert np.isfinite(data["minus_llr_avg"]).all()

    metrics = per_chrom_ap_table(
        data,
        ["minus_llr_avg", "probe_score"],
        n_bootstrap=1000,
        rng=0,
    )
    wanted = {subset for subset, _ in SUBSETS[dataset]}
    metrics = metrics[metrics["subset"].isin(wanted)].copy()
    expected = {(score, subset) for score, *_ in READOUTS for subset in wanted}
    observed = set(zip(metrics["score_type"], metrics["subset"], strict=True))
    assert observed == expected, (
        f"{model}/{dataset}: expected metric cells {sorted(expected)}, "
        f"got {sorted(observed)}"
    )
    assert metrics["value"].between(0, 1).all()
    assert metrics["se"].notna().all()
    return metrics


def load_parameter_scaling_metrics() -> pd.DataFrame:
    """Load all eight endpoints for both datasets and both paired readouts."""
    frames: list[pd.DataFrame] = []
    for row in _model_rows().itertuples(index=False):
        for dataset in SUBSETS:
            metrics = _read_paired_metrics(row.model, dataset)
            metrics["dataset"] = dataset
            metrics["params"] = int(row.params)
            metrics["model"] = row.model
            frames.append(metrics)
    result = pd.concat(frames, ignore_index=True)
    expected_rows = (
        8 * sum(len(subsets) for subsets in SUBSETS.values()) * len(READOUTS)
    )
    assert len(result) == expected_rows, (
        f"expected {expected_rows} endpoint metric rows, got {len(result)}"
    )
    return result


def _plot_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    title: str,
    show_ylabel: bool,
) -> None:
    """Draw paired readout scaling curves for one consequence."""
    assert set(data["score_type"]) == {score for score, *_ in READOUTS}
    for score_type, _label, color, marker, linestyle in READOUTS:
        series = data[data["score_type"] == score_type].sort_values("params")
        assert len(series) == 8
        ax.errorbar(
            series["params"],
            series["value"] * 100.0,
            yerr=series["se"] * 100.0,
            color=color,
            ecolor=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.35,
            elinewidth=0.9,
            capsize=0,
            markersize=5,
            markeredgecolor="#1f1e1b",
            markeredgewidth=0.45,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("Model parameters", labelpad=X_LABEL_PAD)
    if show_ylabel:
        ax.set_ylabel("AUPRC (%)")
    ax.grid(False)
    ax.margins(x=0.06, y=0.13)
    ax.set_box_aspect(1)


def build(data: pd.DataFrame | None = None) -> None:
    """Build the combined Mendelian + SGE parameter-scaling figure."""
    if data is None:
        data = load_parameter_scaling_metrics()

    mosaic = [
        [
            "m_missense",
            "m_missense",
            "m_splicing",
            "m_splicing",
            "m_synonymous",
            "m_synonymous",
        ],
        ["m_promoter", "m_promoter", "m_5utr", "m_5utr", "m_3utr", "m_3utr"],
        [".", "s_missense", "s_missense", "s_splicing", "s_splicing", "."],
    ]
    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=figsize(10.0, 9.2),
        gridspec_kw={"hspace": 0.52, "wspace": 0.16},
    )

    mendelian_axes = (
        ("m_missense", "missense_variant", "Missense"),
        ("m_splicing", "splicing", "Splicing"),
        ("m_synonymous", "synonymous_variant", "Synonymous"),
        ("m_promoter", "tss_proximal", "Promoter"),
        ("m_5utr", "5_prime_UTR_variant", "5′ UTR"),
        ("m_3utr", "3_prime_UTR_variant", "3′ UTR"),
    )
    for index, (axis_name, subset, title) in enumerate(mendelian_axes):
        panel = data[
            (data["dataset"] == "mendelian_traits") & (data["subset"] == subset)
        ]
        _plot_panel(
            axes[axis_name],
            panel,
            title=title,
            show_ylabel=index in (0, 3),
        )

    sge_axes = (
        ("s_missense", "missense_variant", "Missense"),
        ("s_splicing", "splicing", "Splicing"),
    )
    for index, (axis_name, subset, title) in enumerate(sge_axes):
        panel = data[(data["dataset"] == "sge") & (data["subset"] == subset)]
        _plot_panel(
            axes[axis_name],
            panel,
            title=title,
            show_ylabel=index == 0,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.35,
            markeredgecolor="#1f1e1b",
            markeredgewidth=0.45,
            markersize=5,
        )
        for _score, _label, color, marker, linestyle in READOUTS
    ]
    labels = [label for _score, label, *_rest in READOUTS]
    fig.legend(
        handles,
        labels,
        title="Scoring protocol",
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.8,
    )
    # Dataset headers sit above their respective facet blocks. Their vertical
    # positions are derived from the live axes rather than hard-coded pixels.
    fig.canvas.draw()
    mendelian_top = max(axes[name].get_position().y1 for name, *_ in mendelian_axes)
    sge_top = max(axes[name].get_position().y1 for name, *_ in sge_axes)
    fig.text(0.02, mendelian_top + 0.012, "Mendelian", weight="bold")
    fig.text(0.02, sge_top + 0.012, "SGE", weight="bold")
    save(fig, "figure5_params_vs_vep_auprc")


if __name__ == "__main__":
    build()
