"""Add the issue #485 511 and 1023 bp extensions to both context-response figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

TRAINED_CONTEXTS = (31, 63, 127, 255)
EXTENSION_CONTEXTS = (511, 1023)
CONTEXTS = (*TRAINED_CONTEXTS, *EXTENSION_CONTEXTS)
PANEL_ORDER = (
    "_macro_avg_",
    "missense_variant",
    "splicing",
    "synonymous_variant",
    "tss_proximal",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "distal",
    "non_coding_transcript_exon_variant",
)
EXCLUDED_SUBSETS = {"mature_miRNA_variant"}
PANEL_LABELS = {
    "_macro_avg_": "Macro Avg",
    "missense_variant": "Missense",
    "splicing": "Splicing",
    "synonymous_variant": "Synonymous",
    "tss_proximal": "Promoter",
    "5_prime_UTR_variant": "5′ UTR",
    "3_prime_UTR_variant": "3′ UTR",
    "distal": "Distal",
    "non_coding_transcript_exon_variant": "ncRNA",
}
COLOR = "#1f77b4"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _regime(context_bp: pd.Series) -> np.ndarray:
    return np.select(
        [context_bp < 255, context_bp == 255],
        ["context reduction", "training context"],
        default="inference extension",
    )


def load_zero_shot(
    source_path: Path,
    metrics_paths: dict[int, Path],
) -> pd.DataFrame:
    trained = pd.read_csv(source_path)
    trained = trained.loc[trained["context_bp"].isin(TRAINED_CONTEXTS)].copy()
    _require(set(trained["context_bp"]) == set(TRAINED_CONTEXTS), "trained zero-shot contexts mismatch")
    _require(
        set(trained["subset"]) == set(PANEL_ORDER),
        "trained zero-shot subsets mismatch",
    )
    _require(
        not trained.duplicated(["context_bp", "subset"]).any(),
        "duplicate trained zero-shot rows",
    )

    extensions: list[pd.DataFrame] = []
    required = {
        "score_type",
        "subset",
        "value",
        "se",
        "n_groups",
        "n_rows",
        "dataset",
        "split",
    }
    for context_bp, metrics_path in metrics_paths.items():
        metrics = pd.read_parquet(metrics_path)
        _require(required.issubset(metrics.columns), f"{context_bp} bp zero-shot schema mismatch")
        _require(set(metrics["dataset"]) == {"mendelian_traits"}, "wrong zero-shot dataset")
        _require(set(metrics["split"]) == {"train"}, "wrong zero-shot split")
        metrics = metrics.loc[
            (metrics["score_type"] == "minus_llr_avg")
            & (metrics["subset"] != "_global_")
        ].copy()
        _require(
            set(metrics["subset"]) == set(PANEL_ORDER) | EXCLUDED_SUBSETS,
            f"{context_bp} bp zero-shot subsets mismatch",
        )
        extension = metrics.loc[metrics["subset"].isin(PANEL_ORDER)].copy()
        extension["context_bp"] = context_bp
        extensions.append(extension)

    columns = [
        "context_bp",
        "subset",
        "value",
        "se",
        "n_groups",
        "n_rows",
        "dataset",
        "split",
    ]
    source = pd.concat(
        [trained.loc[:, columns], *[item.loc[:, columns] for item in extensions]],
        ignore_index=True,
    )
    _validate_common(source)
    _require(source.groupby("subset")["n_groups"].nunique().eq(1).all(), "zero-shot group counts differ")
    _require(source.groupby("subset")["n_rows"].nunique().eq(1).all(), "zero-shot row counts differ")
    source["regime"] = _regime(source["context_bp"])
    return source.sort_values(["subset", "context_bp"], kind="stable").reset_index(drop=True)


def _probe_diagnostics(model_path: Path, context_bp: int) -> pd.DataFrame:
    classifiers = joblib.load(model_path)
    rows: list[dict[str, object]] = []
    for subset, record in classifiers.items():
        summary = record["c_summary"]
        rows.append(
            {
                "subset": subset,
                "c_min": summary["c_min"],
                "c_med": summary["c_med"],
                "c_max": summary["c_max"],
                "full_c": summary["full_c"],
                "n_at_low_edge": summary["n_at_low_edge"],
                "n_at_high_edge": summary["n_at_high_edge"],
                "low_edge_gain": summary["low_edge_gain"],
                "high_edge_gain": summary["high_edge_gain"],
                "at_edge": summary["at_edge"],
                "truncation_risk": summary["truncation_risk"],
            }
        )
    diagnostics = pd.DataFrame(rows)
    _require(
        not diagnostics.duplicated(["subset"]).any(),
        f"duplicate {context_bp} bp probe diagnostics",
    )
    return diagnostics


def load_probe(
    source_path: Path,
    extension_paths: dict[int, tuple[Path, Path]],
) -> pd.DataFrame:
    trained = pd.read_csv(source_path)
    trained = trained.loc[trained["context_bp"].isin(TRAINED_CONTEXTS)].copy()
    _require(set(trained["context_bp"]) == set(TRAINED_CONTEXTS), "trained probe contexts mismatch")
    _require(set(trained["subset"]) == set(PANEL_ORDER), "trained probe subsets mismatch")
    _require(
        not trained.duplicated(["context_bp", "subset"]).any(),
        "duplicate trained probe rows",
    )

    extensions: list[pd.DataFrame] = []
    required = {
        "score_type",
        "subset",
        "value",
        "se",
        "n",
        "n_pos",
        "n_chrom",
        "dataset",
        "split",
    }
    for context_bp, (metrics_path, model_path) in extension_paths.items():
        metrics = pd.read_parquet(metrics_path)
        _require(required.issubset(metrics.columns), f"{context_bp} bp probe schema mismatch")
        _require(set(metrics["dataset"]) == {"mendelian_traits"}, "wrong probe dataset")
        _require(set(metrics["split"]) == {"train"}, "wrong probe split")
        observed = metrics.loc[metrics["score_type"] == "probe_score"].copy()
        _require(
            set(observed["subset"]) == set(PANEL_ORDER) | EXCLUDED_SUBSETS,
            f"{context_bp} bp probe subsets mismatch",
        )
        observed = observed.loc[observed["subset"].isin(PANEL_ORDER)].copy()
        observed["context_bp"] = context_bp
        diagnostics = _probe_diagnostics(model_path, context_bp)
        extension = observed.merge(
            diagnostics,
            on="subset",
            how="left",
            validate="one_to_one",
        )
        extension["truncation_risk"] = extension["truncation_risk"].eq(True)
        extensions.append(extension)

    columns = [
        "context_bp",
        "subset",
        "value",
        "se",
        "n",
        "n_pos",
        "n_chrom",
        "c_min",
        "c_med",
        "c_max",
        "full_c",
        "n_at_low_edge",
        "n_at_high_edge",
        "low_edge_gain",
        "high_edge_gain",
        "at_edge",
        "truncation_risk",
        "dataset",
        "split",
    ]
    source = pd.concat(
        [trained.loc[:, columns], *[item.loc[:, columns] for item in extensions]],
        ignore_index=True,
    )
    source["truncation_risk"] = source["truncation_risk"].eq(True)
    _validate_common(source)
    for column in ("n", "n_pos", "n_chrom"):
        _require(source.groupby("subset")[column].nunique().eq(1).all(), f"probe {column} differs")
    source["regime"] = _regime(source["context_bp"])
    return source.sort_values(["subset", "context_bp"], kind="stable").reset_index(drop=True)


def _validate_common(source: pd.DataFrame) -> None:
    _require(set(source["context_bp"]) == set(CONTEXTS), "context grid mismatch")
    _require(set(source["subset"]) == set(PANEL_ORDER), "panel grid mismatch")
    _require(len(source) == len(CONTEXTS) * len(PANEL_ORDER), "row-count mismatch")
    _require(not source.duplicated(["context_bp", "subset"]).any(), "duplicate plot rows")
    _require(np.isfinite(source["value"]).all(), "AUPRC contains non-finite values")
    _require(np.isfinite(source["se"]).all(), "SE contains non-finite values")


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            color=COLOR,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.4,
            linewidth=1.7,
            label="Context reduction",
        ),
        Line2D(
            [],
            [],
            color=COLOR,
            marker="D",
            markerfacecolor=COLOR,
            markeredgecolor="white",
            linewidth=0,
            label="Training context (255 bp)",
        ),
        Line2D(
            [],
            [],
            color=COLOR,
            marker="s",
            markerfacecolor="white",
            markeredgewidth=1.5,
            linestyle="--",
            linewidth=1.7,
            label="Inference extension (>255 bp; OOD)",
        ),
    ]


def plot_facets(
    source: pd.DataFrame,
    output_svg: Path,
    preview_png: Path,
    *,
    zero_shot: bool,
) -> None:
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(8.1, 8.7),
        sharex=True,
        sharey=False,
        layout=None,
    )

    for axis, subset in zip(axes.ravel(), PANEL_ORDER, strict=True):
        panel = source.loc[source["subset"] == subset].sort_values("context_bp")
        is_macro = subset == "_macro_avg_"
        linewidth = 2.6 if is_macro else 1.7
        x = panel["context_bp"].to_numpy(dtype=float)
        values = panel["value"].to_numpy(dtype=float)
        errors = panel["se"].to_numpy(dtype=float)
        trained = x <= 255
        extended = x >= 255

        axis.plot(x[trained], values[trained], color=COLOR, linewidth=linewidth, zorder=2)
        axis.plot(
            x[extended],
            values[extended],
            color=COLOR,
            linewidth=linewidth,
            linestyle="--",
            zorder=2,
        )
        axis.errorbar(
            x,
            values,
            yerr=errors,
            fmt="none",
            ecolor=COLOR,
            elinewidth=1.1,
            capsize=0,
            zorder=1,
        )
        reduced = x < 255
        axis.scatter(
            x[reduced],
            values[reduced],
            marker="o",
            s=34 if not is_macro else 42,
            facecolor="white",
            edgecolor=COLOR,
            linewidth=1.4,
            zorder=3,
        )
        native = x == 255
        axis.scatter(
            x[native],
            values[native],
            marker="D",
            s=48 if not is_macro else 60,
            facecolor=COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        extension = x > 255
        axis.scatter(
            x[extension],
            values[extension],
            marker="s",
            s=42 if not is_macro else 52,
            facecolor="white",
            edgecolor=COLOR,
            linewidth=1.5,
            zorder=4,
        )

        if not zero_shot and panel["truncation_risk"].any():
            risky_contexts = panel.loc[panel["truncation_risk"], "context_bp"]
            label = ", ".join(str(int(value)) for value in risky_contexts)
            axis.text(
                0.04,
                0.96,
                f"C-grid risk: {label} bp",
                transform=axis.transAxes,
                ha="left",
                va="top",
                color="#A33A3A",
                fontsize="small",
            )

        axis.set_title(PANEL_LABELS[subset], fontweight="bold" if is_macro else "normal")
        axis.set_xscale("log", base=2)
        axis.set_xticks(CONTEXTS, labels=[str(value) for value in CONTEXTS])
        axis.set_xlim(26, 1160)
        lower = max(0.0, float(np.min(values - errors)))
        upper = min(1.0, float(np.max(values + errors)))
        span = max(upper - lower, 0.05)
        axis.set_ylim(max(0.0, lower - 0.08 * span), min(1.0, upper + 0.08 * span))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.set_box_aspect(1)

    if zero_shot:
        title = "m5.1 zero-shot Mendelian VEP across inference contexts"
        ylabel = "AUPRC (±1 SE)"
    else:
        title = "m5.1 frozen-probe Mendelian VEP across inference contexts"
        ylabel = "AUPRC (±1 SE)"

    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.085,
        top=0.855,
        wspace=0.22,
        hspace=0.25,
    )
    figure.suptitle(title, y=0.985)
    figure.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.545, 0.952),
        ncol=3,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    figure.supxlabel("Inference context (bp; log₂ scale)", y=0.02)
    figure.supylabel(ylabel, x=0.015)
    figure.savefig(output_svg, format="svg")
    figure.savefig(preview_png, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-shot-source", type=Path, required=True)
    parser.add_argument("--probe-source", type=Path, required=True)
    parser.add_argument("--zero-shot-metrics-511", type=Path, required=True)
    parser.add_argument("--probe-metrics-511", type=Path, required=True)
    parser.add_argument("--probe-model-511", type=Path, required=True)
    parser.add_argument("--zero-shot-metrics-1023", type=Path, required=True)
    parser.add_argument("--probe-metrics-1023", type=Path, required=True)
    parser.add_argument("--probe-model-1023", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    zero_shot = load_zero_shot(
        args.zero_shot_source,
        {
            511: args.zero_shot_metrics_511,
            1023: args.zero_shot_metrics_1023,
        },
    )
    probe = load_probe(
        args.probe_source,
        {
            511: (args.probe_metrics_511, args.probe_model_511),
            1023: (args.probe_metrics_1023, args.probe_model_1023),
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zero_shot.to_csv(
        args.output_dir / "zero_shot_source.csv",
        index=False,
        float_format="%.8g",
    )
    probe.to_csv(
        args.output_dir / "probe_source.csv",
        index=False,
        float_format="%.8g",
    )
    plot_facets(
        zero_shot,
        args.output_dir / "zero_shot_context.svg",
        args.output_dir / "zero_shot_context.preview.png",
        zero_shot=True,
    )
    plot_facets(
        probe,
        args.output_dir / "probe_context.svg",
        args.output_dir / "probe_context.preview.png",
        zero_shot=False,
    )


if __name__ == "__main__":
    main()
