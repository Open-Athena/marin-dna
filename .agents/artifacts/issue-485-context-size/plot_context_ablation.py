"""Build the issue #485 context-ablation source tables and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONTEXTS = (31, 63, 127, 255)
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


def _read_table(path: Path, context_bp: int) -> pd.DataFrame:
    _require(path.is_file(), f"missing input: {path}")
    table = pd.read_parquet(path)
    table["context_bp"] = context_bp
    return table


def load_zero_shot(input_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
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
    for context_bp in CONTEXTS:
        table = _read_table(input_dir / f"zero_shot_{context_bp}.parquet", context_bp)
        _require(required.issubset(table.columns), f"zero-shot schema mismatch at {context_bp} bp")
        _require(set(table["dataset"]) == {"mendelian_traits"}, f"wrong dataset at {context_bp} bp")
        _require(set(table["split"]) == {"train"}, f"wrong split at {context_bp} bp")
        table = table.loc[
            (table["score_type"] == "minus_llr_avg") & (table["subset"] != "_global_")
        ].copy()
        _require(
            set(table["subset"]) == set(PANEL_ORDER) | EXCLUDED_SUBSETS,
            f"zero-shot subset mismatch at {context_bp} bp",
        )
        table = table.loc[table["subset"].isin(PANEL_ORDER)].copy()
        _require(set(table["subset"]) == set(PANEL_ORDER), f"zero-shot subset mismatch at {context_bp} bp")
        _require(len(table) == len(PANEL_ORDER), f"duplicate zero-shot rows at {context_bp} bp")
        frames.append(table)

    source = pd.concat(frames, ignore_index=True)
    _require(np.isfinite(source["value"]).all(), "zero-shot AUPRC contains non-finite values")
    _require(np.isfinite(source["se"]).all(), "zero-shot SE contains non-finite values")
    _require(source.groupby("subset")["n_groups"].nunique().eq(1).all(), "zero-shot group counts differ by context")
    _require(source.groupby("subset")["n_rows"].nunique().eq(1).all(), "zero-shot row counts differ by context")
    return source.sort_values(["subset", "context_bp"], kind="stable").reset_index(drop=True)


def load_probe(input_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
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
    for context_bp in CONTEXTS:
        table = _read_table(input_dir / f"probe_{context_bp}.parquet", context_bp)
        _require(required.issubset(table.columns), f"probe schema mismatch at {context_bp} bp")
        _require(set(table["dataset"]) == {"mendelian_traits"}, f"wrong probe dataset at {context_bp} bp")
        _require(set(table["split"]) == {"train"}, f"wrong probe split at {context_bp} bp")
        table = table.loc[table["score_type"] == "probe_score"].copy()
        _require(not table.duplicated(["subset"]).any(), f"duplicate probe rows at {context_bp} bp")
        _require(
            set(table["subset"]) == set(PANEL_ORDER) | EXCLUDED_SUBSETS,
            f"probe subset mismatch at {context_bp} bp",
        )
        table = table.loc[table["subset"].isin(PANEL_ORDER)].copy()
        frames.append(table)

    observed = pd.concat(frames, ignore_index=True)
    diagnostic_rows: list[dict[str, object]] = []
    for context_bp in CONTEXTS:
        path = input_dir / f"probe_model_{context_bp}.joblib"
        _require(path.is_file(), f"missing input: {path}")
        classifiers = joblib.load(path)
        for subset, record in classifiers.items():
            summary = record["c_summary"]
            diagnostic_rows.append(
                {
                    "context_bp": context_bp,
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
    diagnostics = pd.DataFrame(diagnostic_rows)
    _require(
        not diagnostics.duplicated(["context_bp", "subset"]).any(),
        "duplicate probe diagnostic rows",
    )
    grid = pd.MultiIndex.from_product(
        [CONTEXTS, PANEL_ORDER], names=["context_bp", "subset"]
    ).to_frame(index=False)
    source = grid.merge(
        observed,
        on=["context_bp", "subset"],
        how="left",
        validate="one_to_one",
    )
    source = source.merge(
        diagnostics,
        on=["context_bp", "subset"],
        how="left",
        validate="one_to_one",
    )
    source["score_type"] = "probe_score"
    source["truncation_risk"] = source["truncation_risk"].eq(True)

    _require(np.isfinite(source["value"]).all(), "probe AUPRC contains non-finite values")
    _require(np.isfinite(source["se"]).all(), "probe SE contains non-finite values")
    for column in ("n", "n_pos", "n_chrom"):
        _require(
            source.groupby("subset")[column].nunique().eq(1).all(),
            f"probe {column} differs by context",
        )

    return source.sort_values(["subset", "context_bp"], kind="stable").reset_index(drop=True)


def _panel_title(subset: str) -> str:
    return PANEL_LABELS[subset]


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
        figsize=(8.1, 8.5),
        sharex=True,
        sharey=False,
        layout=None,
    )
    flat_axes = axes.ravel()

    for axis, subset in zip(flat_axes, PANEL_ORDER, strict=False):
        panel = source.loc[source["subset"] == subset].sort_values("context_bp")
        is_macro = subset == "_macro_avg_"
        color = COLOR
        linewidth = 2.6 if is_macro else 1.7
        values = panel["value"].to_numpy(dtype=float)
        errors = panel["se"].to_numpy(dtype=float)
        x = panel["context_bp"].to_numpy(dtype=float)
        valid = np.isfinite(values)
        _require(valid.all(), f"non-finite plotted values for {subset}")

        if valid.any():
            axis.plot(x, values, color=color, linewidth=linewidth, zorder=2)
            axis.errorbar(
                x[valid],
                values[valid],
                yerr=errors[valid],
                fmt="none",
                ecolor=color,
                elinewidth=1.1,
                capsize=0,
                zorder=1,
            )
            non_native = valid & (x != 255)
            axis.scatter(
                x[non_native],
                values[non_native],
                marker="o",
                s=34 if not is_macro else 42,
                facecolor="white",
                edgecolor=color,
                linewidth=1.4,
                zorder=3,
            )
            native = valid & (x == 255)
            axis.scatter(
                x[native],
                values[native],
                marker="D",
                s=48 if not is_macro else 60,
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )

        if zero_shot and not is_macro and int(panel["n_groups"].iloc[0]) < 30:
            axis.text(
                0.04,
                0.96,
                f"{int(panel['n_groups'].iloc[0])} groups; excluded from macro",
                transform=axis.transAxes,
                ha="left",
                va="top",
                color="#555555",
                fontsize="small",
            )

        if not zero_shot and panel["truncation_risk"].any():
            risky_contexts = panel.loc[
                panel["truncation_risk"], "context_bp"
            ]
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

        axis.set_title(
            _panel_title(subset),
            fontweight="bold" if is_macro else "normal",
        )
        axis.set_xscale("log", base=2)
        axis.set_xticks(CONTEXTS, labels=[str(value) for value in CONTEXTS])
        axis.set_xlim(26, 300)
        lower = max(0.0, float(np.min(values - errors)))
        upper = min(1.0, float(np.max(values + errors)))
        span = max(upper - lower, 0.05)
        axis.set_ylim(max(0.0, lower - 0.08 * span), min(1.0, upper + 0.08 * span))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.set_box_aspect(1)

    for axis in flat_axes[len(PANEL_ORDER) :]:
        axis.set_visible(False)

    if zero_shot:
        title = "m5.1 zero-shot Mendelian VEP across inference contexts"
        ylabel = "Matched-pair AUPRC (±1 match-group bootstrap SE)"
    else:
        title = "m5.1 frozen-probe Mendelian VEP across inference contexts"
        ylabel = "Per-chromosome-weighted AUPRC (±1 chromosome-bootstrap SE)"

    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.085,
        top=0.90,
        wspace=0.22,
        hspace=0.25,
    )
    figure.suptitle(title, y=0.985)
    figure.supxlabel("Inference context (bp; log₂ scale)", y=0.02)
    figure.supylabel(ylabel, x=0.015)
    figure.savefig(output_svg, format="svg")
    figure.savefig(preview_png, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    zero_shot = load_zero_shot(args.input_dir)
    probe = load_probe(args.input_dir)

    zero_columns = [
        "context_bp",
        "subset",
        "value",
        "se",
        "n_groups",
        "n_rows",
        "dataset",
        "split",
    ]
    probe_columns = [
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
    zero_shot.loc[:, zero_columns].to_csv(
        args.output_dir / "zero_shot_source.csv", index=False, float_format="%.8g"
    )
    probe.loc[:, probe_columns].to_csv(
        args.output_dir / "probe_source.csv", index=False, float_format="%.8g"
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
