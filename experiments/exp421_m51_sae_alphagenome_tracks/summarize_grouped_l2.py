"""Create compact, reproducible summaries of the issue #421 grouped-L2 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

PRIMARY_STATISTIC = "pearson_abs_delta"
BLOCK_ORDER = ["1", "10", "19"]
ORIENTATION_ORDER = ["forward", "reverse_complement"]
RESOLUTION_ORDER = [
    "overall",
    "assay",
    "tissue",
    "cell_lineage",
    "assay_tissue",
    "assay_cell_lineage",
]


def read_primary_summary(result_root: Path) -> pl.DataFrame:
    summary = pl.read_parquet(result_root / "family_summary.parquet")
    required = {
        "report_block",
        "orientation",
        "resolution",
        "statistic",
        "target_id",
        "track_count",
        "eligible_features",
        "q_lt_0_05_and_abs_effect_ge_0_05",
        "max_abs_effect",
        "winning_feature_id",
        "winning_effect",
        "winning_qvalue",
    }
    assert required <= set(summary.columns)
    primary = summary.filter(pl.col("statistic") == PRIMARY_STATISTIC)
    assert set(primary["report_block"].unique()) == set(BLOCK_ORDER)
    assert set(primary["orientation"].unique()) == set(ORIENTATION_ORDER)
    assert set(primary["resolution"].unique()) == set(RESOLUTION_ORDER)
    assert primary.height == 2 * 3 * (1 + 7 + 17 + 12 + 105 + 70)
    assert primary.select(sorted(required)).null_count().sum_horizontal().sum() == 0
    return primary.with_columns(
        pl.col("winning_effect").abs().alias("absolute_winning_effect"),
        pl.col("report_block")
        .replace_strict({"1": "block 1", "10": "block 10", "19": "block 19"})
        .alias("layer"),
        pl.col("orientation")
        .replace_strict({"forward": "FWD", "reverse_complement": "RC"})
        .alias("strand"),
    )


def resolution_overview(primary: pl.DataFrame) -> pl.DataFrame:
    return (
        primary.group_by(
            ["report_block", "layer", "orientation", "strand", "resolution"]
        )
        .agg(
            pl.len().alias("outcomes"),
            pl.first("eligible_features"),
            pl.col("absolute_winning_effect").median().alias("median_winner_abs_r"),
            pl.col("absolute_winning_effect").quantile(0.9).alias("p90_winner_abs_r"),
            pl.col("absolute_winning_effect").max().alias("max_winner_abs_r"),
            (pl.col("q_lt_0_05_and_abs_effect_ge_0_05") > 0)
            .sum()
            .alias("outcomes_with_material_pair"),
            pl.col("q_lt_0_05_and_abs_effect_ge_0_05").sum().alias("material_pairs"),
        )
        .sort(["resolution", "report_block", "orientation"])
    )


def winner_recurrence(primary: pl.DataFrame) -> pl.DataFrame:
    return (
        primary.group_by(
            [
                "report_block",
                "layer",
                "orientation",
                "strand",
                "resolution",
                "winning_feature_id",
            ]
        )
        .agg(
            pl.len().alias("outcomes_won"),
            pl.col("absolute_winning_effect").median().alias("median_abs_r"),
            pl.col("absolute_winning_effect").max().alias("max_abs_r"),
        )
        .with_columns(
            pl.col("outcomes_won")
            .sum()
            .over(["report_block", "orientation", "resolution"])
            .alias("outcomes_in_family")
        )
        .with_columns(
            (pl.col("outcomes_won") / pl.col("outcomes_in_family")).alias(
                "winner_fraction"
            )
        )
        .sort(
            ["report_block", "orientation", "resolution", "outcomes_won"],
            descending=[False, False, False, True],
        )
    )


def orientation_concordance(primary: pl.DataFrame) -> pl.DataFrame:
    keys = ["report_block", "layer", "resolution", "target_id"]
    forward = primary.filter(pl.col("orientation") == "forward").select(
        *keys,
        pl.col("winning_feature_id").alias("fwd_feature"),
        pl.col("winning_effect").alias("fwd_effect"),
    )
    reverse = primary.filter(pl.col("orientation") == "reverse_complement").select(
        *keys,
        pl.col("winning_feature_id").alias("rc_feature"),
        pl.col("winning_effect").alias("rc_effect"),
    )
    joined = forward.join(reverse, on=keys, how="inner", validate="1:1")
    assert joined.height == forward.height == reverse.height
    rows: list[dict[str, object]] = []
    for group_key, group in joined.group_by(
        ["report_block", "layer", "resolution"], maintain_order=True
    ):
        fwd = group["fwd_effect"].to_numpy()
        rc = group["rc_effect"].to_numpy()
        correlation = float(np.corrcoef(fwd, rc)[0, 1]) if len(group) >= 3 else None
        rows.append(
            {
                "report_block": group_key[0],
                "layer": group_key[1],
                "resolution": group_key[2],
                "outcomes": len(group),
                "same_winner": int((group["fwd_feature"] == group["rc_feature"]).sum()),
                "same_winner_fraction": float(
                    (group["fwd_feature"] == group["rc_feature"]).mean()
                ),
                "winner_effect_pearson_r": correlation,
            }
        )
    return pl.DataFrame(rows).sort(["resolution", "report_block"])


def track_count_sensitivity(primary: pl.DataFrame) -> pl.DataFrame:
    combinations = primary.filter(
        pl.col("resolution").is_in(["assay_tissue", "assay_cell_lineage"])
    ).with_columns(
        pl.when(pl.col("track_count") == 1)
        .then(pl.lit("1"))
        .when(pl.col("track_count") < 5)
        .then(pl.lit("2–4"))
        .when(pl.col("track_count") < 10)
        .then(pl.lit("5–9"))
        .otherwise(pl.lit("10+"))
        .alias("track_count_bin")
    )
    return (
        combinations.group_by(
            [
                "report_block",
                "layer",
                "orientation",
                "strand",
                "resolution",
                "track_count_bin",
            ]
        )
        .agg(
            pl.len().alias("outcomes"),
            pl.col("absolute_winning_effect").median().alias("median_winner_abs_r"),
            pl.col("absolute_winning_effect").max().alias("max_winner_abs_r"),
        )
        .sort(["resolution", "report_block", "orientation", "track_count_bin"])
    )


def save_resolution_plot(primary: pl.DataFrame, output_dir: Path) -> None:
    plot_data = primary.to_pandas()
    resolution_labels = {
        "overall": "Overall",
        "assay": "Assay",
        "tissue": "Tissue",
        "cell_lineage": "Cell lineage",
        "assay_tissue": "Assay × tissue",
        "assay_cell_lineage": "Assay × cell lineage",
    }
    plot_data["resolution_label"] = plot_data["resolution"].map(resolution_labels)
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.catplot(
        data=plot_data,
        x="layer",
        y="absolute_winning_effect",
        hue="strand",
        col="resolution_label",
        col_wrap=3,
        col_order=list(resolution_labels.values()),
        kind="box",
        order=["block 1", "block 10", "block 19"],
        hue_order=["FWD", "RC"],
        sharey=True,
        showfliers=False,
        height=4.0,
        aspect=1.1,
    )
    grid.set_axis_labels("", "")
    grid.set_titles("{col_name}")
    grid.figure.supxlabel("SAE layer", y=0.015)
    grid.figure.supylabel("Per-outcome winner |Pearson r|", x=0.01)
    grid.figure.subplots_adjust(top=0.88, bottom=0.14, left=0.07)
    grid.figure.suptitle(
        "AlphaGenome L2 association is much stronger after the first layer"
    )
    for suffix in ("png", "svg"):
        grid.figure.savefig(
            output_dir / f"winner_effect_by_layer_resolution.{suffix}",
            bbox_inches="tight",
            dpi=180,
        )
    plt.close(grid.figure)


def save_assay_plot(primary: pl.DataFrame, output_dir: Path) -> None:
    assay = primary.filter(pl.col("resolution") == "assay").to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=assay,
        x="target_id",
        y="absolute_winning_effect",
        hue="layer",
        hue_order=["block 1", "block 10", "block 19"],
        col="strand",
        col_order=["FWD", "RC"],
        kind="line",
        marker="o",
        dashes=False,
        estimator=None,
        height=4.3,
        aspect=1.35,
    )
    grid.set_axis_labels("AlphaGenome assay", "Winner |Pearson r|")
    grid.set_titles("{col_name}")
    for axis in grid.axes.flat:
        axis.tick_params(axis="x", rotation=35)
    grid.figure.subplots_adjust(top=0.84)
    grid.figure.suptitle("Accessibility and promoter-initiation outcomes dominate")
    for suffix in ("png", "svg"):
        grid.figure.savefig(
            output_dir / f"assay_winner_effects.{suffix}",
            bbox_inches="tight",
            dpi=180,
        )
    plt.close(grid.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    primary = read_primary_summary(args.result_root)
    outputs = {
        "primary_outcome_winners.parquet": primary,
        "resolution_overview.csv": resolution_overview(primary),
        "winner_recurrence.csv": winner_recurrence(primary),
        "orientation_concordance.csv": orientation_concordance(primary),
        "track_count_sensitivity.csv": track_count_sensitivity(primary),
    }
    for name, frame in outputs.items():
        path = args.output_dir / name
        if path.suffix == ".parquet":
            frame.write_parquet(path)
        else:
            frame.write_csv(path)
    save_resolution_plot(primary, args.output_dir)
    save_assay_plot(primary, args.output_dir)
    metadata = {
        "primary_statistic": PRIMARY_STATISTIC,
        "source_result_root": str(args.result_root),
        "source_manifest_sha256": __import__("hashlib")
        .sha256((args.result_root / "manifest.json").read_bytes())
        .hexdigest(),
        "rows": {name: frame.height for name, frame in outputs.items()},
    }
    (args.output_dir / "summary_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
