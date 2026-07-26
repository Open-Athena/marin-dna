#!/usr/bin/env python3
"""Plot indel-mapped versus same-offset attention for issue #402."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

ATTENTION_ROOT = (
    "gs://marin-us-east5/evals/issue402-rag-30k/2026.07.26/indel-attention-ac7016"
)
MODEL_ORDER = ["46M", "104M"]
SCOPE_ORDER = ["All 13 orthologs", "Bos taurus", "Tolypeutes matacus"]
SCOPE_COLORS = {
    "All 13 orthologs": "#252525",
    "Bos taurus": "#2b8cbe",
    "Tolypeutes matacus": "#e6550d",
}
MANUAL_EXAMPLES = {
    "Bos taurus": ("win_18_000432221", 3, "Bos_taurus"),
    "Tolypeutes matacus": ("win_18_000064770", 2, "Tolypeutes_matacus"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=ATTENTION_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_indel_attention"),
    )
    return parser.parse_args()


def _scope_rows(source: pl.DataFrame, scope: str) -> pl.DataFrame:
    if scope == "All 13 orthologs":
        scoped = source
    else:
        anchor_id, slot, species = MANUAL_EXAMPLES[scope]
        scoped = source.filter(
            (pl.col("anchor_id") == anchor_id)
            & (pl.col("slot") == slot)
            & (pl.col("species") == species)
        )
    assert scoped.height > 0
    return scoped.with_columns(pl.lit(scope).alias("scope"))


def load_attention_metrics(
    attention_root: str = ATTENTION_ROOT,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Aggregate shifted-position attention by layer and over all layers."""
    source = pl.concat(
        [
            pl.read_parquet(
                f"{attention_root.rstrip('/')}/{model}/mapped_attention.parquet"
            )
            for model in MODEL_ORDER
        ]
    ).filter(pl.col("shift") != 0)
    assert source.height > 10_000
    assert sorted(source["model"].unique()) == sorted(MODEL_ORDER)
    assert source.select("anchor_id", "slot").n_unique() == 13
    assert source.filter(
        ~pl.col("mapped_attention").is_finite()
        | ~pl.col("naive_attention").is_finite()
        | (pl.col("mapped_attention") < 0)
        | (pl.col("naive_attention") < 0)
    ).is_empty()
    scoped = pl.concat([_scope_rows(source, scope) for scope in SCOPE_ORDER])
    layer_metrics = (
        scoped.group_by("model", "scope", "layer")
        .agg(
            pl.len().alias("n_attention_rows"),
            pl.col("mapped_attention").mean().alias("mapped_attention"),
            pl.col("naive_attention").mean().alias("naive_attention"),
            (pl.col("mapped_attention") > pl.col("naive_attention"))
            .mean()
            .alias("mapped_wins_fraction"),
        )
        .with_columns(
            (pl.col("mapped_attention") / pl.col("naive_attention")).alias(
                "mapped_to_naive_ratio"
            )
        )
        .sort("model", "scope", "layer")
    )
    aggregate_metrics = (
        scoped.group_by("model", "scope")
        .agg(
            pl.len().alias("n_attention_rows"),
            pl.col("mapped_attention").mean().alias("mapped_attention"),
            pl.col("naive_attention").mean().alias("naive_attention"),
            (pl.col("mapped_attention") > pl.col("naive_attention"))
            .mean()
            .alias("mapped_wins_fraction"),
        )
        .with_columns(
            (pl.col("mapped_attention") / pl.col("naive_attention")).alias(
                "mapped_to_naive_ratio"
            )
        )
        .sort("model", "scope")
    )
    assert layer_metrics.filter(
        ~pl.col("mapped_to_naive_ratio").is_finite()
        | (pl.col("mapped_to_naive_ratio") <= 0)
    ).is_empty()
    assert aggregate_metrics.height == len(MODEL_ORDER) * len(SCOPE_ORDER)
    if attention_root.rstrip("/") == ATTENTION_ROOT.rstrip("/"):
        expected_aggregate = {
            ("46M", "All 13 orthologs"): 10.9385,
            ("46M", "Bos taurus"): 5.6197,
            ("46M", "Tolypeutes matacus"): 16.1735,
            ("104M", "All 13 orthologs"): 6.2013,
            ("104M", "Bos taurus"): 4.9024,
            ("104M", "Tolypeutes matacus"): 9.7635,
        }
        observed = {
            (row["model"], row["scope"]): row["mapped_to_naive_ratio"]
            for row in aggregate_metrics.to_dicts()
        }
        assert observed.keys() == expected_aggregate.keys()
        for key, expected in expected_aggregate.items():
            assert abs(observed[key] - expected) < 5e-4, (
                key,
                observed[key],
                expected,
            )
    return layer_metrics, aggregate_metrics


def plot_attention(
    layer_metrics: pl.DataFrame,
    aggregate_metrics: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Render per-layer ratios with the same-offset position as a control."""
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_metrics.write_parquet(
        output_dir / "layer_metrics.parquet", compression="zstd"
    )
    aggregate_metrics.write_parquet(
        output_dir / "aggregate_metrics.parquet", compression="zstd"
    )
    frame = layer_metrics.to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=frame,
        x="layer",
        y="mapped_to_naive_ratio",
        hue="scope",
        hue_order=SCOPE_ORDER,
        style="scope",
        style_order=SCOPE_ORDER,
        col="model",
        col_order=MODEL_ORDER,
        kind="line",
        markers=True,
        dashes=False,
        palette=SCOPE_COLORS,
        height=4.9,
        aspect=1.1,
        facet_kws={"sharex": False, "sharey": True},
    )
    grid.set_axis_labels("Layer", "Attention ratio")
    grid.set_titles("{col_name} model")
    for axis in grid.axes.flat:
        axis.set_yscale("log")
        axis.axhline(1, color="#737373", linewidth=1, linestyle=":")
    grid.figure.suptitle(
        "Attention follows inferred ortholog positions across indel shifts"
    )
    grid.figure.text(
        0.5,
        0.012,
        "Shifted targets only; attention is averaged over heads and sampled human "
        "targets before taking the ratio. Mapping is inferred by global pairwise "
        "alignment.\n“All” pools 13 orthologs across two windows; the other lines "
        "are two manually inspected indel examples. Ratio 1 = same-offset control.",
        ha="center",
        fontsize=9.5,
    )
    grid.figure.subplots_adjust(top=0.82, bottom=0.23, wspace=0.18)
    grid.figure.savefig(output_dir / "figure.svg", bbox_inches="tight", pad_inches=0.25)
    grid.figure.savefig(
        output_dir / "figure.png", dpi=180, bbox_inches="tight", pad_inches=0.25
    )
    plt.close(grid.figure)


def main() -> None:
    args = parse_args()
    layer_metrics, aggregate_metrics = load_attention_metrics(args.input_root)
    plot_attention(layer_metrics, aggregate_metrics, args.output_dir)
    print(aggregate_metrics)


if __name__ == "__main__":
    main()
