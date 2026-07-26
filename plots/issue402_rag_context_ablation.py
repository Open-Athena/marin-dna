#!/usr/bin/env python3
"""Plot issue #402 ortholog-content and special-token loss ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

SANITY_ROOTS = {
    "46M": (
        "gs://marin-us-east5/evals/"
        "dna-exp402-rag-h640-p46m-30k/2026.07.26/sanity-ac7016"
    ),
    "104M": (
        "gs://marin-us-east5/evals/"
        "dna-exp402-rag-h768-p104m-30k/2026.07.26/sanity-ac7016"
    ),
}
MODEL_ORDER = ["46M", "104M"]
MODEL_COLORS = {"46M": "#3366cc", "104M": "#d95f02"}
GROUP_MODES = {
    "Ortholog content": [
        ("full", "Full"),
        ("roll", "Roll +31 bp"),
        ("all_n", "All N"),
        ("unrelated", "Unrelated"),
    ],
    "Special tokens": [
        ("full", "Full"),
        ("bos_to_pad", "BOS→PAD"),
        ("seq_to_unk", "SEQ→UNK"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-46m", default=SANITY_ROOTS["46M"])
    parser.add_argument("--input-104m", default=SANITY_ROOTS["104M"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_context_ablation"),
    )
    return parser.parse_args()


def load_ablations(roots: dict[str, str] = SANITY_ROOTS) -> pl.DataFrame:
    """Load each model's frozen human-token loss ablations."""
    assert set(roots) == set(MODEL_ORDER)
    source = pl.concat(
        [
            pl.read_parquet(f"{root}/validation_context_ablation.parquet")
            for root in roots.values()
        ]
    )
    assert source.height == len(roots) * 6
    rows: list[dict[str, object]] = []
    for group, modes in GROUP_MODES.items():
        for index, (mode, label) in enumerate(modes):
            selected = source.filter(pl.col("mode") == mode)
            assert selected.height == len(roots)
            rows.extend(
                {
                    **row,
                    "group": group,
                    "ablation_index": index,
                    "ablation": label,
                }
                for row in selected.to_dicts()
            )
    data = pl.DataFrame(rows)
    assert data.height == len(roots) * sum(map(len, GROUP_MODES.values()))
    assert data.filter(
        ~pl.col("mean_human_loss").is_finite() | ~pl.col("se_human_loss").is_finite()
    ).is_empty()
    return data


def plot_ablations(data: pl.DataFrame, output_dir: Path) -> None:
    """Render independently scaled content and token ablation facets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    frame = data.to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=frame,
        x="ablation_index",
        y="mean_human_loss",
        hue="model",
        hue_order=MODEL_ORDER,
        style="model",
        style_order=MODEL_ORDER,
        col="group",
        col_order=list(GROUP_MODES),
        kind="line",
        markers=True,
        dashes=False,
        palette=MODEL_COLORS,
        height=4.5,
        aspect=1.2,
        facet_kws={"sharex": False, "sharey": False},
    )
    grid.set_axis_labels("", "Human next-token NLL")
    for group, axis in grid.axes_dict.items():
        subset = frame[frame["group"] == group]
        labels = [label for _, label in GROUP_MODES[group]]
        axis.set_title(group)
        axis.set_xticks(range(len(labels)), labels, rotation=15, ha="right")
        for model in MODEL_ORDER:
            model_rows = subset[subset["model"] == model].sort_values("ablation_index")
            axis.errorbar(
                model_rows["ablation_index"],
                model_rows["mean_human_loss"],
                yerr=model_rows["se_human_loss"],
                fmt="none",
                ecolor=MODEL_COLORS[model],
                elinewidth=1.2,
                capsize=0,
                alpha=0.85,
            )
    grid.figure.suptitle(
        "Ortholog content and [SEQ] markers materially affect human-token loss"
    )
    grid.figure.text(
        0.5,
        0.015,
        "Means ±1 SE over 512 documents. Facets use independent y-axes; "
        "all targets and human input tokens are held fixed.",
        ha="center",
        fontsize=10,
    )
    grid.figure.subplots_adjust(top=0.82, bottom=0.24, wspace=0.25)
    grid.figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    args = parse_args()
    data = load_ablations({"46M": args.input_46m, "104M": args.input_104m})
    plot_ablations(data, args.output_dir)
    print(
        data.sort("group", "model", "ablation_index").select(
            "model", "group", "ablation", "mean_human_loss", "se_human_loss"
        )
    )


if __name__ == "__main__":
    main()
