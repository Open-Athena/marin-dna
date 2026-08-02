"""Plot complete-family SAE associations with broad variant consequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats

EXPECTED_LAYERS = (1, 10, 19)
EXPECTED_ORIENTATIONS = ("forward", "reverse_complement")
EXPECTED_RESPONSES = ("absolute", "signed")
EXPECTED_CONSEQUENCES = 35
CHANCE_AUPRC = 1.0 / EXPECTED_CONSEQUENCES


def read_complete_families(result_root: str) -> pl.DataFrame:
    pattern = f"{result_root.rstrip('/')}/*__one_vs_rest.parquet"
    frame = pl.scan_parquet(pattern).collect(engine="streaming")
    assert frame.height > 0
    assert set(frame["report_block"].unique()) == set(EXPECTED_LAYERS)
    assert set(frame["orientation"].unique()) == set(EXPECTED_ORIENTATIONS)
    assert set(frame["response"].unique()) == set(EXPECTED_RESPONSES)
    assert frame["consequence"].n_unique() == EXPECTED_CONSEQUENCES
    assert frame.null_count().sum_horizontal().sum() == 0
    return frame


def consequence_summaries(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    keys = ["report_block", "orientation", "response", "consequence"]
    auprc = (
        frame.group_by(keys)
        .agg(
            pl.col("direction_aligned_auprc").max().alias("best_auprc"),
            ((pl.col("welch_qvalue") < 0.05) & (pl.col("mwu_qvalue") < 0.05))
            .sum()
            .alias("concordant_hits"),
        )
        .with_columns((pl.col("best_auprc") / CHANCE_AUPRC).alias("auprc_over_chance"))
    )
    effects = (
        frame.filter((pl.col("welch_qvalue") < 0.05) & (pl.col("mwu_qvalue") < 0.05))
        .group_by(keys)
        .agg(pl.col("rank_biserial").abs().max().alias("max_abs_rank_biserial"))
    )
    expected_rows = (
        len(EXPECTED_LAYERS)
        * len(EXPECTED_ORIENTATIONS)
        * len(EXPECTED_RESPONSES)
        * EXPECTED_CONSEQUENCES
    )
    assert auprc.height == expected_rows
    assert effects.height <= expected_rows
    return auprc, effects


def consequence_order(summary: pl.DataFrame) -> list[str]:
    order = (
        summary.group_by("consequence")
        .agg(pl.col("best_auprc").max().alias("best"))
        .sort("best", descending=True)["consequence"]
        .to_list()
    )
    assert len(order) == EXPECTED_CONSEQUENCES
    return order


def strand_concordance(frame: pl.DataFrame) -> pl.DataFrame:
    """Compare like-for-like feature associations across FWD and RC."""

    rows: list[dict[str, float | int | str]] = []
    keys = ["feature_id", "consequence"]
    for layer in EXPECTED_LAYERS:
        for response in EXPECTED_RESPONSES:
            subset = frame.filter(
                (pl.col("report_block") == layer) & (pl.col("response") == response)
            ).with_columns(
                (
                    (pl.col("welch_qvalue") < 0.05)
                    & (pl.col("mwu_qvalue") < 0.05)
                ).alias("discovery")
            )
            forward = subset.filter(pl.col("orientation") == "forward")
            reverse = subset.filter(pl.col("orientation") == "reverse_complement")
            assert forward.select(keys).is_duplicated().sum() == 0
            assert reverse.select(keys).is_duplicated().sum() == 0

            forward_hits = {
                (int(row[0]), str(row[1]))
                for row in forward.filter("discovery").select(keys).iter_rows()
            }
            reverse_hits = {
                (int(row[0]), str(row[1]))
                for row in reverse.filter("discovery").select(keys).iter_rows()
            }
            union = forward_hits | reverse_hits
            intersection = forward_hits & reverse_hits
            assert union

            common = forward.select(keys + ["rank_biserial", "discovery"]).join(
                reverse.select(keys + ["rank_biserial", "discovery"]),
                on=keys,
                how="inner",
                suffix="_rc",
            )
            assert common.height > 1
            forward_effect = common["rank_biserial"].to_numpy()
            reverse_effect = common["rank_biserial_rc"].to_numpy()
            pearson = stats.pearsonr(forward_effect, reverse_effect).statistic
            spearman = stats.spearmanr(forward_effect, reverse_effect).statistic
            assert np.isfinite(pearson) and np.isfinite(spearman)

            discovered_common = common.filter(
                pl.col("discovery") & pl.col("discovery_rc")
            )
            assert discovered_common.height == len(intersection)
            sign_agreement = (
                np.sign(discovered_common["rank_biserial"].to_numpy())
                == np.sign(discovered_common["rank_biserial_rc"].to_numpy())
            ).mean()
            rows.append(
                {
                    "report_block": layer,
                    "response": response,
                    "common_pairs": common.height,
                    "forward_discoveries": len(forward_hits),
                    "reverse_complement_discoveries": len(reverse_hits),
                    "both_strand_discoveries": len(intersection),
                    "discovery_jaccard": len(intersection) / len(union),
                    "effect_pearson": float(pearson),
                    "effect_spearman": float(spearman),
                    "both_strand_effect_sign_agreement": float(sign_agreement),
                }
            )
    output = pl.DataFrame(rows, infer_schema_length=None)
    assert output.height == len(EXPECTED_LAYERS) * len(EXPECTED_RESPONSES)
    return output


def matrix_for(
    frame: pl.DataFrame,
    *,
    orientation: str,
    response: str,
    value: str,
    order: list[str],
) -> np.ndarray:
    subset = frame.filter(
        (pl.col("orientation") == orientation) & (pl.col("response") == response)
    )
    lookup = {
        (str(row["consequence"]), int(row["report_block"])): float(row[value])
        for row in subset.iter_rows(named=True)
    }
    output = np.full((len(order), len(EXPECTED_LAYERS)), np.nan, dtype=np.float64)
    for row_index, consequence in enumerate(order):
        for column_index, layer in enumerate(EXPECTED_LAYERS):
            if (consequence, layer) in lookup:
                output[row_index, column_index] = lookup[(consequence, layer)]
    return output


def draw_heatmaps(
    frame: pl.DataFrame,
    *,
    value: str,
    order: list[str],
    output_path: Path,
    title: str,
    colorbar_label: str,
    palette: str,
    vmin: float,
    vmax: float,
) -> None:
    sns.set_theme(style="white", context="notebook", font_scale=0.9)
    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12.5, 15.5),
        sharex=True,
        sharey=False,
        constrained_layout=False,
    )
    colorbar_axis = figure.add_axes((0.92, 0.22, 0.018, 0.56))
    for row_index, response in enumerate(EXPECTED_RESPONSES):
        for column_index, orientation in enumerate(EXPECTED_ORIENTATIONS):
            axis = axes[row_index, column_index]
            matrix = matrix_for(
                frame,
                orientation=orientation,
                response=response,
                value=value,
                order=order,
            )
            sns.heatmap(
                matrix,
                ax=axis,
                cmap=palette,
                vmin=vmin,
                vmax=vmax,
                mask=~np.isfinite(matrix),
                linewidths=0.25,
                linecolor="white",
                cbar=bool(row_index == 0 and column_index == 0),
                cbar_ax=colorbar_axis if row_index == 0 and column_index == 0 else None,
                xticklabels=[f"Layer {layer}" for layer in EXPECTED_LAYERS],
                yticklabels=order if column_index == 0 else False,
            )
            orientation_title = (
                "Forward" if orientation == "forward" else "Reverse complement"
            )
            response_title = "|Alt − ref|" if response == "absolute" else "Alt − ref"
            axis.set_title(f"{orientation_title} · {response_title}", pad=8)
            axis.set_xlabel("")
            axis.set_ylabel("")
            axis.tick_params(axis="x", rotation=0)
            axis.tick_params(axis="y", labelsize=8)
    colorbar_axis.set_ylabel(colorbar_label, rotation=270, labelpad=18)
    figure.suptitle(title, fontsize=15, y=0.982)
    figure.text(
        0.5,
        0.012,
        "Each cell selects the strongest single SAE feature within a complete layer × strand × response family; "
        "AUPRC is descriptive, while effect cells require both Welch and Mann–Whitney BH q < 0.05.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    figure.subplots_adjust(
        left=0.27, right=0.89, top=0.94, bottom=0.05, hspace=0.10, wspace=0.08
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        default="s3://oa-bolinas/experiments/exp422/retrieval/dna-exp422-complete-family-25m-r1/families",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue422_sae_consequence_associations"),
    )
    args = parser.parse_args()

    families = read_complete_families(args.result_root)
    auprc, effects = consequence_summaries(families)
    concordance = strand_concordance(families)
    order = consequence_order(auprc)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    auprc.write_parquet(args.output_dir / "per_consequence_auprc.parquet")
    effects.write_parquet(args.output_dir / "per_consequence_effect.parquet")
    concordance.write_parquet(args.output_dir / "strand_concordance.parquet")

    draw_heatmaps(
        auprc,
        value="auprc_over_chance",
        order=order,
        output_path=args.output_dir / "figure",
        title="Broad consequence associations are strongest for splice and coding variants",
        colorbar_label="Best direction-aligned AUPRC ÷ chance (1/35)",
        palette="mako",
        vmin=1.0,
        vmax=float(auprc["auprc_over_chance"].max()),
    )
    draw_heatmaps(
        effects,
        value="max_abs_rank_biserial",
        order=order,
        output_path=args.output_dir / "effect",
        title="FDR-supported consequence effect sizes across SAE layers and strands",
        colorbar_label="Maximum |rank-biserial effect|",
        palette="rocket_r",
        vmin=0.0,
        vmax=float(effects["max_abs_rank_biserial"].max()),
    )


if __name__ == "__main__":
    main()
