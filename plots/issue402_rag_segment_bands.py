#!/usr/bin/env python3
"""Explain issue #402's segment-loss bands with per-document sequence features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy.stats import spearmanr

LOSS_ROOTS = {
    "46M": "s3://oa-bolinas/snakemake/analysis/issue402_segment_loss/46m-step-29999",
    "104M": "s3://oa-bolinas/snakemake/analysis/issue402_segment_loss/104m-step-29999",
}
VALIDATION_PARQUET = (
    "s3://oa-bolinas/snakemake/rag_glm/results/dataset/"
    "zoonomia-rag-v1-v1/data/validation/part-00000-of-00001.parquet"
)
S3_OPTIONS = {"aws_region": "us-east-2"}
MODEL_ORDER = ["46M", "104M"]
SPECIES = [
    "Microgale_talazaci",
    "Loxodonta_africana",
    "Tolypeutes_matacus",
    "Bos_taurus",
    "Equus_caballus",
    "Mus_musculus",
    "Microcebus_murinus",
    "Homo_sapiens",
]
SPECIES_DISPLAY = [
    "M. talazaci",
    "L. africana",
    "T. matacus",
    "B. taurus",
    "E. caballus",
    "M. musculus",
    "M. murinus",
    "H. sapiens",
]
SUPERORDERS = [
    "Afrotheria",
    "Afrotheria",
    "Xenarthra",
    "Laurasiatheria",
    "Laurasiatheria",
    "Euarchontoglires",
    "Euarchontoglires",
    "Euarchontoglires",
]
BASE_CODES = np.frombuffer(b"ACGT", dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_segment_bands"),
    )
    return parser.parse_args()


def _read(path: str, *, columns: list[str] | None = None) -> pl.DataFrame:
    return pl.read_parquet(path, columns=columns, storage_options=S3_OPTIONS)


def _row_identity(current: np.ndarray, other: np.ndarray) -> np.ndarray:
    comparable = np.isin(current, BASE_CODES) & np.isin(other, BASE_CODES)
    denominator = comparable.sum(axis=1)
    return np.divide(
        ((current == other) & comparable).sum(axis=1),
        denominator,
        out=np.full(current.shape[0], np.nan),
        where=denominator > 0,
    )


def _max_finite(values: np.ndarray) -> np.ndarray:
    assert values.ndim == 2 and values.shape[0] > 0
    result = np.where(np.isfinite(values), values, -np.inf).max(axis=0)
    result[result == -np.inf] = np.nan
    return result


def load_analysis_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    columns = ["anchor_id"] + [
        column
        for index in range(8)
        for column in (f"sequence_{index}", f"available_{index}")
    ]
    validation = _read(VALIDATION_PARQUET, columns=columns)
    assert validation.height == 2_048
    assert validation["anchor_id"].n_unique() == validation.height

    arrays: list[np.ndarray] = []
    for index in range(8):
        sequence = "".join(validation[f"sequence_{index}"].to_list())
        array = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)
        arrays.append(array.reshape(validation.height, 255))

    feature_frames: list[pl.DataFrame] = []
    for index, current in enumerate(arrays):
        if index:
            prior_identities = np.vstack(
                [_row_identity(current, arrays[prior]) for prior in range(index)]
            )
            max_prior = _max_finite(prior_identities)
        else:
            max_prior = np.full(validation.height, np.nan)
        feature_frames.append(
            pl.DataFrame(
                {
                    "anchor_id": validation["anchor_id"],
                    "segment_index": pl.Series(
                        [index] * validation.height, dtype=pl.Int32
                    ),
                    "superorder": pl.Series([SUPERORDERS[index]] * validation.height),
                    "same_superorder_prior": pl.Series(
                        [
                            any(
                                SUPERORDERS[prior] == SUPERORDERS[index]
                                for prior in range(index)
                            )
                        ]
                        * validation.height
                    ),
                    "max_prior_identity": max_prior,
                    "identity_to_human": _row_identity(current, arrays[-1]),
                }
            )
        )
    features = pl.concat(feature_frames)

    losses = pl.concat(
        [
            _read(f"{root}/validation_document_segment_loss.parquet")
            for root in LOSS_ROOTS.values()
        ]
    )
    assert losses.height == 2 * 2_048 * 8
    data = losses.join(
        features,
        on=["anchor_id", "segment_index"],
        how="left",
        validate="m:1",
    )
    assert data.height == losses.height
    assert data.filter(~pl.col("mean_loss").is_finite()).is_empty()

    position = pl.concat(
        [
            _read(f"{root}/validation_position_loss.parquet")
            for root in LOSS_ROOTS.values()
        ]
    ).filter(pl.col("layout_token_type").is_in(["ortholog_base", "human_base"]))
    assert position.height == 2 * 8 * 255
    return data, position


def summarize_segments(data: pl.DataFrame) -> pl.DataFrame:
    return (
        data.group_by(
            "model",
            "segment_index",
            "segment",
            "superorder",
            "same_superorder_prior",
        )
        .agg(
            pl.col("mean_loss").mean().alias("all_mean_loss"),
            pl.col("mean_loss")
            .filter(pl.col("available"))
            .mean()
            .alias("available_mean_loss"),
            pl.col("mean_loss")
            .filter(~pl.col("available"))
            .mean()
            .alias("missing_mean_loss"),
            pl.col("available").mean().alias("available_fraction"),
            pl.col("max_prior_identity")
            .filter(pl.col("max_prior_identity").is_finite())
            .mean()
            .alias("max_prior_identity"),
            pl.col("identity_to_human")
            .filter(pl.col("identity_to_human").is_finite())
            .mean()
            .alias("identity_to_human"),
            pl.col("available").sum().alias("n_available"),
            pl.len().alias("n_documents"),
        )
        .sort("model", "segment_index")
    )


def identity_deciles(data: pl.DataFrame) -> pl.DataFrame:
    eligible = data.filter(
        (pl.col("segment_index") > 0)
        & pl.col("available")
        & pl.col("max_prior_identity").is_finite()
    ).with_columns(
        (
            pl.col("mean_loss")
            - pl.col("mean_loss").mean().over("model", "segment_index")
        ).alias("centered_loss"),
        (
            (
                pl.col("max_prior_identity")
                .rank(method="ordinal")
                .over("model", "segment_index")
                - 1
            )
            * 10
            / pl.len().over("model", "segment_index")
        )
        .floor()
        .clip(0, 9)
        .cast(pl.Int8)
        .alias("identity_decile"),
    )
    result = (
        eligible.group_by("model", "identity_decile")
        .agg(
            pl.col("centered_loss").mean().alias("centered_loss"),
            (pl.col("centered_loss").std() / pl.len().sqrt()).alias("se_centered_loss"),
            pl.col("max_prior_identity").mean().alias("mean_prior_identity"),
            pl.len().alias("n"),
        )
        .with_columns((pl.col("identity_decile") + 1).alias("identity_decile"))
        .sort("model", "identity_decile")
    )
    assert result.height == 20
    return result


def calculate_summary(
    data: pl.DataFrame, position: pl.DataFrame, segments: pl.DataFrame
) -> dict[str, object]:
    within: dict[str, dict[str, float | int]] = {}
    for model in MODEL_ORDER:
        frame = data.filter(
            (pl.col("model") == model)
            & (pl.col("segment_index") > 0)
            & pl.col("available")
            & pl.col("max_prior_identity").is_finite()
        ).with_columns(
            (
                pl.col("mean_loss") - pl.col("mean_loss").mean().over("segment_index")
            ).alias("centered_loss"),
            (
                pl.col("max_prior_identity")
                - pl.col("max_prior_identity").mean().over("segment_index")
            ).alias("centered_identity"),
        )
        within[model] = {
            "spearman_prior_identity_vs_loss": float(
                spearmanr(frame["centered_identity"], frame["centered_loss"]).statistic
            ),
            "n": frame.height,
        }

    bands = (
        segments.filter(pl.col("segment_index") > 0)
        .group_by("segment_index")
        .agg(
            pl.col("available_mean_loss").mean(),
            pl.col("max_prior_identity").mean(),
        )
    )
    aggregate_rho = float(
        spearmanr(bands["max_prior_identity"], bands["available_mean_loss"]).statistic
    )

    wide = (
        position.select("model", "segment_index", "within_segment_offset", "mean_loss")
        .pivot(
            on=["model", "segment_index"],
            index="within_segment_offset",
            values="mean_loss",
        )
        .sort("within_segment_offset")
    )
    correlations = np.corrcoef(wide.drop("within_segment_offset").to_numpy().T)
    upper = correlations[np.triu_indices(correlations.shape[0], 1)]
    model_46 = position.filter(pl.col("model") == "46M").sort(
        "segment_index", "within_segment_offset"
    )
    model_104 = position.filter(pl.col("model") == "104M").sort(
        "segment_index", "within_segment_offset"
    )
    missing = segments.filter(pl.col("missing_mean_loss").is_not_null())
    return {
        "within_segment_available_only": within,
        "aggregate_segments_1_to_7_spearman_prior_identity_vs_available_loss": aggregate_rho,
        "position_curve_pairwise_pearson_min": float(upper.min()),
        "position_curve_pairwise_pearson_median": float(np.median(upper)),
        "cross_model_per_position_spearman": float(
            spearmanr(model_46["mean_loss"], model_104["mean_loss"]).statistic
        ),
        "missing_all_n_mean_loss_range": [
            float(missing["missing_mean_loss"].min()),
            float(missing["missing_mean_loss"].max()),
        ],
    }


def plot_explanation(
    segments: pl.DataFrame,
    deciles: pl.DataFrame,
    summary: dict[str, object],
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
    colors = sns.color_palette("colorblind", n_colors=3)
    markers = {"46M": "o", "104M": "s"}
    linestyles = {"46M": "--", "104M": "-"}
    figure, axes = plt.subplots(1, 2, figsize=(17.0, 6.4))

    loss_axis = axes[0]
    for model, color in zip(MODEL_ORDER, colors[:2], strict=True):
        frame = segments.filter(pl.col("model") == model).sort("segment_index")
        loss_axis.plot(
            frame["segment_index"],
            frame["available_mean_loss"],
            marker=markers[model],
            linestyle=linestyles[model],
            color=color,
            label=f"{model} NLL",
        )
    loss_axis.set_ylabel("Mean token NLL (available windows)")
    loss_axis.set_xlabel("Segment index and species")
    loss_axis.set_xticks(
        range(8),
        [f"{index}\n{name}" for index, name in enumerate(SPECIES_DISPLAY)],
        rotation=20,
        ha="right",
    )
    loss_axis.set_title("Band level follows similarity to earlier context")
    for boundary in (1.5, 2.5, 4.5):
        loss_axis.axvline(boundary, color="0.75", linewidth=1)

    identity_axis = loss_axis.twinx()
    identity = (
        segments.group_by("segment_index")
        .agg(pl.col("max_prior_identity").mean())
        .sort("segment_index")
    )
    identity_axis.plot(
        identity["segment_index"],
        identity["max_prior_identity"],
        color=colors[2],
        marker="D",
        linewidth=2.5,
        label="Best earlier-slot identity",
    )
    identity_axis.set_ylabel("Best earlier-slot identity")
    loss_axis.tick_params(axis="y", colors=colors[0])
    loss_axis.yaxis.label.set_color(colors[0])
    identity_axis.tick_params(axis="y", colors=colors[2])
    identity_axis.yaxis.label.set_color(colors[2])
    left_handles, left_labels = loss_axis.get_legend_handles_labels()
    right_handles, right_labels = identity_axis.get_legend_handles_labels()
    loss_axis.legend(
        left_handles + right_handles,
        left_labels + right_labels,
        loc="upper right",
        fontsize=10,
    )

    decile_axis = axes[1]
    for model, color in zip(MODEL_ORDER, colors[:2], strict=True):
        frame = deciles.filter(pl.col("model") == model).sort("identity_decile")
        decile_axis.errorbar(
            frame["identity_decile"],
            frame["centered_loss"],
            yerr=frame["se_centered_loss"],
            marker=markers[model],
            linestyle=linestyles[model],
            color=color,
            capsize=0,
            label=model,
        )
    decile_axis.axhline(0, color="0.5", linewidth=1)
    decile_axis.set_xlabel("Prior-identity decile within each segment")
    decile_axis.set_ylabel("Token NLL minus segment mean")
    decile_axis.set_title("The relationship also holds within segments")
    decile_axis.legend(title="Model", loc="lower left")
    rho_46 = summary["within_segment_available_only"]["46M"][
        "spearman_prior_identity_vs_loss"
    ]
    rho_104 = summary["within_segment_available_only"]["104M"][
        "spearman_prior_identity_vs_loss"
    ]
    decile_axis.text(
        0.61,
        0.84,
        f"Pooled within-slot Spearman ρ\n46M {rho_46:.2f} · 104M {rho_104:.2f}",
        transform=decile_axis.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        fontsize=10,
    )

    figure.suptitle("Why segment losses form reproducible bands")
    figure.text(
        0.5,
        0.005,
        "Available projected windows only. Left panel has independent y-axes; "
        "compare trends, not levels. Right error bars are ±1 SE (capless).",
        ha="center",
        fontsize=10,
    )
    figure.subplots_adjust(top=0.82, bottom=0.30, wspace=0.52)
    figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, position = load_analysis_data()
    segments = summarize_segments(data)
    deciles = identity_deciles(data)
    summary = calculate_summary(data, position, segments)

    segments.write_parquet(
        args.output_dir / "segment_summary.parquet", compression="zstd"
    )
    deciles.write_parquet(
        args.output_dir / "identity_deciles.parquet", compression="zstd"
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_explanation(segments, deciles, summary, args.output_dir)
    with pl.Config(tbl_rows=30, tbl_cols=20, tbl_width_chars=200):
        print(segments)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
