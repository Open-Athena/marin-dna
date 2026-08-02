"""Post-hoc mechanism checks for a VEP-annotated SAE feature candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import average_precision_score

from common import ISSUE, write_json

RESPONSES = (
    "forward_abs_delta",
    "reverse_complement_abs_delta",
    "mean_abs_delta",
    "max_abs_delta",
)
PREDICTORS = (
    "sift_deleteriousness",
    "polyphen_deleteriousness",
    "blosum_deleteriousness",
)


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and np.isfinite(values).all()
    assert np.all((0 <= values) & (values <= 1))
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    scaled = ranked * ranked.size / np.arange(1, ranked.size + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.clip(monotone, 0, 1)
    return adjusted


def add_mechanism_columns(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (-pl.col("sift_score")).alias("sift_deleteriousness"),
        pl.col("polyphen_score").alias("polyphen_deleteriousness"),
        (-pl.col("blosum62")).alias("blosum_deleteriousness"),
        pl.concat_str("ref", "alt", separator=">").alias("allele_change"),
        pl.col("clinical_significance").is_not_null().alias("has_clinical_record"),
    )


def correlation_results(frame: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for response in RESPONSES:
        for predictor in PREDICTORS:
            complete = frame.select(response, predictor).drop_nulls()
            assert complete.height >= 100
            x = complete[response].to_numpy()
            y = complete[predictor].to_numpy()
            pearson = stats.pearsonr(x, y)
            spearman = stats.spearmanr(x, y)
            rows.append(
                {
                    "response": response,
                    "predictor": predictor,
                    "n": complete.height,
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                }
            )
    result = pl.DataFrame(rows)
    return result.with_columns(
        pl.Series("pearson_q", bh_adjust(result["pearson_p"].to_numpy())),
        pl.Series("spearman_q", bh_adjust(result["spearman_p"].to_numpy())),
    )


def standardized_mean_difference(positive: np.ndarray, negative: np.ndarray) -> float:
    numerator = positive.mean() - negative.mean()
    pooled_variance = (
        (positive.size - 1) * positive.var(ddof=1)
        + (negative.size - 1) * negative.var(ddof=1)
    ) / (positive.size + negative.size - 2)
    return float(numerator / np.sqrt(pooled_variance))


def label_association_results(frame: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    targets = [("overall", frame)] + [
        (str(change), frame.filter(pl.col("allele_change") == change))
        for change in sorted(frame["allele_change"].unique().to_list())
    ]
    for target, part in targets:
        labels = part["label"].to_numpy()
        if labels.sum() < 10 or (~labels).sum() < 10:
            continue
        for response in RESPONSES:
            scores = part[response].to_numpy()
            positive, negative = scores[labels], scores[~labels]
            welch = stats.ttest_ind(positive, negative, equal_var=False)
            mann = stats.mannwhitneyu(positive, negative, alternative="two-sided")
            rows.append(
                {
                    "target": target,
                    "response": response,
                    "n": part.height,
                    "n_positive": int(labels.sum()),
                    "prevalence": float(labels.mean()),
                    "auprc": float(average_precision_score(labels, scores)),
                    "welch_p": float(welch.pvalue),
                    "mann_whitney_p": float(mann.pvalue),
                    "standardized_mean_difference": standardized_mean_difference(
                        positive, negative
                    ),
                    "rank_biserial": float(
                        2 * mann.statistic / (positive.size * negative.size) - 1
                    ),
                }
            )
    result = pl.DataFrame(rows)
    return result.with_columns(
        pl.Series("welch_q", bh_adjust(result["welch_p"].to_numpy())),
        pl.Series("mann_whitney_q", bh_adjust(result["mann_whitney_p"].to_numpy())),
    )


def codon_change_position(value: str | None) -> int | None:
    if value is None or "/" not in value:
        return None
    reference, alternate = value.split("/", maxsplit=1)
    if len(reference) != 3 or len(alternate) != 3:
        return None
    changed = [
        index
        for index, (ref, alt) in enumerate(zip(reference, alternate, strict=True))
        if ref.upper() != alt.upper()
    ]
    return changed[0] + 1 if len(changed) == 1 else None


def tail_mechanism_results(frame: pl.DataFrame) -> pl.DataFrame:
    codon_position = [codon_change_position(value) for value in frame["codons"]]
    frame = frame.with_columns(pl.Series("codon_position", codon_position, pl.UInt8))
    pieces: list[pl.DataFrame] = []
    for response in RESPONSES:
        ranked = frame.sort(response, descending=True).with_row_index("response_rank")
        for fraction in (0.01, 0.05, 0.10):
            count = max(1, round(ranked.height * fraction))
            top = ranked.head(count)
            pieces.append(
                pl.DataFrame(
                    {
                        "response": [response],
                        "fraction": [fraction],
                        "n": [count],
                        "positive_fraction": [float(top["label"].mean())],
                        "clinical_record_fraction": [
                            float(top["has_clinical_record"].mean())
                        ],
                        "mean_sift_deleteriousness": [
                            float(top["sift_deleteriousness"].mean())
                        ],
                        "mean_polyphen_deleteriousness": [
                            float(top["polyphen_deleteriousness"].mean())
                        ],
                        "mean_blosum_deleteriousness": [
                            float(top["blosum_deleteriousness"].mean())
                        ],
                        "codon_position_1_fraction": [
                            float((top["codon_position"] == 1).mean())
                        ],
                        "codon_position_2_fraction": [
                            float((top["codon_position"] == 2).mean())
                        ],
                        "codon_position_3_fraction": [
                            float((top["codon_position"] == 3).mean())
                        ],
                    }
                )
            )
    return pl.concat(pieces)


def response_deciles(frame: pl.DataFrame) -> pl.DataFrame:
    pieces: list[pl.DataFrame] = []
    for response in RESPONSES:
        ordered = frame.sort(response).with_row_index("rank")
        ordered = ordered.with_columns(
            ((pl.col("rank") * 10 / ordered.height).floor() + 1)
            .clip(1, 10)
            .cast(pl.UInt8)
            .alias("decile")
        )
        pieces.append(
            ordered.group_by("decile")
            .agg(
                pl.len().alias("n"),
                pl.col("label").mean().alias("positive_fraction"),
                pl.col(response).mean().alias("mean_response"),
            )
            .with_columns(pl.lit(response).alias("response"))
        )
    return pl.concat(pieces).select(
        "response", "decile", "n", "positive_fraction", "mean_response"
    )


def run(input_path: Path, output_dir: Path) -> None:
    assert input_path.is_file()
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = add_mechanism_columns(pl.read_parquet(input_path))
    assert frame.height == 2_500 and int(frame["label"].sum()) == 250
    outputs = {
        "correlations.parquet": correlation_results(frame),
        "label_associations.parquet": label_association_results(frame),
        "tail_mechanisms.parquet": tail_mechanism_results(frame),
        "response_deciles.parquet": response_deciles(frame),
        "top_annotated_variants.parquet": frame.sort(
            "mean_abs_delta", descending=True
        ).head(200),
    }
    for name, value in outputs.items():
        value.write_parquet(output_dir / name, compression="zstd")
    metadata = {
        "issue": ISSUE,
        "analysis_status": "post_hoc_descriptive",
        "rows": frame.height,
        "positive_rows": int(frame["label"].sum()),
        "artifacts": sorted(outputs),
    }
    write_json(output_dir / "results.json", metadata)
    (output_dir / "RESULTS.md").write_text(
        "# Feature 1662 missense-mechanism audit\n\n"
        "This is a post-hoc descriptive audit, not a new inferential test.\n\n"
        "```json\n" + json.dumps(metadata, indent=2, sort_keys=True) + "\n```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
