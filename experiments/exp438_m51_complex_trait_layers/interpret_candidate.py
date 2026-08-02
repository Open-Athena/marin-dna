"""Post-hoc descriptive audit of one recurrent SAE feature.

This intentionally does not add inferential claims to the preregistered scan.
It turns the sparse FWD/RC extraction into a compact per-variant table and
reports strand agreement, activation-state transitions, tail enrichment,
allele-change summaries, and top loci for biological follow-up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import average_precision_score

from common import ISSUE, write_json
from prepare_panel import EXPECTED_ROWS


ORIENTATIONS = ("forward", "reverse_complement")
TAIL_FRACTIONS = (0.01, 0.05, 0.10)


def load_sparse_feature(path: Path, feature_id: int, *, rows: int) -> pl.DataFrame:
    """Load one feature with explicit zero rows restored."""
    assert path.is_file() and 0 <= feature_id < 15_360 and rows > 0
    selected = (
        pl.scan_parquet(path)
        .filter(pl.col("feature_id") == feature_id)
        .select("panel_row", "ref_activation", "alt_activation", "delta")
        .collect()
    )
    assert selected["panel_row"].n_unique() == selected.height
    assert selected.height <= rows
    assert selected.select(pl.all().exclude("panel_row").is_finite().all()).row(0) == (
        True,
        True,
        True,
    )
    assert (
        selected.select(
            (pl.col("delta") - (pl.col("alt_activation") - pl.col("ref_activation")))
            .abs()
            .max()
        ).item()
        <= 1e-5
    )

    return (
        pl.DataFrame({"panel_row": pl.arange(0, rows, eager=True, dtype=pl.UInt32)})
        .join(selected, on="panel_row", how="left")
        .with_columns(
            pl.col("ref_activation").fill_null(0.0),
            pl.col("alt_activation").fill_null(0.0),
            pl.col("delta").fill_null(0.0),
        )
    )


def activation_state(ref: str, alt: str) -> pl.Expr:
    return (
        pl.when((pl.col(ref) == 0) & (pl.col(alt) == 0))
        .then(pl.lit("inactive"))
        .when((pl.col(ref) == 0) & (pl.col(alt) > 0))
        .then(pl.lit("activated_by_alt"))
        .when((pl.col(ref) > 0) & (pl.col(alt) == 0))
        .then(pl.lit("deactivated_by_alt"))
        .otherwise(pl.lit("active_both"))
    )


def build_candidate_table(
    panel_path: Path,
    forward_path: Path,
    reverse_complement_path: Path,
    *,
    feature_id: int,
) -> pl.DataFrame:
    panel = pl.read_parquet(panel_path).with_row_index("panel_row")
    assert panel.height == EXPECTED_ROWS
    assert panel["label"].dtype == pl.Boolean
    frame = panel
    for orientation, path in zip(
        ORIENTATIONS, (forward_path, reverse_complement_path), strict=True
    ):
        values = load_sparse_feature(path, feature_id, rows=panel.height).rename(
            {
                "ref_activation": f"{orientation}_ref",
                "alt_activation": f"{orientation}_alt",
                "delta": f"{orientation}_delta",
            }
        )
        frame = frame.join(values, on="panel_row", how="inner", validate="1:1")
        frame = frame.with_columns(
            pl.col(f"{orientation}_delta").abs().alias(f"{orientation}_abs_delta"),
            activation_state(f"{orientation}_ref", f"{orientation}_alt").alias(
                f"{orientation}_state"
            ),
        )
    frame = frame.with_columns(
        pl.mean_horizontal("forward_abs_delta", "reverse_complement_abs_delta").alias(
            "mean_abs_delta"
        ),
        pl.max_horizontal("forward_abs_delta", "reverse_complement_abs_delta").alias(
            "max_abs_delta"
        ),
        pl.mean_horizontal("forward_delta", "reverse_complement_delta").alias(
            "mean_signed_delta"
        ),
        pl.concat_str("ref", "alt", separator=">").alias("allele_change"),
    )
    assert (
        frame.height == EXPECTED_ROWS and frame["panel_row"].n_unique() == frame.height
    )
    return frame


def correlation_summary(frame: pl.DataFrame) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for response in ("abs_delta", "delta"):
        left = frame[f"forward_{response}"].to_numpy()
        right = frame[f"reverse_complement_{response}"].to_numpy()
        pearson = stats.pearsonr(left, right)
        spearman = stats.spearmanr(left, right)
        rows.append(
            {
                "response": response,
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            }
        )
    return rows


def target_summary(frame: pl.DataFrame) -> pl.DataFrame:
    pieces: list[pl.DataFrame] = []
    targets = [("overall", frame)] + [
        (str(subset), frame.filter(pl.col("subset") == subset))
        for subset in sorted(frame["subset"].unique().to_list())
    ]
    for target, part in targets:
        labels = part["label"].cast(pl.UInt8).to_numpy()
        assert 0 < int(labels.sum()) < labels.size
        for orientation in ORIENTATIONS:
            for response in ("abs_delta", "delta"):
                scores = part[f"{orientation}_{response}"].to_numpy()
                ap_higher = average_precision_score(labels, scores)
                ap_lower = average_precision_score(labels, -scores)
                pieces.append(
                    pl.DataFrame(
                        {
                            "target": [target],
                            "orientation": [orientation],
                            "response": [response],
                            "n": [part.height],
                            "n_positive": [int(labels.sum())],
                            "prevalence": [float(labels.mean())],
                            "auprc_higher": [float(ap_higher)],
                            "auprc_lower": [float(ap_lower)],
                            "positive_mean": [
                                float(scores[labels.astype(bool)].mean())
                            ],
                            "negative_mean": [
                                float(scores[~labels.astype(bool)].mean())
                            ],
                            "nonzero_support": [int(np.count_nonzero(scores))],
                        }
                    )
                )
    return pl.concat(pieces)


def tail_summary(frame: pl.DataFrame) -> pl.DataFrame:
    pieces: list[pl.DataFrame] = []
    for response in (
        "forward_abs_delta",
        "reverse_complement_abs_delta",
        "mean_abs_delta",
        "max_abs_delta",
    ):
        ranked = frame.sort(response, descending=True)
        for fraction in TAIL_FRACTIONS:
            count = max(1, round(frame.height * fraction))
            top = ranked.head(count)
            pieces.append(
                pl.DataFrame(
                    {
                        "response": [response],
                        "fraction": [fraction],
                        "n": [count],
                        "positive_fraction": [float(top["label"].mean())],
                        "unique_match_groups": [top["match_group"].n_unique()],
                        "unique_closest_genes": [
                            top["exon_closest_gene_id"].drop_nulls().n_unique()
                        ],
                    }
                )
            )
    return pl.concat(pieces)


def state_summary(frame: pl.DataFrame) -> pl.DataFrame:
    pieces: list[pl.DataFrame] = []
    for orientation in ORIENTATIONS:
        state = f"{orientation}_state"
        pieces.append(
            frame.group_by(state, "label")
            .agg(pl.len().alias("n"), pl.col(f"{orientation}_abs_delta").mean())
            .rename({state: "state", f"{orientation}_abs_delta": "mean_abs_delta"})
            .with_columns(pl.lit(orientation).alias("orientation"))
        )
    return pl.concat(pieces).select(
        "orientation", "state", "label", "n", "mean_abs_delta"
    )


def allele_summary(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.group_by("allele_change")
        .agg(
            pl.len().alias("n"),
            pl.col("label").sum().alias("n_positive"),
            pl.col("label").mean().alias("prevalence"),
            pl.col("forward_abs_delta").mean(),
            pl.col("reverse_complement_abs_delta").mean(),
            pl.col("mean_abs_delta").mean(),
        )
        .sort("allele_change")
    )


def top_variants(frame: pl.DataFrame, *, top_n: int) -> pl.DataFrame:
    assert 0 < top_n <= frame.height
    columns = [
        "panel_row",
        "chrom",
        "pos",
        "ref",
        "alt",
        "rsid",
        "label",
        "subset",
        "match_group",
        "pip",
        "traits",
        "consequence",
        "consequence_cre",
        "exon_closest_gene_id",
        "distance_exon",
        "tss_closest_gene_id",
        "distance_tss",
        "forward_ref",
        "forward_alt",
        "forward_delta",
        "forward_abs_delta",
        "reverse_complement_ref",
        "reverse_complement_alt",
        "reverse_complement_delta",
        "reverse_complement_abs_delta",
        "mean_abs_delta",
        "max_abs_delta",
    ]
    return frame.sort("mean_abs_delta", descending=True).head(top_n).select(columns)


def run(
    panel_path: Path,
    forward_path: Path,
    reverse_complement_path: Path,
    output_dir: Path,
    *,
    feature_id: int,
    top_n: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = build_candidate_table(
        panel_path,
        forward_path,
        reverse_complement_path,
        feature_id=feature_id,
    )
    outputs = {
        "candidate_variants.parquet": frame,
        "target_summary.parquet": target_summary(frame),
        "tail_summary.parquet": tail_summary(frame),
        "state_summary.parquet": state_summary(frame),
        "allele_summary.parquet": allele_summary(frame),
        "top_variants.parquet": top_variants(frame, top_n=top_n),
    }
    for name, value in outputs.items():
        value.write_parquet(output_dir / name, compression="zstd")
    metadata = {
        "issue": ISSUE,
        "feature_id": feature_id,
        "analysis_status": "post_hoc_descriptive",
        "rows": frame.height,
        "positive_rows": int(frame["label"].sum()),
        "correlations": correlation_summary(frame),
        "artifacts": sorted(outputs),
    }
    write_json(output_dir / "results.json", metadata)
    (output_dir / "RESULTS.md").write_text(
        "# Feature candidate audit\n\n"
        f"Feature: block 19 / 25M / {feature_id}\n\n"
        "This is a post-hoc descriptive audit, not a new inferential test.\n\n"
        "```json\n" + json.dumps(metadata, indent=2, sort_keys=True) + "\n```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--reverse-complement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-id", type=int, default=1662)
    parser.add_argument("--top-n", type=int, default=200)
    args = parser.parse_args()
    run(
        args.panel,
        args.forward,
        args.reverse_complement,
        args.output_dir,
        feature_id=args.feature_id,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
