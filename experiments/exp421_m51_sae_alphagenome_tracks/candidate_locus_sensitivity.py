"""Post-hoc recurrent-locus sensitivity for issue #421 candidate features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import average_precision_score

FTL_GENE_ID = "ENSG00000087086"
F9_GENE_ID = "ENSG00000101981"
GENE_LABELS = {FTL_GENE_ID: "FTL", F9_GENE_ID: "F9"}


def _variant_gene_membership(contexts: pl.DataFrame) -> pl.DataFrame:
    return contexts.select(
        "panel_row",
        "label",
        "subset",
        *[
            (
                (pl.col("exon_closest_gene_id") == gene_id)
                | (pl.col("tss_closest_gene_id") == gene_id)
            ).alias(f"is_{label.lower()}")
            for gene_id, label in GENE_LABELS.items()
        ],
    )


def _association(values: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    pathogenic = values[labels]
    benign = values[~labels]
    assert len(pathogenic) >= 10
    assert len(benign) >= 10
    mann = stats.mannwhitneyu(
        pathogenic,
        benign,
        alternative="two-sided",
        method="asymptotic",
    )
    welch = stats.ttest_ind(
        pathogenic,
        benign,
        equal_var=False,
        alternative="two-sided",
    )
    return {
        "n": len(values),
        "n_pathogenic": len(pathogenic),
        "n_benign": len(benign),
        "pathogenic_prevalence": float(labels.mean()),
        "mean_pathogenic": float(pathogenic.mean()),
        "mean_benign": float(benign.mean()),
        "rank_biserial": float(
            2 * mann.statistic / (len(pathogenic) * len(benign)) - 1
        ),
        "mann_whitney_p": float(mann.pvalue),
        "welch_p": float(welch.pvalue),
        "average_precision": float(average_precision_score(labels, values)),
    }


def locus_exclusion_sensitivity(
    responses: pl.DataFrame,
    contexts: pl.DataFrame,
    *,
    report_block: int = 19,
    feature_id: int = 11_928,
) -> pl.DataFrame:
    """Measure label association after excluding recurrent FTL/F9 loci."""
    frame = responses.filter(
        (pl.col("report_block") == report_block) & (pl.col("feature_id") == feature_id)
    ).join(
        _variant_gene_membership(contexts),
        on="panel_row",
        how="inner",
        validate="m:1",
    )
    assert frame.height == contexts.height * frame["orientation"].n_unique()
    exclusions = {
        "none": pl.lit(True),
        "FTL": ~pl.col("is_ftl"),
        "F9": ~pl.col("is_f9"),
        "FTL+F9": ~(pl.col("is_ftl") | pl.col("is_f9")),
    }
    scopes = {
        "all_subsets": pl.lit(True),
        "5_prime_UTR_variant": pl.col("subset") == "5_prime_UTR_variant",
    }
    rows: list[dict[str, float | int | str]] = []
    for orientation in sorted(frame["orientation"].unique()):
        oriented = frame.filter(pl.col("orientation") == orientation)
        for scope, scope_filter in scopes.items():
            for exclusion, exclusion_filter in exclusions.items():
                selected = oriented.filter(scope_filter & exclusion_filter)
                values = selected["abs_delta"].to_numpy()
                labels = selected["label"].to_numpy().astype(bool)
                rows.append(
                    {
                        "report_block": report_block,
                        "feature_id": feature_id,
                        "orientation": orientation,
                        "scope": scope,
                        "excluded_loci": exclusion,
                        **_association(values, labels),
                    }
                )
    result = pl.DataFrame(rows)
    assert result.height == 2 * len(scopes) * len(exclusions)
    return result.sort(["scope", "orientation", "excluded_loci"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = locus_exclusion_sensitivity(
        pl.read_parquet(args.result_root / "selected_feature_responses.parquet"),
        pl.read_parquet(args.result_root / "variant_contexts.parquet"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(args.output)


if __name__ == "__main__":
    main()
