from __future__ import annotations

from pathlib import Path

import polars as pl
from marin_dna_vertebrate_projection.issue_473.comparison import (
    write_policy_comparison,
)


def _sequences(
    accepted: list[tuple[str, str]],
    *,
    aligned_bases: int,
    target_span: int,
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "query_name": query_name,
                "species": species,
                "alignment_source": "zoonomia_cactus",
                "aligned_bases": aligned_bases,
                "pre_resize_t_start": 100,
                "pre_resize_t_end": 100 + target_span,
            }
            for query_name, species in accepted
        ],
        schema={
            "query_name": pl.String,
            "species": pl.String,
            "alignment_source": pl.String,
            "aligned_bases": pl.Int64,
            "pre_resize_t_start": pl.Int64,
            "pre_resize_t_end": pl.Int64,
        },
    )


def _per_anchor(accepted_counts: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "accepted_total_projections": accepted_counts,
            "requested_total_species": [2] * len(accepted_counts),
            "no_mapping_count": [
                2 - accepted_count for accepted_count in accepted_counts
            ],
        }
    )


def test_policy_comparison_reports_recovery_agreement_and_flanks(
    tmp_path: Path,
) -> None:
    sequence_paths: dict[str, str] = {}
    per_anchor_paths: dict[str, str] = {}
    inputs = {
        "full_window": (
            _sequences(
                [("a1", "Mus musculus")],
                aligned_bases=255,
                target_span=255,
            ),
            _per_anchor([1, 0]),
        ),
        "center_1": (
            _sequences(
                [("a1", "Mus musculus"), ("a2", "Mus musculus")],
                aligned_bases=1,
                target_span=1,
            ),
            _per_anchor([1, 1]),
        ),
    }
    for policy_name, (sequences, per_anchor) in inputs.items():
        sequence_path = tmp_path / f"{policy_name}-sequences.parquet"
        per_anchor_path = tmp_path / f"{policy_name}-per-anchor.parquet"
        sequences.write_parquet(sequence_path)
        per_anchor.write_parquet(per_anchor_path)
        sequence_paths[policy_name] = str(sequence_path)
        per_anchor_paths[policy_name] = str(per_anchor_path)

    summary_path = tmp_path / "summary.parquet"
    pairwise_path = tmp_path / "pairwise.parquet"
    write_policy_comparison(
        sequence_paths,
        per_anchor_paths,
        summary_path,
        pairwise_path,
    )

    summary = pl.read_parquet(summary_path).sort("projection_policy")
    center = summary.filter(pl.col("projection_policy") == "center_1").row(
        0, named=True
    )
    assert center["accepted_pairs"] == 2
    assert center["requested_pairs"] == 4
    assert center["recovered_fraction"] == 0.5
    assert center["mean_mapped_landmark_fraction"] == 1.0
    assert center["mean_emitted_flank_bases"] == 254.0

    pairwise = pl.read_parquet(pairwise_path).row(0, named=True)
    assert pairwise["accepted_by_both"] == 1
    assert pairwise["baseline_only"] == 0
    assert pairwise["policy_only"] == 1
    assert pairwise["accepted_union"] == 2
    assert pairwise["accepted_jaccard"] == 0.5
