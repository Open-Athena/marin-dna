from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from characterize_grouped_l2_features import (
    _max_homopolymer,
    build_contexts,
    densify_features,
    feature_summary,
    orientation_concordance,
    reverse_complement,
    target_sensitivity,
)
from pyfaidx import Fasta


def test_sequence_helpers_and_coordinate_boundary(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import characterize_grouped_l2_features as characterize

    monkeypatch.setattr(characterize, "EXPECTED_ROWS", 2)
    sequence = ("ACGT" * 30)[:100]
    fasta_path = tmp_path / "tiny.fa"
    fasta_path.write_text(f">1\n{sequence}\n")
    indexed = Fasta(str(fasta_path))
    indexed.close()

    panel = pl.DataFrame(
        {
            "chrom": ["1", "1"],
            "pos": [32, 40],
            "ref": [sequence[31], sequence[39]],
            "alt": ["A", "C"],
        }
    )
    contexts = build_contexts(panel, fasta_path)
    assert contexts.height == 2
    assert contexts["ref_context"].str.len_chars().to_list() == [63, 63]
    assert contexts["ref_context"][0][31] == sequence[31]
    assert contexts["alt_context"][1][31] == "C"
    assert reverse_complement("ACGTN") == "NACGT"
    assert _max_homopolymer("AACCCGT") == 3


def test_densify_and_feature_summary_preserve_signed_delta() -> None:
    sparse = pl.DataFrame(
        {
            "panel_row": [0, 2],
            "feature_id": [219, 219],
            "ref_activation": [0.0, 4.0],
            "alt_activation": [2.0, 1.0],
            "delta": [2.0, -3.0],
        }
    )
    dense = densify_features(sparse, rows=4, feature_ids=(219,))
    assert dense["delta"].to_list() == [2.0, 0.0, -3.0, 0.0]
    assert dense["abs_delta"].to_list() == [2.0, 0.0, 3.0, 0.0]

    responses = dense.with_columns(
        pl.lit("block19-25m").alias("arm"),
        pl.lit(19).alias("report_block"),
        pl.lit("broad_accessibility").alias("feature_role"),
        pl.lit("forward").alias("orientation"),
    )
    summary = feature_summary(responses).row(0, named=True)
    assert summary["support"] == 2
    assert summary["positive_delta"] == 1
    assert summary["negative_delta"] == 1
    assert summary["prevalence"] == 0.5


def test_orientation_concordance_reports_overlap_and_sign() -> None:
    base = {
        "arm": ["block19-25m"] * 8,
        "report_block": [19] * 8,
        "feature_id": [219] * 8,
        "feature_role": ["broad_accessibility"] * 8,
        "panel_row": [0, 1, 2, 3] * 2,
        "orientation": ["forward"] * 4 + ["reverse_complement"] * 4,
        "ref_activation": [0.0] * 8,
        "alt_activation": [0.0] * 8,
    }
    responses = (
        pl.DataFrame(base)
        .with_columns(pl.Series("delta", [1.0, 2.0, 0.0, -1.0, 1.5, 0.0, 3.0, -2.0]))
        .with_columns(pl.col("delta").abs().alias("abs_delta"))
    )
    result = orientation_concordance(responses).row(0, named=True)
    assert result["active_union"] == 4
    assert result["active_intersection"] == 2
    assert result["active_jaccard"] == 0.5
    assert result["delta_sign_agreement_when_both_active"] == 1.0


def test_target_sensitivity_runs_registered_tail_protocols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import characterize_grouped_l2_features as characterize

    rows = 200
    monkeypatch.setattr(characterize, "EXPECTED_ROWS", rows)
    x = np.linspace(0.0, 10.0, rows, dtype=np.float32)
    responses = pl.DataFrame(
        {
            "arm": ["block19-25m"] * rows,
            "report_block": [19] * rows,
            "feature_id": [11_928] * rows,
            "feature_role": ["tail_sensitive_candidate"] * rows,
            "orientation": ["forward"] * rows,
            "panel_row": np.arange(rows, dtype=np.uint32),
            "abs_delta": x,
        }
    )
    values = np.column_stack((x, x**2, x[::-1])).astype(np.float32)
    catalog = pl.DataFrame(
        {
            "target_index": [0, 1, 2],
            "target_id": ["all_tracks", "RNA_SEQ", "tissue|liver"],
            "target_name": ["all", "RNA", "liver"],
            "group_axis": ["overall", "assay", "tissue"],
            "track_count": [4430, 100, 600],
        }
    )
    result = target_sensitivity(responses, {"selected": (values, catalog)})
    assert result.height == 18
    assert set(result["protocol"]) == {
        "raw",
        "log1p_feature",
        "spearman",
        "trim_feature_top_1pct",
        "trim_outcome_top_1pct",
        "trim_both_top_1pct",
    }
    assert result.filter(pl.col("protocol") == "raw")["n"].unique().item() == rows
    assert result.filter(pl.col("protocol") == "trim_both_top_1pct")["n"].min() < rows
    assert result["qvalue"].is_between(0, 1).all()
