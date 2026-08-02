from __future__ import annotations

from pathlib import Path

import polars as pl

from analyze_motif_contexts import (
    feature_frame,
    metadata_summary,
    motif_rows,
    plot_feature,
    top_variant_rows,
)


def test_feature_frame_is_unique_and_covers_anchor_layers() -> None:
    frame = feature_frame()
    assert set(frame["block"].to_list()) == {1, 10, 19}
    assert (
        frame.select(pl.struct("block", "feature_id").n_unique()).item() == frame.height
    )


def test_top_variant_rows_are_outcome_blind_and_rank_absolute_delta() -> None:
    activation = pl.DataFrame(
        {
            "panel_row": [0, 1, 2],
            "feature_id": [7, 7, 7],
            "ref_activation": [0.0, 4.0, 2.0],
            "alt_activation": [3.0, 0.0, 2.5],
            "delta": [3.0, -4.0, 0.5],
        }
    )
    panel = pl.DataFrame(
        {
            "panel_row": [0, 1, 2],
            "position_status": [
                "focal_repeat",
                "repeat_free_window",
                "near_repeat",
            ],
            "allele_change": ["A>G", "C>T", "G>A"],
        }
    )
    observed, summary = top_variant_rows(
        activation,
        panel,
        block=10,
        feature_id=7,
        orientation="forward",
        reason="test",
    )
    assert "label" not in observed.columns
    assert observed["panel_row"].to_list() == [1, 0, 2]
    assert observed["activation_transition"].to_list() == [
        "active_to_inactive",
        "inactive_to_active",
        "active_to_active_changed",
    ]
    assert summary["paired_nonzero_variants"] == 3


def test_motif_rows_and_plot_integrate(tmp_path: Path) -> None:
    context_rows = []
    for context_id in range(80):
        sequence = list("ACGT" * 63 + "ACG")
        sequence[127] = "A" if context_id < 40 else "C"
        context_rows.append(
            {
                "context_id": context_id,
                "chrom": "1",
                "pos0": 1_000 + context_id,
                "sequence": "".join(sequence),
                "is_repeat": True,
                "repeat_strand": "+",
                "repeat_name": "AluSx",
                "repeat_class": "SINE",
                "repeat_family": "Alu",
                "family_label": "SINE|Alu",
                "subfamily_label": "SINE|Alu|AluSx",
                "milli_div": 50,
                "boundary_distance": 40,
                "overlap_count": 1,
                "gc_fraction": 0.5,
                "gc_bin": 5,
                "cpg_count": 10,
                "shannon_entropy": 2.0,
                "max_homopolymer": 1,
                "repeat_fraction": 1.0,
            }
        )
    contexts = pl.DataFrame(context_rows)
    activations = pl.DataFrame(
        {
            "context_id": list(range(40)),
            "feature_id": [7] * 40,
            "activation": [float(40 - index) for index in range(40)],
        }
    )
    combined, position, kmers, summary = motif_rows(
        contexts=contexts,
        activations=activations,
        block=10,
        feature_id=7,
        orientation="forward",
        reason="synthetic",
    )
    assert combined.height == 80
    assert summary["status"] == "analyzed"
    assert summary["match_chrom_class_gc"] == 40
    assert position.height == 63 * 4
    assert kmers.height > 0
    categories = metadata_summary(combined)
    assert set(categories["role"].to_list()) == {"top", "control"}

    both_orientations = pl.concat(
        [
            position,
            position.with_columns(pl.lit("reverse_complement").alias("orientation")),
        ]
    )
    paths = plot_feature(
        both_orientations,
        block=10,
        feature_id=7,
        output_dir=tmp_path,
    )
    assert {path.suffix for path in paths} == {".png", ".svg"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
