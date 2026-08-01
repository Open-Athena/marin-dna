from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest

import panel
from panel import (
    LOCAL_THREAD_LIMITS,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    RawCdsSegment,
    annotate_transcript_hit,
    assign_blocks,
    balance_panel,
    build_transcript,
    local_heavy_guard,
    variant_hashes,
)


def test_import_caps_native_thread_pools() -> None:
    assert {
        variable: int(os.environ[variable]) for variable in LOCAL_THREAD_LIMITS
    } == {
        "POLARS_MAX_THREADS": 2,
        "RAYON_NUM_THREADS": 2,
        "OMP_NUM_THREADS": 1,
        "MKL_NUM_THREADS": 1,
        "OPENBLAS_NUM_THREADS": 1,
        "NUMEXPR_NUM_THREADS": 1,
    }


def test_local_heavy_guard_rejects_a_second_task(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(panel, "LOCAL_HEAVY_LOCK", tmp_path / "heavy.lock")
    monkeypatch.setattr(
        panel,
        "memory_status_bytes",
        lambda: {"MemAvailable": 10 * 1024**3, "SwapTotal": 0},
    )
    monkeypatch.setattr(panel.os, "getloadavg", lambda: (0.5, 0.5, 0.5))
    monkeypatch.setattr(
        panel,
        "lower_process_priority",
        lambda: {"nice": 10, "ionice_class": 2, "ionice_priority": 7},
    )
    with local_heavy_guard() as policy:
        assert policy["memory_available_at_start_bytes"] == 10 * 1024**3
        with (
            pytest.raises(RuntimeError, match="another task holds"),
            local_heavy_guard(),
        ):
            raise AssertionError("unreachable")


def test_local_heavy_guard_rejects_memory_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        panel,
        "memory_status_bytes",
        lambda: {"MemAvailable": 5 * 1024**3, "SwapTotal": 0},
    )
    with (
        pytest.raises(AssertionError, match="require at least 6 GiB"),
        local_heavy_guard(),
    ):
        raise AssertionError("unreachable")


def test_local_heavy_guard_rejects_high_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        panel,
        "memory_status_bytes",
        lambda: {"MemAvailable": 10 * 1024**3, "SwapTotal": 0},
    )
    monkeypatch.setattr(panel.os, "getloadavg", lambda: (2.1, 1.0, 0.5))
    with (
        pytest.raises(AssertionError, match="require at most 2.0"),
        local_heavy_guard(),
    ):
        raise AssertionError("unreachable")


def test_variant_hashes_are_stable_and_seeded() -> None:
    frame = pl.DataFrame(
        {
            "pos": [10, 10, 11],
            "ref": ["A", "A", "G"],
            "alt": ["C", "G", "T"],
        }
    )
    first = variant_hashes(frame, seed=428)
    second = variant_hashes(frame, seed=428)
    different_seed = variant_hashes(frame, seed=429)
    np.testing.assert_array_equal(first, second)
    assert len(set(first.tolist())) == frame.height
    assert not np.array_equal(first, different_seed)


def test_assign_blocks_is_deterministic_and_has_preregistered_sizes() -> None:
    forward = assign_blocks(range(29))
    reverse = assign_blocks(reversed(range(29)))
    assert forward == reverse
    assert list(forward.values()).count("discovery") == 17
    assert list(forward.values()).count("validation") == 6
    assert list(forward.values()).count("test") == 6


def test_transcript_offsets_cross_plus_strand_exons() -> None:
    transcript = build_transcript(
        transcript_id="plus",
        gene_id="gene",
        gene_name="GENE",
        strand="+",
        segments=[
            RawCdsSegment(start0=10, end0=14, phase=0),
            RawCdsSegment(start0=20, end0=25, phase=2),
        ],
    )
    assert transcript.phase_consistent
    assert transcript.coding_offset(10) == 0
    assert transcript.coding_offset(13) == 3
    assert transcript.coding_offset(20) == 4
    assert transcript.genomic_position0(3) == 13
    assert transcript.genomic_position0(4) == 20


def test_transcript_offsets_follow_negative_transcript_order() -> None:
    transcript = build_transcript(
        transcript_id="minus",
        gene_id="gene",
        gene_name="GENE",
        strand="-",
        segments=[
            RawCdsSegment(start0=10, end0=15, phase=2),
            RawCdsSegment(start0=20, end0=24, phase=0),
        ],
    )
    assert transcript.phase_consistent
    assert transcript.coding_offset(23) == 0
    assert transcript.coding_offset(20) == 3
    assert transcript.coding_offset(14) == 4
    assert transcript.genomic_position0(3) == 20
    assert transcript.genomic_position0(4) == 14


def test_annotation_reconstructs_plus_exon_spanning_missense() -> None:
    transcript = build_transcript(
        transcript_id="plus",
        gene_id="gene",
        gene_name="GENE",
        strand="+",
        segments=[
            RawCdsSegment(start0=10, end0=14, phase=0),
            RawCdsSegment(start0=20, end0=25, phase=2),
        ],
    )
    sequence = list("A" * 30)
    sequence[13] = "G"
    sequence[20] = "A"
    sequence[21] = "A"
    hit = annotate_transcript_hit(
        transcript,
        position0=20,
        ref="A",
        alt="G",
        sequence="".join(sequence),
    )
    assert hit is not None
    assert hit["codon_position"] == 2
    assert hit["ref_codon"] == "GAA"
    assert hit["alt_codon"] == "GGA"
    assert hit["amino_acid_change"] == "E>G"
    assert hit["predicted_consequence"] == POSITIVE_CLASS


def test_annotation_orients_negative_strand_codon() -> None:
    transcript = build_transcript(
        transcript_id="minus",
        gene_id="gene",
        gene_name="GENE",
        strand="-",
        segments=[
            RawCdsSegment(start0=10, end0=15, phase=2),
            RawCdsSegment(start0=20, end0=24, phase=0),
        ],
    )
    sequence = list("A" * 30)
    sequence[20] = "C"
    sequence[14] = "T"
    sequence[13] = "T"
    hit = annotate_transcript_hit(
        transcript,
        position0=14,
        ref="T",
        alt="C",
        sequence="".join(sequence),
    )
    assert hit is not None
    assert hit["strand"] == "-"
    assert hit["codon_position"] == 2
    assert hit["ref_codon"] == "GAA"
    assert hit["alt_codon"] == "GGA"
    assert hit["predicted_consequence"] == POSITIVE_CLASS


def test_balance_panel_is_exact_with_global_block_splits() -> None:
    rows: list[dict[str, object]] = []
    position = 1
    available = {
        "discovery": (71, 67),
        "validation": (66, 69),
        "test": (65, 68),
    }
    blocks = {"discovery": 1, "validation": 18, "test": 24}
    for split, (positive_count, negative_count) in available.items():
        for label, count in (
            (POSITIVE_CLASS, positive_count),
            (NEGATIVE_CLASS, negative_count),
        ):
            for _ in range(count):
                rows.append(
                    {
                        "chrom": "21",
                        "pos": position,
                        "ref": "A",
                        "alt": "G",
                        "block_id": blocks[split],
                        "split": split,
                        "consequence_cre": label,
                        "sample_hash": position * 17,
                        "consensus_strand": "+",
                        "consensus_codon_position": 2,
                        "transcript_substitution": "A>G",
                    }
                )
                position += 1
    panel, metadata = balance_panel(pl.DataFrame(rows))
    assert metadata["retained_strata"] == ["2|A>G"]
    assert panel.height == 2 * (67 + 66 + 65)
    counts = panel.group_by(["split", "consequence_cre"]).len()
    for split, expected in (("discovery", 67), ("validation", 66), ("test", 65)):
        observed = counts.filter(pl.col("split") == split)
        assert observed.height == 2
        assert observed["len"].to_list() == [expected, expected]
    assert (
        panel.group_by("block_id").agg(pl.col("split").n_unique())["split"].max() == 1
    )
