"""Frozen design helpers for repeat-feature single-base saturation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from common import ISSUE, sha256_file
from extract_common import FOCAL_INDEX, ORIENTATIONS, WINDOW_BP
from motif_context_common import MOTIF_RADIUS, NUCLEOTIDES

RUN_ID = "dna-exp435-repeat-saturation-r1"
MOTIF_ARCHIVE_RUN_ID = "dna-exp435-repeat-motif-context-r1"
MOTIF_ARCHIVE_MANIFEST_SHA256 = (
    "50bae7719bdc3f5ad448497c8775672dfa5f1690dd433c188dc4f036ecd649f4"
)
TOP_CONTEXTS_PER_VIEW = 64
MIN_CONTEXTS = 32
OFFSETS = tuple(range(-MOTIF_RADIUS, MOTIF_RADIUS + 1))
STATES_PER_CONTEXT = 1 + 3 * len(OFFSETS)
MUTATIONS_PER_CONTEXT = STATES_PER_CONTEXT - 1

assert MOTIF_RADIUS == 31
assert STATES_PER_CONTEXT == 190 and MUTATIONS_PER_CONTEXT == 189
assert FOCAL_INDEX - MOTIF_RADIUS >= 0
assert FOCAL_INDEX + MOTIF_RADIUS < WINDOW_BP


@dataclass(frozen=True)
class SaturationFeature:
    block: int
    feature_id: int


FEATURES = (
    SaturationFeature(1, 10488),
    SaturationFeature(10, 6903),
    SaturationFeature(10, 7168),
    SaturationFeature(10, 9767),
    SaturationFeature(10, 11265),
    SaturationFeature(10, 13092),
    SaturationFeature(19, 219),
    SaturationFeature(19, 13311),
    SaturationFeature(19, 13894),
)
FEATURE_KEYS = tuple((item.block, item.feature_id) for item in FEATURES)
VIEW_KEYS = tuple(
    (item.block, item.feature_id, orientation)
    for item in FEATURES
    for orientation in ORIENTATIONS
)

assert len(set(FEATURE_KEYS)) == len(FEATURE_KEYS) == 9
assert len(set(VIEW_KEYS)) == len(VIEW_KEYS) == 18


def verify_motif_archive(
    root: Path,
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    """Verify and load the exact prior motif archive."""

    archive_path = root / "archive_manifest.json"
    assert archive_path.is_file()
    assert sha256_file(archive_path) == MOTIF_ARCHIVE_MANIFEST_SHA256
    archive = json.loads(archive_path.read_text())
    assert archive["issue"] == ISSUE and archive["run_id"] == MOTIF_ARCHIVE_RUN_ID
    assert archive["analysis_status"] == "posthoc_repeat_motif_and_context_description"
    for relative, expected in archive["artifacts"].items():
        path = root / relative
        assert path.is_file() and path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]
    motif_root = root / "motif_context"
    top_contexts = pl.read_parquet(motif_root / "top_contexts.parquet")
    kmers = pl.read_parquet(motif_root / "kmer_enrichment.parquet")
    return archive, top_contexts, kmers


def select_contexts(frame: pl.DataFrame) -> pl.DataFrame:
    """Select the frozen top-64 contexts per feature/orientation view."""

    required = {
        "context_id",
        "chrom",
        "pos0",
        "activation",
        "role",
        "block",
        "arm",
        "feature_id",
        "orientation",
        "model_sequence",
    }
    assert required <= set(frame.columns)
    frames: list[pl.DataFrame] = []
    for block, feature_id, orientation in VIEW_KEYS:
        selected = (
            frame.filter(
                (pl.col("role") == "top")
                & (pl.col("block") == block)
                & (pl.col("feature_id") == feature_id)
                & (pl.col("orientation") == orientation)
            )
            .sort("activation", "context_id", descending=[True, False])
            .head(TOP_CONTEXTS_PER_VIEW)
        )
        assert selected.height == TOP_CONTEXTS_PER_VIEW
        assert selected["activation"].is_not_null().all()
        frames.append(selected)
    result = pl.concat(frames, how="vertical_relaxed").with_row_index(
        "saturation_context_id"
    )
    assert result.height == len(VIEW_KEYS) * TOP_CONTEXTS_PER_VIEW
    assert result["saturation_context_id"].to_list() == list(range(result.height))
    assert result["model_sequence"].str.len_chars().unique().to_list() == [WINDOW_BP]
    assert result.filter(
        ~pl.col("model_sequence").str.contains(r"^[ACGT]+$")
    ).is_empty()
    counts = result.group_by("block", "feature_id", "orientation").len()
    assert counts["len"].unique().to_list() == [TOP_CONTEXTS_PER_VIEW]
    return result


def qualifying_kmer_sets(
    frame: pl.DataFrame,
) -> dict[tuple[int, int, str], frozenset[str]]:
    """Return the preregistered positive, FDR-significant k-mer dictionaries."""

    required = {"kmer", "q_value", "log2_odds", "block", "feature_id", "orientation"}
    assert required <= set(frame.columns)
    filtered = frame.filter((pl.col("q_value") < 0.05) & (pl.col("log2_odds") > 0))
    result: dict[tuple[int, int, str], frozenset[str]] = {}
    for block, feature_id, orientation in VIEW_KEYS:
        kmers = frozenset(
            filtered.filter(
                (pl.col("block") == block)
                & (pl.col("feature_id") == feature_id)
                & (pl.col("orientation") == orientation)
            )["kmer"].to_list()
        )
        assert kmers
        assert all(
            3 <= len(kmer) <= 6 and set(kmer) <= set(NUCLEOTIDES) for kmer in kmers
        )
        result[(block, feature_id, orientation)] = kmers
    assert set(result) == set(VIEW_KEYS)
    return result


def kmer_occurrence_counts(sequence: str, dictionary: frozenset[str]) -> Counter[str]:
    """Count dictionary k-mer occurrences in a fixed sequence window."""

    assert len(sequence) == 2 * MOTIF_RADIUS + 1
    assert set(sequence) <= set(NUCLEOTIDES)
    lengths = {len(kmer) for kmer in dictionary}
    observed = [
        sequence[index : index + length]
        for length in lengths
        for index in range(len(sequence) - length + 1)
    ]
    return Counter(kmer for kmer in observed if kmer in dictionary)


def enumerate_context_states(
    row: dict[str, Any], dictionary: frozenset[str]
) -> list[dict[str, Any]]:
    """Enumerate baseline and all single-base edits for one oriented context."""

    sequence = str(row["model_sequence"])
    assert len(sequence) == WINDOW_BP and set(sequence) <= set(NUCLEOTIDES)
    orientation = str(row["orientation"])
    assert orientation in ORIENTATIONS
    motif_start = FOCAL_INDEX - MOTIF_RADIUS
    motif_stop = FOCAL_INDEX + MOTIF_RADIUS + 1
    baseline_motif = sequence[motif_start:motif_stop]
    baseline_kmers = kmer_occurrence_counts(baseline_motif, dictionary)
    common = {
        "saturation_context_id": int(row["saturation_context_id"]),
        "block": int(row["block"]),
        "arm": str(row["arm"]),
        "feature_id": int(row["feature_id"]),
        "orientation": orientation,
    }
    states: list[dict[str, Any]] = [
        {
            **common,
            "is_baseline": True,
            "model_offset": None,
            "reference_offset": None,
            "model_ref": None,
            "model_alt": None,
            "baseline_enriched_kmers": sum(baseline_kmers.values()),
            "destroyed_kmers": 0,
            "created_kmers": 0,
            "net_kmers_lost": 0,
            "motif_loss": False,
            "motif_gain": False,
            "neutral": True,
            "sequence": sequence,
        }
    ]
    for model_offset in OFFSETS:
        sequence_index = FOCAL_INDEX + model_offset
        model_ref = sequence[sequence_index]
        for model_alt in NUCLEOTIDES:
            if model_alt == model_ref:
                continue
            mutant = (
                sequence[:sequence_index] + model_alt + sequence[sequence_index + 1 :]
            )
            assert sum(a != b for a, b in zip(sequence, mutant, strict=True)) == 1
            mutant_kmers = kmer_occurrence_counts(
                mutant[motif_start:motif_stop], dictionary
            )
            destroyed = baseline_kmers - mutant_kmers
            created = mutant_kmers - baseline_kmers
            destroyed_occurrences = sum(destroyed.values())
            created_occurrences = sum(created.values())
            net_lost = destroyed_occurrences - created_occurrences
            states.append(
                {
                    **common,
                    "is_baseline": False,
                    "model_offset": model_offset,
                    "reference_offset": (
                        model_offset if orientation == "forward" else -model_offset
                    ),
                    "model_ref": model_ref,
                    "model_alt": model_alt,
                    "baseline_enriched_kmers": sum(baseline_kmers.values()),
                    "destroyed_kmers": destroyed_occurrences,
                    "created_kmers": created_occurrences,
                    "net_kmers_lost": net_lost,
                    "motif_loss": net_lost > 0,
                    "motif_gain": net_lost < 0,
                    "neutral": net_lost == 0,
                    "sequence": mutant,
                }
            )
    assert len(states) == STATES_PER_CONTEXT
    assert sum(not state["is_baseline"] for state in states) == MUTATIONS_PER_CONTEXT
    return states


def build_state_table(
    contexts: pl.DataFrame,
    kmer_sets: dict[tuple[int, int, str], frozenset[str]],
) -> pl.DataFrame:
    """Materialize deterministic sequence states and baseline indices."""

    rows: list[dict[str, Any]] = []
    for row in contexts.iter_rows(named=True):
        key = (int(row["block"]), int(row["feature_id"]), str(row["orientation"]))
        rows.extend(enumerate_context_states(row, kmer_sets[key]))
    states = pl.DataFrame(rows).with_row_index("state_index")
    states = states.with_columns(
        (pl.col("state_index") - (pl.col("state_index") % STATES_PER_CONTEXT)).alias(
            "baseline_state_index"
        )
    )
    assert states.height == contexts.height * STATES_PER_CONTEXT
    assert states["state_index"].to_list() == list(range(states.height))
    assert states.group_by("saturation_context_id").len()["len"].unique().to_list() == [
        STATES_PER_CONTEXT
    ]
    assert states.filter(pl.col("is_baseline")).height == contexts.height
    assert (
        states.filter(~pl.col("is_baseline")).height
        == contexts.height * MUTATIONS_PER_CONTEXT
    )
    assert states.filter(
        pl.col("is_baseline")
        & (pl.col("state_index") != pl.col("baseline_state_index"))
    ).is_empty()
    return states
