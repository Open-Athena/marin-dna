"""Pure sequence-state construction for feature-1662 saturation mutagenesis."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from saturation_common import (
    FOCAL_INDEX,
    NUCLEOTIDES,
    ORIENTATIONS,
    POSITIONS,
    WINDOW_BP,
    mutate_transcript_codon,
)

STATES_PER_CONTEXT = 1 + 3 * len(POSITIONS)
MUTATIONS_PER_CONTEXT = STATES_PER_CONTEXT - 1

assert STATES_PER_CONTEXT == 94 and MUTATIONS_PER_CONTEXT == 93


def enumerate_context_states(
    row: dict[str, Any], reference_sequence: str
) -> list[dict[str, Any]]:
    """Enumerate one baseline and all single-nucleotide edits in the 31-bp span."""

    reference_sequence = reference_sequence.upper()
    assert len(reference_sequence) == WINDOW_BP
    assert set(reference_sequence) <= set(NUCLEOTIDES)
    assert reference_sequence[FOCAL_INDEX] == str(row["ref"]).upper()
    context_index = int(row["context_index"])
    strand = int(row["strand"])
    focal_codon_position = int(row["focal_codon_position"])
    ref_codon = str(row["ref_codon"])
    baseline = {
        "context_index": context_index,
        "is_baseline": True,
        "genomic_offset": None,
        "transcript_offset": None,
        "genomic_ref": None,
        "genomic_alt": None,
        "edited_codon_position": None,
        "alternate_codon": None,
        "alternate_amino_acid": None,
        "consequence": None,
        "sequence": reference_sequence,
    }
    states = [baseline]
    for genomic_offset in POSITIONS:
        sequence_index = FOCAL_INDEX + genomic_offset
        genomic_ref = reference_sequence[sequence_index]
        for genomic_alt in NUCLEOTIDES:
            if genomic_alt == genomic_ref:
                continue
            sequence = (
                reference_sequence[:sequence_index]
                + genomic_alt
                + reference_sequence[sequence_index + 1 :]
            )
            assert (
                sum(
                    left != right
                    for left, right in zip(reference_sequence, sequence, strict=True)
                )
                == 1
            )
            annotation = mutate_transcript_codon(
                ref_codon,
                focal_codon_position=focal_codon_position,
                strand=strand,
                genomic_offset=genomic_offset,
                genomic_alt=genomic_alt,
            )
            if annotation is None:
                edited_codon_position = None
                alternate_codon = None
                alternate_amino_acid = None
                consequence = None
            else:
                (
                    edited_codon_position,
                    alternate_codon,
                    alternate_amino_acid,
                    consequence,
                ) = annotation
            states.append(
                {
                    "context_index": context_index,
                    "is_baseline": False,
                    "genomic_offset": genomic_offset,
                    "transcript_offset": genomic_offset * strand,
                    "genomic_ref": genomic_ref,
                    "genomic_alt": genomic_alt,
                    "edited_codon_position": edited_codon_position,
                    "alternate_codon": alternate_codon,
                    "alternate_amino_acid": alternate_amino_acid,
                    "consequence": consequence,
                    "sequence": sequence,
                }
            )
    assert len(states) == STATES_PER_CONTEXT
    assert sum(not state["is_baseline"] for state in states) == MUTATIONS_PER_CONTEXT
    return states


def build_response_table(
    states: pl.DataFrame, activations: dict[str, np.ndarray]
) -> pl.DataFrame:
    """Pair every mutant activation with its context-specific baseline."""

    mutations = states.filter(~pl.col("is_baseline")).drop("sequence")
    state_indices = mutations["state_index"].to_numpy()
    baseline_indices = mutations["baseline_state_index"].to_numpy()
    frames: list[pl.DataFrame] = []
    for orientation in ORIENTATIONS:
        values = activations[orientation]
        assert values.shape == (states.height,)
        baseline = values[baseline_indices]
        mutant = values[state_indices]
        frames.append(
            mutations.with_columns(
                pl.lit(orientation).alias("orientation"),
                pl.Series("baseline_activation", baseline, pl.Float32),
                pl.Series("mutant_activation", mutant, pl.Float32),
                pl.Series("delta", mutant - baseline, pl.Float32),
                pl.Series("abs_delta", np.abs(mutant - baseline), pl.Float32),
            )
        )
    result = pl.concat(frames).sort("orientation", "state_index")
    assert result.height == 2 * MUTATIONS_PER_CONTEXT * (
        states.height // STATES_PER_CONTEXT
    )
    assert set(result["orientation"].unique()) == set(ORIENTATIONS)
    assert result.select(
        pl.col("baseline_activation").is_finite().all(),
        pl.col("mutant_activation").is_finite().all(),
        pl.col("delta").is_finite().all(),
        pl.col("abs_delta").is_finite().all(),
    ).row(0) == (True, True, True, True)
    return result
