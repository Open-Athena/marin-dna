"""Materialize per-variant (context, ref_completion, alt_completion) for the
online lm_eval VEP scorer. Emits two rows per variant: one per strand."""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import datasets

from marin_dna.data.dna import complement_base
from marin_dna.data.genome import Genome
from marin_dna.data.transforms import _get_variant_window


def _add_eval_harness_fields(
    example: dict[str, Any],
    genome: Genome,
    window_size: int,
    strand: Literal["+", "-"],
) -> dict[str, Any]:
    """Per-example transform for one strand. Assumes SNVs."""
    window, var_pos = _get_variant_window(example, genome, window_size, strand=strand)
    alt = str(example["alt"]).upper()
    alt_in_strand = alt if strand == "+" else complement_base(alt)
    right_flank = window[var_pos + 1 :]
    return {
        "context": window[:var_pos],
        # window[var_pos] equals ref_in_strand by _get_variant_window's assert.
        "ref_completion": window[var_pos:],
        "alt_completion": alt_in_strand + right_flank,
        "strand": strand,
    }


def materialize_sequences(
    dataset: datasets.Dataset,
    genome: Genome,
    window_size: int,
) -> datasets.Dataset:
    """Add materialized sequence fields to a variant dataset.

    Each input variant emits two output rows (one per strand). Renames
    ``label`` → ``target`` and adds ``[context, ref_completion, alt_completion,
    strand]``. Rows are sorted by ``(chrom, pos, ref, alt, strand)`` so per-
    variant pairs are adjacent.
    """
    strands: tuple[Literal["+", "-"], ...] = ("+", "-")
    parts = [
        dataset.map(
            partial(
                _add_eval_harness_fields,
                genome=genome,
                window_size=window_size,
                strand=strand,
            )
        )
        for strand in strands
    ]
    out = datasets.concatenate_datasets(parts)
    if "label" in out.column_names:
        out = out.rename_column("label", "target")
    sort_keys = [
        c for c in ("chrom", "pos", "ref", "alt", "strand") if c in out.column_names
    ]
    return out.sort(sort_keys)
