"""Fixed historical loss views and conservative source-interval holdout selection."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np


def loss_views(
    sequences: Sequence[str], per_target_losses: np.ndarray
) -> dict[str, float | int]:
    """Partition original, unaugmented 8191 targets per 8192-base window.

    N stays in the original loss and uppercase N is in the unmasked partition.
    Uppercase means unmasked, not independently verified nonrepetitive biology.
    Inputs are token losses, before any sample-normalized training loss weights.
    """
    if not sequences or any(len(s) != 8192 for s in sequences):
        raise ValueError("Validation requires original 8192-base windows")
    losses = np.asarray(per_target_losses, dtype=np.float64)
    if losses.shape != (len(sequences), 8191) or not np.all(np.isfinite(losses)):
        raise ValueError("Expected finite losses for every original next-base target")
    chars = np.stack(
        [np.frombuffer(s[1:].encode("ascii"), dtype=np.uint8) for s in sequences]
    )
    repeat = (chars >= ord("a")) & (chars <= ord("z"))
    unmasked = (chars >= ord("A")) & (chars <= ord("Z"))
    if not np.all(repeat | unmasked) or not repeat.any() or not unmasked.any():
        raise ValueError("Expected both native case partitions and alphabetic DNA")
    nr, nu = int(repeat.sum()), int(unmasked.sum())
    sr, su = float(losses[repeat].sum()), float(losses[unmasked].sum())
    return {
        "eval/loss": float(losses.mean()),
        "eval/loss_softmasked": sr / nr,
        "eval/loss_nonsoftmasked": su / nu,
        "eval/loss_weighted_10to1": (10 * su + sr) / (10 * nu + nr),
        "eval/softmasked_targets": nr,
        "eval/nonsoftmasked_targets": nu,
        "eval/unknown_N_targets": int(
            ((chars == ord("N")) | (chars == ord("n"))).sum()
        ),
        "eval/examples": len(sequences),
    }


def select_disjoint_validation(
    records: Sequence[dict[str, Any]], count: int = 512, seed: int = 593
) -> list[dict[str, Any]]:
    """Select from a previously screened validation candidate pool.

    Retain all species (when count permits), one example per source locus and no
    overlapping selected source intervals, considering both strands together.
    Source coordinates are opaque for rendering, but compared in the same raw
    dataset convention, so no unverified coordinate conversion is introduced.
    """
    if count < 1:
        raise ValueError("Validation count must be positive")
    rng = np.random.default_rng(seed)
    ordered = list(records)
    if any(r["split"] != "validation" for r in ordered):
        raise ValueError("Holdout selection must use only validation rows")
    rng.shuffle(ordered)
    by_species: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ordered:
        by_species[r["assembly"]].append(r)
    selected = []
    intervals: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    # Round-robin gives every species a chance without oversampling a rare one.
    species_order = sorted(by_species)
    rng.shuffle(species_order)
    while len(selected) < count:
        added = False
        for species in species_order:
            candidates = by_species[species]
            while candidates:
                r = candidates.pop()
                key = (r["assembly"], r["chrom"])
                if any(
                    r["start"] < end and r["end"] > start
                    for start, end in intervals[key]
                ):
                    continue
                selected.append(r)
                intervals[key].append((r["start"], r["end"]))
                added = True
                break
            if len(selected) == count:
                break
        if not added:
            raise ValueError("Insufficient disjoint source regions for requested count")
    return sorted(selected, key=lambda r: r["row"])


def verify_disjoint_metadata(
    selected: Sequence[dict[str, Any]],
    metadata_root: Path,
    expected_rows: int = 2_638_656,
) -> int:
    """Recheck the chosen holdout against every decoded training metadata row."""
    import pyarrow.parquet as pq

    intervals: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for r in selected:
        intervals[r["assembly"], r["chrom"]].append((r["start"], r["end"]))
    for values in intervals.values():
        values.sort()
        if any(a[1] > b[0] for a, b in pairwise(values)):
            raise ValueError("Selected validation source intervals intersect")
    checked = 0
    for path in sorted(metadata_root.glob("*.parquet")):
        for r in pq.read_table(
            path, columns=["assembly", "chrom", "start", "end"]
        ).to_pylist():
            values = intervals.get((r["assembly"], r["chrom"]), [])
            prior = bisect_left(values, (r["end"],)) - 1
            if prior >= 0 and values[prior][1] > r["start"]:
                raise ValueError(
                    "Validation source region intersects a training source region"
                )
            checked += 1
    if checked != expected_rows:
        raise ValueError("Expected every training metadata row from the pinned dataset")
    return checked
