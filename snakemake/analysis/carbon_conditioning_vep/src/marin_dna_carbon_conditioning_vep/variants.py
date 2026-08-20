"""Mendelian dataset validation and centered 8,192 bp window materialization."""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

REQUIRED_VARIANT_COLUMNS = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "subset",
    "match_group",
)
CANONICAL_BASES = frozenset("ACGT")


@dataclass(frozen=True)
class WindowFailure:
    """One row-level exclusion that triggers whole-match-group removal."""

    match_group: int
    variant_id: str
    reason: str
    detail: str


def variant_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Build the stable variant key used to align prompt conditions."""
    return f"{chrom}:{pos}:{ref}>{alt}"


def validate_mendelian_dataset(
    dataset: pd.DataFrame,
    *,
    expected_rows: int,
    expected_positives: int,
    expected_groups: int,
    expected_group_size: int,
    expected_chroms: set[str],
) -> pd.DataFrame:
    """Validate the pinned development split without reading any other split."""
    missing = sorted(set(REQUIRED_VARIANT_COLUMNS) - set(dataset.columns))
    assert not missing, f"Mendelian dataset missing columns: {missing}"
    df = dataset.copy()
    df["chrom"] = df["chrom"].astype(str)
    assert len(df) == expected_rows, f"row count {len(df)} != {expected_rows}"
    labels = df["label"].astype(bool)
    assert int(labels.sum()) == expected_positives, (
        f"positive count {int(labels.sum())} != {expected_positives}"
    )
    observed_chroms = set(df["chrom"])
    assert observed_chroms == expected_chroms, (
        f"chromosomes {sorted(observed_chroms)} != {sorted(expected_chroms)}"
    )
    assert not any(chrom.startswith("chr") for chrom in observed_chroms), (
        "Ensembl sequence names required; chr-prefix rewriting is forbidden"
    )
    for allele_col in ("ref", "alt"):
        alleles = df[allele_col].astype(str)
        assert alleles.str.len().eq(1).all(), f"{allele_col} contains non-SNV alleles"
        assert alleles.isin(CANONICAL_BASES).all(), (
            f"{allele_col} contains non-canonical alleles"
        )
        df[allele_col] = alleles
    assert (df["ref"] != df["alt"]).all(), "REF and ALT must differ for every SNV"
    assert pd.api.types.is_integer_dtype(df["pos"]), "pos must be integer and 1-based"
    assert (df["pos"] >= 1).all(), "1-based pos must be positive"

    group_sizes = df.groupby("match_group", sort=False).size()
    assert len(group_sizes) == expected_groups, (
        f"match_group count {len(group_sizes)} != {expected_groups}"
    )
    assert group_sizes.eq(expected_group_size).all(), (
        f"match_group sizes differ from {expected_group_size}: "
        f"{group_sizes[~group_sizes.eq(expected_group_size)].head().to_dict()}"
    )
    positives_per_group = labels.groupby(df["match_group"]).sum()
    assert positives_per_group.eq(1).all(), (
        "every 1:9 match_group must contain exactly one pathogenic positive"
    )
    subsets_per_group = df.groupby("match_group")["subset"].nunique()
    assert subsets_per_group.eq(1).all(), (
        "a match_group may not span consequence subsets"
    )

    ids = [
        variant_id(str(row.chrom), int(row.pos), str(row.ref), str(row.alt))
        for row in df.itertuples(index=False)
    ]
    assert len(ids) == len(set(ids)), "derived variant IDs are not unique"
    df.insert(0, "variant_id", ids)
    return df


def centered_window_bounds(pos_1based: int, window_size: int) -> tuple[int, int, int]:
    """Convert one 1-based SNV coordinate to a 0-based half-open centered window."""
    assert pos_1based >= 1, f"1-based position must be positive, got {pos_1based}"
    assert window_size > 0, f"window size must be positive, got {window_size}"
    center_0based = pos_1based - 1
    variant_index = window_size // 2
    start = center_0based - variant_index
    return start, start + window_size, variant_index


def replace_centered_allele(
    reference_sequence: str,
    variant_index: int,
    ref: str,
    alt: str,
) -> str:
    """Replace exactly the centered reference base with the alternate allele."""
    assert 0 <= variant_index < len(reference_sequence)
    assert reference_sequence[variant_index] == ref, (
        f"reference mismatch at window index {variant_index}: "
        f"FASTA={reference_sequence[variant_index]!r}, dataset={ref!r}"
    )
    assert ref in CANONICAL_BASES and alt in CANONICAL_BASES and ref != alt
    alternate = (
        reference_sequence[:variant_index]
        + alt
        + reference_sequence[variant_index + 1 :]
    )
    assert len(alternate) == len(reference_sequence)
    differences = sum(
        a != b for a, b in zip(reference_sequence, alternate, strict=True)
    )
    assert differences == 1, (
        f"ALT window changed {differences} positions instead of one"
    )
    return alternate


def _materialize_one_window(
    row: Any,
    genome: Callable[[str, int, int, str], str],
    chrom_lengths: Mapping[str, int],
    window_size: int,
) -> dict[str, Any]:
    chrom = str(row.chrom)
    if chrom not in chrom_lengths:
        raise ValueError(f"noncanonical_contig:{chrom}")
    start, end, variant_index = centered_window_bounds(int(row.pos), window_size)
    if start < 0 or end > chrom_lengths[chrom]:
        raise ValueError(
            f"boundary:{chrom}:{start}-{end}:contig_length={chrom_lengths[chrom]}"
        )
    sequence = str(genome(chrom, start, end, "+"))
    if len(sequence) != window_size:
        raise ValueError(f"window_length:{len(sequence)}:{window_size}")
    sequence = sequence.upper()
    if not set(sequence) <= CANONICAL_BASES:
        invalid = sorted(set(sequence) - CANONICAL_BASES)
        raise ValueError(f"noncanonical_sequence:{invalid}")
    if sequence[variant_index] != row.ref:
        raise ValueError(
            f"reference_allele:FASTA={sequence[variant_index]}:dataset={row.ref}"
        )
    alternate = replace_centered_allele(
        sequence,
        variant_index,
        str(row.ref),
        str(row.alt),
    )
    return {
        "window_start": start,
        "window_end": end,
        "variant_index": variant_index,
        "ref_sequence": sequence,
        "alt_sequence": alternate,
    }


def materialize_variant_windows(
    dataset: pd.DataFrame,
    genome: Callable[[str, int, int, str], str],
    chrom_lengths: Mapping[str, int],
    *,
    window_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build valid windows and drop every group touched by a row-level failure."""
    materialized: dict[str, dict[str, Any]] = {}
    failures: list[WindowFailure] = []
    for row in dataset.itertuples(index=False):
        try:
            materialized[str(row.variant_id)] = _materialize_one_window(
                row,
                genome,
                chrom_lengths,
                window_size,
            )
        except (AssertionError, ValueError) as exc:
            message = str(exc)
            reason = message.split(":", 1)[0] or type(exc).__name__
            failures.append(
                WindowFailure(
                    match_group=int(row.match_group),
                    variant_id=str(row.variant_id),
                    reason=reason,
                    detail=message,
                )
            )

    failed_groups = {failure.match_group for failure in failures}
    kept = dataset.loc[~dataset["match_group"].isin(failed_groups)].copy()
    assert not kept.empty, "window validation excluded every match_group"
    assert not kept["variant_id"].duplicated().any()
    window_rows = [materialized[variant_key] for variant_key in kept["variant_id"]]
    windows = pd.concat(
        [kept.reset_index(drop=True), pd.DataFrame(window_rows)], axis=1
    )
    assert windows.groupby("match_group").size().nunique() == 1, (
        "whole-group exclusion left ragged match groups"
    )

    failure_columns = ["match_group", "variant_id", "reason", "detail"]
    failure_frame = pd.DataFrame(
        [asdict(failure) for failure in failures], columns=failure_columns
    )
    if failed_groups:
        group_rows = dataset.loc[dataset["match_group"].isin(failed_groups)]
        assert set(group_rows["match_group"]) == failed_groups
    return windows, failure_frame


def validate_reference_contigs(
    observed: Mapping[str, int], expected: Mapping[str, int]
) -> None:
    """Require exact GRCh38 lengths for every development-split contig."""
    missing = sorted(set(expected) - set(observed))
    assert not missing, f"reference FASTA missing required Ensembl contigs: {missing}"
    mismatches = {
        chrom: (int(observed[chrom]), int(length))
        for chrom, length in expected.items()
        if int(observed[chrom]) != int(length)
    }
    assert not mismatches, f"reference contig length mismatches: {mismatches}"
