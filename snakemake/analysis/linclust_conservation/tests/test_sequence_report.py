from marin_dna_linclust_conservation.sequence_report import (
    SourceSequence,
    parse_sequence_records,
    sample_tiled_intervals,
)


def _record(
    accession: str,
    sequence: str,
    *,
    unit: str,
    role: str,
    length: int = 1_000,
    location: str = "Chromosome",
) -> dict[str, object]:
    return {
        "assembly_accession": accession,
        "assembly_unit": unit,
        "assigned_molecule_location_type": location,
        "length": length,
        "refseq_accession": sequence,
        "role": role,
    }


def test_sequence_filter_keeps_primary_roles_and_excludes_alt_and_mitochondria() -> (
    None
):
    records = [
        _record(
            "GCF_1.1", "NC_1.1", unit="Primary Assembly", role="assembled-molecule"
        ),
        _record("GCF_1.1", "NC_STRAIN.1", unit="C57BL/6J", role="assembled-molecule"),
        _record("GCF_1.1", "NW_1.1", unit="Primary Assembly", role="unplaced-scaffold"),
        _record("GCF_1.1", "NT_1.1", unit="ALT_REF_LOCI_1", role="alt-scaffold"),
        _record(
            "GCF_1.1",
            "NC_MT.1",
            unit="non-nuclear",
            role="assembled-molecule",
            location="Mitochondrion",
        ),
    ]
    selected = parse_sequence_records(records)
    assert [sequence.sequence_accession for sequence in selected] == [
        "NC_1.1",
        "NC_STRAIN.1",
        "NW_1.1",
    ]
    assert not any(sequence.is_mitochondrial for sequence in selected)


def test_uniform_grid_sampling_is_deterministic_and_on_stride() -> None:
    sequences = [
        SourceSequence(
            "GCF_1.1",
            "NC_A.1",
            1_000,
            "assembled-molecule",
            "Primary Assembly",
            "Chromosome",
        ),
        SourceSequence(
            "GCF_1.1",
            "NC_B.1",
            2_000,
            "assembled-molecule",
            "Primary Assembly",
            "Chromosome",
        ),
    ]
    first = sample_tiled_intervals(
        sequences,
        window_length=255,
        stride=128,
        sample_size=8,
        seed=521,
    )
    second = sample_tiled_intervals(
        reversed(sequences),
        window_length=255,
        stride=128,
        sample_size=8,
        seed=521,
    )
    assert first == second
    assert all(interval.start % 128 == 0 for interval in first)
    assert all(interval.end - interval.start == 255 for interval in first)
