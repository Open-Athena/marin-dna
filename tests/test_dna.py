from marin_dna.data.dna import complement_base, reverse_complement


def test_reverse_complement_preserves_case_and_unknown_bases() -> None:
    assert reverse_complement("AaCGTNx") == "xNACGtT"


def test_complement_base_handles_canonical_and_unknown_bases() -> None:
    assert [complement_base(base) for base in "ACGT"] == list("TGCA")
    assert complement_base("N") == "N"
