from marin_dna_carbon_conditioning_vep.dna import reverse_complement


def test_reverse_complement_preserves_length_case_and_unknowns() -> None:
    sequence = "AaCcGTN?"
    reverse = reverse_complement(sequence)
    assert reverse == "?NACgGtT"
    assert len(reverse) == len(sequence)
    assert reverse_complement(reverse) == sequence
