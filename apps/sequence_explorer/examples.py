"""Built-in GRCh38 examples for the MarinDNA sequence explorer.

Coordinates are 0-based half-open. Negative-strand loci are stored already
reverse-complemented, so every sequence reads 5'→3' along its annotated strand.
The source intervals match ``nuc_dep.loci`` in the evals_v2 configuration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecommendedExample:
    name: str
    chrom: str
    start: int
    end: int
    strand: str
    sequence: str

    def __post_init__(self) -> None:
        assert self.end > self.start >= 0
        assert self.strand in {"+", "-"}
        assert len(self.sequence) == self.end - self.start
        assert set(self.sequence) <= set("ACGT")

    @property
    def label(self) -> str:
        return (
            f"{self.name} · GRCh38 chr{self.chrom}:{self.start}-{self.end} "
            f"({self.strand}, {len(self.sequence)} bp)"
        )


EXAMPLES: tuple[RecommendedExample, ...] = (
    RecommendedExample(
        name="LDLR",
        chrom="19",
        start=11089299,
        end=11089425,
        strand="+",
        sequence=(
            "AGTGGGAATCAGAGCTTCACGGGTTAAAAAGCCGATGTCACATCGGCCGTTCGAAACTCCT"
            "CCTCTTGCAGTGAGGTGAAGACATTTGAAAATCACCCCACTGCAAACTCCTCCCCCTGCTAGAAA"
        ),
    ),
    RecommendedExample(
        name="TH",
        chrom="11",
        start=2171682,
        end=2171868,
        strand="-",
        sequence=(
            "GGGGGCTTTGACGTCAGCTCAGCTTATAAGAGGCTGCTGGGCCAGGGCTGTGGAGACGGAG"
            "CCCGGACCTCCACACTGAGCCATGCCCACCCCCGACGCCACCACGCCACAGGCCAAGGGCTT"
            "CCGCAGGGCCGTGTCTGAGCTGGACGCCAAGCAGGCAGAGGCCATCATGGTAAGAGGGCAGGT"
        ),
    ),
    RecommendedExample(
        name="GRIA4",
        chrom="11",
        start=105609444,
        end=105609472,
        strand="+",
        sequence="AGCAGAGTGAGCATTCCAGAGTCCCAGA",
    ),
    RecommendedExample(
        name="HBA1",
        chrom="16",
        start=176699,
        end=176954,
        strand="+",
        sequence=(
            "CTCAGAGAGAACCCACCATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCCTG"
            "GGGTAAGGTCGGCGCGCACGCTGGCGAGTATGGTGCGGAGGCCCTGGAGAGGTGAGGCTCCC"
            "TCCCCTGCTCCGACCCGGGCTCCTCGCCCGCCCGGACCCACAGGCCACCCTCAACCGTCCTG"
            "GCCCCGGACCCAAACCCCACCCCTCACTCTGCTTCTCCCCGCAGGATGTTCCTGTCCTTCCCC"
            "ACCACCA"
        ),
    ),
    RecommendedExample(
        name="tRNA-Arg-TCT-4-1",
        chrom="1",
        start=159141610,
        end=159141684,
        strand="-",
        sequence=(
            "GTCTCTGTGGCGCAATGGACGAGCGCGCTGGACTTCTAATCCAGAGGTTCCGGGTTCGAGTCCCGGCAGAGATG"
        ),
    ),
)

EXAMPLES_BY_LABEL = {example.label: example for example in EXAMPLES}
DEFAULT_EXAMPLE = EXAMPLES[0]
