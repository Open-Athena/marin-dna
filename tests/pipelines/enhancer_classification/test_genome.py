"""Tests for the Genome class with FASTA and 2bit backends."""

from pathlib import Path

import pytest
from marin_dna.data.genome import Genome as BFGenome

from marin_dna.pipelines.enhancer_classification.genome import Genome

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"
FASTA_PATH = FIXTURES / "mini_genome.fa"
TWOBIT_PATH = FIXTURES / "mini_genome.2bit"

CHROMS = {"1": 52, "2": 52}

CHROM1_SEQ = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
CHROM2_SEQ = "TTTTTAAAAACCCCCGGGGGACGTACGTACGTACGTACGTACGTACGTACGT"


@pytest.fixture(params=["fasta", "2bit"], ids=["fasta", "2bit"])
def genome(request: pytest.FixtureRequest) -> Genome:
    if request.param == "fasta":
        return Genome(FASTA_PATH)
    return Genome(TWOBIT_PATH)


class TestGenomeBasic:
    def test_chroms(self, genome: Genome) -> None:
        assert genome.chroms == CHROMS

    def test_full_chromosome(self, genome: Genome) -> None:
        assert genome("1", 0, 52) == CHROM1_SEQ
        assert genome("2", 0, 52) == CHROM2_SEQ

    def test_subsequence(self, genome: Genome) -> None:
        assert genome("1", 0, 10) == "ACGTACGTAC"
        assert genome("2", 5, 15) == "AAAAACCCCC"

    def test_reverse_complement(self, genome: Genome) -> None:
        assert genome("1", 0, 4, strand="-") == "ACGT"
        assert genome("2", 0, 5, strand="-") == "AAAAA"

    def test_left_padding(self, genome: Genome) -> None:
        seq = genome("1", -3, 5)
        assert seq == "NNNACGTA"
        assert len(seq) == 8

    def test_right_padding(self, genome: Genome) -> None:
        seq = genome("1", 48, 56)
        assert seq == "ACGTNNNN"
        assert len(seq) == 8

    def test_both_padding(self, genome: Genome) -> None:
        seq = genome("1", -2, 54)
        assert seq.startswith("NN")
        assert seq.endswith("NN")
        assert len(seq) == 56

    def test_invalid_chrom(self, genome: Genome) -> None:
        with pytest.raises(ValueError, match="not found"):
            genome("nonexistent", 0, 10)

    def test_invalid_strand(self, genome: Genome) -> None:
        with pytest.raises(ValueError, match="strand"):
            genome("1", 0, 10, strand="*")  # type: ignore[arg-type]

    def test_start_greater_than_end(self, genome: Genome) -> None:
        with pytest.raises(ValueError, match="start"):
            genome("1", 10, 5)

    def test_start_beyond_chrom_size(self, genome: Genome) -> None:
        # chrom 1 has size 52
        with pytest.raises(ValueError, match="out of range"):
            genome("1", 100, 110)

    def test_negative_end(self, genome: Genome) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            genome("1", -10, -5)

    def test_subset_chroms(self) -> None:
        g = Genome(FASTA_PATH, subset_chroms={"1"})
        assert "1" in g.chroms
        assert "2" not in g.chroms


class TestBackendEquivalence:
    """Verify FASTA and 2bit backends produce identical results."""

    @pytest.fixture
    def fasta_genome(self) -> Genome:
        return Genome(FASTA_PATH)

    @pytest.fixture
    def twobit_genome(self) -> Genome:
        return Genome(TWOBIT_PATH)

    @pytest.mark.parametrize(
        "chrom,start,end",
        [
            ("1", 0, 10),
            ("1", 5, 15),
            ("1", 0, 52),
            ("1", 25, 40),
            ("2", 0, 10),
            ("2", 5, 15),
            ("2", 0, 52),
            ("2", 20, 35),
        ],
    )
    def test_forward_strand(
        self,
        fasta_genome: Genome,
        twobit_genome: Genome,
        chrom: str,
        start: int,
        end: int,
    ) -> None:
        assert fasta_genome(chrom, start, end) == twobit_genome(chrom, start, end)

    @pytest.mark.parametrize("chrom", ["1", "2"])
    def test_reverse_strand(
        self, fasta_genome: Genome, twobit_genome: Genome, chrom: str
    ) -> None:
        assert fasta_genome(chrom, 0, 20, strand="-") == twobit_genome(
            chrom, 0, 20, strand="-"
        )

    def test_left_padding(self, fasta_genome: Genome, twobit_genome: Genome) -> None:
        assert fasta_genome("1", -5, 10) == twobit_genome("1", -5, 10)

    def test_right_padding(self, fasta_genome: Genome, twobit_genome: Genome) -> None:
        assert fasta_genome("1", 45, 60) == twobit_genome("1", 45, 60)


class TestBiofoundationEquivalence:
    """Verify our Genome matches biofoundation.data.Genome for FASTA."""

    @pytest.fixture
    def our_genome(self) -> Genome:
        return Genome(FASTA_PATH)

    @pytest.fixture
    def bf_genome(self) -> BFGenome:
        return BFGenome(FASTA_PATH)

    @pytest.mark.parametrize(
        "chrom,start,end,strand",
        [
            ("1", 0, 10, "+"),
            ("1", 5, 15, "+"),
            ("1", 0, 52, "+"),
            ("1", 0, 10, "-"),
            ("2", 0, 10, "+"),
            ("2", 0, 52, "+"),
            ("2", 0, 10, "-"),
            ("1", -5, 10, "+"),
            ("1", 45, 60, "+"),
        ],
    )
    def test_matches_biofoundation(
        self,
        our_genome: Genome,
        bf_genome: BFGenome,
        chrom: str,
        start: int,
        end: int,
        strand: str,
    ) -> None:
        assert our_genome(chrom, start, end, strand) == bf_genome(
            chrom, start, end, strand
        )
