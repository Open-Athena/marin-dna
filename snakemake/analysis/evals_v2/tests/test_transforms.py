"""Tests for ``marin_dna_evals.transforms`` CLM-side helpers.

Vendored from biofoundation/tests/test_data.py at commit 834dd4c (May 2026):
only the CLM transform tests are migrated (MLM transforms were dropped from
``marin_dna_evals.transforms``). The ``_SpecialTokensTokenizer`` and
``_StubCharTokenizer`` helpers are plain classes here — the ``Tokenizer``
abstract base no longer exists; transforms duck-type their tokenizer arg.
"""

import pytest
import torch
from Bio.Seq import Seq
from transformers import AutoTokenizer

from marin_dna.data.dna import complement_base
from marin_dna.data.genome import Genome
from marin_dna_evals.transforms import (
    NUCLEOTIDES,
    _get_nucleotide_token_ids,
    _get_special_token_counts,
    transform_ll_clm,
    transform_llr_clm,
    transform_reflogprob_clm,
    transform_variant_marginal_clm,
)


class _SpecialTokensTokenizer:
    """Wrap an HF tokenizer to optionally prepend BOS / append EOS.

    Uses synthetic IDs that don't collide with real DNA tokens. After the
    biofoundation -> marin-dna migration there is no ``Tokenizer`` abstract
    base; transforms accept any duck-typed tokenizer.
    """

    def __init__(self, base, *, bos_id=None, eos_id=None):
        self._base = base
        self._bos = bos_id
        self._eos = eos_id

    def encode(self, text):
        ids = list(self._base.encode(text))
        if self._bos is not None:
            ids = [self._bos] + ids
        if self._eos is not None:
            ids = ids + [self._eos]
        return ids

    @property
    def mask_token_id(self):
        return self._base.mask_token_id

    @property
    def bos_token_id(self) -> int:
        if self._bos is None:
            raise AttributeError("no BOS configured")
        return self._bos

    @property
    def eos_token_id(self) -> int:
        if self._eos is None:
            raise AttributeError("no EOS configured")
        return self._eos


_BOS_ID = 100
_EOS_ID = 101


def _write_test_fasta(tmp_path):
    fasta = ">chr1\nACGTACGTAC\n"
    path = tmp_path / "llr_test.fa"
    path.write_text(fasta)
    return path


def _write_long_test_fasta(tmp_path):
    """400-bp FASTA — long enough to extract any reasonable window without N-padding."""
    fasta = ">chr1\n" + ("ACGT" * 100) + "\n"
    path = tmp_path / "long.fa"
    path.write_text(fasta)
    return path


# --- transform_llr_clm tests ------------------------------------------------


def test_transform_llr_clm_basic_functionality(tmp_path):
    """Test basic functionality of transform_llr_clm"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))
    window_size = 16
    example = {"chrom": "chr1", "pos": 6, "ref": "C", "alt": "A"}

    result = transform_llr_clm(example, tokenizer, genome, window_size)

    # Check return structure: ref-only input_ids [L] + alt_token_id (int).
    assert isinstance(result, dict)
    assert "input_ids" in result
    assert "alt_token_id" in result
    assert isinstance(result["input_ids"], torch.Tensor)
    assert result["input_ids"].shape == (window_size,)


def test_transform_llr_clm_records_ref_and_alt(tmp_path):
    """Ref-suffix carries the ref nucleotide at the variant position; the
    alt nucleotide travels separately as a scalar token ID (the kernel
    reconstructs the alt suffix on the fly via a single-token swap)."""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))
    window_size = 16
    example = {"chrom": "chr1", "pos": 6, "ref": "C", "alt": "G"}

    result = transform_llr_clm(example, tokenizer, genome, window_size)
    input_ids = result["input_ids"]

    # In-sequence variant position is window_size // 2 (no BOS for songlab).
    var_pos = window_size // 2
    ref_token_id = tokenizer.encode(example["ref"])[0]
    alt_token_id = tokenizer.encode(example["alt"])[0]

    assert input_ids[var_pos].item() == ref_token_id
    assert result["alt_token_id"] == alt_token_id


def test_transform_llr_clm_token_ids_are_valid(tmp_path):
    """All ref-sequence token IDs are non-negative."""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))
    window_size = 16
    example = {"chrom": "chr1", "pos": 6, "ref": "C", "alt": "T"}

    result = transform_llr_clm(example, tokenizer, genome, window_size)
    assert (result["input_ids"] >= 0).all()


def test_transform_llr_clm_different_window_sizes(tmp_path):
    """Shape scales with window_size; ref nucleotide at var_pos always matches."""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))

    for window_size in [16, 18, 20]:
        example = {"chrom": "chr1", "pos": 6, "ref": "C", "alt": "T"}
        result = transform_llr_clm(example, tokenizer, genome, window_size)
        assert result["input_ids"].shape == (window_size,)
        var_pos = window_size // 2
        ref_token_id = tokenizer.encode(example["ref"])[0]
        assert result["input_ids"][var_pos].item() == ref_token_id


@pytest.mark.parametrize(
    "bos_id,eos_id,counts",
    [
        (None, None, (0, 0)),
        (_BOS_ID, None, (1, 0)),
        (None, _EOS_ID, (0, 1)),
        (_BOS_ID, _EOS_ID, (1, 1)),
    ],
)
def test_get_special_token_counts(bos_id, eos_id, counts):
    base = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    tokenizer = _SpecialTokensTokenizer(base, bos_id=bos_id, eos_id=eos_id)
    assert _get_special_token_counts(tokenizer) == counts


@pytest.mark.parametrize(
    "tokenizer_name,counts",
    [
        ("songlab/tokenizer-dna-mlm", (0, 0)),
        ("songlab/tokenizer-dna-clm", (0, 0)),
        ("bolinas-dna/tokenizer-char-bos", (1, 0)),
        ("bolinas-dna/tokenizer-char-bos-eos", (1, 1)),
    ],
)
def test_get_special_token_counts_real_tokenizers(tokenizer_name, counts):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    assert _get_special_token_counts(tokenizer) == counts


def test_transform_llr_clm_exp136_recipe(tmp_path):
    """Regression for issue #19: window_size=255 + marin_dna BOS tokenizer.

    With BOS (n_prefix=1), the token-level var_pos is window_size // 2 + 1 = 128."""
    tokenizer = AutoTokenizer.from_pretrained("bolinas-dna/tokenizer-char-bos")
    fasta_path = tmp_path / "g.fa"
    fasta_path.write_text(">chr1\n" + ("ACGT" * 100) + "\n")
    genome = Genome(fasta_path)
    example = {"chrom": "chr1", "pos": 200, "ref": "T", "alt": "G"}

    result = transform_llr_clm(example, tokenizer, genome, window_size=255)

    assert result["input_ids"].shape == (256,)  # 255 bp + 1 BOS
    var_pos_tokens = 255 // 2 + 1  # 128 (in-seq pos + n_prefix)
    nuc_ids = _get_nucleotide_token_ids(tokenizer)
    assert result["input_ids"][var_pos_tokens].item() == nuc_ids["T"]
    assert result["alt_token_id"] == nuc_ids["G"]


def test_transform_llr_clm_window_size_one(tmp_path):
    """Smallest window: just the variant base itself."""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))
    example = {"chrom": "chr1", "pos": 6, "ref": "C", "alt": "A"}

    result = transform_llr_clm(example, tokenizer, genome, window_size=1)

    assert result["input_ids"].shape == (1,)
    assert result["input_ids"][0].item() == tokenizer.encode("C")[0]
    assert result["alt_token_id"] == tokenizer.encode("A")[0]


@pytest.mark.parametrize(
    "bos_id,eos_id",
    [(None, None), (_BOS_ID, None), (None, _EOS_ID), (_BOS_ID, _EOS_ID)],
)
@pytest.mark.parametrize("window_size", [15, 16])
def test_transform_llr_clm_handles_bos_eos(tmp_path, bos_id, eos_id, window_size):
    base = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    tokenizer = _SpecialTokensTokenizer(base, bos_id=bos_id, eos_id=eos_id)
    genome = Genome(_write_test_fasta(tmp_path))
    example = {"chrom": "chr1", "pos": 6, "ref": "C", "alt": "A"}

    result = transform_llr_clm(example, tokenizer, genome, window_size)

    has_bos = bos_id is not None
    has_eos = eos_id is not None
    expected_len = window_size + int(has_bos) + int(has_eos)
    assert result["input_ids"].shape == (expected_len,)
    var_pos_tokens = window_size // 2 + int(has_bos)
    assert result["input_ids"][var_pos_tokens].item() == base.encode("C")[0]
    assert result["alt_token_id"] == base.encode("A")[0]


@pytest.mark.parametrize(
    "bos_id,eos_id",
    [(None, None), (_BOS_ID, None), (None, _EOS_ID), (_BOS_ID, _EOS_ID)],
)
def test_transform_reflogprob_clm_handles_bos_eos(bos_id, eos_id):
    base = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    tokenizer = _SpecialTokensTokenizer(base, bos_id=bos_id, eos_id=eos_id)
    pos = 3
    example = {"seq": "ATCGATCG", "pos": pos}

    result = transform_reflogprob_clm(example, tokenizer)

    has_bos = bos_id is not None
    has_eos = eos_id is not None
    expected_len = len(example["seq"]) + int(has_bos) + int(has_eos)
    input_ids = result["input_ids"]
    assert input_ids.shape == (4, expected_len)

    tokenized_pos = pos + int(has_bos)
    nucleotides = ["A", "C", "G", "T"]
    for i, nuc in enumerate(nucleotides):
        assert input_ids[i, tokenized_pos].item() == base.encode(nuc)[0]
        for j in range(expected_len):
            if j == tokenized_pos:
                continue
            assert input_ids[i, j].item() == input_ids[0, j].item()
    assert nucleotides[result["ref"]] == example["seq"][pos]


def test_complement_base():
    assert complement_base("A") == "T"
    assert complement_base("C") == "G"
    assert complement_base("G") == "C"
    assert complement_base("T") == "A"
    # Non-ACGT round-trips unchanged
    assert complement_base("N") == "N"
    assert complement_base("M") == "M"
    assert complement_base("R") == "R"


@pytest.mark.parametrize(
    "bos_id,eos_id",
    [(None, None), (_BOS_ID, None), (None, _EOS_ID), (_BOS_ID, _EOS_ID)],
)
@pytest.mark.parametrize("window_size", [5, 6, 15, 16])
def test_transform_llr_clm_strand_rc_matrix(tmp_path, bos_id, eos_id, window_size):
    """Full {even/odd window_size} × {BOS y/n} × {EOS y/n} coverage for the
    RC-strand path of transform_llr_clm: shape, variant index, complemented
    tokens, and that the ref sequence is the revcomp of the FWD ref sequence."""
    base = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    tokenizer = _SpecialTokensTokenizer(base, bos_id=bos_id, eos_id=eos_id)
    genome = Genome(_write_long_test_fasta(tmp_path))
    example = {"chrom": "chr1", "pos": 200, "ref": "T", "alt": "G"}

    fwd = transform_llr_clm(example, tokenizer, genome, window_size, strand="+")
    rc = transform_llr_clm(example, tokenizer, genome, window_size, strand="-")

    has_bos = bos_id is not None
    has_eos = eos_id is not None
    expected_len = window_size + int(has_bos) + int(has_eos)

    assert rc["input_ids"].shape == (expected_len,)

    rc_dna_pos = (
        window_size - 1 - window_size // 2
    )  # asymmetric when window_size is even
    rc_token_idx = rc_dna_pos + int(has_bos)

    nuc_ids = {n: base.encode(n)[0] for n in "ACGT"}
    # Ref nuc on RC strand is complement(T) = A; alt nuc is complement(G) = C.
    assert rc["input_ids"][rc_token_idx].item() == nuc_ids["A"]
    assert rc["alt_token_id"] == nuc_ids["C"]

    # Body of the rc ref sequence equals revcomp of the body of fwd ref sequence.
    body_slice = slice(int(has_bos), expected_len - int(has_eos))
    id_to_nuc = {v: k for k, v in nuc_ids.items()}
    fwd_body_dna = "".join(id_to_nuc[t.item()] for t in fwd["input_ids"][body_slice])
    rc_body_dna = "".join(id_to_nuc[t.item()] for t in rc["input_ids"][body_slice])
    assert str(Seq(fwd_body_dna).reverse_complement()) == rc_body_dna


@pytest.mark.parametrize("window_size", [1, 5, 15, 255])
def test_transform_llr_clm_odd_window_strand_symmetric(tmp_path, window_size):
    """For odd window_size, the variant DNA index is identical on both strands."""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_long_test_fasta(tmp_path))
    example = {"chrom": "chr1", "pos": 200, "ref": "T", "alt": "G"}

    fwd = transform_llr_clm(example, tokenizer, genome, window_size, strand="+")
    rc = transform_llr_clm(example, tokenizer, genome, window_size, strand="-")

    var_pos = window_size // 2  # equal on both strands for odd window_size
    nuc_ids = {n: tokenizer.encode(n)[0] for n in "ACGT"}
    # FWD: ref="T" at var_pos
    assert fwd["input_ids"][var_pos].item() == nuc_ids["T"]
    # RC: complement(ref) = "A" at var_pos
    assert rc["input_ids"][var_pos].item() == nuc_ids["A"]


def test_transform_llr_clm_strand_rc_n_padding(tmp_path):
    """Variant near chrom start — FWD window N-pads on the left, RC on the right."""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))  # 10-bp chrom: ACGTACGTAC
    window_size = 8
    example = {"chrom": "chr1", "pos": 2, "ref": "C", "alt": "T"}

    fwd = transform_llr_clm(example, tokenizer, genome, window_size, strand="+")
    rc = transform_llr_clm(example, tokenizer, genome, window_size, strand="-")

    assert fwd["input_ids"].shape == (window_size,)
    assert rc["input_ids"].shape == (window_size,)

    base = tokenizer
    nuc_ids = {n: base.encode(n)[0] for n in "ACGTN"}
    fwd_var_pos = window_size // 2
    rc_var_pos = window_size - 1 - window_size // 2
    assert fwd["input_ids"][fwd_var_pos].item() == nuc_ids["C"]
    assert rc["input_ids"][rc_var_pos].item() == nuc_ids["G"]  # complement("C")
    assert fwd["alt_token_id"] == nuc_ids["T"]
    assert rc["alt_token_id"] == nuc_ids["A"]  # complement("T")

    # The N-padded body on FWD reverse-complements to the N-padded body on RC
    # (N maps to N under reverse_complement). Sanity: at least one N appears
    # on each strand (chrom boundary).
    id_to_nuc = {v: k for k, v in nuc_ids.items()}
    fwd_dna = "".join(id_to_nuc.get(t.item(), "?") for t in fwd["input_ids"])
    rc_dna = "".join(id_to_nuc.get(t.item(), "?") for t in rc["input_ids"])
    assert "N" in fwd_dna
    assert "N" in rc_dna


# --- transform_reflogprob_clm tests ----------------------------------------


@pytest.mark.parametrize(
    "bos_id,eos_id",
    [(None, None), (_BOS_ID, None), (None, _EOS_ID), (_BOS_ID, _EOS_ID)],
)
@pytest.mark.parametrize("seq", ["ATCGATCG", "ACGTAC"])
def test_transform_reflogprob_clm_strand_rc_matrix(bos_id, eos_id, seq):
    """Full matrix for the RC-strand path of transform_reflogprob_clm."""
    base = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    tokenizer = _SpecialTokensTokenizer(base, bos_id=bos_id, eos_id=eos_id)
    pos = 2
    example = {"seq": seq, "pos": pos}

    rc = transform_reflogprob_clm(example, tokenizer, strand="-")

    has_bos = bos_id is not None
    has_eos = eos_id is not None
    expected_len = len(seq) + int(has_bos) + int(has_eos)
    rc_pos_dna = len(seq) - 1 - pos
    rc_token_idx = rc_pos_dna + int(has_bos)

    assert rc["input_ids"].shape == (4, expected_len)
    # Each of the 4 sequences has the corresponding nucleotide at the RC index
    for i, nuc in enumerate(NUCLEOTIDES):
        assert rc["input_ids"][i, rc_token_idx].item() == base.encode(nuc)[0]
    # ref is the index of the complement of the original ref base
    assert NUCLEOTIDES[rc["ref"]] == complement_base(seq[pos])


def test_transform_reflogprob_clm_basic_functionality():
    """Test basic functionality of transform_reflogprob_clm"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    pos = 2
    example = {"seq": "ATCGATCG", "pos": pos}

    result = transform_reflogprob_clm(example, tokenizer)

    # Check return structure
    assert isinstance(result, dict)
    assert "input_ids" in result
    assert "ref" in result

    # Check types
    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["ref"], int)

    # Check shape: should be [4, L] for four nucleotide variants
    assert result["input_ids"].shape[0] == 4
    assert result["input_ids"].shape[1] == len(example["seq"])


def test_transform_reflogprob_clm_creates_four_sequences():
    """Test that transform_reflogprob_clm creates 4 sequences (one per nucleotide)"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    example = {"seq": "AAACCCGGG", "pos": 4}

    result = transform_reflogprob_clm(example, tokenizer)

    input_ids = result["input_ids"]

    # Should have exactly 4 sequences (A, C, G, T)
    assert input_ids.shape[0] == 4

    # All sequences should have the same length
    for i in range(1, 4):
        assert input_ids[i].shape == input_ids[0].shape


def test_transform_reflogprob_clm_correct_nucleotides_at_position():
    """Test that each sequence has the correct nucleotide at the specified position"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    pos = 3
    example = {"seq": "ATCGATCG", "pos": pos}

    result = transform_reflogprob_clm(example, tokenizer)

    input_ids = result["input_ids"]

    # Each of the 4 sequences should have a different nucleotide at pos
    nucleotides = ["A", "C", "G", "T"]
    for i, nuc in enumerate(nucleotides):
        # Get the token ID for this nucleotide
        expected_token_id = tokenizer.encode(nuc)[0]
        actual_token_id = input_ids[i, pos].item()
        assert actual_token_id == expected_token_id


def test_transform_reflogprob_clm_ref_index_mapping():
    """Test that ref index correctly maps to nucleotide (0=A, 1=C, 2=G, 3=T)"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")

    nucleotides = ["A", "C", "G", "T"]

    for expected_idx, nuc in enumerate(nucleotides):
        pos = 2
        example = {"seq": f"NN{nuc}NNNNN", "pos": pos}

        result = transform_reflogprob_clm(example, tokenizer)

        # The ref should be the index corresponding to the nucleotide
        assert result["ref"] == expected_idx


def test_transform_reflogprob_clm_different_nucleotides():
    """Test with each nucleotide (A, C, G, T) as the reference"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")

    test_cases = [
        ("AAATTTCCC", 3, "T"),  # T at position 3
        ("GCGCGCGC", 2, "G"),  # G at position 2
        ("TTCCGGAA", 1, "T"),  # T at position 1
        ("ACGTACGT", 4, "A"),  # A at position 4
    ]

    for seq, pos, expected_nuc in test_cases:
        example = {"seq": seq, "pos": pos}

        result = transform_reflogprob_clm(example, tokenizer)

        # Check that ref maps to the correct nucleotide
        nucleotides = ["A", "C", "G", "T"]
        assert nucleotides[result["ref"]] == expected_nuc


def test_transform_reflogprob_clm_different_positions():
    """Test with different positions in the sequence"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    seq = "ACGTACGT"

    for pos in range(len(seq)):
        example = {"seq": seq, "pos": pos}

        result = transform_reflogprob_clm(example, tokenizer)

        # Should always create 4 sequences
        assert result["input_ids"].shape[0] == 4

        # ref should be valid index (0-3)
        assert 0 <= result["ref"] < 4

        # The nucleotide at pos should match what's in the sequence
        nucleotides = ["A", "C", "G", "T"]
        assert nucleotides[result["ref"]] == seq[pos]


# --- transform_variant_marginal_clm tests ----------------------------------


def test_transform_variant_marginal_clm_basic(tmp_path):
    """Returns just ``{input_ids: [L]}`` — the window, no allele materialization."""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_long_test_fasta(tmp_path))
    window_size = 16
    example = {"chrom": "chr1", "pos": 200, "ref": "T"}

    result = transform_variant_marginal_clm(example, tokenizer, genome, window_size)

    assert set(result.keys()) == {"input_ids"}
    assert isinstance(result["input_ids"], torch.Tensor)
    assert result["input_ids"].shape == (window_size,)
    # Variant base sits at window_size // 2 on the forward strand (no BOS).
    assert result["input_ids"][window_size // 2].item() == tokenizer.encode("T")[0]


@pytest.mark.parametrize("window_size", [15, 16])
def test_transform_variant_marginal_clm_strand_rc(tmp_path, window_size):
    """RC window is the reverse complement; the variant base is complemented."""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_long_test_fasta(tmp_path))
    example = {"chrom": "chr1", "pos": 200, "ref": "T"}

    fwd = transform_variant_marginal_clm(
        example, tokenizer, genome, window_size, strand="+"
    )
    rc = transform_variant_marginal_clm(
        example, tokenizer, genome, window_size, strand="-"
    )

    nuc_ids = {n: tokenizer.encode(n)[0] for n in "ACGT"}
    fwd_var_pos = window_size // 2
    rc_var_pos = window_size - 1 - window_size // 2
    assert fwd["input_ids"][fwd_var_pos].item() == nuc_ids["T"]
    assert rc["input_ids"][rc_var_pos].item() == nuc_ids["A"]  # complement(T)
    # The whole window reverse-complements between strands.
    id_to_nuc = {v: k for k, v in nuc_ids.items()}
    fwd_dna = "".join(id_to_nuc[t.item()] for t in fwd["input_ids"])
    rc_dna = "".join(id_to_nuc[t.item()] for t in rc["input_ids"])
    assert str(Seq(fwd_dna).reverse_complement()) == rc_dna


def test_transform_variant_marginal_clm_asserts_ref_matches_genome(tmp_path):
    """A ref disagreeing with the genome base trips _get_variant_window's assertion."""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_long_test_fasta(tmp_path))
    example = {"chrom": "chr1", "pos": 200, "ref": "A"}  # genome base at pos 200 is T

    with pytest.raises(AssertionError):
        transform_variant_marginal_clm(example, tokenizer, genome, 16)


def test_transform_variant_marginal_clm_n_padding(tmp_path):
    """A window near the chromosome start N-pads to full width."""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))  # 10-bp chrom: ACGTACGTAC
    window_size = 8
    example = {"chrom": "chr1", "pos": 2, "ref": "C"}  # pos 2 (1-based) → "C"

    result = transform_variant_marginal_clm(example, tokenizer, genome, window_size)

    assert result["input_ids"].shape == (window_size,)
    nuc_ids = {n: tokenizer.encode(n)[0] for n in "ACGTN"}
    id_to_nuc = {v: k for k, v in nuc_ids.items()}
    dna = "".join(id_to_nuc.get(t.item(), "?") for t in result["input_ids"])
    assert "N" in dna  # left flank padded past the chrom start


# --- transform_ll_clm tests -------------------------------------------------


class _StubCharTokenizer:
    """Char-level tokenizer stub with configurable BOS/EOS for testing.

    Plain class — no abstract base; transforms duck-type their tokenizer arg.
    """

    def __init__(self, bos: int | None = None, eos: int | None = None):
        # Map A/C/G/T (case-insensitive) to ids 10..13; lowercase too.
        self._vocab = {"a": 10, "c": 11, "g": 12, "t": 13, "n": 14}
        self._bos = bos
        self._eos = eos

    def encode(self, text: str) -> list[int]:
        body = [self._vocab[c.lower()] for c in text]
        out = []
        if self._bos is not None:
            out.append(self._bos)
        out.extend(body)
        if self._eos is not None:
            out.append(self._eos)
        return out

    @property
    def bos_token_id(self) -> int:
        if self._bos is None:
            raise AttributeError("no BOS")
        return self._bos

    @property
    def eos_token_id(self) -> int:
        if self._eos is None:
            raise AttributeError("no EOS")
        return self._eos


def test_transform_ll_clm_no_specials():
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    seq = "ACgtAC"
    out = transform_ll_clm({"seq": seq}, tokenizer)

    assert out["input_ids"].dtype == torch.long
    assert out["is_upper"].dtype == torch.bool
    # tokenizer is char-level with no BOS/EOS, so length == len(seq)
    assert out["input_ids"].shape == (len(seq),)
    assert out["is_upper"].shape == (len(seq),)
    expected_upper = torch.tensor([True, True, False, False, True, True])
    assert torch.equal(out["is_upper"], expected_upper)


@pytest.mark.parametrize(
    "bos,eos",
    [(None, None), (101, None), (None, 102), (101, 102)],
)
def test_transform_ll_clm_special_tokens(bos, eos):
    tokenizer = _StubCharTokenizer(bos=bos, eos=eos)
    seq = "ACgt"
    out = transform_ll_clm({"seq": seq}, tokenizer)

    body_upper = [True, True, False, False]
    expected_upper = []
    if bos is not None:
        expected_upper.append(False)
    expected_upper.extend(body_upper)
    if eos is not None:
        expected_upper.append(False)
    assert torch.equal(out["is_upper"], torch.tensor(expected_upper))

    expected_ids = []
    if bos is not None:
        expected_ids.append(bos)
    expected_ids.extend([10, 11, 12, 13])  # A C g t
    if eos is not None:
        expected_ids.append(eos)
    assert out["input_ids"].tolist() == expected_ids


def test_transform_ll_clm_rejects_non_char_level():
    """A tokenizer that splits a single character into multiple tokens should fail."""

    class _BPELikeTokenizer:
        def encode(self, text: str) -> list[int]:
            # Pretend each character maps to two tokens — char-level assertion must fire.
            return [ord(c) for c in text for _ in range(2)]

    with pytest.raises(AssertionError, match="Char-level"):
        transform_ll_clm({"seq": "ACGT"}, _BPELikeTokenizer())


def test_transform_ll_clm_all_lower_and_all_upper():
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    out_upper = transform_ll_clm({"seq": "ACGTAC"}, tokenizer)
    out_lower = transform_ll_clm({"seq": "acgtac"}, tokenizer)
    assert out_upper["is_upper"].all()
    assert (~out_lower["is_upper"]).all()
    # Tokenizer is case-insensitive, so input_ids should match.
    assert torch.equal(out_upper["input_ids"], out_lower["input_ids"])


def test_transform_ll_clm_honors_disabled_auto_insertion():
    """A tokenizer with bos/eos defined but auto-insertion disabled (a
    common HF setup, e.g. GPT-2-style) must not gain extra special-token
    targets. We honor whatever ``add_special_tokens=True`` returns; if
    the tokenizer chose not to insert, neither do we."""

    class _NoAutoInsertTokenizer:
        # bos/eos IDs are defined, but encode never inserts them.
        def encode(self, text: str) -> list[int]:
            return [{"a": 10, "c": 11, "g": 12, "t": 13}[c.lower()] for c in text]

        @property
        def bos_token_id(self) -> int:
            return 99

        @property
        def eos_token_id(self) -> int:
            return 98

    out = transform_ll_clm({"seq": "ACgt"}, _NoAutoInsertTokenizer())
    # No specials in the encoding → no specials in input_ids; is_upper is
    # purely the per-char case. Crucially, n_total = len(seq), not len(seq)+2.
    assert out["input_ids"].tolist() == [10, 11, 12, 13]
    assert out["is_upper"].tolist() == [True, True, False, False]


def test_transform_ll_clm_byte_level_tokenizer_uppercases():
    """Case-sensitive byte-level tokenizer (e.g. Evo2's vortex
    CharLevelTokenizer) must still produce correct, identical input_ids
    for upper / lower / mixed sequences — transform_ll_clm uppercases
    before tokenizing so the model only ever sees uppercase bytes.
    """

    class _ByteLevelTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(text.encode("utf-8"))

    tokenizer = _ByteLevelTokenizer()
    seqs = ["ACGTAC", "acgtac", "AcGtAc"]
    outs = [transform_ll_clm({"seq": s}, tokenizer) for s in seqs]
    # All three sequences must produce identical input_ids — that's what
    # makes Evo2 happy. The is_upper masks differ as expected.
    for o in outs[1:]:
        assert torch.equal(o["input_ids"], outs[0]["input_ids"])
    # Sanity: input_ids correspond to ASCII codes for uppercase ACGTAC.
    assert outs[0]["input_ids"].tolist() == [65, 67, 71, 84, 65, 67]
    assert outs[0]["is_upper"].tolist() == [True, True, True, True, True, True]
    assert outs[1]["is_upper"].tolist() == [False, False, False, False, False, False]
    assert outs[2]["is_upper"].tolist() == [True, False, True, False, True, False]
