"""Tests for sequence materialization into eval harness format.

Each input variant produces two output rows: ``strand="+"`` (FWD) and
``strand="-"`` (RC of the same window). The tests pin down per-strand sequence
construction and the doubled row count, plus equivalence with
``marin_dna.data.transforms._get_variant_window`` (the offline-batched-VEP path's
golden reference).
"""

from pathlib import Path

import pytest
from datasets import Dataset

from marin_dna.data.dna import complement_base, reverse_complement
from marin_dna.data.genome import Genome
from marin_dna.data.transforms import _get_variant_window, in_seq_var_pos
from marin_dna.pipelines.evals.materialize import (
    _add_eval_harness_fields,
    materialize_sequences,
)

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"


@pytest.fixture
def mini_genome():
    """Load a small fake genome for deterministic tests.

    Chromosome "1": ACGTACGTACGTACGT... (repeating 4-mer pattern, 50bp)
    Chromosome "2": TTTTTAAAAACCCCCGGGGG + ACGTACGT... (50bp)
    """
    return Genome(FIXTURES_DIR / "mini_genome.fa")


# --------------------------------------------------------------------------- #
# FWD strand (existing semantics, with new ``strand`` parameter)              #
# --------------------------------------------------------------------------- #


def test_add_eval_harness_fields_fwd(mini_genome):
    """FWD strand: pinned-down sequences on a known SNV.

    Chrom "1" sequence: ACGTACGTACGTACGT...
    Variant at pos=11 (1-based) => 0-based index 10 => "G"
    With window_size=10: start=10-5=5, end=10+5=15
    context = genome("1", 5, 10) = "CGTAC"  (positions 5-9)
    right_flank = genome("1", 11, 15) = "TACG"  (positions 11-14)
    ref_completion = "G" + "TACG" = "GTACG"
    alt_completion = "A" + "TACG" = "ATACG"
    """
    example = {"chrom": "1", "pos": 11, "ref": "G", "alt": "A", "label": 1}
    result = _add_eval_harness_fields(
        example, genome=mini_genome, window_size=10, strand="+"
    )

    assert result["context"] == "CGTAC"
    assert result["ref_completion"] == "GTACG"
    assert result["alt_completion"] == "ATACG"
    assert result["strand"] == "+"


def test_window_centering_fwd(mini_genome):
    """FWD context and completion lengths sum to window_size."""
    example = {"chrom": "1", "pos": 25, "ref": "A", "alt": "T", "label": 0}
    result = _add_eval_harness_fields(
        example, genome=mini_genome, window_size=20, strand="+"
    )

    assert len(result["context"]) == 10  # window_size // 2
    assert len(result["ref_completion"]) == 10  # window_size // 2
    assert len(result["alt_completion"]) == 10


def test_odd_window_size_fwd(mini_genome):
    """FWD odd window sizes: extra base goes to the completion."""
    example = {"chrom": "1", "pos": 25, "ref": "A", "alt": "T", "label": 0}
    result = _add_eval_harness_fields(
        example, genome=mini_genome, window_size=11, strand="+"
    )

    assert len(result["context"]) == 5  # 11 // 2
    assert len(result["ref_completion"]) == 6  # 11 - 11 // 2
    assert len(result["context"]) + len(result["ref_completion"]) == 11

    full_ref_window = result["context"] + result["ref_completion"]
    expected = mini_genome("1", 19, 30).upper()
    assert full_ref_window == expected


def test_context_plus_ref_completion_matches_genome_fwd(mini_genome):
    """FWD context + ref_completion should reconstruct the reference window."""
    example = {"chrom": "1", "pos": 25, "ref": "A", "alt": "T", "label": 0}
    result = _add_eval_harness_fields(
        example, genome=mini_genome, window_size=20, strand="+"
    )

    full_ref_window = result["context"] + result["ref_completion"]
    expected = mini_genome("1", 14, 34).upper()
    assert full_ref_window == expected


# --------------------------------------------------------------------------- #
# RC strand                                                                   #
# --------------------------------------------------------------------------- #


def test_add_eval_harness_fields_rc(mini_genome):
    """RC strand: full window is RC of FWD; ref/alt are complemented.

    Same variant as test_add_eval_harness_fields_fwd: chrom 1, pos 11, ref G,
    alt A, window 10 (even). FWD window = genome[5:15] = "CGTACGTACG" (10 bp);
    RC window = reverse_complement("CGTACGTACG") = "CGTACGTACG" (palindrome of
    the repeating 4-mer pattern).

    Even window_size=10: FWD var_pos = 5; RC var_pos = 10 - 1 - 5 = 4.
    RC window[4] = complement_base("G") = "C". So:
      context        = RC[:4] = "CGTA"          (length 4)
      ref_completion = "C" + RC[5:] = "CGTACG"  (length 6)
      alt_completion = "T" + RC[5:] = "TGTACG"  (length 6)
    """
    example = {"chrom": "1", "pos": 11, "ref": "G", "alt": "A", "label": 1}
    result = _add_eval_harness_fields(
        example, genome=mini_genome, window_size=10, strand="-"
    )

    fwd_window = mini_genome("1", 5, 15).upper()
    rc_window = reverse_complement(fwd_window)
    rc_var_pos = in_seq_var_pos(10, "-")
    assert rc_window[rc_var_pos] == complement_base("G") == "C"

    assert result["context"] == rc_window[:rc_var_pos]
    assert result["ref_completion"] == "C" + rc_window[rc_var_pos + 1 :]
    assert result["alt_completion"] == "T" + rc_window[rc_var_pos + 1 :]
    assert result["strand"] == "-"


def test_rc_matches_get_variant_window(mini_genome):
    """RC sequences must match what marin_dna.data.transforms._get_variant_window
    produces — the golden reference used by the offline batched VEP path.
    """
    example = {"chrom": "1", "pos": 25, "ref": "A", "alt": "T"}
    for window_size in (10, 11, 20, 21):
        rc_seq, rc_var_pos = _get_variant_window(
            example, mini_genome, window_size, strand="-"
        )
        result = _add_eval_harness_fields(
            example, genome=mini_genome, window_size=window_size, strand="-"
        )
        assert result["context"] == rc_seq[:rc_var_pos], (
            f"window_size={window_size} context mismatch"
        )
        assert (
            result["ref_completion"] == complement_base("A") + rc_seq[rc_var_pos + 1 :]
        )
        assert (
            result["alt_completion"] == complement_base("T") + rc_seq[rc_var_pos + 1 :]
        )


def test_rc_assertion_fires_on_bad_ref(mini_genome):
    """The internal assertion catches a wrong ref allele on the RC strand."""
    # Real ref at pos=11 is "G"; pretending it's "T" should fail loud.
    example = {"chrom": "1", "pos": 11, "ref": "T", "alt": "A"}
    with pytest.raises(AssertionError):
        _add_eval_harness_fields(
            example, genome=mini_genome, window_size=10, strand="-"
        )


def test_rc_odd_window_symmetric(mini_genome):
    """For odd window_size, FWD and RC have identical context/completion lengths."""
    example = {"chrom": "1", "pos": 25, "ref": "A", "alt": "T"}
    fwd = _add_eval_harness_fields(
        example, genome=mini_genome, window_size=11, strand="+"
    )
    rc = _add_eval_harness_fields(
        example, genome=mini_genome, window_size=11, strand="-"
    )
    assert len(fwd["context"]) == len(rc["context"])
    assert len(fwd["ref_completion"]) == len(rc["ref_completion"])
    assert len(fwd["alt_completion"]) == len(rc["alt_completion"])


def test_rc_even_window_asymmetric(mini_genome):
    """For even window_size, RC context is 1 bp shorter, completion 1 bp longer."""
    example = {"chrom": "1", "pos": 25, "ref": "A", "alt": "T"}
    fwd = _add_eval_harness_fields(
        example, genome=mini_genome, window_size=10, strand="+"
    )
    rc = _add_eval_harness_fields(
        example, genome=mini_genome, window_size=10, strand="-"
    )
    assert len(rc["context"]) == len(fwd["context"]) - 1
    assert len(rc["ref_completion"]) == len(fwd["ref_completion"]) + 1
    assert len(rc["alt_completion"]) == len(fwd["alt_completion"]) + 1
    # Total reconstructed window is the same length on both strands.
    assert len(rc["context"]) + len(rc["ref_completion"]) == 10
    assert len(fwd["context"]) + len(fwd["ref_completion"]) == 10


# --------------------------------------------------------------------------- #
# materialize_sequences (end-to-end)                                          #
# --------------------------------------------------------------------------- #


def test_materialize_sequences_doubles_rows(mini_genome):
    """End-to-end: each input variant emits 2 rows (one per strand).

    Output is sorted by (chrom, pos, ref, alt, strand).
    """
    ds = Dataset.from_dict(
        {
            "chrom": ["1", "1"],
            "pos": [11, 15],
            "ref": ["G", "G"],
            "alt": ["A", "T"],
            "label": [1, 0],
        }
    )

    result = materialize_sequences(ds, mini_genome, window_size=10)

    assert "context" in result.column_names
    assert "ref_completion" in result.column_names
    assert "alt_completion" in result.column_names
    assert "strand" in result.column_names
    assert "target" in result.column_names
    assert "label" not in result.column_names
    assert len(result) == 2 * len(ds)
    assert sorted(set(result["strand"])) == ["+", "-"]
    by_variant: dict[tuple, list[str]] = {}
    for row in result:
        key = (row["chrom"], row["pos"], row["ref"], row["alt"])
        by_variant.setdefault(key, []).append(row["strand"])
    for key, strands in by_variant.items():
        assert sorted(strands) == ["+", "-"], f"variant {key} has strands {strands}"


def test_materialize_sequences_preserves_subset_and_match_group(mini_genome):
    """Subset and match_group must be preserved on both per-strand rows."""
    ds = Dataset.from_dict(
        {
            "chrom": ["1", "1"],
            "pos": [11, 15],
            "ref": ["G", "G"],
            "alt": ["A", "T"],
            "label": [1, 0],
            "subset": ["5UTR", "5UTR"],
            "match_group": [42, 42],
        }
    )

    result = materialize_sequences(ds, mini_genome, window_size=10)

    assert "subset" in result.column_names
    assert "match_group" in result.column_names
    assert sorted(set(result["subset"])) == ["5UTR"]
    assert sorted(set(result["match_group"])) == [42]
    # 2x rows per variant => 4 rows total, all in match_group=42 5UTR.
    assert len(result) == 4
