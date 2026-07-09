"""Tests for the #369 LoRA fine-tuning harness — folds, window construction, feature parity.

These cover the correctness-critical pure logic (no model / GPU needed): the leak-proof
chromosome folds, the variant-position bookkeeping in window building (an off-by-one here
would silently score the wrong site), and that the model's ``concat_ref_delta`` feature is
identical to the frozen probe's — the apples-to-apples invariant.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from marin_dna.data.dna import complement_base, reverse_complement
from marin_dna.data.transforms import (
    _get_nucleotide_token_ids,
    _get_special_token_counts,
    in_seq_var_pos,
)
from marin_dna.pipelines.evals.variant_probe import pair_feature
from marin_dna.pipelines.finetune.data import (
    build_variant_windows,
    chrom_fold_masks,
    nested_loco_folds,
)


class FakeTokenizer:
    """Minimal BOS-prepending nucleotide tokenizer (A/C/G/T/N -> 1..5, BOS=0)."""

    bos_token_id = 0
    eos_token_id = None
    _map = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}

    def encode(self, seq: str) -> list[int]:
        return [self.bos_token_id] + [self._map[c] for c in seq]


class FakeGenome:
    def __init__(self, seqs: dict[str, str]) -> None:
        self.seqs = seqs

    def __call__(self, chrom: str, start: int, end: int, strand: str = "+") -> str:
        s = self.seqs[chrom][start:end]
        assert len(s) == end - start, "test genome has no edge padding; keep in-bounds"
        return reverse_complement(s) if strand == "-" else s


# --------------------------------------------------------------------------- folds
def test_chrom_fold_masks_partition_and_disjoint():
    chrom = np.array(["1", "1", "3", "3", "5", "5"])
    train, val, test = chrom_fold_masks(chrom, "1", "3")
    assert test.tolist() == [True, True, False, False, False, False]
    assert val.tolist() == [False, False, True, True, False, False]
    assert train.tolist() == [False, False, False, False, True, True]
    # every variant in exactly one partition
    assert np.all(train.astype(int) + val.astype(int) + test.astype(int) == 1)


def test_chrom_fold_masks_rejects_bad_splits():
    chrom = np.array(["1", "3", "5"])
    with pytest.raises(AssertionError):
        chrom_fold_masks(chrom, "1", "1")  # test == val
    with pytest.raises(AssertionError):
        chrom_fold_masks(chrom, "9", "3")  # test chrom absent -> empty test


def test_nested_loco_covers_all_with_distinct_val():
    chrom = np.array(["1", "3", "5", "1", "3", "5"])
    folds = list(nested_loco_folds(chrom))
    assert [f[0] for f in folds] == ["1", "3", "5"]  # every chrom is test once
    for test_c, val_c, (tr, va, te) in folds:
        assert test_c != val_c
        assert set(chrom[tr]) and not (set(chrom[tr]) & {test_c, val_c})


# ------------------------------------------------------------------ window building
def _one_variant_df(chrom: str, pos: int, ref: str, alt: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
          "label": 1, "subset": "missense_variant", "match_group": "g0"}]
    )


def test_build_windows_variant_position_and_alleles():
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), size=400))
    genome = FakeGenome({"1": seq})
    tok = FakeTokenizer()
    window = 7
    pos1 = 50  # 1-based; ref = seq[49]
    ref = seq[pos1 - 1]
    alt = "A" if ref != "A" else "C"
    vw = build_variant_windows(_one_variant_df("1", pos1, ref, alt), tok, genome, window)

    n_prefix, _ = _get_special_token_counts(tok)
    nuc = _get_nucleotide_token_ids(tok)
    var_pos = n_prefix + in_seq_var_pos(window, "+")
    assert vw.pool_lo == n_prefix and vw.pool_hi == n_prefix + window
    assert vw.ref_fwd.shape == (1, window + n_prefix)
    # exactly one changed token, at the variant position, ref->alt
    assert int((vw.alt_fwd[0] != vw.ref_fwd[0]).sum()) == 1
    assert vw.ref_fwd[0, var_pos].item() == nuc[ref]
    assert vw.alt_fwd[0, var_pos].item() == nuc[alt]
    # RC strand: the variant token is the complement of ref/alt
    assert vw.ref_rc[0, var_pos].item() == nuc[complement_base(ref)]
    assert vw.alt_rc[0, var_pos].item() == nuc[complement_base(alt)]
    assert int((vw.alt_rc[0] != vw.ref_rc[0]).sum()) == 1


# ----------------------------------------------------------------- feature parity
def test_concat_ref_delta_matches_probe():
    """The model's feature must equal the frozen probe's ``concat_ref_delta`` exactly."""
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((5, 8)).astype(np.float32)
    alt = rng.standard_normal((5, 8)).astype(np.float32)
    expected = pair_feature(ref, alt, "concat_ref_delta")  # probe's numpy feature
    pr, pa = torch.tensor(ref), torch.tensor(alt)
    got = torch.cat([pr, pa - pr], dim=-1).numpy()  # the model's _head input
    assert got.shape == (5, 16)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
