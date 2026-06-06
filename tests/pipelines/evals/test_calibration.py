"""Tests for the cLLR stage-3 calibration-table helpers (#270).

The pure pieces (``expand_sites_to_variants``, ``aggregate_llr_neutral_mean``)
are tested directly; the orchestrator ``compute_llr_neutral_mean`` is tested with
``compute_variant_scores`` patched, so no model/genome/GPU is touched (mirrors
``tests/pipelines/evals/test_inference.py``)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from marin_dna.pipelines.evals.calibration import (
    aggregate_llr_neutral_mean,
    compute_llr_neutral_mean,
    expand_sites_to_variants,
)

EXPECTED_TABLE_COLUMNS = [
    "pentanuc_mut",
    "pentanuc",
    "ref",
    "alt",
    "n_sites",
    "llr_neutral_mean_fwd",
    "llr_neutral_mean_rc",
    "llr_neutral_mean_avg",
    "llr_neutral_std_avg",
    "subsample_n",
]


def _sites() -> pd.DataFrame:
    """Three neutral sites with distinct refs; pentanuc center == ref."""
    return pd.DataFrame(
        {
            "chrom": ["1", "1", "2"],
            "pos": [100, 200, 300],
            "ref": ["A", "C", "G"],
            "pentanuc": ["TTATT", "GGCGG", "AAGAA"],
        }
    )


# --- expand_sites_to_variants ----------------------------------------------


def test_expand_three_alts_per_site():
    variants = expand_sites_to_variants(_sites())
    assert len(variants) == 3 * 3
    assert list(variants.columns) == [
        "chrom",
        "pos",
        "ref",
        "alt",
        "pentanuc",
        "pentanuc_mut",
    ]
    # Exactly 3 rows per original site, never alt == ref.
    assert (variants["ref"] != variants["alt"]).all()
    assert (variants.groupby(["chrom", "pos"]).size() == 3).all()
    # pentanuc_mut formatting + the alts for the ref-A site.
    a_site = variants[(variants["chrom"] == "1") & (variants["pos"] == 100)]
    assert set(a_site["alt"]) == {"C", "G", "T"}
    assert set(a_site["pentanuc_mut"]) == {"TTATT_C", "TTATT_G", "TTATT_T"}


def test_expand_rejects_center_ref_mismatch():
    bad = _sites()
    bad.loc[0, "pentanuc"] = "TTCTT"  # center C != ref A
    with pytest.raises(AssertionError, match="center"):
        expand_sites_to_variants(bad)


def test_expand_rejects_non_acgt_pentanuc():
    bad = _sites()
    bad.loc[0, "pentanuc"] = "TTANT"  # N flank
    with pytest.raises(AssertionError, match="ACGT"):
        expand_sites_to_variants(bad)


def test_expand_rejects_missing_column():
    with pytest.raises(AssertionError, match="pentanuc"):
        expand_sites_to_variants(_sites().drop(columns=["pentanuc"]))


# --- aggregate_llr_neutral_mean --------------------------------------------


def _scored() -> pd.DataFrame:
    """Two cells: AAAAA_C (3 obs) and TTTTT_G (2 obs), with hand-checkable LLRs."""
    return pd.DataFrame(
        {
            "pentanuc_mut": ["AAAAA_C", "AAAAA_C", "AAAAA_C", "TTTTT_G", "TTTTT_G"],
            "pentanuc": ["AAAAA", "AAAAA", "AAAAA", "TTTTT", "TTTTT"],
            "ref": ["A", "A", "A", "T", "T"],
            "alt": ["C", "C", "C", "G", "G"],
            "llr_fwd": [1.0, 2.0, 3.0, -1.0, -3.0],
            "llr_rc": [1.0, 0.0, 1.0, 1.0, 1.0],
        }
    )


def test_aggregate_means_and_counts():
    table = aggregate_llr_neutral_mean(_scored(), min_bin_count=2, subsample_n=100)
    assert list(table.columns) == EXPECTED_TABLE_COLUMNS
    assert list(table["pentanuc_mut"]) == ["AAAAA_C", "TTTTT_G"]  # sorted

    cpg = table[table["pentanuc_mut"] == "AAAAA_C"].iloc[0]
    assert cpg["n_sites"] == 3
    assert cpg["llr_neutral_mean_fwd"] == pytest.approx(2.0)  # mean(1,2,3)
    assert cpg["llr_neutral_mean_rc"] == pytest.approx(2.0 / 3.0)  # mean(1,0,1)
    # avg = mean of per-variant (fwd+rc)/2 = mean(1,1,2) = 4/3
    assert cpg["llr_neutral_mean_avg"] == pytest.approx(4.0 / 3.0)
    # and equals (mean_fwd + mean_rc)/2
    assert cpg["llr_neutral_mean_avg"] == pytest.approx(
        (cpg["llr_neutral_mean_fwd"] + cpg["llr_neutral_mean_rc"]) / 2
    )
    assert cpg["llr_neutral_std_avg"] == pytest.approx(np.std([1, 1, 2], ddof=1))
    assert (table["subsample_n"] == 100).all()

    tg = table[table["pentanuc_mut"] == "TTTTT_G"].iloc[0]
    assert tg["n_sites"] == 2
    assert tg["llr_neutral_mean_avg"] == pytest.approx(-0.5)  # mean(0, -1)


def test_aggregate_min_bin_count_assertion():
    # TTTTT_G has only 2 obs; a floor of 3 must trip.
    with pytest.raises(AssertionError, match="min_bin_count"):
        aggregate_llr_neutral_mean(_scored(), min_bin_count=3, subsample_n=100)


def test_aggregate_nan_guard():
    bad = _scored()
    bad.loc[0, "llr_fwd"] = np.nan
    with pytest.raises(AssertionError, match="NaN"):
        aggregate_llr_neutral_mean(bad, min_bin_count=2, subsample_n=100)


# --- compute_llr_neutral_mean (orchestrator, mocked scoring) ---------------


def test_compute_llr_neutral_mean_end_to_end():
    # Two sites sharing one 5-mer → 3 cells (one per alt), each with 2 obs.
    sites = pd.DataFrame(
        {
            "chrom": ["1", "1"],
            "pos": [100, 200],
            "ref": ["A", "A"],
            "pentanuc": ["TTATT", "TTATT"],
        }
    )
    # 6 variants (2 sites × 3 alts); canned per-strand atoms (order matches the
    # alt-major concat in expand_sites_to_variants, but the test only checks
    # shape / columns / counts, not which value lands in which cell).
    canned = pd.DataFrame(
        {
            "llr_fwd": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "llr_rc": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "jsd_fwd": [0.0] * 6,
            "jsd_rc": [0.0] * 6,
        }
    )
    with patch(
        "marin_dna.pipelines.evals.calibration.compute_variant_scores",
        return_value=canned,
    ) as mock_score:
        table = compute_llr_neutral_mean(
            checkpoint_path="ckpt",
            sites=sites,
            genome_path="s3://genome.fa.gz",
            window_size=255,
            subsample_n=2,
            min_bin_count=2,
            rc=True,
        )
    mock_score.assert_called_once()
    assert mock_score.call_args.kwargs["rc"] is True
    assert mock_score.call_args.kwargs["context_size"] == 255
    assert list(table.columns) == EXPECTED_TABLE_COLUMNS
    assert len(table) == 3  # TTATT_{C,G,T}
    assert (table["n_sites"] == 2).all()
    assert set(table["pentanuc_mut"]) == {"TTATT_C", "TTATT_G", "TTATT_T"}


def test_compute_llr_neutral_mean_requires_rc():
    with pytest.raises(AssertionError, match="rc must be True"):
        compute_llr_neutral_mean(
            checkpoint_path="ckpt",
            sites=_sites(),
            genome_path="g",
            window_size=255,
            subsample_n=100,
            min_bin_count=1,
            rc=False,
        )
