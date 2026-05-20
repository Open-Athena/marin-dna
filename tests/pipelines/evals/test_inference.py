"""Tests for ``compute_variant_scores`` — focused on the per-strand column
contract under ``rc=False`` and ``rc=True``. Model + tokenizer + genome
loading are mocked so the test runs on CPU in milliseconds.

End-to-end inference smoke tests (real model + real genome) live in
``tests/model/test_scoring.py``."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from marin_dna.pipelines.evals.inference import compute_variant_scores


def _stub_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chrom": ["chr1"] * 4,
            "pos": [100, 200, 300, 400],
            "ref": ["A", "C", "G", "T"],
            "alt": ["T", "G", "A", "C"],
            "label": [1, 0, 1, 0],
        }
    )


def _patched_model_load():
    """Patch the heavy model/tokenizer/genome loaders so the test never
    actually downloads a checkpoint or opens a FASTA."""
    return (
        patch(
            "marin_dna.pipelines.evals.inference.AutoTokenizer.from_pretrained",
            return_value=object(),
        ),
        patch(
            "marin_dna.pipelines.evals.inference.AutoModelForCausalLM.from_pretrained",
            return_value=object(),
        ),
        patch(
            "marin_dna.pipelines.evals.inference.Genome",
            return_value=object(),
        ),
    )


def test_compute_variant_scores_rc_false_returns_two_cols():
    ds = _stub_dataset()
    fwd_arr = np.array([[0.1, 0.01], [0.2, 0.02], [0.3, 0.03], [0.4, 0.04]])

    tok_patch, model_patch, genome_patch = _patched_model_load()
    with (
        tok_patch,
        model_patch,
        genome_patch,
        patch(
            "marin_dna.pipelines.evals.inference.run_variant_score_bundle",
            return_value={"fwd": fwd_arr},
        ),
    ):
        scores = compute_variant_scores(
            checkpoint_path="/unused",
            dataset=ds,
            genome_path="/unused.fa",
            rc=False,
        )

    assert list(scores.columns) == ["llr_fwd", "jsd_fwd"]
    np.testing.assert_array_equal(scores["llr_fwd"].values, fwd_arr[:, 0])
    np.testing.assert_array_equal(scores["jsd_fwd"].values, fwd_arr[:, 1])
    assert len(scores) == len(ds)


def test_compute_variant_scores_rc_true_returns_four_cols():
    ds = _stub_dataset()
    fwd_arr = np.array([[0.1, 0.01], [0.2, 0.02], [0.3, 0.03], [0.4, 0.04]])
    rc_arr = np.array([[-0.1, 0.05], [-0.2, 0.06], [-0.3, 0.07], [-0.4, 0.08]])

    tok_patch, model_patch, genome_patch = _patched_model_load()
    with (
        tok_patch,
        model_patch,
        genome_patch,
        patch(
            "marin_dna.pipelines.evals.inference.run_variant_score_bundle",
            return_value={"fwd": fwd_arr, "rc": rc_arr},
        ),
    ):
        scores = compute_variant_scores(
            checkpoint_path="/unused",
            dataset=ds,
            genome_path="/unused.fa",
            rc=True,
        )

    assert set(scores.columns) == {"llr_fwd", "llr_rc", "jsd_fwd", "jsd_rc"}
    np.testing.assert_array_equal(scores["llr_fwd"].values, fwd_arr[:, 0])
    np.testing.assert_array_equal(scores["jsd_fwd"].values, fwd_arr[:, 1])
    np.testing.assert_array_equal(scores["llr_rc"].values, rc_arr[:, 0])
    np.testing.assert_array_equal(scores["jsd_rc"].values, rc_arr[:, 1])
    assert len(scores) == len(ds)


def test_compute_variant_scores_avg_derivable_from_atoms():
    """The metrics rule materializes _avg = (fwd+rc)/2 downstream; sanity
    check that the atoms emitted by compute_variant_scores are sufficient
    to recover the previous AVG behavior."""
    ds = _stub_dataset()
    fwd_arr = np.array([[1.0, 0.5], [2.0, 0.6], [3.0, 0.7], [4.0, 0.8]])
    rc_arr = np.array([[-1.0, 0.1], [0.0, 0.2], [1.0, 0.3], [2.0, 0.4]])

    tok_patch, model_patch, genome_patch = _patched_model_load()
    with (
        tok_patch,
        model_patch,
        genome_patch,
        patch(
            "marin_dna.pipelines.evals.inference.run_variant_score_bundle",
            return_value={"fwd": fwd_arr, "rc": rc_arr},
        ),
    ):
        scores = compute_variant_scores(
            checkpoint_path="/unused",
            dataset=ds,
            genome_path="/unused.fa",
            rc=True,
        )

    expected_llr_avg = (fwd_arr[:, 0] + rc_arr[:, 0]) / 2
    expected_jsd_avg = (fwd_arr[:, 1] + rc_arr[:, 1]) / 2
    np.testing.assert_allclose(
        (scores["llr_fwd"] + scores["llr_rc"]) / 2, expected_llr_avg
    )
    np.testing.assert_allclose(
        (scores["jsd_fwd"] + scores["jsd_rc"]) / 2, expected_jsd_avg
    )
