"""Tests for ``compute_variant_scores`` — focused on the per-strand column
contract under ``rc=False`` and ``rc=True``. Model + tokenizer + genome
loading are mocked so the test runs on CPU in milliseconds.

End-to-end inference smoke tests (real model + real genome) live in
``tests/model/test_scoring.py``."""

from __future__ import annotations

from inspect import signature
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from marin_dna_evals.inference import (
    compute_variant_scores,
    fwd_rc_average_f16,
)


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
            "marin_dna_evals.inference.AutoTokenizer.from_pretrained",
            return_value=object(),
        ),
        patch(
            "marin_dna_evals.inference.AutoModelForCausalLM.from_pretrained",
            return_value=object(),
        ),
        patch(
            "marin_dna_evals.inference.Genome",
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
            "marin_dna_evals.inference.run_variant_score_bundle",
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
            "marin_dna_evals.inference.run_variant_score_bundle",
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


def test_compute_variant_scores_threads_execution_settings():
    ds = _stub_dataset()
    fwd = np.zeros((len(ds), 2), dtype=np.float32)
    rc = np.zeros((len(ds), 2), dtype=np.float32)
    tok_patch, model_patch, genome_patch = _patched_model_load()
    with (
        tok_patch,
        model_patch,
        genome_patch,
        patch(
            "marin_dna_evals.inference.run_variant_score_bundle",
            return_value={"fwd": fwd, "rc": rc},
        ) as runner,
    ):
        compute_variant_scores(
            checkpoint_path="/unused",
            dataset=ds,
            genome_path="/unused.fa",
            batch_size=7,
            num_workers=2,
            torch_compile=True,
            bf16=False,
            rc=True,
            eval_accumulation_steps=3,
        )

    kwargs = runner.call_args.kwargs["inference_kwargs"]
    assert kwargs["per_device_eval_batch_size"] == 7
    assert kwargs["dataloader_num_workers"] == 2
    assert kwargs["torch_compile"] is True
    assert kwargs["bf16_full_eval"] is False
    assert kwargs["eval_accumulation_steps"] == 3


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
            "marin_dna_evals.inference.run_variant_score_bundle",
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


# --- fwd_rc_average_f16 + return_embeddings (issue #318) ---------------------


def test_fwd_rc_average_f16_is_fp32_average_then_cast():
    rng = np.random.default_rng(1)
    fwd = rng.standard_normal((5, 4)).astype(np.float32)
    rc = rng.standard_normal((5, 4)).astype(np.float32)
    out = fwd_rc_average_f16([fwd, rc])
    assert out.dtype == np.float16
    # fp32 average, then a single cast to f16.
    np.testing.assert_array_equal(out, ((fwd + rc) / 2).astype(np.float16))


def test_fwd_rc_average_f16_accumulates_in_fp32_not_f16():
    """Summing many large terms overflows an f16 accumulator but not fp32 — pins
    the fp32-accumulation contract (production passes ≤2 strands, but the property
    is what protects the pooled means from rounding)."""
    n = 256
    strands = [np.full((1, 3), 300.0, dtype=np.float32) for _ in range(n)]
    out = fwd_rc_average_f16(strands)
    np.testing.assert_allclose(out, np.full((1, 3), 300.0, dtype=np.float16))
    acc16 = np.zeros((1, 3), dtype=np.float16)
    with np.errstate(over="ignore"):  # the overflow is the point of the test
        for s in strands:
            acc16 = (acc16 + s.astype(np.float16)).astype(np.float16)
    assert not np.isfinite(acc16).all()  # f16 running sum overflows to inf


def _patched_model_load_with_hidden(hidden_size: int):
    """Like ``_patched_model_load`` but the model exposes ``config.hidden_size``
    (the driver reads it to slice the embedding block)."""
    return (
        patch(
            "marin_dna_evals.inference.AutoTokenizer.from_pretrained",
            return_value=object(),
        ),
        patch(
            "marin_dna_evals.inference.AutoModelForCausalLM.from_pretrained",
            return_value=SimpleNamespace(
                config=SimpleNamespace(hidden_size=hidden_size)
            ),
        ),
        patch(
            "marin_dna_evals.inference.Genome",
            return_value=object(),
        ),
    )


def test_compute_variant_scores_embeddings_columns_and_fp32_average():
    ds = _stub_dataset()
    n, d = len(ds), 3
    rng = np.random.default_rng(0)
    fwd = np.zeros((n, 2 + 2 * d), dtype=np.float32)
    rc = np.zeros((n, 2 + 2 * d), dtype=np.float32)
    fwd[:, 0], fwd[:, 1] = [0.1, 0.2, 0.3, 0.4], [0.01, 0.02, 0.03, 0.04]
    rc[:, 0], rc[:, 1] = [-0.1, -0.2, -0.3, -0.4], [0.05, 0.06, 0.07, 0.08]
    fwd[:, 2:] = rng.standard_normal((n, 2 * d)).astype(np.float32)
    rc[:, 2:] = rng.standard_normal((n, 2 * d)).astype(np.float32)

    tok_patch, model_patch, genome_patch = _patched_model_load_with_hidden(d)
    with (
        tok_patch,
        model_patch,
        genome_patch,
        patch(
            "marin_dna_evals.inference.run_variant_score_bundle",
            return_value={"fwd": fwd, "rc": rc},
        ),
    ):
        scores = compute_variant_scores(
            checkpoint_path="/unused",
            dataset=ds,
            genome_path="/unused.fa",
            rc=True,
            return_embeddings=True,
        )

    assert set(scores.columns) == {
        "llr_fwd",
        "llr_rc",
        "jsd_fwd",
        "jsd_rc",
        "emb_ref",
        "emb_alt",
    }
    # Scalar score atoms unchanged by the embedding path.
    np.testing.assert_array_equal(scores["llr_fwd"].values, fwd[:, 0])
    np.testing.assert_array_equal(scores["jsd_rc"].values, rc[:, 1])
    # emb_ref/emb_alt: per-row length-D f16 = fp32 FWD+RC average, then cast.
    got_ref = np.stack(scores["emb_ref"].to_numpy())
    got_alt = np.stack(scores["emb_alt"].to_numpy())
    assert got_ref.dtype == np.float16 and got_ref.shape == (n, d)
    exp_ref = ((fwd[:, 2 : 2 + d] + rc[:, 2 : 2 + d]) / 2).astype(np.float16)
    exp_alt = ((fwd[:, 2 + d :] + rc[:, 2 + d :]) / 2).astype(np.float16)
    np.testing.assert_array_equal(got_ref, exp_ref)
    np.testing.assert_array_equal(got_alt, exp_alt)


def test_compute_variant_scores_embeddings_parquet_roundtrip(tmp_path):
    """The emb columns survive the production storage path (the rule concats the
    variant frame + scores and writes parquet) as length-D float16 lists."""
    ds = _stub_dataset()
    n, d = len(ds), 4
    rng = np.random.default_rng(2)
    fwd = np.zeros((n, 2 + 2 * d), dtype=np.float32)
    rc = np.zeros((n, 2 + 2 * d), dtype=np.float32)
    fwd[:, 2:] = rng.standard_normal((n, 2 * d)).astype(np.float32)
    rc[:, 2:] = rng.standard_normal((n, 2 * d)).astype(np.float32)

    tok_patch, model_patch, genome_patch = _patched_model_load_with_hidden(d)
    with (
        tok_patch,
        model_patch,
        genome_patch,
        patch(
            "marin_dna_evals.inference.run_variant_score_bundle",
            return_value={"fwd": fwd, "rc": rc},
        ),
    ):
        scores = compute_variant_scores(
            checkpoint_path="/unused",
            dataset=ds,
            genome_path="/unused.fa",
            rc=True,
            return_embeddings=True,
        )

    out = pd.concat([ds.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    path = tmp_path / "scores.parquet"
    out.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    got = np.stack(back["emb_ref"].to_numpy())
    assert got.dtype == np.float16 and got.shape == (n, d)
    np.testing.assert_array_equal(got, np.stack(scores["emb_ref"].to_numpy()))


def test_compute_variant_scores_return_embeddings_requires_rc():
    ds = _stub_dataset()
    tok_patch, model_patch, genome_patch = _patched_model_load_with_hidden(3)
    with (
        tok_patch,
        model_patch,
        genome_patch,
        pytest.raises(AssertionError, match="requires rc=True"),
    ):
        compute_variant_scores(
            checkpoint_path="/unused",
            dataset=ds,
            genome_path="/unused.fa",
            rc=False,
            return_embeddings=True,
        )


def test_compute_variant_scores_embeddings_width_mismatch_asserts():
    """A bundle width that isn't ``2 + 2*hidden_size`` is a loud error (guards
    against a silent mis-slice if the kernel/runner ever drift)."""
    ds = _stub_dataset()
    n, d = len(ds), 3
    # Model says hidden_size=d but the array is 2 + 2*(d+1) wide.
    bad = np.zeros((n, 2 + 2 * (d + 1)), dtype=np.float32)
    tok_patch, model_patch, genome_patch = _patched_model_load_with_hidden(d)
    with (
        tok_patch,
        model_patch,
        genome_patch,
        patch(
            "marin_dna_evals.inference.run_variant_score_bundle",
            return_value={"fwd": bad, "rc": bad},
        ),
        pytest.raises(AssertionError, match="hidden_size"),
    ):
        compute_variant_scores(
            checkpoint_path="/unused",
            dataset=ds,
            genome_path="/unused.fa",
            rc=True,
            return_embeddings=True,
        )


def test_compute_variant_scores_preserves_legacy_positional_order():
    parameters = list(signature(compute_variant_scores).parameters)
    assert parameters == [
        "checkpoint_path",
        "dataset",
        "genome_path",
        "context_size",
        "batch_size",
        "num_workers",
        "data_transform_on_the_fly",
        "torch_compile",
        "rc",
        "return_embeddings",
        "eval_accumulation_steps",
        "bf16",
    ]
