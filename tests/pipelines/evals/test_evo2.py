"""Tests for Evo2 variant scoring helpers.

The real inference path needs ``evo2`` + an H100/GH200 + ~40 GB of downloaded
weights, so this file only exercises the lightweight contract (signature,
import, aggregator math). The heavy lifting is checked at runtime on the
SkyPilot cluster.
"""

import numpy as np

from marin_dna.pipelines.evals.evo2 import aggregate_ll_gap, compute_evo2_ll


def test_evo2_helpers_importable():
    """Sanity-check that the public LL-gap functions exist and are importable
    without evo2 installed.

    Variant scoring lives in ``scripts/evo2_eval/_evo2_scoring.py``
    (intentionally not testable from here without an evo2 install)."""
    assert callable(compute_evo2_ll)
    assert callable(aggregate_ll_gap)


def test_aggregate_ll_gap_token_weighted_mean_and_sign():
    """Per-row sums and counts must aggregate to a token-weighted dataset
    mean (sum-then-divide), and the gap convention must be
    ``LL_upper - LL_lower``: positive when uppercase log-probs are
    closer to 0 (easier) than lowercase log-probs.
    """
    # Two rows, one all-upper and one all-lower — the edge case that
    # motivates returning sums+counts (per-row means would NaN here).
    # All-upper row: 4 target tokens, total log p = -2.0
    # All-lower row: 6 target tokens, total log p = -9.0
    pred = np.array(
        [
            [-2.0, 0.0, 4.0, 0.0],
            [0.0, -9.0, 0.0, 6.0],
        ],
        dtype=np.float32,
    )
    out = aggregate_ll_gap(pred)

    # token-weighted mean: -11.0 / 10 = -1.1
    assert out["LL_all"] == -1.1
    # mean LL on uppercase: -2 / 4 = -0.5
    assert out["LL_upper"] == -0.5
    # mean LL on lowercase: -9 / 6 = -1.5
    assert out["LL_lower"] == -1.5
    # gap = LL_upper - LL_lower = -0.5 - (-1.5) = +1.0
    # (positive ⇒ uppercase is easier to predict, the expected sign on CDS)
    assert out["gap"] == 1.0
    assert out["n_upper"] == 4
    assert out["n_lower"] == 6


def test_aggregate_ll_gap_uses_fp64():
    """Aggregator must cast to fp64 before the cross-row sum (PR #18 of
    biofoundation). We verify by feeding a fp32 input large enough that
    a naive fp32 ``sum`` would drop the bottom bits, then checking the
    result has fp64 precision relative to a fully-fp64 reference.
    """
    rng = np.random.default_rng(0)
    n_rows = 50_000
    upper_logp = rng.uniform(-3.0, -0.1, size=n_rows).astype(np.float32)
    lower_logp = rng.uniform(-3.0, -0.1, size=n_rows).astype(np.float32)
    pred = np.stack(
        [
            upper_logp,
            lower_logp,
            np.ones(n_rows, dtype=np.float32),
            np.ones(n_rows, dtype=np.float32),
        ],
        axis=1,
    )

    out = aggregate_ll_gap(pred)
    # All four returned floats are Python floats (fp64).
    for k in ("LL_all", "LL_upper", "LL_lower", "gap"):
        assert isinstance(out[k], float)
    # Reference: same arithmetic but fully fp64 from the start. Any miss
    # in the cast will show up as a >fp32-eps difference here on a 50k-row
    # sample where the per-token mean is ~-1.5.
    ref_upper = upper_logp.astype(np.float64).sum() / n_rows
    ref_lower = lower_logp.astype(np.float64).sum() / n_rows
    assert abs(out["LL_upper"] - ref_upper) < 1e-12
    assert abs(out["LL_lower"] - ref_lower) < 1e-12


def test_aggregate_ll_gap_rejects_bad_shape():
    import pytest

    with pytest.raises(AssertionError):
        aggregate_ll_gap(np.zeros((5, 3)))  # not [N, 4]


def test_aggregate_ll_gap_rejects_zero_count_buckets():
    import pytest

    # all upper, no lower at all
    pred = np.array([[-1.0, 0.0, 5.0, 0.0]], dtype=np.float32)
    with pytest.raises(AssertionError, match="non-functional"):
        aggregate_ll_gap(pred)

    # all lower, no upper at all
    pred = np.array([[0.0, -1.0, 0.0, 5.0]], dtype=np.float32)
    with pytest.raises(AssertionError, match="functional"):
        aggregate_ll_gap(pred)


# --------------------------------------------------------------------------- #
# scripts/evo2_eval/compute_auprc_metrics.py contract tests
#
# Light-touch: verify the COL_RENAME / SCORE_COLUMNS coupling and a single
# end-to-end pass on a synthetic 50-row predictions parquet, so a future
# typo in either constant (or a schema drift in `compute_auprc_metrics`)
# trips a fast unit test instead of a silent dashboard regression.
# --------------------------------------------------------------------------- #


def _load_driver():
    """Load the script as a module by path (it's not a package)."""
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts" / "evo2_eval" / "compute_auprc_metrics.py"
    spec = importlib.util.spec_from_file_location("compute_auprc_metrics", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_score_columns_subset_of_col_rename_values():
    """SCORE_COLUMNS must reference columns that actually emerge from the
    rename — otherwise the driver would assert empty score_cols at runtime
    on every input.
    """
    mod = _load_driver()
    rename_targets = set(mod.COL_RENAME.values())
    missing = [c for c in mod.SCORE_COLUMNS if c not in rename_targets]
    assert not missing, (
        f"SCORE_COLUMNS contains entries not produced by COL_RENAME: {missing}; "
        f"available targets: {sorted(rename_targets)}"
    )


def test_driver_end_to_end_on_synthetic_predictions(tmp_path):
    """Round-trip: write a tiny synthetic predictions parquet in the evo2
    naming, invoke the driver, read back the metrics parquet, check schema
    and a couple of value invariants (AUPRC in [0,1], 6 score_types × 11
    subset rows for 9 subsets + global + macro).
    """
    import subprocess
    import sys

    import pandas as pd

    mod = _load_driver()

    # Synthetic predictions in evo2 naming. 30 match_groups across 2
    # subsets so both subsets clear n_min=30 separately? No — n_min=30 is
    # per-subset, so we use 30 groups in ONE subset (qualifying), and 5
    # in a smaller subset (not qualifying for macro). _global_ row still
    # contributes. Each group: 1 positive + 9 negatives = 10 rows.
    rng = np.random.default_rng(42)
    rows = []
    for gid in range(30):
        score_signal = rng.normal(loc=2.0)  # positives skew high
        rows.append(
            {
                "chrom": "1",
                "pos": gid,
                "ref": "A",
                "alt": "C",
                "label": 1,
                "subset": "big",
                "match_group": gid,
                "minus_llr": score_signal,
                "minus_llr_fwd": score_signal,
                "minus_llr_rev": score_signal,
                "next_token_jsd_mean": abs(score_signal) * 0.01,
                "next_token_jsd_mean_fwd": abs(score_signal) * 0.01,
                "next_token_jsd_mean_rev": abs(score_signal) * 0.01,
            }
        )
        for _ in range(9):
            noise = rng.normal(loc=0.0)
            rows.append(
                {
                    "chrom": "1",
                    "pos": gid,
                    "ref": "A",
                    "alt": "G",
                    "label": 0,
                    "subset": "big",
                    "match_group": gid,
                    "minus_llr": noise,
                    "minus_llr_fwd": noise,
                    "minus_llr_rev": noise,
                    "next_token_jsd_mean": abs(noise) * 0.01,
                    "next_token_jsd_mean_fwd": abs(noise) * 0.01,
                    "next_token_jsd_mean_rev": abs(noise) * 0.01,
                }
            )
    df = pd.DataFrame(rows)

    in_path = tmp_path / "synth_train.parquet"
    out_path = tmp_path / "synth_metrics.parquet"
    df.to_parquet(in_path, index=False)

    # Invoke the script as a subprocess so the argparse-driven `main()`
    # path is exercised end-to-end (the way Snakemake / a user would
    # invoke it). Small n_bootstrap so the test runs in <1s.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib.util,sys; "
            f"spec=importlib.util.spec_from_file_location('m', "
            f"'{mod.__file__}'); m=importlib.util.module_from_spec(spec); "
            f"spec.loader.exec_module(m); sys.argv=['m', '--input', "
            f"'{in_path}', '--output', '{out_path}', '--model', 'evo2_test', "
            f"'--n-bootstrap', '50']; m.main()",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"driver failed: {result.stderr}"

    metrics = pd.read_parquet(out_path)
    expected_cols = {
        "score_type",
        "subset",
        "value",
        "se",
        "n_groups",
        "n_rows",
        "model",
        "dataset",
        "split",
    }
    assert set(metrics.columns) == expected_cols, (
        f"metrics columns: {sorted(metrics.columns)}"
    )

    score_types = set(metrics["score_type"].unique())
    assert score_types == set(mod.SCORE_COLUMNS), (
        f"missing score_types: {set(mod.SCORE_COLUMNS) - score_types}"
    )

    # 1 subset + _global_ + _macro_avg_ = 3 rows per score_type
    assert (metrics["score_type"] == mod.SCORE_COLUMNS[0]).sum() == 3

    # AUPRC bounded [0, 1]; SE non-negative.
    assert (metrics["value"] >= 0).all() and (metrics["value"] <= 1).all()
    assert (metrics["se"] >= 0).all()

    # Stamped metadata round-trips.
    assert (metrics["model"] == "evo2_test").all()
    assert (metrics["dataset"] == "mendelian_traits").all()  # default
    assert (metrics["split"] == "train").all()


# --------------------------------------------------------------------------- #
# scripts/evo2_eval/_evo2_scoring.py — embedding-bundle kernel (issue #131)
#
# The full inference path needs evo2 + a GPU, but the kernel takes the model as
# an ARGUMENT, so the pooling math + the f16 FWD/RC aggregator are testable with a
# fake Evo2 stub on CPU (mirrors the gLM #325 kernel tests). The real forward is
# checked at runtime on the GH200 smoke.
# --------------------------------------------------------------------------- #


def _load_scoring():
    """Load the script-local Evo2 scoring module by path (not a package)."""
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts" / "evo2_eval" / "_evo2_scoring.py"
    spec = importlib.util.spec_from_file_location("_evo2_scoring", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeEvo2:
    """Minimal Evo2-shaped stub: ``model(ids[, return_embeddings, layer_names])``
    returns ``(outputs, emb)`` where ``outputs[0]`` is logits and ``emb`` is
    ``{layer: hidden}`` (empty unless embeddings requested).

    Both logits and hidden are deterministic table look-ups of the token ids, so
    the test can recompute the expected entire-window mean-pool independently and
    confirm the logits (hence scores) don't depend on the embeddings flag.
    """

    def __init__(self, logit_table, emb_table):
        self._logit_table = logit_table  # [V, V]
        self._emb_table = emb_table  # [V, D]

    def __call__(self, input_ids, return_embeddings=False, layer_names=None):
        logits = self._logit_table[input_ids]  # [2B, L, V]
        emb = {}
        if return_embeddings:
            assert layer_names is not None and len(layer_names) == 1
            emb[layer_names[0]] = self._emb_table[input_ids]  # [2B, L, D]
        return (logits,), emb


def _fake_kernel_inputs(seed=0):
    import torch

    rng = np.random.default_rng(seed)
    V, D, B, L = 8, 5, 3, 8  # vocab, hidden, batch, window (var_pos = L//2 = 4)
    nuc_token_ids = torch.arange(4)  # A,C,G,T -> 0,1,2,3
    input_ids = torch.from_numpy(rng.integers(0, 4, size=(B, L))).long()
    alt_token_id = torch.from_numpy(rng.integers(0, 4, size=B)).long()
    logit_table = torch.from_numpy(rng.standard_normal((V, V))).float()
    emb_table = torch.from_numpy(rng.standard_normal((V, D))).float()
    model = _FakeEvo2(logit_table, emb_table)
    return model, input_ids, alt_token_id, nuc_token_ids, B, D, L


def test_evo2_kernel_embeddings_entire_window_pool_and_layout():
    """``return_embeddings`` returns ``[B, 2 + 2D]``: cols [0:2] = scores
    (unchanged vs the no-embedding call), [2:2+D] = ref entire-window mean-pool,
    [2+D:2+2D] = alt entire-window mean-pool.
    """
    import torch

    mod = _load_scoring()
    model, input_ids, alt_token_id, nuc, B, D, L = _fake_kernel_inputs()
    var_pos = L // 2

    scores = mod._compute_evo2_kernel(
        model, input_ids, alt_token_id, var_pos=var_pos, nuc_token_ids=nuc
    )
    bundle = mod._compute_evo2_kernel(
        model,
        input_ids,
        alt_token_id,
        var_pos=var_pos,
        nuc_token_ids=nuc,
        return_embeddings=True,
        emb_layer="norm",
    )
    assert scores.shape == (B, 2)
    assert bundle.shape == (B, 2 + 2 * D)
    # Scores must be byte-identical whether or not embeddings ride along.
    assert torch.allclose(bundle[:, :2], scores, atol=0, rtol=0)

    # Independently recompute the entire-window mean-pool. The kernel forms
    # combined = cat([ref, alt]) where alt = ref with var_pos swapped to alt_token.
    alt_seq = input_ids.clone()
    alt_seq[:, var_pos] = alt_token_id
    emb_table = model._emb_table
    emb_ref_expected = emb_table[input_ids].float().mean(dim=1)  # [B, D]
    emb_alt_expected = emb_table[alt_seq].float().mean(dim=1)  # [B, D]
    assert torch.allclose(bundle[:, 2 : 2 + D], emb_ref_expected, atol=1e-5)
    assert torch.allclose(bundle[:, 2 + D :], emb_alt_expected, atol=1e-5)


def test_evo2_kernel_ref_alt_pools_differ_only_downstream():
    """ref and alt windows are identical except at/after var_pos, so the pooled
    delta is non-zero (the variant changes >=1 position) — guards against a
    stub/pool bug that would collapse emb_ref == emb_alt.
    """
    import torch

    mod = _load_scoring()
    model, input_ids, alt_token_id, nuc, B, D, L = _fake_kernel_inputs(seed=1)
    var_pos = L // 2
    # Force a real allele change at var_pos for every row (alt != ref).
    alt_token_id = (input_ids[:, var_pos] + 1) % 4
    bundle = mod._compute_evo2_kernel(
        model,
        input_ids,
        alt_token_id,
        var_pos=var_pos,
        nuc_token_ids=nuc,
        return_embeddings=True,
        emb_layer="norm",
    )
    emb_ref = bundle[:, 2 : 2 + D]
    emb_alt = bundle[:, 2 + D :]
    assert not torch.allclose(emb_ref, emb_alt), "ref/alt pools should differ"


def test_fwd_rc_average_f16_fp32_accumulation_and_f16_cast():
    """The aggregator averages strands in fp32 and casts to f16 only at the end."""
    mod = _load_scoring()
    rng = np.random.default_rng(3)
    fwd = rng.standard_normal((20, 7)).astype(np.float32)
    rev = rng.standard_normal((20, 7)).astype(np.float32)
    out = mod.fwd_rc_average_f16([fwd, rev])
    assert out.dtype == np.float16
    # Reference: fp32 mean then cast — must match bit-for-bit.
    ref = ((fwd + rev) / 2).astype(np.float16)
    assert np.array_equal(out, ref)
    # Single strand → identity (just the f16 cast).
    one = mod.fwd_rc_average_f16([fwd])
    assert np.array_equal(one, fwd.astype(np.float16))


def test_fwd_rc_average_f16_float32_out_dtype_is_lossless():
    """out_dtype='float32' (the #131 massive-activation escape hatch) returns the
    exact fp32 FWD+RC mean — no rounding."""
    mod = _load_scoring()
    rng = np.random.default_rng(4)
    fwd = rng.standard_normal((12, 5)).astype(np.float32)
    rev = rng.standard_normal((12, 5)).astype(np.float32)
    out = mod.fwd_rc_average_f16([fwd, rev], out_dtype=np.float32)
    assert out.dtype == np.float32
    assert np.array_equal(out, ((fwd + rev) / 2).astype(np.float32))
    # A value that overflows f16 is fine in f32 (no overflow assert tripped).
    big = np.full((2, 3), 1e6, dtype=np.float32)
    out_big = mod.fwd_rc_average_f16([big, big], out_dtype=np.float32)
    assert np.isfinite(out_big).all() and out_big.dtype == np.float32


def test_fwd_rc_average_f16_rejects_overflow():
    import pytest

    mod = _load_scoring()
    # A channel beyond float16's +-65504 overflows to inf on the cast → assert.
    big = np.full((2, 3), 1e6, dtype=np.float32)
    with pytest.raises(AssertionError, match="non-finite"):
        mod.fwd_rc_average_f16([big, big])
