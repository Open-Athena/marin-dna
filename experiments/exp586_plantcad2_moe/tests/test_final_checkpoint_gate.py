import jax.numpy as jnp
import numpy as np
import pytest

from exp586_moe.final_checkpoint_gate import (
    admission,
    compare_readouts,
    windows_for_parity,
)
from exp586_moe.stable_scatter_reference import ordered_combine


def readouts() -> dict[str, np.ndarray]:
    return {
        "probabilities": np.full((8, 3, 4), 0.25),
        "full_logs": np.full((8, 31), -1.4),
        "acgt_logs": np.full((8, 31), -1.38),
        "mean": np.ones((8, 32)),
        "variant": np.ones((8, 32)),
    }


def test_parity_checks_suffix_likelihoods_and_features_as_well_as_probabilities():
    reference = readouts()
    assert compare_readouts(reference, readouts())["accepted"]
    for field in ("probabilities", "full_logs", "mean", "variant"):
        bad = readouts()
        bad[field] *= 2
        assert not compare_readouts(reference, bad)["accepted"]
    bad = readouts()
    bad["variant"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_readouts(reference, bad)


def test_parity_cases_include_reverse_complements_variants_and_ambiguous_bases():
    windows = windows_for_parity(["ACGT" * 2048] * 16)
    assert len(windows) == 56
    assert all(len(window) == 8192 for window in windows)
    assert windows[32][4096] != windows[0][4096]
    assert windows[-1] == "N" * 8192


def test_ordered_combine_restores_token_assignments_and_uses_fp32_sum():
    dispatched = jnp.asarray(
        [[256, 256], [1, 2], [-256, -256], [7, 8]], dtype=jnp.bfloat16
    )
    positions = jnp.asarray([[0, 1, 2], [3, 2, 0]])
    weights = jnp.asarray([[1, 1, 1], [0.5, 0, 0]], dtype=jnp.float32)
    combined = ordered_combine(dispatched, positions, weights)
    np.testing.assert_array_equal(
        np.asarray(combined, dtype=np.float32), [[1, 2], [3.5, 4]]
    )
    assert combined.dtype == jnp.bfloat16


def test_atomic_outliers_do_not_replace_exact_independent_reference_requirement():
    import copy

    exact = compare_readouts(readouts(), readouts())
    noisy = copy.deepcopy(exact)
    noisy.update(probability_max_abs=0.098, accepted=False)
    report = {
        "readout_comparisons": {"2048": exact, "8192": exact},
        "atomic_scatter_comparisons": {"2048": noisy, "8192": noisy},
        "backend_loss_abs": 0.00003,
    }
    assert admission(report)
    bad = copy.deepcopy(report)
    bad["readout_comparisons"]["8192"]["variant"]["max_abs"] = 0.5
    assert not admission(bad)
    bad = copy.deepcopy(report)
    bad["atomic_scatter_comparisons"]["8192"]["probability_p99_abs"] = 0.03
    assert not admission(bad)
    bad = copy.deepcopy(report)
    bad["backend_loss_abs"] = 0.0001
    assert not admission(bad)
