import numpy as np
import pytest

from exp586_moe.final_checkpoint_gate import compare_readouts, windows_for_parity


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
