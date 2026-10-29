import numpy as np

from exp586_moe.inference.correctness import acgt_parity


def fixture() -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(586)
    sequences = ["".join(rng.choice(list("ACGT"), size=8192)) for _ in range(2)]
    logits = rng.normal(size=(2, 8192, 4)).astype(np.float32)
    return logits, sequences


def test_exact_acgt_parity_is_accepted() -> None:
    logits, sequences = fixture()
    report = acgt_parity(logits, logits.copy(), sequences)
    assert report["accepted"] is True
    assert report["probability_max_abs_error"] == 0


def test_material_probability_change_is_rejected() -> None:
    expected, sequences = fixture()
    actual = expected.copy()
    actual[..., 0] += 1
    report = acgt_parity(actual, expected, sequences)
    assert report["accepted"] is False
