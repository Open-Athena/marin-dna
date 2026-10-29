"""Task-relevant parity checks for dense A/C/G/T logits."""

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def acgt_parity(
    actual: np.ndarray, expected: np.ndarray, sequences: list[str]
) -> dict[str, float | bool | int]:
    """Compare backend outputs using the next-base workload they will serve."""
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape or actual.shape != (len(sequences), 8192, 4):
        raise ValueError("Parity arrays must be [sequence, 8192, A/C/G/T]")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Parity arrays must be finite")

    raw_error = np.abs(actual - expected)
    actual_probabilities = _softmax(actual)
    expected_probabilities = _softmax(expected)
    probability_error = np.abs(actual_probabilities - expected_probabilities)
    argmax_agreement = float(
        np.mean(
            actual_probabilities.argmax(axis=-1)
            == expected_probabilities.argmax(axis=-1)
        )
    )

    labels = np.asarray(
        [["ACGT".find(base) for base in sequence[1:]] for sequence in sequences],
        dtype=np.int32,
    )
    valid = labels >= 0
    safe_labels = np.maximum(labels, 0)

    def objective(probabilities: np.ndarray) -> tuple[float, float]:
        predictions = probabilities[:, :-1]
        chosen = np.take_along_axis(predictions, safe_labels[..., None], axis=-1)[
            ..., 0
        ]
        nll = float(-np.log(chosen[valid]).mean())
        accuracy = float((predictions.argmax(axis=-1)[valid] == labels[valid]).mean())
        return nll, accuracy

    actual_nll, actual_accuracy = objective(actual_probabilities)
    expected_nll, expected_accuracy = objective(expected_probabilities)
    report: dict[str, float | bool | int] = {
        "positions": int(np.prod(actual.shape[:-1])),
        "raw_max_abs_error": float(raw_error.max()),
        "raw_mean_abs_error": float(raw_error.mean()),
        "raw_p99_abs_error": float(np.quantile(raw_error, 0.99)),
        "probability_max_abs_error": float(probability_error.max()),
        "probability_mean_abs_error": float(probability_error.mean()),
        "probability_p99_abs_error": float(np.quantile(probability_error, 0.99)),
        "argmax_agreement": argmax_agreement,
        "actual_next_base_nll": actual_nll,
        "expected_next_base_nll": expected_nll,
        "next_base_nll_abs_delta": abs(actual_nll - expected_nll),
        "actual_next_base_accuracy": actual_accuracy,
        "expected_next_base_accuracy": expected_accuracy,
        "next_base_accuracy_abs_delta": abs(actual_accuracy - expected_accuracy),
    }
    report["accepted"] = bool(
        report["probability_mean_abs_error"] <= 0.003
        and report["probability_p99_abs_error"] <= 0.03
        and argmax_agreement >= 0.99
        and report["next_base_nll_abs_delta"] <= 0.001
        and report["next_base_accuracy_abs_delta"] <= 0.001
    )
    return report
