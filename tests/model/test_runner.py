"""Tests for the HF inference-harness plumbing in ``marin_dna.model.runner``.

These exercise the wiring only (``Trainer`` is mocked) so they run on CPU in
milliseconds. End-to-end forward-pass smokes with a real model live in
``tests/model/test_scoring.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import datasets
import numpy as np

from marin_dna.model.runner import _run_inference


def _tiny_dataset(n: int) -> datasets.Dataset:
    return datasets.Dataset.from_dict({"x": list(range(n))})


def test_run_inference_forwards_callbacks_to_trainer():
    """A caller-supplied callback list reaches the ``Trainer`` constructor."""
    ds = _tiny_dataset(4)
    sentinel = object()
    preds = np.zeros((4, 2), dtype=np.float32)

    with patch("marin_dna.model.runner.Trainer") as MockTrainer:
        MockTrainer.return_value.predict.return_value = MagicMock(predictions=preds)
        out = _run_inference(
            MagicMock(),
            ds,
            callbacks=[sentinel],
            per_device_eval_batch_size=1,
        )

    _, kwargs = MockTrainer.call_args
    assert kwargs["callbacks"] == [sentinel]
    np.testing.assert_array_equal(out, preds)


def test_run_inference_default_callbacks_none():
    """Omitting ``callbacks`` passes ``None`` — i.e. the prior behaviour exactly."""
    ds = _tiny_dataset(3)
    preds = np.zeros((3, 2), dtype=np.float32)

    with patch("marin_dna.model.runner.Trainer") as MockTrainer:
        MockTrainer.return_value.predict.return_value = MagicMock(predictions=preds)
        _run_inference(MagicMock(), ds, per_device_eval_batch_size=1)

    _, kwargs = MockTrainer.call_args
    assert kwargs["callbacks"] is None
