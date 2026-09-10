"""Preserve caller precision across Trainer/Accelerate construction."""

from types import SimpleNamespace

import datasets
import numpy as np
import pytest
import torch
from marin_dna_evals.model.runner import _run_inference


@pytest.mark.parametrize("requested", [False, None])
def test_runner_preserves_explicit_tf32_after_trainer_setup(monkeypatch, requested):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    class PrecisionOverridingTrainer:
        def __init__(self, *, model, args):
            # Reproduce Accelerate's compilation-state initialization.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        def predict(self, *, test_dataset):
            expected = requested is None
            assert torch.backends.cuda.matmul.allow_tf32 is expected
            assert torch.backends.cudnn.allow_tf32 is expected
            return SimpleNamespace(predictions=np.asarray([[1.0], [2.0]]))

    monkeypatch.setattr(
        "marin_dna_evals.model.runner.Trainer", PrecisionOverridingTrainer
    )
    result = _run_inference(
        torch.nn.Identity(),
        datasets.Dataset.from_dict({"input_ids": [[1], [2]]}),
        use_cpu=True,
        per_device_eval_batch_size=2,
        tf32=requested,
        report_to="none",
    )
    np.testing.assert_array_equal(result, [[1.0], [2.0]])
