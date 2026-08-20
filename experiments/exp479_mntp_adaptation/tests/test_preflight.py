from __future__ import annotations

import os

import pytest
import torch

from exp479_mntp.preflight import enable_training_determinism


def test_enable_training_determinism_matches_lightning_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", calls.append)
    torch.backends.cudnn.benchmark = True

    enable_training_determinism()

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert not torch.backends.cudnn.benchmark
    assert calls == [True]
