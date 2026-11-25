from types import SimpleNamespace

import numpy as np
import pytest

from exp586_moe import tracking


def test_resumed_history_uses_server_step_when_local_step_is_reset(monkeypatch):
    monkeypatch.setattr(tracking.jax, "process_index", lambda: 0)
    monkeypatch.setattr(tracking.multihost_utils, "broadcast_one_to_all", lambda x: x)
    monkeypatch.setattr(tracking.tracker, "log_summary", lambda x: None)
    monkeypatch.setattr(
        tracking.tracker,
        "get_tracker",
        lambda name: SimpleNamespace(
            run=SimpleNamespace(entity="eric-czech", project="marin", id="test", step=0)
        ),
    )
    paths = []

    def run(path):
        paths.append(path)
        return SimpleNamespace(lastHistoryStep=271)

    monkeypatch.setattr(
        tracking.wandb, "Api", lambda **kwargs: SimpleNamespace(run=run)
    )
    assert tracking.wandb_history_offset() + 1 == 272
    assert paths == ["eric-czech/marin/test"]


def test_history_lookup_failure_is_broadcast_and_never_defaults_to_zero(monkeypatch):
    monkeypatch.setattr(tracking.jax, "process_index", lambda: 0)

    def fail(name):
        raise ConnectionError("unavailable")

    monkeypatch.setattr(tracking.tracker, "get_tracker", fail)
    signals = []

    def broadcast(value):
        signals.append(int(value))
        return value

    monkeypatch.setattr(tracking.multihost_utils, "broadcast_one_to_all", broadcast)
    with pytest.raises(RuntimeError, match="history offset"):
        tracking.wandb_history_offset()
    assert signals == [-1]


def test_nonprimary_rank_uses_broadcast_without_querying_wandb(monkeypatch):
    monkeypatch.setattr(tracking.jax, "process_index", lambda: 1)
    monkeypatch.setattr(
        tracking.tracker, "get_tracker", lambda name: pytest.fail("nonprimary lookup")
    )
    monkeypatch.setattr(tracking.tracker, "log_summary", lambda x: None)
    monkeypatch.setattr(
        tracking.multihost_utils, "broadcast_one_to_all", lambda x: np.asarray(271)
    )
    assert tracking.wandb_history_offset() == 271
