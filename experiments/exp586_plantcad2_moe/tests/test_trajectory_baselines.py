"""Keep added development rows unscored while preserving benchmark vectors."""

import hashlib
import json
import sys
from types import ModuleType

import numpy as np
import pytest

from exp586_moe import trajectory_baselines


def test_added_development_rows_have_no_fabricated_zero_shot_scores(
    tmp_path, monkeypatch
):
    spec = {"key": "task", "file": "input"}
    requests = {"tasks": [spec], "inputs": {"input": {"rows": 5}}}
    payload = json.dumps(requests).encode()
    monkeypatch.setattr(trajectory_baselines, "read_bytes", lambda uri: payload)
    module = ModuleType("exp586_evals.classification_baselines")

    def existing_reducer(preparation, cached, historical, output):
        assert preparation["tasks"] == [spec]
        assert json.loads((cached / "receipts.json").read_text()) == [
            {"verified": True}
        ]
        output.mkdir()
        np.savez(output / "task.npz", row_id=np.arange(3), natural=[0.2, 0.7, 0.4])

    module.prepare_baselines = existing_reducer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cached = tmp_path / "artifacts/context-8192/tasks/chunks"
    cached.mkdir(parents=True)
    (cached / "chunk.json").write_text('{"verified": true}')
    output = tmp_path / "output"
    output.mkdir()
    trajectory_baselines.prepare(
        {},
        tmp_path / "artifacts",
        output,
        "s3://input",
        hashlib.sha256(payload).hexdigest(),
    )
    with np.load(output / "classification-baselines/task.npz") as result:
        np.testing.assert_array_equal(result["row_id"], np.arange(5))
        np.testing.assert_array_equal(result["natural"][:3], [0.2, 0.7, 0.4])
        assert np.isnan(result["natural"][3:]).all()
    assert not (cached / "receipts.json").exists()


def test_changed_classification_inventory_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(trajectory_baselines, "read_bytes", lambda uri: b"changed")
    monkeypatch.setitem(
        sys.modules, "exp586_evals.classification_baselines", ModuleType("stub")
    )
    sys.modules["exp586_evals.classification_baselines"].prepare_baselines = lambda *a: (
        None
    )
    with pytest.raises(ValueError, match="manifest changed"):
        trajectory_baselines.prepare({}, tmp_path, tmp_path, "s3://input", "0" * 64)
