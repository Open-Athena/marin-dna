"""Reject incomplete, mismatched and corrupted resumable evaluation chunks."""

import hashlib
import io
import json

import numpy as np
import pytest

from exp586_moe.real_eval import (
    LEGACY_SCRATCH_CHECKPOINT,
    LEGACY_SCRATCH_DIGEST,
    RealEvalConfig,
    commit_chunk,
    load_chunk,
)


def test_chunk_commit_and_binding(tmp_path):
    root = str(tmp_path)
    (tmp_path / "chunks").mkdir()
    binding = {"requests": "x", "checkpoint": "y"}
    chunk = {
        "key": "test--000000-000001",
        "start": 0,
        "stop": 1,
        "task": "test",
        "worker": 0,
    }
    assert load_chunk(root, chunk, binding) is None
    stream = io.BytesIO()
    np.savez_compressed(stream, left__scores=np.array([1.0]))
    blob = stream.getvalue()
    path = tmp_path / "chunks" / (chunk["key"] + ".npz")
    path.write_bytes(blob)
    receipt = {
        **chunk,
        "binding": binding,
        "array_sha256": hashlib.sha256(blob).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(receipt))
    assert load_chunk(root, chunk, binding) == receipt
    # Partial AF recovery must preserve already committed embeddings and metadata.
    replay = {**receipt, "started_epoch": 123}
    assert commit_chunk(root, replay, {"left__scores": np.array([1.0])}) == receipt
    assert json.loads(path.with_suffix(".json").read_text()) == receipt
    with pytest.raises(AssertionError):
        commit_chunk(root, replay, {"left__scores": np.array([2.0])})

    with pytest.raises(ValueError, match="another"):
        load_chunk(root, chunk, {"requests": "different"})
    with pytest.raises(ValueError, match="another"):
        load_chunk(root, {**chunk, "stop": 2}, binding)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_chunk(root, chunk, binding)


def test_legacy_scratch_requires_exact_binding():
    args = {
        "condition": "scratch",
        "run_id": "exp586-real-smoke-test",
        "cluster": "cw-rno2a",
        "requests_sha256": "a" * 64,
        "checkpoint_metadata_digest": LEGACY_SCRATCH_DIGEST,
        "checkpoint": LEGACY_SCRATCH_CHECKPOINT,
        "allow_legacy_scratch": True,
    }
    RealEvalConfig(**args)
    args["condition"] = "pretrained"
    with pytest.raises(ValueError, match="Legacy"):
        RealEvalConfig(**args)


def test_startup_error_is_not_masked(monkeypatch):
    from exp586_moe import real_eval

    original = TimeoutError("W&B startup timeout")

    def fail(config):
        raise original

    def missing(name):
        raise RuntimeError("No global tracker set")

    monkeypatch.setattr(real_eval, "run_pilot", fail)
    monkeypatch.setattr(real_eval.tracker, "get_tracker", missing)
    with pytest.raises(TimeoutError) as result:
        real_eval.run(None)
    assert result.value is original
