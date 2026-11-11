"""A standby gate remains valid when its admitted writer resumes after preemption."""

import hashlib
import io
import json
import sys

import pytest

from exp593_dna import anneal_standby
from exp593_dna.anneal_standby import _verify_recovery_pointer
from exp593_dna.anneal_train import TrainingConfig


class ObjectStore:
    def __init__(self, objects):
        self.objects = objects

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[f"s3://{Bucket}/{Key}"])}


def test_recovery_gate_accepts_newer_verified_pointer():
    config = TrainingConfig(trial="181b", cluster="cw-rno2a", nodes=8)
    root = config.checkpoint_root
    released = root + "/step-560168"
    current = root + "/step-609531"
    metadata = {"step": 609531, "exp593_anneal_clock": {"stage": {"updates": 123}}}
    metadata_bytes = json.dumps(metadata).encode()
    pointer = {
        "checkpoint": current,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "clock": metadata["exp593_anneal_clock"],
    }
    store = ObjectStore(
        {
            root + "/latest.json": json.dumps(pointer).encode(),
            current + "/metadata.json": metadata_bytes,
        }
    )
    gate = {"checkpoint": released, "metadata_sha256": "old-checkpoint-digest"}
    _verify_recovery_pointer(store, config, gate)

    pointer["checkpoint"] = root + "/step-560167"
    store.objects[root + "/latest.json"] = json.dumps(pointer).encode()
    with pytest.raises(ValueError, match="precedes"):
        _verify_recovery_pointer(store, config, gate)

    pointer["checkpoint"] = current
    pointer["metadata_sha256"] = "invalid-digest"
    store.objects[root + "/latest.json"] = json.dumps(pointer).encode()
    with pytest.raises(ValueError, match="does not bind"):
        _verify_recovery_pointer(store, config, gate)


def test_qualified_181b_east_one_node_production_standby(monkeypatch):
    dispatched = []
    monkeypatch.setattr(
        anneal_standby, "dispatch", lambda config, gate: dispatched.append(config)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "anneal_standby",
            "--trial",
            "181b",
            "--cluster",
            "cw-us-east-02a",
            "--nodes",
            "1",
            "--gate-uri",
            f"{anneal_standby.OWNER}/annealing/operations/gates/test.json",
            "--gate-nonce",
            "x" * 24,
        ],
    )
    anneal_standby.main()
    assert len(dispatched) == 1
    assert (dispatched[0].trial, dispatched[0].cluster, dispatched[0].nodes) == (
        "181b",
        "cw-us-east-02a",
        1,
    )
