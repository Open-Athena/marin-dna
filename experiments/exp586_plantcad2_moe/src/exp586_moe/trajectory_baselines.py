"""Preserve matched zero-shot vectors alongside a checkpoint's full reduction."""

import hashlib
import json
from pathlib import Path

import numpy as np

from exp586_moe.real_eval import read_bytes


def prepare(
    preparation: dict,
    artifacts: Path,
    output: Path,
    requests_uri: str,
    requests_sha256: str,
) -> None:
    """Reuse original scoring and validate full-benchmark metric parity.

    Classification development rows can extend the original zero-shot inventory.
    They receive missing zero-shot scores, never fabricated model predictions.
    The unchanged classification runner requires all held-out scores to be finite.
    """
    from exp586_evals.classification_baselines import prepare_baselines

    payload = read_bytes(requests_uri)
    if hashlib.sha256(payload).hexdigest() != requests_sha256:
        raise ValueError("Classification request manifest changed")
    features = json.loads(payload)
    cached = artifacts / "context-8192/tasks/chunks"
    receipts = [json.loads(p.read_text()) for p in sorted(cached.glob("*.json"))]
    receipt_path = cached / "receipts.json"
    if receipt_path.exists():
        raise ValueError("Unexpected preexisting derived receipt collection")
    receipt_path.write_text(json.dumps(receipts))
    target = output / "classification-baselines"
    try:
        prepare_baselines(
            {**preparation, "tasks": features["tasks"]},
            cached,
            output / "result.json",
            target,
        )
    finally:
        receipt_path.unlink()
    for spec in features["tasks"]:
        expected = features["inputs"][spec["file"]]["rows"]
        path = target / (spec["key"] + ".npz")
        with np.load(path, allow_pickle=False) as source:
            values = {name: source[name] for name in source.files}
        observed = len(values["row_id"])
        if observed > expected:
            raise ValueError("Zero-shot inventory exceeds classification inventory")
        if observed < expected:
            values["row_id"] = np.arange(expected, dtype=np.int64)
            for name in values.keys() - {"row_id"}:
                values[name] = np.pad(
                    values[name].astype(np.float64),
                    (0, expected - observed),
                    constant_values=np.nan,
                )
            np.savez_compressed(path, **values)
