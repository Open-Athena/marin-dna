"""Immutable scoring protocols and auditable predictions under the existing owner."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from exp593_dna.development_readout import (
    DEVELOPMENT_MANIFEST_SHA256,
    DEVELOPMENT_MANIFEST_URI,
)
from exp593_dna.publish_inputs import BUCKET, OWNER, put_immutable
from exp593_dna.recovery import canonical_json
from exp593_dna.schedule import TokenClock
from exp593_dna.storage import coreweave_client


def _publish(root: str, name: str, payload: bytes) -> dict[str, Any]:
    expected = f"s3://{BUCKET}/{OWNER}/"
    if not root.startswith(expected) or not root.endswith("/development"):
        raise ValueError("Development outputs require the experiment's native owner")
    digest = hashlib.sha256(payload).hexdigest()
    key = f"{root.removeprefix('s3://' + BUCKET + '/')}/{digest}/{name}"
    with TemporaryDirectory(prefix="exp593-development-") as temporary:
        path = Path(temporary) / name
        path.write_bytes(payload)
        return put_immutable(coreweave_client(), path, key, digest, len(payload))


def publish_development_protocol(root: str) -> dict[str, Any]:
    """Freeze implementation/precision bindings before looking at any prediction."""
    protocol = {
        "schema": "exp593-development-v1",
        "data_manifest_uri": DEVELOPMENT_MANIFEST_URI,
        "data_manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
        "precision": "BF16 model; FP32 vocabulary projection and logsumexp; full original vocabulary, never ACGT-renormalized",
        "conservation": "mean bare left/right-RC true-base log-probability; separate minimal FIM true-base log-probability; AUROC and tie-grouped AP; no supervised fitting",
        "junction": "blind minimal FIM; macro gap NLL, teacher-forced base accuracy, independent greedy EM and case-folded EM; donor/acceptor/random separately",
        "source_sha256": {
            name: hashlib.sha256(
                files("exp593_dna").joinpath(name).read_bytes()
            ).hexdigest()
            for name in (
                "development_readout.py",
                "development_evaluate.py",
                "development_results.py",
                "evaluate.py",
            )
        },
    }
    return _publish(root, "protocol.json", canonical_json(protocol).encode())


def publish_development_result(
    root: str,
    clock: TokenClock,
    event: str,
    protocol: dict[str, Any],
    bindings: dict[str, Any],
    metrics: dict[str, float | int],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Read back every prediction object, then commit a content-addressed result."""
    predictions = _publish(
        root,
        "predictions.jsonl",
        ("\n".join(canonical_json(r) for r in records) + "\n").encode(),
    )
    result = {
        "schema": "exp593-development-v1",
        "clock": asdict(clock),
        "event": event,
        "scoring_protocol": protocol,
        "training_bindings_sha256": hashlib.sha256(
            canonical_json(bindings).encode()
        ).hexdigest(),
        "metrics": metrics,
        "predictions": predictions,
    }
    return _publish(root, "result.json", canonical_json(result).encode())
