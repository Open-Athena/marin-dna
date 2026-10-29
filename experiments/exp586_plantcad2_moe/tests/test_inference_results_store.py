from pathlib import Path

import fsspec
import pytest

from exp586_moe.inference.results_store import upload_trial_artifacts


def test_upload_trial_artifacts_persists_available_evidence(tmp_path: Path) -> None:
    output = tmp_path / "trial.json"
    reference = tmp_path / "trial.reference.npz"
    ledger = tmp_path / "torch.jsonl"
    output.write_text("result")
    reference.write_bytes(b"reference")
    ledger.write_text("ledger")
    filesystem = fsspec.filesystem("memory")

    uploaded = upload_trial_artifacts(
        output,
        ledger,
        "bucket/results",
        filesystem=filesystem,
    )

    assert uploaded == [
        "bucket/results/trial.json",
        "bucket/results/trial.reference.npz",
        "bucket/results/torch.jsonl",
    ]
    assert filesystem.cat("bucket/results/trial.json") == b"result"
    assert filesystem.cat("bucket/results/trial.reference.npz") == b"reference"
    assert filesystem.cat("bucket/results/torch.jsonl") == b"ledger"


def test_upload_trial_artifacts_rejects_ambiguous_prefix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="trailing slash"):
        upload_trial_artifacts(
            tmp_path / "trial.json",
            tmp_path / "torch.jsonl",
            "bucket/results/",
            filesystem=fsspec.filesystem("memory"),
        )
