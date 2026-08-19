from __future__ import annotations

from pathlib import Path
from typing import Any

from exp479_mntp import publishing
from exp479_mntp.publishing import remote_arm_is_complete


class FakeApi:
    def __init__(self) -> None:
        self.files: set[str] = set()
        self.uploads: list[str] = []

    def file_exists(self, *, filename: str, **kwargs: object) -> bool:
        del kwargs
        return filename in self.files

    def upload_file(self, *, path_in_repo: str, **kwargs: object) -> None:
        del kwargs
        self.files.add(path_in_repo)
        self.uploads.append(path_in_repo)


def test_checkpoint_upload_is_remote_idempotent(tmp_path: Path, monkeypatch: Any) -> None:
    api = FakeApi()
    monkeypatch.setattr(publishing, "HfApi", lambda: api)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "step-0100.ckpt").touch()
    callback = publishing.CheckpointUploadCallback(
        checkpoint_dir=checkpoint_dir,
        repo_id="org/repo",
        arm="transferred_mntp",
    )
    callback._upload_new()
    callback.load_state_dict({})
    callback._upload_new()
    assert api.uploads == ["lightning/transferred_mntp/step-0100.ckpt"]


def test_remote_completion_requires_manifest_and_export() -> None:
    files = {
        "runs/transferred_mntp/manifest.json",
        "hf/transferred_mntp/step-1000/config.json",
    }
    assert remote_arm_is_complete(files, "transferred_mntp")
    assert not remote_arm_is_complete(
        files - {"runs/transferred_mntp/manifest.json"}, "transferred_mntp"
    )
