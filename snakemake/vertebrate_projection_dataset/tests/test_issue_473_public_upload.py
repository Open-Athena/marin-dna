from pathlib import Path

import pytest
from marin_dna_vertebrate_projection.issue_473 import public_upload


def test_public_upload_creates_and_rechecks_public_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class PublicInfo:
        private = False

    class FakeApi:
        def __init__(self) -> None:
            self.created: list[tuple[str, str, bool, bool]] = []
            self.info_calls = 0

        def repo_exists(self, repo_id: str, *, repo_type: str) -> bool:
            assert repo_id == "marin-dna/test-public"
            assert repo_type == "dataset"
            return False

        def create_repo(
            self,
            repo_id: str,
            *,
            repo_type: str,
            private: bool,
            exist_ok: bool,
        ) -> None:
            self.created.append((repo_id, repo_type, private, exist_ok))

        def dataset_info(self, repo_id: str) -> PublicInfo:
            assert repo_id == "marin-dna/test-public"
            self.info_calls += 1
            return PublicInfo()

    api = FakeApi()
    uploads: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(public_upload, "HfApi", lambda: api)
    monkeypatch.setattr(
        public_upload,
        "upload_validated_dataset",
        lambda *args, **kwargs: uploads.append((args, kwargs)),
    )

    public_upload.upload_public_validated_dataset(
        tmp_path / "artifacts",
        tmp_path / "manifest.json",
        tmp_path / "upload.done",
        cohort="center1_cds",
        repo_id="marin-dna/test-public",
        workers=4,
    )

    assert api.created == [("marin-dna/test-public", "dataset", False, False)]
    assert api.info_calls == 2
    assert len(uploads) == 1
    assert uploads[0][1] == {
        "cohort": "center1_cds",
        "repo_id": "marin-dna/test-public",
        "workers": 4,
    }


def test_public_upload_rejects_preexisting_private_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PrivateInfo:
        private = True

    class FakeApi:
        def repo_exists(self, repo_id: str, *, repo_type: str) -> bool:
            return True

        def dataset_info(self, repo_id: str) -> PrivateInfo:
            return PrivateInfo()

    monkeypatch.setattr(public_upload, "HfApi", FakeApi)
    monkeypatch.setattr(
        public_upload,
        "upload_validated_dataset",
        lambda *args, **kwargs: pytest.fail("private repository must not upload"),
    )

    with pytest.raises(RuntimeError, match="refusing to upload.*privately"):
        public_upload.upload_public_validated_dataset(
            "artifacts",
            "manifest.json",
            "upload.done",
            cohort="center1_cds",
            repo_id="marin-dna/private",
            workers=4,
        )
