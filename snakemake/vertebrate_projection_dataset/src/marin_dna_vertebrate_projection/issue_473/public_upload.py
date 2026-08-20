"""Public-only Hugging Face upload boundary for issue #473."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

from marin_dna_vertebrate_projection.publication import upload_validated_dataset


def upload_public_validated_dataset(
    artifact_dir: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    cohort: str,
    repo_id: str,
    workers: int,
) -> None:
    """Create or verify a public repo, upload, then recheck visibility."""
    api = HfApi()
    if not api.repo_exists(repo_id, repo_type="dataset"):
        api.create_repo(
            repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=False,
        )
    before = api.dataset_info(repo_id)
    if before.private is not False:
        raise RuntimeError(f"refusing to upload #473 data privately: {repo_id}")

    upload_validated_dataset(
        artifact_dir,
        manifest_path,
        output_path,
        cohort=cohort,
        repo_id=repo_id,
        workers=workers,
    )

    after = api.dataset_info(repo_id)
    if after.private is not False:
        raise RuntimeError(f"#473 dataset became private: {repo_id}")
