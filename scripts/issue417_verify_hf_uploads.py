#!/usr/bin/env python3
"""Verify issue #417 Hugging Face datasets after publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def verify_hf_uploads(
    artifact_dir: Path,
    artifact_manifest_path: Path,
    output_path: Path,
    *,
    hf_owner: str,
    hf_repo_prefix: str,
    check_loading: bool,
) -> dict[str, object]:
    """Compare Hub objects with the validated local artifacts."""
    manifest_bytes = artifact_manifest_path.read_bytes()
    artifact_manifest = json.loads(manifest_bytes)
    cohorts = artifact_manifest["cohorts"]
    assert isinstance(cohorts, dict) and cohorts

    datasets_version: str | None = None
    if check_loading:
        import datasets

        datasets_version = datasets.__version__

    api = HfApi()
    repositories: dict[str, object] = {}
    for cohort, cohort_manifest in cohorts.items():
        assert isinstance(cohort, str)
        assert isinstance(cohort_manifest, dict)
        repo_id = f"{hf_owner}/{hf_repo_prefix}-{cohort}"
        info = api.dataset_info(repo_id, files_metadata=True)
        assert info.sha is not None and len(info.sha) == 40
        assert not info.private
        assert not info.gated

        remote_files = {sibling.rfilename: sibling for sibling in info.siblings}
        expected_files = {".gitattributes", "README.md"}
        expected_shards: dict[str, dict[str, object]] = {}
        for split in ["train", "validation"]:
            split_manifest = cohort_manifest["splits"][split]
            for shard in split_manifest["shards"]:
                path = Path(shard["path"])
                relative = str(path.relative_to(artifact_dir / cohort))
                expected_files.add(relative)
                expected_shards[relative] = shard
        assert set(remote_files) == expected_files, {
            "repo": repo_id,
            "missing": sorted(expected_files - set(remote_files)),
            "unexpected": sorted(set(remote_files) - expected_files),
        }

        for relative, shard in expected_shards.items():
            remote = remote_files[relative]
            assert remote.size == shard["compressed_bytes"], (repo_id, relative)
            assert remote.lfs is not None, (repo_id, relative)
            assert remote.lfs.sha256 == shard["sha256"], (repo_id, relative)

        local_card = (artifact_dir / cohort / "README.md").read_bytes()
        remote_card = Path(
            hf_hub_download(
                repo_id,
                "README.md",
                repo_type="dataset",
                revision=info.sha,
            )
        ).read_bytes()
        assert remote_card == local_card, repo_id

        loaded_records: dict[str, object] = {}
        if check_loading:
            expected_columns = set(cohort_manifest["schema"])
            for split in ["train", "validation"]:
                loaded = datasets.load_dataset(
                    repo_id,
                    split=split,
                    revision=info.sha,
                    streaming=True,
                )
                record = next(iter(loaded))
                assert set(record) == expected_columns, (repo_id, split)
                assert len(record["sequence"]) == artifact_manifest["target_length"]
                if split == "train":
                    assert record["source_chrom"] != artifact_manifest["validation_chrom"]
                    assert record["augmentation"] in {"+", "-"}
                else:
                    assert record["source_chrom"] == artifact_manifest["validation_chrom"]
                    assert record["augmentation"] == "+"
                loaded_records[split] = {
                    "query_name": record["query_name"],
                    "source_chrom": record["source_chrom"],
                    "augmentation": record["augmentation"],
                }

        repositories[cohort] = {
            "repo_id": repo_id,
            "revision": info.sha,
            "files": len(remote_files),
            "compressed_bytes": sum(
                int(shard["compressed_bytes"])
                for shard in expected_shards.values()
            ),
            "loaded_records": loaded_records,
        }

    result: dict[str, object] = {
        "artifact_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "artifact_pipeline_commit": artifact_manifest["pipeline_commit"],
        "datasets_version": datasets_version,
        "repositories": repositories,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-owner", default="marin-dna")
    parser.add_argument("--hf-repo-prefix", default="vertebrate-v1")
    parser.add_argument("--check-loading", action="store_true")
    args = parser.parse_args()
    result = verify_hf_uploads(
        args.artifact_dir,
        args.artifact_manifest,
        args.output,
        hf_owner=args.hf_owner,
        hf_repo_prefix=args.hf_repo_prefix,
        check_loading=args.check_loading,
    )
    repositories = result["repositories"]
    assert isinstance(repositories, dict)
    print(f"verified {len(repositories)} Hugging Face repositories: {args.output}")


if __name__ == "__main__":
    main()
