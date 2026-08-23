#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub",
#     "tqdm",
# ]
# ///
"""
Upload sharded dataset to HuggingFace Hub.

Uploads the sharded Angiosperm_16_genomes dataset created by shard_dataset.py
to a new HuggingFace dataset repository.

Usage:
    uv run upload_dataset.py [--input-dir INPUT_DIR] [--repo-id REPO_ID]
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser(description="Upload sharded dataset to HuggingFace Hub")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("sharded_data"),
        help="Input directory with sharded files (default: sharded_data)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="gonzalobenegas/Angiosperm_16_genomes_sharded",
        help="HuggingFace repo ID (default: gonzalobenegas/Angiosperm_16_genomes_sharded)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    api = HfApi()

    # Create the repository if it doesn't exist
    print(f"Creating/checking repository: {args.repo_id}")
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
    )

    # Upload each split
    splits = ["train", "valid", "test"]
    for split in splits:
        split_dir = args.input_dir / split
        if not split_dir.exists():
            print(f"Warning: Split directory not found: {split_dir}, skipping")
            continue

        shard_files = sorted(split_dir.glob("shard_*.jsonl.zst"))
        if not shard_files:
            print(f"Warning: No shard files found in {split_dir}, skipping")
            continue

        print(f"\nUploading {split} split ({len(shard_files)} shards)...")

        # Upload the entire split directory
        api.upload_folder(
            folder_path=str(split_dir),
            path_in_repo=f"data/{split}",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Upload {split} split ({len(shard_files)} shards)",
        )
        print(f"  Uploaded {split} split")

    print(f"\nDone! Dataset uploaded to: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
