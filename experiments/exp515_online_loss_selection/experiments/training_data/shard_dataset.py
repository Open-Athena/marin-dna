#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "huggingface_hub",
#     "tqdm",
#     "zstandard",
# ]
# ///
"""
Download and shard the Angiosperm_16_genomes dataset.

Downloads the dataset from kuleshov-group/Angiosperm_16_genomes and creates
64 shards per split in jsonl.zst format.

Usage:
    uv run shard_dataset.py [--output-dir OUTPUT_DIR] [--n-shards N_SHARDS]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def download_split(repo_id: str, split: str, cache_dir: Path) -> Path:
    """Download a single split from HuggingFace."""
    filename = f"data/{split}/{split}.jsonl.zst"
    return Path(
        hf_hub_download(  # nosec B615
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    )


def shard_split(
    input_path: Path,
    output_dir: Path,
    split: str,
    n_shards: int,
    seed: int = 42,
) -> None:
    """Load a split and create sharded output files."""
    print(f"Loading {split} split from {input_path}...")
    df = pd.read_json(input_path, lines=True)
    print(f"  Loaded {len(df):,} rows")

    # Shuffle the data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Create output directory
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Split into shards and save
    print(f"Writing {n_shards} shards...")
    shards = np.array_split(df, n_shards)
    for i, df_shard in enumerate(tqdm(shards, desc=f"Sharding {split}")):
        shard_path = split_dir / f"shard_{i:04d}.jsonl.zst"
        df_shard.to_json(
            shard_path,
            orient="records",
            lines=True,
            compression={"method": "zstd", "threads": -1},
        )

    print(f"  Wrote {n_shards} shards to {split_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and shard Angiosperm_16_genomes dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sharded_data"),
        help="Output directory for sharded files (default: sharded_data)",
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=64,
        help="Number of shards per split (default: 64)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for HuggingFace downloads",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    args = parser.parse_args()

    repo_id = "kuleshov-group/Angiosperm_16_genomes"
    splits = ["train", "valid", "test"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")

        # Download
        input_path = download_split(repo_id, split, args.cache_dir)

        # Shard
        shard_split(
            input_path=input_path,
            output_dir=args.output_dir,
            split=split,
            n_shards=args.n_shards,
            seed=args.seed,
        )

    print(f"\nDone! Sharded data written to {args.output_dir}")


if __name__ == "__main__":
    main()
