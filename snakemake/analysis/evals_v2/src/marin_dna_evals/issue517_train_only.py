"""Explicit development-file loading for the issue-517 order-control retry."""

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download


REVISIONS = {
    "bolinas-dna/evals_mendelian_traits": "4aed58e50c5dea0b878a665007af2ef9e5108e9f",
    "bolinas-dna/evals_complex_traits": "22f86a89c65cb8f3007ac3cc2739f40efefa4340",
}


def load_order_development_frame(
    repo_id: str, revision: str, split: str
) -> pd.DataFrame:
    """Fetch only the immutable train.parquet; reject other scopes before I/O."""
    if split != "train" or REVISIONS.get(repo_id) != revision:
        raise ValueError("only the pinned issue-517 development datasets are allowed")
    train_file = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="train.parquet",
        revision=revision,
    )
    frame = load_dataset(
        "parquet", data_files={"train": train_file}, split="train"
    ).to_pandas()
    observed = {str(chrom).removeprefix("chr") for chrom in frame["chrom"].unique()}
    allowed = {str(chrom) for chrom in range(1, 23, 2)} | {"X"}
    if observed - allowed:
        raise ValueError("development input contains a non-development chromosome")
    return frame
