"""Held-out-safe evaluation loading for issue #515."""

from __future__ import annotations

import logging
import urllib.request
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from biofoundation.data import Genome, transform_llr_mlm
from biofoundation.model.base import Tokenizer
from datasets import Dataset, load_dataset

log = logging.getLogger(__name__)

DEVELOPMENT_CHROMS = frozenset({*(str(value) for value in range(1, 23, 2)), "X"})
TRAITGYM_PROMOTER_SUBSET = "tss_proximal"
TRAITGYM_MIRNA_SUBSET = "mature_miRNA_variant"


def assert_development_split(frame: pd.DataFrame) -> None:
    """Reject labeled variants outside odd autosomes and X."""

    required = {"chrom", "label", "subset", "match_group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"evaluation dataset lacks required columns: {sorted(missing)}"
        )
    observed = {str(value) for value in frame["chrom"].unique()}
    forbidden = observed - DEVELOPMENT_CHROMS
    if forbidden:
        raise RuntimeError(
            "evaluation train split crosses the development boundary: "
            f"{sorted(forbidden)}"
        )
    if not observed:
        raise ValueError("evaluation train split is empty")


def filter_traitgym_promoter(dataset: Dataset) -> Dataset:
    """Select Mendelian TSS-proximal variants from the development split.

    Complete match groups containing a mature-miRNA variant are removed before
    filtering, as required by the registered TraitGym evaluation protocol.
    """

    frame = dataset.to_pandas()
    frame["chrom"] = frame["chrom"].astype(str)
    assert_development_split(frame)
    excluded_groups = set(
        frame.loc[frame["subset"] == TRAITGYM_MIRNA_SUBSET, "match_group"]
    )
    if excluded_groups:
        frame = frame.loc[~frame["match_group"].isin(excluded_groups)]
    frame = frame.loc[frame["subset"] == TRAITGYM_PROMOTER_SUBSET].copy()
    if frame.empty:
        raise ValueError("TraitGym development split has no TSS-proximal variants")
    if frame["label"].nunique() != 2:
        raise ValueError("TraitGym promoter endpoint requires both label classes")
    log.info(
        "TraitGym promoter development endpoint: %d variants; %d match groups",
        len(frame),
        frame["match_group"].nunique(),
    )
    return Dataset.from_pandas(frame, preserve_index=False)


EVAL_FILTERS = {
    "traitgym_promoter": filter_traitgym_promoter,
    "none": lambda dataset: dataset,
}


def download_genome(url: str, data_dir: str | Path = "data") -> Path:
    """Download a reference genome when it is not already present."""

    path = Path(data_dir) / Path(url).name
    if path.exists():
        log.info("Genome already exists at %s", path)
        return path
    log.info("Downloading genome from %s to %s", url, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)  # nosec B310
    return path


def transform_llr_clm_odd(
    example: dict[str, Any],
    *,
    tokenizer: Tokenizer,
    genome: Genome,
    window_size: int,
) -> dict[str, torch.Tensor]:
    """Create ref/alt CLM inputs for an odd genomic window.

    Input ``pos`` follows VCF's 1-based convention.
    Reference access is converted at this boundary to 0-based, half-open
    coordinates.
    """

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("the issue #515 CLM window must be a positive odd number")
    center_zero_based = int(example["pos"]) - 1
    start = center_zero_based - window_size // 2
    end = start + window_size
    sequence = genome(str(example["chrom"]), start, end).upper()
    variant_index = center_zero_based - start
    reference = str(example["ref"]).upper()
    alternate = str(example["alt"]).upper()
    if len(sequence) != window_size:
        raise ValueError(f"short reference window: observed {len(sequence)} bases")
    if sequence[variant_index] != reference:
        raise ValueError(
            f"reference mismatch at {example['chrom']}:{example['pos']}: "
            f"dataset={reference}, genome={sequence[variant_index]}"
        )
    alternate_sequence = (
        sequence[:variant_index] + alternate + sequence[variant_index + 1 :]
    )
    reference_ids = torch.tensor(tokenizer.encode(sequence), dtype=torch.long)
    alternate_ids = torch.tensor(tokenizer.encode(alternate_sequence), dtype=torch.long)
    expected_tokens = window_size + 1
    if reference_ids.numel() != expected_tokens:
        raise ValueError(
            "issue #515 requires exactly one leading BOS and one token per base; "
            f"observed {reference_ids.numel()} tokens for {window_size} bases"
        )
    return {"input_ids": torch.stack((reference_ids, alternate_ids))}


def load_eval_dataset(
    tokenizer: Tokenizer,
    dataset_name: str,
    genome_url: str,
    filter_name: str = "none",
    dataset_config: str | None = None,
    dataset_revision: str | None = None,
    split: str = "train",
    window_size: int = 255,
    cache_dir: str | Path = "data/evals_cache",
    objective: str = "mlm",
    data_dir: str | Path = "data",
    label_column: str = "label",
) -> Dataset:
    """Load, filter, transform, and cache one pinned evaluation dataset."""

    from datasets import load_from_disk

    if filter_name not in EVAL_FILTERS:
        raise ValueError(
            f"Unknown filter_name: {filter_name}. Must be one of {list(EVAL_FILTERS)}"
        )
    if dataset_revision is None:
        raise ValueError("evaluation datasets must pin dataset_revision")
    revision_key = dataset_revision[:12]
    cache_parts = [dataset_name.replace("/", "_"), revision_key]
    if dataset_config:
        cache_parts.append(dataset_config)
    cache_parts.extend([split, filter_name, f"window{window_size}", objective])
    cache_path = Path(cache_dir) / "_".join(cache_parts)
    if cache_path.exists():
        dataset = load_from_disk(str(cache_path))
        dataset.set_format(type="torch")
        return dataset

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        revision=dataset_revision,
        split=split,
    )  # nosec B615
    dataset = EVAL_FILTERS[filter_name](dataset)
    genome = Genome(download_genome(genome_url, data_dir))
    original_columns = dataset.column_names
    if objective in {"mlm", "dlm"}:
        transform = transform_llr_mlm
    elif objective == "clm":
        transform = transform_llr_clm_odd
    else:
        raise ValueError(f"Unknown objective: {objective}")
    dataset = dataset.map(
        partial(
            transform,
            tokenizer=tokenizer,
            genome=genome,
            window_size=window_size,
        ),
        remove_columns=[
            column for column in original_columns if column != label_column
        ],
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_path))
    dataset.set_format(type="torch")
    return dataset
