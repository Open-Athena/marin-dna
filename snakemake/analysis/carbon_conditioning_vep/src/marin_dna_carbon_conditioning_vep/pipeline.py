"""Pipeline-level data loading, provenance, and label-blind smoke sampling."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from marin_dna_carbon_conditioning_vep.variants import (
    materialize_variant_windows,
    validate_mendelian_dataset,
    validate_reference_contigs,
)


def load_development_dataset(
    dataset_config: dict[str, Any],
    *,
    downloader: Callable[..., str] | None = None,
    parquet_reader: Callable[[str], pd.DataFrame] = pd.read_parquet,
) -> pd.DataFrame:
    """Download and read exactly the pinned development parquet file."""
    split = str(dataset_config["split"])
    filename = str(dataset_config["filename"])
    assert split == "train", "held-out Mendelian labels are forbidden"
    assert filename == "train.parquet", "only the development parquet is allowed"
    if downloader is None:
        downloader = hf_hub_download
    path = downloader(
        repo_id=str(dataset_config["repo"]),
        filename=filename,
        repo_type="dataset",
        revision=str(dataset_config["revision"]),
    )
    return parquet_reader(path)


def select_analysis_subset(
    dataset: pd.DataFrame,
    analysis_config: dict[str, Any],
) -> pd.DataFrame:
    """Select and validate the fixed match-group-complete analysis subset."""
    subset = str(analysis_config["subset"])
    selected = dataset.loc[dataset["subset"].astype(str) == subset].copy()
    assert len(selected) == int(analysis_config["expected_rows"]), (
        f"{subset} row count changed: {len(selected)}"
    )
    assert int(selected["label"].astype(int).sum()) == int(
        analysis_config["expected_positives"]
    ), f"{subset} positive count changed"
    assert selected["match_group"].nunique() == int(
        analysis_config["expected_groups"]
    ), f"{subset} match-group count changed"
    group_sizes = selected.groupby("match_group").size()
    assert group_sizes.eq(int(analysis_config["expected_group_size"])).all(), (
        f"{subset} contains incomplete match groups"
    )
    assert selected["subset"].astype(str).eq(subset).all()
    return selected.reset_index(drop=True)


def stage_analysis_windows(
    source_path: str | Path,
    *,
    dataset_config: dict[str, Any],
    reference_config: dict[str, Any],
    analysis_config: dict[str, Any],
) -> pd.DataFrame:
    """Filter a validated development-window artifact to the fixed pilot subset."""
    subset = str(analysis_config["subset"])
    selected_batches: list[pa.Table] = []
    source_file = pq.ParquetFile(source_path)
    for batch in source_file.iter_batches(batch_size=256):
        table = pa.Table.from_batches([batch])
        selected_batch = table.filter(pc.equal(table["subset"], subset))
        if selected_batch.num_rows:
            selected_batches.append(selected_batch)
    assert selected_batches, f"validated window artifact lacks subset {subset!r}"
    source = pa.concat_tables(selected_batches).to_pandas()
    selected = select_analysis_subset(source, analysis_config)
    expected_provenance = {
        "dataset_repo": str(dataset_config["repo"]),
        "dataset_revision": str(dataset_config["revision"]),
        "dataset_split": str(dataset_config["split"]),
        "reference_path": str(reference_config["path"]),
        "reference_assembly": str(reference_config["assembly"]),
        "reference_ensembl_release": int(reference_config["ensembl_release"]),
        "reference_masking": str(reference_config["masking"]),
    }
    for column, expected in expected_provenance.items():
        assert column in selected, f"validated window artifact lacks {column!r}"
        observed = selected[column].drop_duplicates().tolist()
        assert observed == [expected], (
            f"validated window provenance changed for {column}: {observed!r}"
        )
    selected["analysis_subset"] = subset
    return selected


def build_validated_windows(
    *,
    dataset_config: dict[str, Any],
    reference_config: dict[str, Any],
    analysis_config: dict[str, Any],
    window_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the pinned development split and materialize validated windows."""
    split = str(dataset_config["split"])
    assert split == "train", "held-out Mendelian labels are forbidden"
    dataset = load_development_dataset(dataset_config)
    validated = validate_mendelian_dataset(
        dataset,
        expected_rows=int(dataset_config["expected_rows"]),
        expected_positives=int(dataset_config["expected_positives"]),
        expected_groups=int(dataset_config["expected_groups"]),
        expected_group_size=int(dataset_config["expected_group_size"]),
        expected_chroms={str(value) for value in dataset_config["expected_chroms"]},
    )
    selected = select_analysis_subset(validated, analysis_config)
    assert reference_config["assembly"] == "GRCh38"
    assert int(reference_config["ensembl_release"]) == 115
    assert reference_config["masking"] == "soft-masked"
    assert reference_config["sequence_names"] == "ensembl"
    from marin_dna.data.genome import Genome

    genome = Genome(str(reference_config["path"]))
    expected_lengths = {
        str(chrom): int(length)
        for chrom, length in reference_config["required_contig_lengths"].items()
    }
    validate_reference_contigs(genome.chroms, expected_lengths)
    windows, failures = materialize_variant_windows(
        selected,
        genome,
        genome.chroms,
        window_size=window_size,
    )
    windows["dataset_repo"] = str(dataset_config["repo"])
    windows["dataset_revision"] = str(dataset_config["revision"])
    windows["dataset_split"] = split
    windows["analysis_subset"] = str(analysis_config["subset"])
    windows["reference_path"] = str(reference_config["path"])
    windows["reference_assembly"] = str(reference_config["assembly"])
    windows["reference_ensembl_release"] = int(reference_config["ensembl_release"])
    windows["reference_masking"] = str(reference_config["masking"])
    windows["coordinates"] = "source pos 1-based; windows 0-based half-open"
    windows["window_size_bp"] = window_size
    return windows, failures


def label_blind_smoke_sample(
    windows: pd.DataFrame,
    *,
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Select rows without labels or consequence metadata entering the smoke scorer."""
    assert 0 < n_rows <= len(windows)
    keys = [
        hashlib.sha256(f"{seed}:{variant_key}".encode()).hexdigest()
        for variant_key in windows["variant_id"]
    ]
    sampled = (
        windows.assign(_sample_key=keys)
        .sort_values("_sample_key")
        .head(n_rows)
        .drop(columns="_sample_key")
        .copy()
    )
    label_columns = {
        "label",
        "subset",
        "match_group",
        "source",
        "clinvar_id",
        "trait",
        "consequence",
        "consequence_cre",
        "consequence_final",
        "consequence_group",
    }
    sampled = sampled.drop(
        columns=[column for column in label_columns if column in sampled.columns]
    )
    sampled["label_blind_smoke"] = True
    assert not label_columns.intersection(sampled.columns)
    return sampled.reset_index(drop=True)
