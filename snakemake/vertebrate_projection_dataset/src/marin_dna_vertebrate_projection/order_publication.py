"""Isolated publication helpers for the issue #517 order enhancer control."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.contract import TARGET_LENGTH
from marin_dna_vertebrate_projection.order_manifest import read_order_manifest
from marin_dna_vertebrate_projection.projection.dataset import reverse_complement_col
from marin_dna_vertebrate_projection.split import (
    VALIDATION_IDENTITY_COLUMNS,
    build_validation_composition,
    select_uniform_validation_rows,
)

SELECTION_SALT = "region=enhancer|species_scope=all"
TARGET_SELECTION = "one_per_ncbi_order_including_human_reference"


def write_order_dataset_split_files(
    combined_path: str | Path,
    order_manifest_path: str | Path,
    source_manifest_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    selection_path: str | Path,
    composition_path: str | Path,
    summary_path: str | Path,
    *,
    add_rc: bool,
    validation_rows: int,
    seed: int,
) -> None:
    """Filter to the complete one-per-order corpus, then split and augment it."""
    manifest = read_order_manifest(order_manifest_path, source_manifest_path)
    target_names = frozenset(manifest["alignment_name"].to_list())
    assert "hg38" not in target_names
    original = pl.scan_parquet(combined_path).filter(
        (pl.col("region_label") == "enhancer")
        & (
            (pl.col("alignment_name") == "hg38")
            | pl.col("alignment_name").is_in(target_names)
        )
    )
    schema = original.collect_schema()
    assert "augmentation" not in schema
    source_rows = int(original.select(pl.len()).collect(engine="streaming").item())
    assert source_rows > validation_rows

    selection = select_uniform_validation_rows(
        original,
        validation_rows=validation_rows,
        seed=seed,
        selection_salt=SELECTION_SALT,
    )
    selected_keys = selection.select(*VALIDATION_IDENTITY_COLUMNS)
    train_original = original.join(
        selected_keys.lazy(),
        on=list(VALIDATION_IDENTITY_COLUMNS),
        how="anti",
    )
    validation = (
        original.join(
            selection.select(
                *VALIDATION_IDENTITY_COLUMNS,
                "selection_rank",
            ).lazy(),
            on=list(VALIDATION_IDENTITY_COLUMNS),
            how="inner",
        )
        .sort("selection_rank")
        .drop("selection_rank")
        .with_columns(pl.lit("+").alias("augmentation"))
        .collect(engine="streaming")
    )
    assert validation.height == validation_rows

    train = (
        pl.concat(
            [
                train_original.with_columns(pl.lit("+").alias("augmentation")),
                train_original.with_columns(
                    reverse_complement_col(pl.col("sequence")).alias("sequence"),
                    pl.lit("-").alias("augmentation"),
                ),
            ],
            how="vertical",
        )
        if add_rc
        else train_original.with_columns(pl.lit("+").alias("augmentation"))
    )

    for path in [
        train_path,
        validation_path,
        selection_path,
        composition_path,
        summary_path,
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    train.sink_parquet(train_path)
    validation.write_parquet(validation_path)
    selection.write_csv(selection_path, separator="\t")
    build_validation_composition(original, validation).write_csv(
        composition_path,
        separator="\t",
    )

    train_rows = int(
        pl.scan_parquet(train_path).select(pl.len()).collect(engine="streaming").item()
    )
    train_original_rows = source_rows - validation_rows
    assert train_rows == train_original_rows * (2 if add_rc else 1)
    assert pl.read_parquet_schema(train_path) == pl.read_parquet_schema(validation_path)
    assert set(validation["augmentation"].to_list()) == {"+"}
    assert (
        pl.scan_parquet(train_path)
        .join(
            selected_keys.lazy(),
            on=list(VALIDATION_IDENTITY_COLUMNS),
            how="semi",
        )
        .select(pl.len())
        .collect(engine="streaming")
        .item()
        == 0
    )
    Path(summary_path).write_text(
        json.dumps(
            {
                "add_reverse_complements": add_rc,
                "realized_token_count": validation_rows * (TARGET_LENGTH + 1),
                "region_label": "enhancer",
                "seed": seed,
                "selection_salt": SELECTION_SALT,
                "source_rows": source_rows,
                "species_scope": "all",
                "split_strategy": "uniform_row_random_before_reverse_complement",
                "target_selection": TARGET_SELECTION,
                "train_original_rows": train_original_rows,
                "train_rows": train_rows,
                "validation_rows": validation_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_order_dataset_card(
    train_path: str | Path,
    validation_path: str | Path,
    order_manifest_path: str | Path,
    source_manifest_path: str | Path,
    output_path: str | Path,
    *,
    pipeline_commit: str,
    hf_repo: str,
    validation_seed: int,
    source_pipeline_commit: str,
    source_config_sha256: str,
) -> None:
    """Write the complete public card for the one-per-order dataset."""
    assert len(pipeline_commit) == 40
    assert len(source_pipeline_commit) == 40
    assert len(source_config_sha256) == 64
    manifest = read_order_manifest(order_manifest_path, source_manifest_path)
    assert manifest.height == 39
    train_rows = int(
        pl.scan_parquet(train_path).select(pl.len()).collect(engine="streaming").item()
    )
    validation_rows = int(
        pl.scan_parquet(validation_path)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    train_schema = pl.read_parquet_schema(train_path)
    assert train_schema == pl.read_parquet_schema(validation_path)
    species_counts = (
        manifest.group_by("backend", "clade")
        .len(name="species")
        .sort("backend", "clade")
    )
    species_lines = "\n".join(
        f"| {backend} | {clade} | {count} |"
        for backend, clade, count in species_counts.iter_rows()
    )
    schema_lines = "\n".join(
        f"- `{column}`: `{dtype}`" for column, dtype in train_schema.items()
    )
    pipeline_url = (
        "https://github.com/Open-Athena/marin-dna/blob/"
        f"{pipeline_commit}/snakemake/vertebrate_projection_dataset/README.md"
    )
    text = f"""---
tags:
- biology
- genomics
- dna
license: openmdw-1.1
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/*.jsonl.zst
  - split: validation
    path: data/validation/*.jsonl.zst
---

# `{hf_repo}`

Human-anchored 255 bp vertebrate enhancer sequences from the Zoonomia 447-mammal Cactus alignment and UCSC hg38 MultiZ 100-way alignment.
The complete dataset contains exactly one sequence source per represented NCBI order: human is the sole Primates source, and 39 non-human projection targets cover the remaining orders.
The target set includes 18 mammals and 21 non-mammalian vertebrates.

Non-human rows project only the central human nucleotide and extract the 255 bp target window centered on its unique mapped locus.
Human reference rows are emitted once per anchor and are not duplicated as a projected species.

Anchor eligibility uses issue #326 Arm A enhancer assignments on the uniform GRCh38 255 bp grid at 128 bp stride.
At least 51 of 255 human positions must satisfy `hg38.phyloP447way >= 2.2162`.
The authoritative projection table was produced by source commit `{source_pipeline_commit}` with source config SHA-256 `{source_config_sha256}`.
Produced by the [commit-pinned vertebrate projection pipeline]({pipeline_url}).

## Coordinates and sequence semantics

Human source coordinates use GRCh38/hg38 primary chromosomes.
All human and target coordinates are 0-based and half-open.
Every emitted sequence is exactly 255 bases, preserves source FASTA/2bit letter case, and is oriented to the human anchor.
Lowercase bases preserve repeat masking; conservation scores do not alter sequence letters or case.

The published format is zstd-compressed JSON Lines under `data/train/` and `data/validation/`.
The release is validated before upload against a producer-keyed manifest of paths, sizes, row counts, and SHA-256 checksums.
This processed dataset is released under OpenMDW 1.1; source genome assemblies, annotations, and alignments retain their own terms.

## Splits

- `train`: {train_rows:,} rows after removing validation and applying reverse-complement augmentation.
- `validation`: {validation_rows:,} original-orientation rows sampled uniformly without replacement with seed {validation_seed} before augmentation ({validation_rows * 256:,} tokens including BOS).

The split is row-level and does not stratify by chromosome, species, or human anchor.
Different species projections from one human anchor may occur on opposite sides of the split.
The reverse complement of a selected validation row is excluded from training.

| Projection backend | Clade | Selected non-human species |
|---|---|---:|
{species_lines}

## Schema

{schema_lines}

## Intended use and limitations

This dataset is intended for genomic language-model research and is not a clinical resource.
Assembly quality, alignment gaps, repeat masking, one-per-order species selection, human conservation selection, and the center-nucleotide acceptance contract affect the observed distribution.
Projecting the center nucleotide does not establish that both 127 bp target flanks are homologous to the full human anchor.
"""
    Path(output_path).write_text(text)


def write_order_source_audit(
    combined_path: str | Path,
    order_manifest_path: str | Path,
    source_manifest_path: str | Path,
    output_path: str | Path,
) -> None:
    """Audit the post-hoc order subset without materializing sequence rows."""
    manifest = read_order_manifest(order_manifest_path, source_manifest_path)
    selected_targets = set(manifest["alignment_name"].to_list())
    counts = (
        pl.scan_parquet(combined_path)
        .filter(pl.col("region_label") == "enhancer")
        .group_by("alignment_name")
        .len(name="rows")
        .collect(engine="streaming")
        .sort("alignment_name")
    )
    observed = set(counts["alignment_name"].to_list())
    expected = selected_targets | {"hg38"}
    missing = expected - observed
    assert not missing, f"order representatives missing enhancer rows: {sorted(missing)}"
    selected_counts = counts.filter(pl.col("alignment_name").is_in(expected))
    metadata = manifest.select(
        "alignment_name", "scientific_name", "order", "clade", "backend"
    )
    target_counts = selected_counts.filter(
        pl.col("alignment_name") != "hg38"
    ).join(metadata, on="alignment_name", how="left", validate="1:1")
    assert sum(target_counts.null_count().row(0)) == 0
    by_clade = (
        target_counts.group_by("backend", "clade")
        .agg(pl.len().alias("targets"), pl.col("rows").sum().alias("rows"))
        .sort("backend", "clade")
    )
    source_rows = int(selected_counts["rows"].sum())
    human_rows = int(
        selected_counts.filter(pl.col("alignment_name") == "hg38")["rows"].item()
    )
    payload = {
        "candidate_region_rows": int(counts["rows"].sum()),
        "human_is_sole_primates_source": True,
        "human_rows": human_rows,
        "nonhuman_rows": source_rows - human_rows,
        "order_manifest_targets": manifest.height,
        "represented_orders_including_human": manifest.height + 1,
        "region_label": "enhancer",
        "selected_alignment_count_including_human": selected_counts.height,
        "source_rows": source_rows,
        "target_counts_by_backend_and_clade": by_clade.to_dicts(),
        "target_rows": target_counts.select(
            "alignment_name", "scientific_name", "order", "clade", "backend", "rows"
        ).to_dicts(),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
