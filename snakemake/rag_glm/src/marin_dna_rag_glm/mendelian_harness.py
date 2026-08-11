"""Materialize issue #402's retrieval-conditioned Mendelian SNV harness.

The raw Mendelian ``pos`` column is 1-based. All coordinates introduced by
this module are 0-based, half-open. Exact centered human windows are projected
directly through the pinned Zoonomia HAL; training-anchor coordinates are not
used as a surrogate join.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl
from marin_dna_zoonomia_projection.projection.filter import (
    filter_length,
    filter_single_chrom_strand,
)
from marin_dna_zoonomia_projection.projection.hal import (
    attach_src_size,
    parse_halliftover_bed,
    run_halliftover,
)
from marin_dna_zoonomia_projection.projection.resize import resize_dataframe

from marin_dna.data.dna import reverse_complement
from marin_dna_rag_glm.dataset import (
    BASES_PER_SLOT,
    DOCUMENT_TOKENS,
    HUMAN_VARIANT_TOKEN_INDEX,
    MISSING_SEQUENCE,
    PROVISIONAL_SPECIES_ORDER,
    SEQUENCE_BOUNDARY,
    SPECIES_ORDER_VERSION,
    validate_species_order,
)

VARIANT_KEY_COLUMNS = ["chrom", "pos", "ref", "alt"]
NON_HUMAN_SPECIES = PROVISIONAL_SPECIES_ORDER[:-1]
PROJECTION_VERSION = "zoonomia-rag-direct-hal-v1"
SOURCE_SPECIES = "Homo_sapiens"
PRE_RESIZE_MIN = 128
PRE_RESIZE_MAX = 512

RAW_MENDELIAN_REVISION = "4aed58e50c5dea0b878a665007af2ef9e5108e9f"
SOURCE_HARNESS_REVISION = "7b92f047f9a36f90e9ac47886afa2a99264ee35c"

PROJECTION_SCHEMA: dict[str, pl.DataType] = {
    "query_name": pl.String,
    "species": pl.String,
    "t_chrom": pl.String,
    "t_start": pl.Int64,
    "t_end": pl.Int64,
    "t_strand": pl.String,
    "t_src_size": pl.Int64,
}

_SOURCE_HARNESS_COLUMNS = {
    *VARIANT_KEY_COLUMNS,
    "target",
    "match_group",
    "context",
    "ref_completion",
    "alt_completion",
    "strand",
}


def _variant_id_expr() -> pl.Expr:
    return pl.concat_str(
        [
            pl.col("chrom"),
            pl.col("pos").cast(pl.String),
            pl.col("ref") + ">" + pl.col("alt"),
        ],
        separator=":",
    )


def _normalize_chrom_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.String).str.replace(r"^chr", "")


def validate_source_harness(rows: pl.DataFrame) -> None:
    """Assert the pinned 255-base Mendelian harness row contract."""
    missing = _SOURCE_HARNESS_COLUMNS - set(rows.columns)
    assert not missing, f"source harness is missing columns {sorted(missing)}"
    assert rows.height > 0, "source harness is empty"
    assert rows.filter(pl.col("pos") <= 0).is_empty(), "pos must be 1-based positive"
    assert rows.filter(pl.col("ref").str.len_chars() != 1).is_empty()
    assert rows.filter(pl.col("alt").str.len_chars() != 1).is_empty()
    assert rows.filter(pl.col("context").str.len_chars() != 127).is_empty()
    assert rows.filter(pl.col("ref_completion").str.len_chars() != 128).is_empty()
    assert rows.filter(pl.col("alt_completion").str.len_chars() != 128).is_empty()
    assert set(rows["strand"].unique()) == {"+", "-"}

    pair_counts = rows.group_by(VARIANT_KEY_COLUMNS).agg(
        pl.len().alias("n_rows"),
        pl.col("strand").n_unique().alias("n_strands"),
        pl.col("target").n_unique().alias("n_targets"),
        pl.col("match_group").n_unique().alias("n_match_groups"),
    )
    assert pair_counts.filter(
        (pl.col("n_rows") != 2)
        | (pl.col("n_strands") != 2)
        | (pl.col("n_targets") != 1)
        | (pl.col("n_match_groups") != 1)
    ).is_empty(), "every variant must have one consistent row per strand"

    forward = rows.filter(pl.col("strand") == "+")
    assert forward.filter(
        pl.col("ref_completion").str.slice(0, 1) != pl.col("ref")
    ).is_empty(), "forward reference completion must start with the reference allele"


def build_mendelian_variant_windows(
    *,
    source_harness_urls: Sequence[str],
    variants_path: str | Path,
    bed_path: str | Path,
) -> int:
    """Write exact unique centered variant windows as Parquet and HAL BED6.

    Source ``pos`` is converted from 1-based to ``variant_pos0`` at this
    boundary. The source window is ``[variant_pos0 - 127, variant_pos0 + 128)``.
    """
    assert source_harness_urls
    forward_frames: list[pl.DataFrame] = []
    for source_url in source_harness_urls:
        rows = pl.read_parquet(source_url)
        validate_source_harness(rows)
        forward_frames.append(
            rows.filter(pl.col("strand") == "+").select(
                *VARIANT_KEY_COLUMNS,
                (pl.col("context") + pl.col("ref_completion")).alias(
                    "human_reference_sequence"
                ),
            )
        )

    variants = pl.concat(forward_frames)
    assert variants.height == sum(frame.height for frame in forward_frames)
    assert variants.select(VARIANT_KEY_COLUMNS).unique().height == variants.height
    variants = variants.with_columns(_normalize_chrom_expr("chrom").alias("chrom"))
    assert variants.filter(pl.col("chrom").str.starts_with("chr")).is_empty()
    variants = (
        variants.with_columns(
            _variant_id_expr().alias("variant_id"),
            (pl.col("pos") - 1).alias("variant_pos0"),
        )
        .with_columns(
            (pl.col("variant_pos0") - BASES_PER_SLOT // 2).alias("human_start"),
            (pl.col("variant_pos0") + BASES_PER_SLOT // 2 + 1).alias("human_end"),
            pl.lit(PROJECTION_VERSION).alias("projection_version"),
        )
        .sort(VARIANT_KEY_COLUMNS)
    )
    assert variants.filter(pl.col("human_start") < 0).is_empty()
    assert variants.filter(
        pl.col("human_end") - pl.col("human_start") != BASES_PER_SLOT
    ).is_empty()
    assert variants.filter(
        pl.col("human_reference_sequence").str.len_chars() != BASES_PER_SLOT
    ).is_empty()
    assert variants.filter(
        pl.col("human_reference_sequence").str.slice(BASES_PER_SLOT // 2, 1)
        != pl.col("ref")
    ).is_empty()
    variants = variants.with_columns(
        pl.Series(
            "query_name",
            [f"mendelian_{index:08d}" for index in range(variants.height)],
            dtype=pl.String,
        )
    )
    assert variants["query_name"].n_unique() == variants.height
    assert variants["variant_id"].n_unique() == variants.height

    variants_output = Path(variants_path)
    variants_output.parent.mkdir(parents=True, exist_ok=True)
    variants.write_parquet(variants_output, compression="zstd", statistics=True)

    bed_output = Path(bed_path)
    bed_output.parent.mkdir(parents=True, exist_ok=True)
    variants.select(
        (pl.lit("chr") + pl.col("chrom")).alias("chrom"),
        pl.col("human_start").alias("start"),
        pl.col("human_end").alias("end"),
        "query_name",
        pl.lit(0).alias("score"),
        pl.lit("+").alias("strand"),
    ).write_csv(bed_output, separator="\t", include_header=False)
    return variants.height


def project_mendelian_variant_windows(
    *,
    hal_path: str | Path,
    source_bed: str | Path,
    target_species: str,
    target_chrom_sizes: str | Path,
    output_parquet: str | Path,
    raw_bed_path: str | Path,
) -> int:
    """Project exact variant windows and apply the canonical v1 quality gate."""
    assert target_species in NON_HUMAN_SPECIES
    raw_output = Path(raw_bed_path)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    run_halliftover(
        hal_path,
        SOURCE_SPECIES,
        source_bed,
        target_species,
        raw_output,
        no_dupes=True,
    )

    rows = parse_halliftover_bed(raw_output, species=target_species)
    rows = attach_src_size(rows, target_chrom_sizes)
    rows = filter_single_chrom_strand(rows)
    rows = filter_length(rows, min_len=PRE_RESIZE_MIN, max_len=PRE_RESIZE_MAX)
    rows = rows.filter(pl.col("t_src_size") >= BASES_PER_SLOT)
    if rows.is_empty():
        projected = pl.DataFrame(schema=PROJECTION_SCHEMA)
    else:
        projected = resize_dataframe(rows, target_len=BASES_PER_SLOT).select(
            list(PROJECTION_SCHEMA)
        )

    assert projected.filter(
        pl.col("t_end") - pl.col("t_start") != BASES_PER_SLOT
    ).is_empty()
    assert projected.filter(
        (pl.col("t_start") < 0) | (pl.col("t_end") > pl.col("t_src_size"))
    ).is_empty()
    assert projected["query_name"].n_unique() == projected.height
    assert set(projected["species"].unique()).issubset({target_species})

    output = Path(output_parquet)
    output.parent.mkdir(parents=True, exist_ok=True)
    projected.write_parquet(output, compression="zstd", statistics=True)
    raw_output.unlink(missing_ok=True)
    return projected.height


def extract_ortholog_sequences_from_twobit(
    projection_parquet: str | Path,
    twobit_path: str | Path,
    output_parquet: str | Path,
) -> int:
    """Extract target windows and normalize them to human-window orientation."""
    import py2bit

    rows = pl.read_parquet(projection_parquet).sort("query_name")
    required = {"t_chrom", "t_start", "t_end", "t_strand", "query_name"}
    assert required <= set(rows.columns)
    assert rows.filter(pl.col("t_end") - pl.col("t_start") != BASES_PER_SLOT).is_empty()
    assert set(rows["t_strand"].unique()).issubset({"+", "-"})
    assert rows["query_name"].n_unique() == rows.height

    genome = py2bit.open(str(twobit_path))
    try:
        chrom_sizes = genome.chroms()
        sequences = []
        for chrom, start, end in rows.select("t_chrom", "t_start", "t_end").iter_rows():
            assert chrom in chrom_sizes, f"2bit is missing chromosome {chrom!r}"
            assert 0 <= start < end <= chrom_sizes[chrom]
            sequence = genome.sequence(chrom, start, end)
            assert sequence is not None
            sequences.append(sequence.upper())
    finally:
        genome.close()

    assert all(len(sequence) == BASES_PER_SLOT for sequence in sequences)
    oriented = [
        reverse_complement(sequence) if strand == "-" else sequence
        for sequence, strand in zip(sequences, rows["t_strand"].to_list())
    ]
    assert all(len(sequence) == BASES_PER_SLOT for sequence in oriented)
    rows = rows.with_columns(pl.Series("sequence", oriented, dtype=pl.String))
    output = Path(output_parquet)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(output, compression="zstd", statistics=True)
    return rows.height


def _materialize_split(
    source_harness_url: str,
    variants: pl.DataFrame,
    species_sequences: Mapping[str, pl.DataFrame],
    species_order: tuple[str, ...],
) -> pl.DataFrame:
    rows = pl.read_parquet(source_harness_url)
    validate_source_harness(rows)
    rows = rows.with_columns(_normalize_chrom_expr("chrom").alias("chrom"))
    rows = rows.with_columns(_variant_id_expr().alias("variant_id")).rename(
        {"context": "_human_context"}
    )
    rows = rows.join(
        variants.select(
            *VARIANT_KEY_COLUMNS,
            "variant_id",
            "query_name",
            "variant_pos0",
            "human_start",
            "human_end",
            "projection_version",
        ),
        on=[*VARIANT_KEY_COLUMNS, "variant_id"],
        how="left",
    )
    assert rows.filter(pl.col("query_name").is_null()).is_empty()

    for slot, species in enumerate(species_order[:-1]):
        sequences = species_sequences[species].select(
            "query_name",
            pl.col("sequence").alias(f"_forward_sequence_{slot}"),
            pl.col("t_chrom").alias(f"projection_chrom_{slot}"),
            pl.col("t_start").alias(f"projection_start_{slot}"),
            pl.col("t_end").alias(f"projection_end_{slot}"),
            pl.col("t_strand").alias(f"projection_strand_{slot}"),
        )
        assert sequences["query_name"].n_unique() == sequences.height
        rows = rows.join(sequences, on="query_name", how="left")
        rows = rows.with_columns(
            pl.col(f"_forward_sequence_{slot}")
            .is_not_null()
            .alias(f"available_{slot}"),
            pl.col(f"_forward_sequence_{slot}")
            .is_not_null()
            .alias(f"quality_pass_{slot}"),
            pl.when(pl.col("strand") == "-")
            .then(
                pl.col(f"_forward_sequence_{slot}").map_elements(
                    lambda sequence: (
                        reverse_complement(sequence) if sequence is not None else None
                    ),
                    return_dtype=pl.String,
                    skip_nulls=False,
                )
            )
            .otherwise(pl.col(f"_forward_sequence_{slot}"))
            .fill_null(MISSING_SEQUENCE)
            .alias(f"sequence_{slot}"),
        ).drop(f"_forward_sequence_{slot}")

    rows = rows.with_columns(
        (pl.col("_human_context") + pl.col("ref_completion")).alias("sequence_7"),
        pl.lit(True).alias("available_7"),
        pl.lit(True).alias("quality_pass_7"),
        pl.lit(SPECIES_ORDER_VERSION).alias("species_order_version"),
        (pl.col("variant_id") + "|" + pl.col("strand")).alias("document_id"),
    )
    context_parts: list[pl.Expr] = []
    for slot in range(len(species_order) - 1):
        context_parts.extend([pl.col(f"sequence_{slot}"), pl.lit(SEQUENCE_BOUNDARY)])
    context_parts.append(pl.col("_human_context"))
    rows = rows.with_columns(pl.concat_str(context_parts).alias("context")).drop(
        "_human_context"
    )

    sequence_columns = [f"sequence_{slot}" for slot in range(len(species_order))]
    assert rows.filter(
        pl.any_horizontal(
            [
                pl.col(column).str.len_chars() != BASES_PER_SLOT
                for column in sequence_columns
            ]
        )
    ).is_empty()
    assert rows.filter(pl.col("context").str.count_matches(r"\[SEQ\]") != 7).is_empty()
    assert rows.filter(
        (pl.col("context").str.replace_all(r"\[SEQ\]", "").str.len_chars() + 7) != 1919
    ).is_empty()
    assert rows.filter(pl.col("ref_completion").str.len_chars() != 128).is_empty()
    assert rows.filter(pl.col("alt_completion").str.len_chars() != 128).is_empty()
    assert rows.select(pl.col("document_id").n_unique()).item() == rows.height

    from marin_dna_rag_glm.tokenizer import create_rag_char_tokenizer

    tokenizer = create_rag_char_tokenizer()
    sample = rows.head(16)
    for completion in ("ref_completion", "alt_completion"):
        texts = (sample["context"] + sample[completion]).to_list()
        assert all(len(ids) == DOCUMENT_TOKENS for ids in tokenizer(texts)["input_ids"])
    return rows.sort([*VARIANT_KEY_COLUMNS, "strand"])


def write_mendelian_harness_readme(
    output_path: str | Path,
    *,
    commit_sha: str,
    hf_repo: str,
    manifest: Mapping[str, object],
) -> None:
    """Write the reviewed Hugging Face dataset card."""
    assert len(commit_sha) == 40
    split_rows = manifest["split_rows"]
    split_variants = manifest["split_variants"]
    text = f"""---
license: apache-2.0
tags:
- biology
- genomics
- dna
---

# {hf_repo}

Retrieval-conditioned, eval-harness-ready Mendelian SNV benchmark. Each row
contains seven fully materialized Zoonomia ortholog slots followed by the
shared 127-base human prefix, plus separate reference and alternate
completions. Every variant has forward and reverse-complement rows.

Produced by the commit-pinned [issue #402 RAG pipeline](https://github.com/Open-Athena/marin-dna/tree/{commit_sha}/snakemake/rag_glm).
Model scoring needs only this pinned dataset and a model checkpoint; it does
not access the HAL, genomes, or projection Parquets.

## Splits

| Split | Variants | Rows |
| --- | ---: | ---: |
| train | {split_variants["train"]:,} | {split_rows["train"]:,} |
| test | {split_variants["test"]:,} | {split_rows["test"]:,} |

## Frozen document contract

- Species order/version: `{SPECIES_ORDER_VERSION}`; human is slot 7.
- Projection/version: `{PROJECTION_VERSION}`.
- `context` has seven complete non-human slots, seven atomic `[SEQ]`
  boundaries, and the shared 127-base human prefix (1,919 tokens before BOS).
- Each completion is 128 bases, so `context + completion` is 2,047 tokens
  before BOS and 2,048 after the BOS/CLS token.
- The centered SNV is absolute token index {HUMAN_VARIANT_TOKEN_INDEX}.
- Missing or quality-filtered non-human projections are full 255-base `N` slots.

Every exact unique 255-base human variant window was projected directly with
`halLiftover --noDupes` against the pinned Zoonomia 447-mammalian 2022 v1 HAL.
The build applies the canonical projection pipeline's single-chromosome,
single-strand, pre-resize length `[128, 512]`, midpoint-resize, bounds, and
strand-normalized sequence conventions. It does not approximate the variant
window through a conservation-filtered training anchor.

## Provenance

- Raw Mendelian source: `bolinas-dna/evals_mendelian_traits` at
  `{RAW_MENDELIAN_REVISION}` (GRCh38, 1-based SNV `pos`).
- Human sequence-materialized source: `bolinas-dna/evals_mendelian_traits_harness_255`
  at `{SOURCE_HARNESS_REVISION}`.
- Alignment: Zoonomia 447-mammalian 2022 v1 HAL; source leaf `Homo_sapiens`.
- Derived genomic coordinates: 0-based, half-open.

`manifest.json` records exact counts, revisions, projection settings, and
per-species availability.
"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def materialize_mendelian_rag_harness(
    *,
    source_harness_urls: Mapping[str, str],
    variants_path: str | Path,
    species_sequence_paths: Mapping[str, str | Path],
    output_split_paths: Mapping[str, str | Path],
    manifest_path: str | Path,
    readme_path: str | Path,
    commit_sha: str,
    hf_repo: str,
    species_order: Sequence[str] = PROVISIONAL_SPECIES_ORDER,
) -> None:
    """Assemble and validate both pinned RAG Mendelian harness splits."""
    assert len(commit_sha) == 40
    order = validate_species_order(species_order)
    assert tuple(species_sequence_paths) == order[:-1]
    assert set(source_harness_urls) == {"train", "test"}
    assert set(output_split_paths) == {"train", "test"}

    variants = pl.read_parquet(variants_path)
    assert variants["variant_id"].n_unique() == variants.height
    assert variants["query_name"].n_unique() == variants.height
    assert set(variants["projection_version"].unique()) == {PROJECTION_VERSION}
    species_sequences = {
        species: pl.read_parquet(path)
        for species, path in species_sequence_paths.items()
    }

    materialized: dict[str, pl.DataFrame] = {}
    for split in ("train", "test"):
        rows = _materialize_split(
            source_harness_urls[split], variants, species_sequences, order
        )
        output = Path(output_split_paths[split])
        output.parent.mkdir(parents=True, exist_ok=True)
        rows.write_parquet(output, compression="zstd", statistics=True)
        materialized[split] = rows

    all_forward = pl.concat(
        [rows.filter(pl.col("strand") == "+") for rows in materialized.values()],
        how="vertical_relaxed",
    )
    assert all_forward.height == variants.height
    missingness = []
    for slot, species in enumerate(order):
        n_available = int(all_forward.select(pl.col(f"available_{slot}").sum()).item())
        missingness.append(
            {
                "slot": slot,
                "species": species,
                "n_available": n_available,
                "n_missing": all_forward.height - n_available,
                "missing_fraction": 1.0 - n_available / all_forward.height,
            }
        )

    manifest: dict[str, object] = {
        "producing_commit": commit_sha,
        "raw_mendelian_revision": RAW_MENDELIAN_REVISION,
        "source_harness_revision": SOURCE_HARNESS_REVISION,
        "projection_version": PROJECTION_VERSION,
        "projection": {
            "alignment": "Zoonomia 447-mammalian 2022 v1 HAL",
            "source_species": SOURCE_SPECIES,
            "hal_liftover_no_dupes": True,
            "pre_resize_min_len": PRE_RESIZE_MIN,
            "pre_resize_max_len": PRE_RESIZE_MAX,
            "target_len": BASES_PER_SLOT,
            "filter": "single target chromosome and strand per query",
        },
        "species_order_version": SPECIES_ORDER_VERSION,
        "species_order": list(order),
        "split_rows": {split: rows.height for split, rows in materialized.items()},
        "split_variants": {
            split: rows.height // 2 for split, rows in materialized.items()
        },
        "n_variants": all_forward.height,
        "context_tokens_without_bos": 1919,
        "completion_tokens": 128,
        "document_tokens_with_bos": DOCUMENT_TOKENS,
        "centered_variant_token_index": HUMAN_VARIANT_TOKEN_INDEX,
        "coordinate_system": (
            "source Mendelian pos is 1-based; all derived intervals are 0-based, half-open"
        ),
        "missingness": missingness,
    }
    manifest_output = Path(manifest_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")
    write_mendelian_harness_readme(
        readme_path,
        commit_sha=commit_sha,
        hf_repo=hf_repo,
        manifest=manifest,
    )
