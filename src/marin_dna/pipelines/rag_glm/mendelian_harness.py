"""Materialize issue #402's retrieval-conditioned Mendelian SNV harness.

The raw Mendelian ``pos`` column is 1-based. All coordinates introduced by
this module are 0-based, half-open. No lift-over is performed here: variants
are mapped through an already-projected, containing Zoonomia anchor.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from marin_dna.data.dna import reverse_complement
from marin_dna.pipelines.rag_glm.dataset import (
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
MAPPING_VERSION = "zoonomia-rag-containing-anchor-offset-v1"
SOURCE_ANCHOR_STEP = 128

RAW_MENDELIAN_REVISION = "4aed58e50c5dea0b878a665007af2ef9e5108e9f"
SOURCE_HARNESS_REVISION = "7b92f047f9a36f90e9ac47886afa2a99264ee35c"

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


def select_containing_projection_anchors(
    variants: pl.DataFrame,
    human_projections: pl.DataFrame,
    *,
    anchor_step: int = SOURCE_ANCHOR_STEP,
) -> pl.DataFrame:
    """Choose the containing conserved anchor whose center is nearest each SNV.

    The immutable projection's human windows are 255-base tiles with 128-base
    step. An arbitrary Mendelian SNV can be contained by at most two tiles.
    Ties are resolved by source start and then ``query_name``.
    """
    assert anchor_step > 0
    assert set(VARIANT_KEY_COLUMNS) <= set(variants.columns)
    assert {
        "query_name",
        "t_chrom",
        "t_start",
        "t_end",
        "t_strand",
    } <= set(human_projections.columns)

    unique_variants = variants.select(VARIANT_KEY_COLUMNS).unique()
    assert unique_variants.height == variants.height, "variants must be unique"
    unique_variants = unique_variants.with_columns(
        _normalize_chrom_expr("chrom").alias("chrom"),
        _variant_id_expr().alias("variant_id"),
        (pl.col("pos") - 1).alias("variant_pos0"),
    )
    assert (
        unique_variants.select(pl.col("variant_id").n_unique()).item()
        == variants.height
    )

    anchors = human_projections.select(
        pl.col("query_name").alias("source_anchor_id"),
        _normalize_chrom_expr("t_chrom").alias("chrom"),
        pl.col("t_start").alias("source_anchor_start"),
        pl.col("t_end").alias("source_anchor_end"),
        pl.col("t_strand").alias("source_anchor_strand"),
    )
    assert (
        anchors.select(pl.col("source_anchor_id").n_unique()).item() == anchors.height
    )
    assert anchors.filter(pl.col("source_anchor_strand") != "+").is_empty()
    assert anchors.filter(
        pl.col("source_anchor_end") - pl.col("source_anchor_start") != BASES_PER_SLOT
    ).is_empty()
    assert anchors.filter(
        pl.col("source_anchor_start") % anchor_step != 0
    ).is_empty(), "human projection starts do not match the frozen 128-base tiling grid"

    tiled_start = (pl.col("variant_pos0") // anchor_step) * anchor_step
    candidates = (
        unique_variants.with_columns(
            pl.concat_list([tiled_start, tiled_start - anchor_step]).alias(
                "source_anchor_start"
            )
        )
        .explode("source_anchor_start")
        .filter(pl.col("source_anchor_start") >= 0)
        .join(anchors, on=["chrom", "source_anchor_start"], how="inner")
        .filter(
            (pl.col("variant_pos0") >= pl.col("source_anchor_start"))
            & (pl.col("variant_pos0") < pl.col("source_anchor_end"))
        )
        .with_columns(
            (
                pl.col("variant_pos0")
                - (pl.col("source_anchor_start") + BASES_PER_SLOT // 2)
            )
            .abs()
            .alias("source_anchor_center_distance")
        )
        .sort(
            [
                "variant_id",
                "source_anchor_center_distance",
                "source_anchor_start",
                "source_anchor_id",
            ]
        )
        .unique("variant_id", keep="first", maintain_order=True)
        .select(
            "variant_id",
            "source_anchor_id",
            "source_anchor_start",
            "source_anchor_end",
            "source_anchor_center_distance",
        )
    )
    mapped = unique_variants.join(candidates, on="variant_id", how="left").with_columns(
        (pl.col("variant_pos0") - pl.col("source_anchor_start")).alias(
            "source_anchor_offset"
        ),
        pl.lit(MAPPING_VERSION).alias("mapping_version"),
    )
    chosen = mapped.filter(pl.col("source_anchor_id").is_not_null())
    assert chosen.filter(
        (pl.col("source_anchor_offset") < 0)
        | (pl.col("source_anchor_offset") >= BASES_PER_SLOT)
    ).is_empty()
    assert mapped.height == variants.height
    return mapped.sort(VARIANT_KEY_COLUMNS)


def derive_projected_variant_intervals(
    mapping: pl.DataFrame,
    projection_rows: pl.DataFrame,
    *,
    species: str,
) -> pl.DataFrame:
    """Propagate an SNV's source-anchor offset into one projected species.

    The projection interval has already been strand-filtered and resized to
    255 bases. On ``+`` the target base is ``t_start + offset``; on ``-`` it
    is ``t_end - 1 - offset``. A centered 255-base extraction is retained only
    when it stays within the target chromosome.
    """
    assert species != "Homo_sapiens"
    assert {
        "variant_id",
        "source_anchor_id",
        "source_anchor_offset",
    } <= set(mapping.columns)
    assert {
        "query_name",
        "species",
        "t_chrom",
        "t_start",
        "t_end",
        "t_strand",
        "t_src_size",
    } <= set(projection_rows.columns)

    species_rows = projection_rows.filter(pl.col("species") == species)
    assert (
        species_rows.group_by("query_name").len().filter(pl.col("len") != 1).is_empty()
    )
    species_rows = species_rows.select(
        pl.col("query_name").alias("source_anchor_id"),
        pl.col("t_chrom").alias("projection_chrom"),
        pl.col("t_start").alias("projection_start"),
        pl.col("t_end").alias("projection_end"),
        pl.col("t_strand").alias("projection_strand"),
        pl.col("t_src_size").alias("projection_chrom_size"),
    )

    rows = mapping.filter(pl.col("source_anchor_id").is_not_null()).join(
        species_rows, on="source_anchor_id", how="inner"
    )
    assert rows.filter(~pl.col("projection_strand").is_in(["+", "-"])).is_empty()
    rows = rows.with_columns(
        pl.when(pl.col("projection_strand") == "+")
        .then(pl.col("projection_start") + pl.col("source_anchor_offset"))
        .otherwise(pl.col("projection_end") - 1 - pl.col("source_anchor_offset"))
        .alias("projected_variant_pos0")
    ).with_columns(
        (pl.col("projected_variant_pos0") - BASES_PER_SLOT // 2).alias(
            "extraction_start"
        ),
        (pl.col("projected_variant_pos0") + BASES_PER_SLOT // 2 + 1).alias(
            "extraction_end"
        ),
    )
    rows = rows.filter(
        (pl.col("extraction_start") >= 0)
        & (pl.col("extraction_end") <= pl.col("projection_chrom_size"))
    ).with_columns(pl.lit(species).alias("species"))

    assert rows.filter(
        pl.col("extraction_end") - pl.col("extraction_start") != BASES_PER_SLOT
    ).is_empty()
    assert rows.select(pl.col("variant_id").n_unique()).item() == rows.height
    return rows.sort(VARIANT_KEY_COLUMNS)


def extract_ortholog_sequences_from_twobit(
    interval_parquet: str | Path,
    twobit_path: str | Path,
    output_parquet: str | Path,
) -> int:
    """Extract target windows and normalize them to human-anchor orientation."""
    import py2bit

    rows = pl.read_parquet(interval_parquet).sort(VARIANT_KEY_COLUMNS)
    required = {
        "projection_chrom",
        "extraction_start",
        "extraction_end",
        "projection_strand",
    }
    assert required <= set(rows.columns)
    assert rows.filter(
        pl.col("extraction_end") - pl.col("extraction_start") != BASES_PER_SLOT
    ).is_empty()
    assert set(rows["projection_strand"].unique()).issubset({"+", "-"})

    genome = py2bit.open(str(twobit_path))
    try:
        chrom_sizes = genome.chroms()
        sequences = []
        for chrom, start, end in rows.select(
            "projection_chrom", "extraction_start", "extraction_end"
        ).iter_rows():
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
        for sequence, strand in zip(sequences, rows["projection_strand"].to_list())
    ]
    assert all(len(sequence) == BASES_PER_SLOT for sequence in oriented)
    rows = rows.with_columns(pl.Series("sequence", oriented, dtype=pl.String))
    output = Path(output_parquet)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(output, compression="zstd", statistics=True)
    return rows.height


def build_mendelian_projection_mapping(
    *,
    source_harness_urls: Sequence[str],
    projection_parquet: str,
    mapping_path: str | Path,
    interval_paths: Mapping[str, str | Path],
    audit_path: str | Path,
    species_order: Sequence[str] = PROVISIONAL_SPECIES_ORDER,
) -> None:
    """Build the additive no-HAL variant→anchor→species extraction mapping."""
    order = validate_species_order(species_order)
    assert tuple(interval_paths) == order[:-1]
    assert source_harness_urls

    source_frames = []
    for source_url in source_harness_urls:
        rows = pl.read_parquet(source_url)
        validate_source_harness(rows)
        source_frames.append(rows.select(VARIANT_KEY_COLUMNS))
    variants = pl.concat(source_frames).unique().sort(VARIANT_KEY_COLUMNS)
    assert variants.height == sum(frame.height // 2 for frame in source_frames)

    source = pl.scan_parquet(projection_parquet)
    human = (
        source.filter(pl.col("species") == "Homo_sapiens")
        .select("query_name", "t_chrom", "t_start", "t_end", "t_strand")
        .collect(engine="streaming")
    )
    mapping = select_containing_projection_anchors(variants, human)
    selected_anchor_ids = mapping.filter(pl.col("source_anchor_id").is_not_null())[
        "source_anchor_id"
    ].unique()

    selected_projection_rows = (
        pl.scan_parquet(projection_parquet)
        .filter(
            pl.col("query_name").is_in(selected_anchor_ids)
            & pl.col("species").is_in(order[:-1])
        )
        .select(
            "query_name",
            "species",
            "t_chrom",
            "t_start",
            "t_end",
            "t_strand",
            "t_src_size",
        )
        .collect(engine="streaming")
    )

    mapping_output = Path(mapping_path)
    mapping_output.parent.mkdir(parents=True, exist_ok=True)
    mapping.write_parquet(mapping_output, compression="zstd", statistics=True)

    species_counts: dict[str, int] = {}
    for species in order[:-1]:
        intervals = derive_projected_variant_intervals(
            mapping, selected_projection_rows, species=species
        )
        interval_output = Path(interval_paths[species])
        interval_output.parent.mkdir(parents=True, exist_ok=True)
        intervals.write_parquet(interval_output, compression="zstd", statistics=True)
        species_counts[species] = intervals.height

    mapped_count = mapping.filter(pl.col("source_anchor_id").is_not_null()).height
    audit = {
        "mapping_version": MAPPING_VERSION,
        "coordinate_system": "Mendelian pos is 1-based; derived coordinates are 0-based, half-open",
        "n_variants": variants.height,
        "n_variants_with_containing_anchor": mapped_count,
        "containing_anchor_fraction": mapped_count / variants.height,
        "selection": (
            "containing 255-base conserved anchor with minimum absolute center distance; "
            "ties by start then query_name"
        ),
        "offset_rule": (
            "plus: t_start + source_offset; minus: t_end - 1 - source_offset"
        ),
        "n_valid_species_intervals": species_counts,
    }
    audit_output = Path(audit_path)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2) + "\n")


def _materialize_split(
    source_harness_url: str,
    mapping: pl.DataFrame,
    species_sequences: Mapping[str, pl.DataFrame],
    species_order: tuple[str, ...],
) -> pl.DataFrame:
    rows = pl.read_parquet(source_harness_url)
    validate_source_harness(rows)
    rows = rows.with_columns(_variant_id_expr().alias("variant_id")).rename(
        {"context": "_human_context"}
    )
    rows = rows.join(mapping, on=[*VARIANT_KEY_COLUMNS, "variant_id"], how="left")
    assert rows.filter(pl.col("mapping_version").is_null()).is_empty()

    for slot, species in enumerate(species_order[:-1]):
        sequences = species_sequences[species].select(
            "variant_id",
            pl.col("sequence").alias(f"_forward_sequence_{slot}"),
            pl.col("projection_chrom").alias(f"projection_chrom_{slot}"),
            pl.col("extraction_start").alias(f"extraction_start_{slot}"),
            pl.col("extraction_end").alias(f"extraction_end_{slot}"),
            pl.col("projection_strand").alias(f"projection_strand_{slot}"),
            pl.col("projected_variant_pos0").alias(f"projected_variant_pos0_{slot}"),
        )
        assert (
            sequences.select(pl.col("variant_id").n_unique()).item() == sequences.height
        )
        rows = rows.join(sequences, on="variant_id", how="left")
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

    from marin_dna.pipelines.rag_glm.tokenizer import create_rag_char_tokenizer

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
not access the HAL, genomes, or projection Parquet.

## Splits

| Split | Variants | Rows |
| --- | ---: | ---: |
| train | {split_variants["train"]:,} | {split_rows["train"]:,} |
| test | {split_variants["test"]:,} | {split_rows["test"]:,} |

## Frozen document contract

- Species order/version: `{SPECIES_ORDER_VERSION}`; human is slot 7.
- Mapping/version: `{MAPPING_VERSION}`.
- `context` has seven complete non-human slots, seven atomic `[SEQ]`
  boundaries, and the shared 127-base human prefix (1,919 tokens before BOS).
- Each completion is 128 bases, so `context + completion` is 2,047 tokens
  before BOS and 2,048 after the BOS/CLS token.
- The centered SNV is absolute token index {HUMAN_VARIANT_TOKEN_INDEX}.
- Missing non-human projections are full 255-base `N` slots.

The arbitrary SNV windows are not keyed directly to the conserved projection
tiles. The build deterministically selects the containing 255-base human
anchor with the closest center (ties by start and anchor ID), propagates the
SNV offset through each already-projected strand-aware interval, and extracts a
centered target window from the archived species 2bit genome. This is an
additive derivation from existing projection coordinates; no `halLiftover` is
run.

## Provenance

- Raw Mendelian source: `bolinas-dna/evals_mendelian_traits` at
  `{RAW_MENDELIAN_REVISION}` (GRCh38, 1-based SNV `pos`).
- Human sequence-materialized source: `bolinas-dna/evals_mendelian_traits_harness_255`
  at `{SOURCE_HARNESS_REVISION}`.
- Zoonomia projection: existing 447-mammalian 2022 v1 `min0.20` artifact.
- Derived genomic coordinates: 0-based, half-open.

`manifest.json` records exact counts, revisions, mapping coverage, and
per-species availability.
"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def materialize_mendelian_rag_harness(
    *,
    source_harness_urls: Mapping[str, str],
    mapping_path: str | Path,
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

    mapping = pl.read_parquet(mapping_path)
    assert mapping.select(pl.col("variant_id").n_unique()).item() == mapping.height
    species_sequences = {
        species: pl.read_parquet(path)
        for species, path in species_sequence_paths.items()
    }

    materialized: dict[str, pl.DataFrame] = {}
    for split in ("train", "test"):
        rows = _materialize_split(
            source_harness_urls[split], mapping, species_sequences, order
        )
        output = Path(output_split_paths[split])
        output.parent.mkdir(parents=True, exist_ok=True)
        rows.write_parquet(output, compression="zstd", statistics=True)
        materialized[split] = rows

    all_forward = pl.concat(
        [rows.filter(pl.col("strand") == "+") for rows in materialized.values()],
        how="vertical_relaxed",
    )
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
    n_with_anchor = int(
        all_forward.select(pl.col("source_anchor_id").is_not_null().sum()).item()
    )
    manifest: dict[str, object] = {
        "producing_commit": commit_sha,
        "raw_mendelian_revision": RAW_MENDELIAN_REVISION,
        "source_harness_revision": SOURCE_HARNESS_REVISION,
        "mapping_version": MAPPING_VERSION,
        "species_order_version": SPECIES_ORDER_VERSION,
        "species_order": list(order),
        "split_rows": {split: rows.height for split, rows in materialized.items()},
        "split_variants": {
            split: rows.height // 2 for split, rows in materialized.items()
        },
        "n_variants": all_forward.height,
        "n_variants_with_containing_anchor": n_with_anchor,
        "containing_anchor_fraction": n_with_anchor / all_forward.height,
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
