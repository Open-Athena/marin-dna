"""Fixed-layout document construction for the issue #402 RAG prototype.

All genomic coordinates are 0-based, half-open. The source projection already
normalizes negative-strand ortholog sequences to the human-anchor orientation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from marin_dna_zoonomia_projection.projection.dataset import reverse_complement_col

from marin_dna.data.utils import get_array_split_pairs

BASES_PER_SLOT = 255
N_SEQUENCE_SLOTS = 8
N_NON_HUMAN_SLOTS = 7
SEQUENCE_BOUNDARY = "[SEQ]"
DOCUMENT_TOKENS_WITHOUT_CLS = N_SEQUENCE_SLOTS * BASES_PER_SLOT + (N_SEQUENCE_SLOTS - 1)
DOCUMENT_TOKENS = 1 + DOCUMENT_TOKENS_WITHOUT_CLS
HUMAN_SEGMENT_START = 1 + N_NON_HUMAN_SLOTS * (BASES_PER_SLOT + 1)
HUMAN_VARIANT_TOKEN_INDEX = HUMAN_SEGMENT_START + BASES_PER_SLOT // 2
MISSING_SEQUENCE = "N" * BASES_PER_SLOT

SPECIES_ORDER_VERSION = "zoonomia-rag-v1"
PROVISIONAL_SPECIES_ORDER: tuple[str, ...] = (
    "Microgale_talazaci",
    "Loxodonta_africana",
    "Tolypeutes_matacus",
    "Bos_taurus",
    "Equus_caballus",
    "Mus_musculus",
    "Microcebus_murinus",
    "Homo_sapiens",
)

CONSERVATION_SOURCE = "phyloP_447m"
CONSERVATION_THRESHOLD = 2.2162
WINDOW_MIN_PROPORTION_CONSERVED = 0.20
REFERENCE_ASSEMBLY = "GRCh38 (Ensembl release 115)"
HAL_SOURCE = "Zoonomia 447-mammalian 2022 v1 Cactus HAL"

_SOURCE_COLUMNS = {
    "query_name",
    "species",
    "t_chrom",
    "t_start",
    "t_end",
    "t_strand",
    "sequence",
}


def validate_species_order(species_order: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze an eight-slot species order."""
    order = tuple(species_order)
    assert len(order) == N_SEQUENCE_SLOTS, (
        f"expected {N_SEQUENCE_SLOTS} species slots, got {len(order)}"
    )
    assert len(set(order)) == len(order), f"species slots are not unique: {order}"
    assert order[-1] == "Homo_sapiens", (
        f"human must occupy slot 7, got final species {order[-1]!r}"
    )
    assert "Homo_sapiens" not in order[:-1], "human appears before the final slot"
    return order


def validate_slot_sequences(sequences: Sequence[str]) -> tuple[str, ...]:
    """Assert the raw fixed-slot sequence invariants."""
    slots = tuple(sequences)
    assert len(slots) == N_SEQUENCE_SLOTS, (
        f"expected {N_SEQUENCE_SLOTS} sequences, got {len(slots)}"
    )
    lengths = [len(sequence) for sequence in slots]
    assert lengths == [BASES_PER_SLOT] * N_SEQUENCE_SLOTS, (
        f"every sequence must be {BASES_PER_SLOT} bases, got {lengths}"
    )
    return slots


def assemble_document(sequences: Sequence[str]) -> str:
    """Join eight 255-base slots with seven literal ``[SEQ]`` markers."""
    slots = validate_slot_sequences(sequences)
    document = SEQUENCE_BOUNDARY.join(slots)
    assert document.count(SEQUENCE_BOUNDARY) == N_SEQUENCE_SLOTS - 1
    return document


def reverse_complement_sequence(sequence: str) -> str:
    """Reverse-complement ACGTN while preserving other ambiguity symbols."""
    translation = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return sequence.translate(translation)[::-1]


def reverse_complement_document_slots(sequences: Sequence[str]) -> tuple[str, ...]:
    """Apply one shared reverse-complement orientation without reordering slots."""
    slots = validate_slot_sequences(sequences)
    reverse = tuple(reverse_complement_sequence(sequence) for sequence in slots)
    assert tuple(reverse_complement_sequence(sequence) for sequence in reverse) == slots
    return reverse


def stable_anchor_rank(anchor_id: str, seed: int) -> int:
    """Return a version-stable 63-bit rank for deterministic subsampling."""
    digest = hashlib.blake2b(
        f"{seed}:{anchor_id}".encode(), digest_size=8, person=b"dna-exp402"
    ).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _assert_projection_source(rows: pl.DataFrame) -> None:
    missing_columns = _SOURCE_COLUMNS - set(rows.columns)
    assert not missing_columns, (
        f"projection source is missing columns {missing_columns}"
    )
    assert rows.height > 0, "projection source is empty"


def assemble_fixed_layout_documents(
    projection_rows: pl.DataFrame,
    *,
    species_order: Sequence[str] = PROVISIONAL_SPECIES_ORDER,
    species_order_version: str = SPECIES_ORDER_VERSION,
) -> pl.DataFrame:
    """Pivot canonical projection rows into one fixed-layout document per anchor.

    The canonical source has one row per successful ``(query_name, species)``
    projection. Missing non-human rows become full-window ``N`` slots. Human
    rows define the anchor cohort and provide the 0-based, half-open source
    coordinates.
    """
    _assert_projection_source(projection_rows)
    order = validate_species_order(species_order)

    rows = projection_rows
    if "augmentation" in rows.columns:
        rows = rows.filter(pl.col("augmentation") == "+")

    rows = rows.filter(pl.col("species").is_in(order))
    assert rows.height > 0, "none of the requested species occur in the source"

    duplicate_pairs = (
        rows.group_by(["query_name", "species"]).len().filter(pl.col("len") != 1)
    )
    assert duplicate_pairs.is_empty(), (
        "expected at most one canonical row per (query_name, species): "
        f"{duplicate_pairs.head(10).to_dicts()}"
    )

    bad_lengths = rows.filter(pl.col("sequence").str.len_chars() != BASES_PER_SLOT)
    assert bad_lengths.is_empty(), (
        f"source contains non-{BASES_PER_SLOT}-base projections: "
        f"{bad_lengths.select('query_name', 'species').head(10).to_dicts()}"
    )

    human_rows = rows.filter(pl.col("species") == order[-1])
    assert human_rows.height > 0, "source contains no Homo_sapiens rows"
    assert (
        human_rows.select(pl.col("query_name").n_unique()).item() == human_rows.height
    )
    assert human_rows.filter(pl.col("t_strand") != "+").is_empty(), (
        "human self-projections must remain in the forward anchor orientation"
    )
    assert human_rows.filter(
        pl.col("t_end") - pl.col("t_start") != BASES_PER_SLOT
    ).is_empty()

    documents = human_rows.select(
        pl.col("query_name").alias("anchor_id"),
        pl.col("t_chrom").str.replace(r"^chr", "").alias("chrom"),
        pl.col("t_start").alias("start"),
        pl.col("t_end").alias("end"),
    )

    for slot, species in enumerate(order):
        species_rows = rows.filter(pl.col("species") == species).select(
            pl.col("query_name").alias("anchor_id"),
            pl.col("sequence").alias(f"sequence_{slot}"),
        )
        documents = documents.join(species_rows, on="anchor_id", how="left")
        available = pl.col(f"sequence_{slot}").is_not_null()
        documents = documents.with_columns(
            available.alias(f"available_{slot}"),
            available.alias(f"quality_pass_{slot}"),
            pl.col(f"sequence_{slot}").fill_null(MISSING_SEQUENCE),
        )

    assert documents.filter(~pl.col("available_7")).is_empty(), (
        "human must be available for every assembled anchor"
    )
    assert documents.filter(
        pl.any_horizontal(
            [
                pl.col(f"sequence_{slot}").str.len_chars() != BASES_PER_SLOT
                for slot in range(N_SEQUENCE_SLOTS)
            ]
        )
    ).is_empty()

    sequence_columns = [f"sequence_{slot}" for slot in range(N_SEQUENCE_SLOTS)]
    documents = documents.with_columns(
        pl.lit(species_order_version).alias("species_order_version"),
        pl.lit(CONSERVATION_SOURCE).alias("conservation_source"),
        pl.lit(CONSERVATION_THRESHOLD).alias("conservation_threshold"),
        pl.lit(WINDOW_MIN_PROPORTION_CONSERVED).alias(
            "window_min_proportion_conserved"
        ),
        pl.lit(REFERENCE_ASSEMBLY).alias("reference_assembly"),
        pl.lit(HAL_SOURCE).alias("hal_source"),
        pl.concat_str(sequence_columns, separator=SEQUENCE_BOUNDARY).alias("seq"),
    )

    assert documents.filter(
        pl.col("end") - pl.col("start") != BASES_PER_SLOT
    ).is_empty()
    return documents.sort("anchor_id")


def split_training_validation(
    documents: pl.DataFrame,
    *,
    validation_chrom: str = "18",
    validation_size: int = 2_048,
    validation_seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split by human chromosome and deterministically sample validation loci."""
    required = {"anchor_id", "chrom", "start", "end", "seq"}
    assert required <= set(documents.columns), (
        f"documents missing columns {required - set(documents.columns)}"
    )
    assert documents.select(pl.col("anchor_id").n_unique()).item() == documents.height

    validation_candidates = documents.filter(pl.col("chrom") == validation_chrom)
    assert validation_candidates.height >= validation_size, (
        f"chromosome {validation_chrom} has only {validation_candidates.height} "
        f"eligible anchors; need {validation_size}"
    )
    validation = (
        validation_candidates.with_columns(
            pl.col("anchor_id")
            .map_elements(
                lambda anchor_id: stable_anchor_rank(anchor_id, validation_seed),
                return_dtype=pl.Int64,
            )
            .alias("_validation_rank")
        )
        .sort(["_validation_rank", "anchor_id"])
        .head(validation_size)
        .drop("_validation_rank")
        .with_columns(pl.lit("+").alias("augmentation"))
    )
    training = documents.filter(pl.col("chrom") != validation_chrom)

    assert training.filter(pl.col("chrom") == validation_chrom).is_empty()
    assert validation.height == validation_size
    assert validation.select(pl.col("anchor_id").n_unique()).item() == validation_size
    assert set(validation["chrom"].to_list()) == {validation_chrom}
    assert set(training["anchor_id"].to_list()).isdisjoint(
        validation["anchor_id"].to_list()
    )
    return training, validation


def add_document_reverse_complements(training: pl.DataFrame) -> pl.DataFrame:
    """Double training rows with one consistent whole-document RC orientation."""
    sequence_columns = [f"sequence_{slot}" for slot in range(N_SEQUENCE_SLOTS)]
    assert set(sequence_columns) <= set(training.columns)
    assert "augmentation" not in training.columns

    forward = training.with_columns(pl.lit("+").alias("augmentation"))
    reverse = training.with_columns(
        *[
            reverse_complement_col(pl.col(column)).alias(column)
            for column in sequence_columns
        ],
        pl.lit("-").alias("augmentation"),
    ).with_columns(
        pl.concat_str(sequence_columns, separator=SEQUENCE_BOUNDARY).alias("seq")
    )
    augmented = pl.concat([forward, reverse], how="vertical")

    assert augmented.height == 2 * training.height
    assert (
        augmented.group_by("anchor_id")
        .agg(pl.col("augmentation").sort())
        .filter(pl.col("augmentation") != pl.Series([["+", "-"]]))
        .is_empty()
    )
    return augmented


def _dataset_manifest(
    documents: pl.DataFrame,
    training: pl.DataFrame,
    validation: pl.DataFrame,
    *,
    source_parquet: str,
    species_order: tuple[str, ...],
    species_order_version: str,
    validation_seed: int,
    shuffle_seed: int,
    commit_sha: str,
) -> dict[str, object]:
    """Build an auditable manifest from the materialized document frames."""
    missingness = []
    for slot, species in enumerate(species_order):
        n_available = documents.select(pl.col(f"available_{slot}").sum()).item()
        missingness.append(
            {
                "slot": slot,
                "species": species,
                "n_available": int(n_available),
                "n_missing": documents.height - int(n_available),
                "missing_fraction": 1.0 - int(n_available) / documents.height,
            }
        )
    return {
        "source_parquet": source_parquet,
        "producing_commit": commit_sha,
        "coordinate_system": "0-based, half-open",
        "species_order_version": species_order_version,
        "species_order": list(species_order),
        "n_distinct_anchors": documents.height,
        "n_training_anchors": training.select(pl.col("anchor_id").n_unique()).item(),
        "n_training_documents": training.height,
        "n_validation_anchors": validation.select(
            pl.col("anchor_id").n_unique()
        ).item(),
        "n_validation_documents": validation.height,
        "validation_tokens": validation.height * DOCUMENT_TOKENS,
        "validation_chrom": "18",
        "validation_seed": validation_seed,
        "validation_rank": "blake2b-64-v1",
        "training_shuffle_seed": shuffle_seed,
        "document_tokens": DOCUMENT_TOKENS,
        "next_token_targets": DOCUMENT_TOKENS - 1,
        "human_segment": [HUMAN_SEGMENT_START, DOCUMENT_TOKENS],
        "centered_variant_token_index": HUMAN_VARIANT_TOKEN_INDEX,
        "conservation": {
            "source": CONSERVATION_SOURCE,
            "base_threshold": CONSERVATION_THRESHOLD,
            "window_min_proportion": WINDOW_MIN_PROPORTION_CONSERVED,
        },
        "reference_assembly": REFERENCE_ASSEMBLY,
        "hal_source": HAL_SOURCE,
        "missingness": missingness,
    }


def write_training_dataset_readme(
    output_path: str | Path,
    *,
    manifest: dict[str, object],
    commit_sha: str,
    hf_repo: str,
) -> None:
    """Write the reviewed Hugging Face dataset card for the training corpus."""
    assert len(commit_sha) == 40
    assert manifest["n_validation_documents"] == 2_048
    species_lines = "\n".join(
        f"{row['slot']}. *{row['species'].replace('_', ' ')}* — "
        f"{row['missing_fraction']:.2%} missing"
        for row in manifest["missingness"]
    )
    pipeline_url = (
        f"https://github.com/Open-Athena/marin-dna/tree/{commit_sha}/snakemake/rag_glm"
    )
    text = f"""---
tags:
- biology
- genomics
- dna
---

# {hf_repo}

Fixed-layout 2,048-token documents built from conservation-filtered GRCh38
255-base anchors, seven fixed Zoonomia mammalian ortholog slots, and a final
human slot. Missing non-human projections are filled with 255 `N` bases;
chromosome 18 is validation-only.

Produced by the commit-pinned [issue #402 RAG pipeline]({pipeline_url}). The
immutable upstream input is the existing Zoonomia v1
`min0.20/all_species_with_sequence.parquet` projection. No `halLiftover` was
run for this dataset.

## Splits

| Split | Documents | Distinct human anchors | Orientation |
| --- | ---: | ---: | --- |
| train | {manifest["n_training_documents"]:,} | {manifest["n_training_anchors"]:,} | forward + whole-document reverse complement |
| validation | {manifest["n_validation_documents"]:,} | {manifest["n_validation_anchors"]:,} | forward only |

Training excludes every chromosome 18 anchor. Validation is a deterministic
2,048-anchor chromosome 18 sample (`blake2b-64-v1`, seed
{manifest["validation_seed"]}) and contains exactly
{manifest["validation_tokens"]:,} input tokens.

## Fixed slot order (`{manifest["species_order_version"]}`)

{species_lines}

Slot 7 is always human. Every raw slot is exactly 255 bases. Documents are
assembled as `slot_0[SEQ]... [SEQ]slot_7`; `[SEQ]` is one atomic token and one
BOS token is prepended as CLS. The final human segment is token interval
`[1793, 2048)`. Raw `N` bases map to the tokenizer unknown token.

## Provenance

- Reference: {manifest["reference_assembly"]}
- Conservation: `{CONSERVATION_SOURCE} >= {CONSERVATION_THRESHOLD}` at least
  `{WINDOW_MIN_PROPORTION_CONSERVED:.2f}` of positions in each 255-base window
- Alignment: {manifest["hal_source"]}
- Coordinates: 0-based, half-open
- Producing commit: `{commit_sha}`

`manifest.json` contains exact row counts, seeds, token positions, and
per-slot missingness.
"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def materialize_training_dataset(
    *,
    source_parquet: str,
    training_shard_paths: list[str],
    validation_path: str,
    manifest_path: str,
    readme_path: str,
    species_order: Sequence[str],
    species_order_version: str,
    validation_size: int,
    validation_seed: int,
    shuffle_seed: int,
    commit_sha: str,
    hf_repo: str,
) -> None:
    """Build, validate, and shard the complete RAG training dataset."""
    assert training_shard_paths
    assert len(commit_sha) == 40, f"expected a full commit SHA, got {commit_sha!r}"
    order = validate_species_order(species_order)

    source = pl.scan_parquet(source_parquet)
    if "augmentation" in source.collect_schema().names():
        source = source.filter(pl.col("augmentation") == "+")
    projection_rows = (
        source.filter(pl.col("species").is_in(order))
        .select(sorted(_SOURCE_COLUMNS))
        .collect(engine="streaming")
    )
    documents = assemble_fixed_layout_documents(
        projection_rows,
        species_order=order,
        species_order_version=species_order_version,
    )
    training_forward, validation = split_training_validation(
        documents,
        validation_size=validation_size,
        validation_seed=validation_seed,
    )
    training = add_document_reverse_complements(training_forward).sample(
        fraction=1.0,
        shuffle=True,
        seed=shuffle_seed,
    )

    assert validation.height == 2_048
    assert validation.height * DOCUMENT_TOKENS == 4_194_304
    assert validation.filter(pl.col("augmentation") != "+").is_empty()
    assert training.filter(pl.col("chrom") == "18").is_empty()
    assert training.select(pl.col("anchor_id").n_unique()).item() * 2 == training.height

    from marin_dna_rag_glm.tokenizer import create_rag_char_tokenizer

    tokenizer = create_rag_char_tokenizer()
    validation_sample = validation["seq"].head(32).to_list()
    tokenized = tokenizer(validation_sample)["input_ids"]
    assert all(len(ids) == DOCUMENT_TOKENS for ids in tokenized)

    for path, (start, end) in zip(
        training_shard_paths,
        get_array_split_pairs(training.height, len(training_shard_paths)),
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        training.slice(start, end - start).write_parquet(
            path, compression="zstd", statistics=True
        )
    Path(validation_path).parent.mkdir(parents=True, exist_ok=True)
    validation.write_parquet(validation_path, compression="zstd", statistics=True)

    manifest = _dataset_manifest(
        documents,
        training,
        validation,
        source_parquet=source_parquet,
        species_order=order,
        species_order_version=species_order_version,
        validation_seed=validation_seed,
        shuffle_seed=shuffle_seed,
        commit_sha=commit_sha,
    )
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
    write_training_dataset_readme(
        readme_path,
        manifest=manifest,
        commit_sha=commit_sha,
        hf_repo=hf_repo,
    )
