"""Materialize Complex-traits and SGE ortholog-RAG evaluation harnesses.

Both source datasets use 1-based GRCh38 SNV positions.  This module converts
them once at the dataset-build boundary, extracts exact 255-base human windows,
and assembles two fully materialized strand rows per source variant after the
same direct-HAL projection used by issue #402's Mendelian harness.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

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
from marin_dna_rag_glm.mendelian_harness import PROJECTION_VERSION

VARIANT_KEY_COLUMNS = ["chrom", "pos", "ref", "alt"]
SUPPORTED_BENCHMARKS = {"complex_traits", "sge"}
BENCHMARK_REQUIRED_COLUMNS = {
    "complex_traits": {"label", "subset", "match_group"},
    "sge": {"label", "subset", "mavedb_urn", "gene"},
}
BENCHMARK_SCORE_PROTOCOL = {
    "complex_traits": "abs_llr",
    "sge": "minus_llr",
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


def _normalize_source_variants(rows: pl.DataFrame, benchmark: str) -> pl.DataFrame:
    assert benchmark in SUPPORTED_BENCHMARKS
    required = {*VARIANT_KEY_COLUMNS, *BENCHMARK_REQUIRED_COLUMNS[benchmark]}
    missing = required - set(rows.columns)
    assert not missing, f"{benchmark} source is missing columns {sorted(missing)}"
    assert rows.height > 0
    assert rows.filter(pl.col("pos") <= 0).is_empty(), (
        "source pos must be 1-based positive"
    )
    assert rows.filter(pl.col("ref").str.len_chars() != 1).is_empty()
    assert rows.filter(pl.col("alt").str.len_chars() != 1).is_empty()
    assert rows.filter(pl.col("ref") == pl.col("alt")).is_empty()
    assert rows.filter(~pl.col("ref").str.to_uppercase().is_in(list("ACGT"))).is_empty()
    assert rows.filter(~pl.col("alt").str.to_uppercase().is_in(list("ACGT"))).is_empty()
    assert rows.select(VARIANT_KEY_COLUMNS).unique().height == rows.height
    assert rows.filter(pl.col("label").is_null()).is_empty()
    assert rows.filter(pl.col("subset").is_null()).is_empty()
    return rows.with_columns(
        pl.col("chrom").cast(pl.String).str.replace(r"^chr", "").alias("chrom"),
        pl.col("ref").str.to_uppercase().alias("ref"),
        pl.col("alt").str.to_uppercase().alias("alt"),
    )


def build_benchmark_variant_windows(
    *,
    benchmark: str,
    source_urls: Mapping[str, str],
    human_twobit_path: str | Path,
    variants_path: str | Path,
    bed_path: str | Path,
) -> int:
    """Extract unique exact human windows and write a HAL BED6.

    Source ``pos`` is converted from 1-based to ``variant_pos0``.  Every
    derived interval is 0-based, half-open ``[variant_pos0-127,
    variant_pos0+128)``.
    """
    import py2bit

    assert benchmark in SUPPORTED_BENCHMARKS
    assert set(source_urls) == {"train", "test"}
    split_variants: list[pl.DataFrame] = []
    for split in ("train", "test"):
        rows = _normalize_source_variants(
            pl.read_parquet(source_urls[split]), benchmark
        )
        split_variants.append(
            rows.select(VARIANT_KEY_COLUMNS).with_columns(pl.lit(split).alias("split"))
        )

    with_splits = pl.concat(split_variants)
    overlap = with_splits.group_by(VARIANT_KEY_COLUMNS).agg(
        pl.col("split").n_unique().alias("n_splits")
    )
    assert overlap.filter(pl.col("n_splits") != 1).is_empty(), (
        f"{benchmark} train/test variants must be disjoint"
    )
    variants = (
        with_splits.drop("split")
        .sort(VARIANT_KEY_COLUMNS)
        .with_columns(
            _variant_id_expr().alias("variant_id"),
            (pl.col("pos") - 1).alias("variant_pos0"),
        )
        .with_columns(
            (pl.col("variant_pos0") - BASES_PER_SLOT // 2).alias("human_start"),
            (pl.col("variant_pos0") + BASES_PER_SLOT // 2 + 1).alias("human_end"),
            pl.lit(PROJECTION_VERSION).alias("projection_version"),
        )
        .with_columns(
            pl.Series(
                "query_name",
                [f"{benchmark}_{index:08d}" for index in range(with_splits.height)],
                dtype=pl.String,
            )
        )
    )
    assert variants["variant_id"].n_unique() == variants.height
    assert variants["query_name"].n_unique() == variants.height
    assert variants.filter(pl.col("human_start") < 0).is_empty()
    assert variants.filter(
        pl.col("human_end") - pl.col("human_start") != BASES_PER_SLOT
    ).is_empty()

    genome = py2bit.open(str(human_twobit_path))
    try:
        chrom_sizes = genome.chroms()
        sequences: list[str] = []
        for chrom, start, end in variants.select(
            "chrom", "human_start", "human_end"
        ).iter_rows():
            twobit_chrom = f"chr{chrom}"
            assert twobit_chrom in chrom_sizes, f"human 2bit is missing {twobit_chrom}"
            assert 0 <= start < end <= chrom_sizes[twobit_chrom]
            sequence = genome.sequence(twobit_chrom, start, end)
            assert sequence is not None
            sequences.append(sequence.upper())
    finally:
        genome.close()

    variants = variants.with_columns(
        pl.Series("human_reference_sequence", sequences, dtype=pl.String)
    )
    assert variants.filter(
        pl.col("human_reference_sequence").str.len_chars() != BASES_PER_SLOT
    ).is_empty()
    assert variants.filter(
        pl.col("human_reference_sequence").str.slice(BASES_PER_SLOT // 2, 1)
        != pl.col("ref")
    ).is_empty(), "source REF does not match the archived GRCh38/HAL human sequence"

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


def _reverse_complement_expr(column: str) -> pl.Expr:
    return pl.col(column).map_elements(reverse_complement, return_dtype=pl.String)


def materialize_benchmark_split(
    *,
    benchmark: str,
    source_url: str,
    variants: pl.DataFrame,
    species_sequences: Mapping[str, pl.DataFrame],
    species_order: tuple[str, ...],
) -> pl.DataFrame:
    """Assemble both materialized strand rows for one benchmark split."""
    source = _normalize_source_variants(pl.read_parquet(source_url), benchmark)
    rows = source.join(
        variants.select(
            *VARIANT_KEY_COLUMNS,
            "variant_id",
            "query_name",
            "variant_pos0",
            "human_start",
            "human_end",
            "human_reference_sequence",
            "projection_version",
        ),
        on=VARIANT_KEY_COLUMNS,
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
        rows = rows.join(sequences, on="query_name", how="left").with_columns(
            pl.col(f"_forward_sequence_{slot}")
            .is_not_null()
            .alias(f"available_{slot}"),
            pl.col(f"_forward_sequence_{slot}")
            .is_not_null()
            .alias(f"quality_pass_{slot}"),
        )

    rows = pl.concat(
        [
            rows.with_columns(pl.lit("+").alias("strand")),
            rows.with_columns(pl.lit("-").alias("strand")),
        ]
    )
    for slot in range(len(species_order) - 1):
        rows = rows.with_columns(
            pl.when(pl.col("strand") == "-")
            .then(_reverse_complement_expr(f"_forward_sequence_{slot}"))
            .otherwise(pl.col(f"_forward_sequence_{slot}"))
            .fill_null(MISSING_SEQUENCE)
            .alias(f"sequence_{slot}")
        ).drop(f"_forward_sequence_{slot}")

    rows = rows.with_columns(
        pl.when(pl.col("strand") == "-")
        .then(_reverse_complement_expr("human_reference_sequence"))
        .otherwise(pl.col("human_reference_sequence"))
        .alias("sequence_7"),
        pl.when(pl.col("strand") == "-")
        .then(_reverse_complement_expr("alt"))
        .otherwise(pl.col("alt"))
        .alias("_oriented_alt"),
    )
    context_parts: list[pl.Expr] = []
    for slot in range(len(species_order) - 1):
        context_parts.extend([pl.col(f"sequence_{slot}"), pl.lit(SEQUENCE_BOUNDARY)])
    context_parts.append(pl.col("sequence_7").str.slice(0, BASES_PER_SLOT // 2))
    rows = rows.with_columns(
        pl.concat_str(context_parts).alias("context"),
        pl.col("sequence_7").str.slice(BASES_PER_SLOT // 2).alias("ref_completion"),
        (
            pl.col("_oriented_alt")
            + pl.col("sequence_7").str.slice(BASES_PER_SLOT // 2 + 1)
        ).alias("alt_completion"),
        pl.col("label").alias("target"),
        pl.lit(True).alias("available_7"),
        pl.lit(True).alias("quality_pass_7"),
        pl.col("chrom").alias("projection_chrom_7"),
        pl.col("human_start").alias("projection_start_7"),
        pl.col("human_end").alias("projection_end_7"),
        pl.col("strand").alias("projection_strand_7"),
        pl.lit(SPECIES_ORDER_VERSION).alias("species_order_version"),
        (
            pl.lit(benchmark + "|")
            + pl.col("variant_id")
            + pl.lit("|")
            + pl.col("strand")
        ).alias("document_id"),
    ).drop("_oriented_alt")

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
    assert rows.filter(
        pl.col("ref_completion").str.slice(1) != pl.col("alt_completion").str.slice(1)
    ).is_empty()
    assert rows.filter(
        pl.col("ref_completion").str.slice(0, 1)
        == pl.col("alt_completion").str.slice(0, 1)
    ).is_empty()
    assert rows["document_id"].n_unique() == rows.height
    strand_counts = rows.group_by(VARIANT_KEY_COLUMNS).agg(
        pl.len().alias("n_rows"), pl.col("strand").n_unique().alias("n_strands")
    )
    assert strand_counts.filter(
        (pl.col("n_rows") != 2) | (pl.col("n_strands") != 2)
    ).is_empty()

    from marin_dna_rag_glm.tokenizer import create_rag_char_tokenizer

    tokenizer = create_rag_char_tokenizer()
    sample = rows.head(16)
    for completion in ("ref_completion", "alt_completion"):
        assert all(
            len(ids) == DOCUMENT_TOKENS
            for ids in tokenizer((sample["context"] + sample[completion]).to_list())[
                "input_ids"
            ]
        )
    return rows.sort([*VARIANT_KEY_COLUMNS, "strand"])


def write_benchmark_harness_readme(
    output_path: str | Path,
    *,
    benchmark: str,
    commit_sha: str,
    hf_repo: str,
    source_revision: str,
    leaderboard_url: str,
    manifest: Mapping[str, object],
) -> None:
    """Write the reviewed Hugging Face dataset card for one benchmark."""
    assert benchmark in SUPPORTED_BENCHMARKS
    assert len(commit_sha) == 40
    title = "Complex-traits" if benchmark == "complex_traits" else "SGE"
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

Retrieval-conditioned, eval-harness-ready {title} SNV benchmark for the
[MarinDNA {title} leaderboard]({leaderboard_url}). Each row contains seven
fully materialized Zoonomia ortholog slots, the shared 127-base human prefix,
and separate reference/alternate completions. Every source variant has forward
and reverse-complement rows.

Produced by the commit-pinned [issue #402 RAG pipeline](https://github.com/Open-Athena/marin-dna/tree/{commit_sha}/snakemake/rag_glm).

| Split | Source variants | Materialized rows |
| --- | ---: | ---: |
| train | {split_variants["train"]:,} | {split_rows["train"]:,} |
| test | {split_variants["test"]:,} | {split_rows["test"]:,} |

## Frozen document contract

- Species order/version: `{SPECIES_ORDER_VERSION}`; human is slot 7.
- Projection/version: `{PROJECTION_VERSION}`.
- `context` is 1,919 tokens before BOS: seven 255-base non-human slots,
  seven atomic `[SEQ]` boundaries, and the shared 127-base human prefix.
- Each completion is 128 bases; BOS + context + completion is exactly 2,048 tokens.
- The centered SNV is absolute token index {HUMAN_VARIANT_TOKEN_INDEX}.
- Missing projections are exactly 255 `N` bases.

Every exact unique human window was projected directly with
`halLiftover --noDupes` against the pinned Zoonomia 447-mammalian 2022 v1 HAL,
then filtered and strand-normalized with the same issue #402 quality contract.
Model scoring needs only this pinned dataset and a checkpoint.

## Provenance

- Source: `marin-dna/evals_{benchmark}` at `{source_revision}` (GRCh38,
  source `pos` is 1-based).
- Score protocol: `{BENCHMARK_SCORE_PROTOCOL[benchmark]}`.
- Alignment source leaf: `Homo_sapiens`.
- Derived genomic coordinates: 0-based, half-open.

`manifest.json` records exact revisions, counts, projection settings, and
per-species availability. All original source columns are retained.
"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def materialize_benchmark_rag_harness(
    *,
    benchmark: str,
    source_urls: Mapping[str, str],
    source_revision: str,
    variants_path: str | Path,
    species_sequence_paths: Mapping[str, str | Path],
    output_split_paths: Mapping[str, str | Path],
    manifest_path: str | Path,
    readme_path: str | Path,
    commit_sha: str,
    hf_repo: str,
    leaderboard_url: str,
    species_order: Sequence[str] = PROVISIONAL_SPECIES_ORDER,
) -> None:
    """Assemble, validate, and describe both splits of one RAG benchmark."""
    assert benchmark in SUPPORTED_BENCHMARKS
    assert len(commit_sha) == 40
    assert set(source_urls) == {"train", "test"}
    assert set(output_split_paths) == {"train", "test"}
    order = validate_species_order(species_order)
    assert tuple(species_sequence_paths) == order[:-1]
    variants = pl.read_parquet(variants_path)
    assert variants["query_name"].n_unique() == variants.height
    species_sequences = {
        species: pl.read_parquet(path)
        for species, path in species_sequence_paths.items()
    }

    materialized: dict[str, pl.DataFrame] = {}
    for split in ("train", "test"):
        rows = materialize_benchmark_split(
            benchmark=benchmark,
            source_url=source_urls[split],
            variants=variants,
            species_sequences=species_sequences,
            species_order=order,
        )
        output = Path(output_split_paths[split])
        output.parent.mkdir(parents=True, exist_ok=True)
        rows.write_parquet(output, compression="zstd", statistics=True)
        materialized[split] = rows

    all_forward = pl.concat(
        [rows.filter(pl.col("strand") == "+") for rows in materialized.values()],
        how="diagonal_relaxed",
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
        "benchmark": benchmark,
        "source_repo": f"marin-dna/evals_{benchmark}",
        "source_revision": source_revision,
        "score_protocol": BENCHMARK_SCORE_PROTOCOL[benchmark],
        "projection_version": PROJECTION_VERSION,
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
        "coordinate_system": "source pos is 1-based; all derived intervals are 0-based, half-open",
        "projection": {
            "alignment": "Zoonomia 447-mammalian 2022 v1 HAL",
            "source_species": "Homo_sapiens",
            "hal_liftover_no_dupes": True,
            "pre_resize_min_len": 128,
            "pre_resize_max_len": 512,
            "target_len": BASES_PER_SLOT,
            "filter": "single target chromosome and strand per query",
        },
        "missingness": missingness,
    }
    manifest_output = Path(manifest_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")
    write_benchmark_harness_readme(
        readme_path,
        benchmark=benchmark,
        commit_sha=commit_sha,
        hf_repo=hf_repo,
        source_revision=source_revision,
        leaderboard_url=leaderboard_url,
        manifest=manifest,
    )
