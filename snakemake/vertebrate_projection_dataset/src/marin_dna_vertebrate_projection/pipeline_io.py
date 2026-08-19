"""Thin, assertion-heavy file wrappers used by the Snakemake pipeline."""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from marin_dna_vertebrate_projection.adapters import (
    hal_records_to_fragments,
)
from marin_dna_vertebrate_projection.contract import (
    ACCEPTED_SCHEMA,
    REJECTION_SCHEMA,
    apply_projection_contract,
)
from marin_dna_vertebrate_projection.inspection import (
    assert_zrs_broad_recovery,
    build_inspection_sample,
    build_rejection_inspection_sample,
    render_inspection_report,
)
from marin_dna_vertebrate_projection.maf import (
    FRAGMENT_SCHEMA,
    iter_projected_anchor_fragments,
)
from marin_dna_vertebrate_projection.manifest import (
    read_species_manifest,
)
from marin_dna_vertebrate_projection.policy import (
    ANCHOR_COLUMNS,
    PROJECTION_REQUEST_COLUMNS,
)
from marin_dna_vertebrate_projection.projection.dataset import reverse_complement_col
from marin_dna_vertebrate_projection.projection.sequence import (
    attach_sequences_to_parquet,
    parquet_to_bed6,
    parse_wrapped_fasta_output,
)
from marin_dna_vertebrate_projection.qc import (
    write_projection_qc_tables_streaming,
)
from marin_dna_vertebrate_projection.split import (
    assign_train_validation_splits,
)


def write_filtered_anchor_bed(
    scored_paths: list[str],
    output_path: str | Path,
    *,
    min_proportion_conserved: float,
) -> None:
    """Write a deterministic, integrity-checked gzip BED of retained anchors."""
    assert scored_paths
    assert 0.0 <= min_proportion_conserved <= 1.0
    scored = pl.concat([pl.read_parquet(path) for path in scored_paths])
    required = {"chrom", "start", "end", "name", "proportion_conserved"}
    assert required <= set(scored.columns)
    kept = scored.filter(
        pl.col("proportion_conserved") >= min_proportion_conserved
    ).select(
        pl.col("chrom").str.strip_prefix("chr"),
        "start",
        "end",
        "name",
    )
    assert 0 < kept.height <= scored.height
    assert kept["name"].n_unique() == kept.height
    assert (kept["start"] >= 0).all()
    assert (kept["end"] > kept["start"]).all()
    assert (~kept["chrom"].str.starts_with("chr")).all()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as temp:
        temporary = Path(temp)
        plain_path = temporary / "anchors.bed"
        compressed_path = temporary / "anchors.bed.gz"
        kept.write_csv(plain_path, separator="\t", include_header=False)
        with (
            plain_path.open("rb") as source,
            compressed_path.open("wb") as raw,
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as destination,
        ):
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        with gzip.open(compressed_path, "rt") as handle:
            written_rows = sum(1 for _ in handle)
        assert written_rows == kept.height
        compressed_path.replace(output)


def read_anchor_catalog(path: str | Path, *, target_length: int = 255) -> pl.DataFrame:
    """Read a TSV/Parquet anchor catalog and assert 0-based half-open invariants."""
    anchor_path = Path(path)
    frame = (
        pl.read_parquet(anchor_path)
        if anchor_path.suffix == ".parquet"
        else pl.read_csv(anchor_path, separator="\t")
    )
    required = set(ANCHOR_COLUMNS)
    missing = required - set(frame.columns)
    assert not missing, f"anchor catalog missing columns: {sorted(missing)}"
    assert frame["query_name"].n_unique() == frame.height
    assert frame["source_chrom"].str.starts_with("chr").all()
    assert (frame["source_start"] >= 0).all()
    assert (frame["source_end"] - frame["source_start"] == target_length).all()
    projection_columns = set(PROJECTION_REQUEST_COLUMNS) - required
    present_projection_columns = projection_columns & set(frame.columns)
    assert (
        not present_projection_columns
        or present_projection_columns == projection_columns
    ), "projection request columns must be supplied together"
    selected_columns = [*ANCHOR_COLUMNS]
    if present_projection_columns:
        assert frame["projection_policy"].n_unique() == 1
        assert (frame["landmark_width"] > 0).all()
        assert (frame["landmark_width"] % 2 == 1).all()
        assert (frame["projection_start"] >= frame["source_start"]).all()
        assert (frame["projection_end"] <= frame["source_end"]).all()
        assert (
            frame["projection_end"] - frame["projection_start"]
            == frame["landmark_width"]
        ).all()
        selected_columns.extend(
            column
            for column in PROJECTION_REQUEST_COLUMNS
            if column not in ANCHOR_COLUMNS
        )
    return frame.select(selected_columns).sort(
        "source_chrom", "source_start", "query_name"
    )


def write_hal_bed6(anchors_path: str | Path, output_path: str | Path) -> None:
    anchors = read_anchor_catalog(anchors_path)
    start_column = (
        "projection_start" if "projection_start" in anchors.columns else "source_start"
    )
    end_column = (
        "projection_end" if "projection_end" in anchors.columns else "source_end"
    )
    anchors.select(
        pl.col("source_chrom"),
        pl.col(start_column),
        pl.col(end_column),
        pl.col("query_name"),
        pl.lit(0).alias("score"),
        pl.lit("+").alias("strand"),
    ).write_csv(output_path, separator="\t", include_header=False)


def write_maf_candidates(
    maf_path: str | Path,
    anchors_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    rows_per_batch: int = 5_000,
) -> None:
    """Stream MAF candidates into species-clustered Parquet row groups.

    Full chromosome MAFs can yield millions of Python fragment records. Small
    per-species buffers keep parsing bounded in memory, and species-clustered
    row groups let downstream per-species contract jobs prune almost all I/O.
    """
    assert rows_per_batch > 0
    anchors = read_anchor_catalog(anchors_path)
    manifest = read_species_manifest(str(manifest_path))
    selected = manifest.filter(
        (pl.col("backend") == "ucsc_multiz100way") & pl.col("selected")
    )
    alignment_names = sorted(selected["alignment_name"].to_list())
    assert alignment_names
    buffers: dict[str, list[dict[str, object]]] = {name: [] for name in alignment_names}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    with TemporaryDirectory(prefix=".maf-fragments-", dir=output.parent) as temp_dir:
        temporary = Path(temp_dir)
        writers: dict[str, pq.ParquetWriter] = {}

        def flush(alignment_name: str) -> None:
            rows = buffers[alignment_name]
            if not rows:
                return
            table = pl.DataFrame(rows, schema=FRAGMENT_SCHEMA).to_arrow()
            writer = writers.get(alignment_name)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary / f"{alignment_name}.parquet",
                    table.schema,
                    compression="zstd",
                    write_statistics=True,
                )
                writers[alignment_name] = writer
            writer.write_table(table)
            rows.clear()

        try:
            for fragment in iter_projected_anchor_fragments(
                maf_path, anchors, manifest
            ):
                alignment_name = str(fragment["alignment_name"])
                assert alignment_name in buffers
                buffers[alignment_name].append(fragment)
                if len(buffers[alignment_name]) >= rows_per_batch:
                    flush(alignment_name)
            for alignment_name in alignment_names:
                flush(alignment_name)
        finally:
            for writer in writers.values():
                writer.close()

        output_writer: pq.ParquetWriter | None = None
        try:
            for alignment_name in alignment_names:
                part_path = temporary / f"{alignment_name}.parquet"
                if not part_path.exists():
                    continue
                for batch in pq.ParquetFile(part_path).iter_batches(
                    batch_size=rows_per_batch
                ):
                    table = pa.Table.from_batches([batch])
                    if output_writer is None:
                        output_writer = pq.ParquetWriter(
                            output,
                            table.schema,
                            compression="zstd",
                            write_statistics=True,
                        )
                    output_writer.write_table(table)
        finally:
            if output_writer is not None:
                output_writer.close()

    if not output.exists():
        pl.DataFrame(schema=FRAGMENT_SCHEMA).write_parquet(output)

    stats = (
        pl.scan_parquet(output)
        .select(
            pl.len().alias("rows"),
            (pl.col("source_fragment_start") < pl.col("source_start"))
            .sum()
            .alias("invalid_source_starts"),
            (pl.col("source_fragment_end") > pl.col("source_end"))
            .sum()
            .alias("invalid_source_ends"),
            (pl.col("t_start") < 0).sum().alias("invalid_target_starts"),
            (pl.col("t_end") > pl.col("t_src_size")).sum().alias("invalid_target_ends"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    assert all(
        int(stats[column]) == 0
        for column in [
            "invalid_source_starts",
            "invalid_source_ends",
            "invalid_target_starts",
            "invalid_target_ends",
        ]
    )


def write_hal_fragments(
    hal_records: pl.DataFrame,
    anchors_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
) -> None:
    anchors = read_anchor_catalog(anchors_path)
    manifest = read_species_manifest(str(manifest_path))
    hal_records_to_fragments(hal_records, anchors, manifest).write_parquet(output_path)


def write_contract_outputs(
    fragments_path: str | Path,
    accepted_path: str | Path,
    rejected_path: str | Path,
    *,
    target_length: int,
    pre_resize_min_length: int,
    pre_resize_max_length: int,
) -> None:
    """Apply the vectorized/fragmented contract in one read of the Parquet."""
    schema = pl.read_parquet_schema(fragments_path)
    missing = set(FRAGMENT_SCHEMA) - set(schema)
    assert not missing, f"projection fragments missing columns: {sorted(missing)}"
    fragments = pl.read_parquet(fragments_path)
    result = apply_projection_contract(
        fragments,
        target_length=target_length,
        pre_resize_min_length=pre_resize_min_length,
        pre_resize_max_length=pre_resize_max_length,
    )

    accepted_output = Path(accepted_path)
    rejected_output = Path(rejected_path)
    accepted_output.parent.mkdir(parents=True, exist_ok=True)
    rejected_output.parent.mkdir(parents=True, exist_ok=True)
    result.accepted.write_parquet(accepted_output)
    result.rejected.write_parquet(rejected_output)


def write_contract_outputs_for_alignment(
    fragments_path: str | Path,
    alignment_name: str,
    accepted_path: str | Path,
    rejected_path: str | Path,
    *,
    target_length: int,
    pre_resize_min_length: int,
    pre_resize_max_length: int,
) -> None:
    """Apply the shared contract to one species from clustered MAF fragments."""
    fragments = (
        pl.scan_parquet(fragments_path)
        .filter(pl.col("alignment_name") == alignment_name)
        .collect(engine="streaming")
    )
    result = apply_projection_contract(
        fragments,
        target_length=target_length,
        pre_resize_min_length=pre_resize_min_length,
        pre_resize_max_length=pre_resize_max_length,
    )
    Path(accepted_path).parent.mkdir(parents=True, exist_ok=True)
    Path(rejected_path).parent.mkdir(parents=True, exist_ok=True)
    result.accepted.write_parquet(accepted_path)
    result.rejected.write_parquet(rejected_path)


def merge_parquets_streaming(input_paths: list[str], output_path: str | Path) -> None:
    """Concatenate schema-identical Parquets with bounded peak memory."""
    assert input_paths
    schemas = [pl.read_parquet_schema(path) for path in input_paths]
    assert all(schema == schemas[0] for schema in schemas)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(
        [pl.scan_parquet(path) for path in input_paths], how="vertical"
    ).sink_parquet(output)


def write_twobit_sequences(
    accepted_path: str | Path,
    two_bit_path: str | Path,
    sequence_path: str | Path,
    rejected_path: str | Path,
    *,
    target_length: int = 255,
) -> None:
    """Extract one genome's accepted rows in a single compiled twoBitToFa call."""
    accepted = Path(accepted_path)
    two_bit = Path(two_bit_path)
    sequence_output = Path(sequence_path)
    rejection_output = Path(rejected_path)
    assert accepted.is_file()
    assert two_bit.is_file()
    sequence_output.parent.mkdir(parents=True, exist_ok=True)
    rejection_output.parent.mkdir(parents=True, exist_ok=True)

    expected_names = pl.read_parquet(accepted, columns=["query_name"])[
        "query_name"
    ].to_list()
    with TemporaryDirectory(
        prefix=".twobit-sequences-", dir=sequence_output.parent
    ) as temp_dir:
        temporary = Path(temp_dir)
        bed_path = temporary / "intervals.bed"
        fasta_path = temporary / "sequences.fa"
        row_count = parquet_to_bed6(accepted, bed_path)
        assert row_count == len(expected_names)
        if row_count:
            subprocess.run(
                [
                    "twoBitToFa",
                    str(two_bit),
                    f"-bed={bed_path}",
                    str(fasta_path),
                ],
                check=True,
            )
            records = parse_wrapped_fasta_output(fasta_path)
            observed_names = [name for name, _sequence in records]
            assert observed_names == expected_names, (
                "twoBitToFa output order/names differ from the accepted Parquet"
            )
            sequences = [sequence for _name, sequence in records]
        else:
            sequences = []
        written = attach_sequences_to_parquet(
            accepted,
            sequences,
            sequence_output,
            target_len=target_length,
        )
        assert written == row_count

    # Coordinates were bounds-checked by the projection contract. A missing or
    # short sequence is therefore corruption and fails above instead of being
    # silently removed from the dataset.
    pl.DataFrame(schema=REJECTION_SCHEMA).write_parquet(rejection_output)


def write_human_reference_sequences(
    anchors_path: str | Path,
    two_bit_path: str | Path,
    chrom_sizes_path: str | Path,
    output_path: str | Path,
    *,
    target_length: int = 255,
) -> None:
    anchors = read_anchor_catalog(anchors_path, target_length=target_length)
    sizes = pl.read_csv(
        chrom_sizes_path,
        separator="\t",
        has_header=False,
        new_columns=["source_chrom", "t_src_size"],
        schema_overrides={"source_chrom": pl.String, "t_src_size": pl.Int64},
    )
    assert sizes["source_chrom"].n_unique() == sizes.height
    accepted = (
        anchors.join(sizes, on="source_chrom", how="inner", validate="m:1")
        .with_columns(
            pl.lit("Homo sapiens").alias("species"),
            pl.lit("hg38").alias("alignment_name"),
            pl.lit("hg38").alias("assembly"),
            pl.lit(9606, dtype=pl.Int64).alias("taxonomy_id"),
            pl.lit("Hominidae").alias("family"),
            pl.lit("mammals").alias("clade"),
            pl.lit(0, dtype=pl.Int64).alias("phylogenetic_rank"),
            pl.lit("human_reference").alias("alignment_source"),
            pl.col("source_chrom").alias("t_chrom"),
            pl.col("source_start").alias("t_start"),
            pl.col("source_end").alias("t_end"),
            pl.lit("+").alias("t_strand"),
            pl.col("source_start").alias("pre_resize_t_start"),
            pl.col("source_end").alias("pre_resize_t_end"),
            pl.lit(1, dtype=pl.Int64).alias("fragment_count"),
            pl.lit(target_length, dtype=pl.Int64).alias("aligned_bases"),
        )
        .select(ACCEPTED_SCHEMA.names())
        .cast(ACCEPTED_SCHEMA)
        .sort("query_name")
    )
    assert accepted.height == anchors.height
    assert (accepted["t_end"] <= accepted["t_src_size"]).all()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".human-reference-", dir=output.parent) as temp_dir:
        temporary = Path(temp_dir)
        accepted_path = temporary / "accepted.parquet"
        rejected_path = temporary / "rejected.parquet"
        accepted.write_parquet(accepted_path)
        write_twobit_sequences(
            accepted_path,
            two_bit_path,
            output,
            rejected_path,
            target_length=target_length,
        )
        assert pl.read_parquet(rejected_path).is_empty()


def combine_sequence_parquets(input_paths: list[str], output_path: str | Path) -> None:
    assert input_paths
    schemas = [pl.read_parquet_schema(path) for path in input_paths]
    assert all(schema == schemas[0] for schema in schemas)
    observed_species: set[str] = set()
    expected_rows = 0
    for path in input_paths:
        stats = (
            pl.scan_parquet(path)
            .select(
                pl.len().alias("rows"),
                pl.col("query_name").n_unique().alias("queries"),
                pl.col("species").n_unique().alias("species_count"),
                pl.col("species").first().alias("species"),
                (pl.col("sequence").str.len_bytes() != 255)
                .sum()
                .alias("invalid_lengths"),
            )
            .collect(engine="streaming")
            .row(0, named=True)
        )
        rows = int(stats["rows"])
        assert rows == int(stats["queries"])
        assert int(stats["invalid_lengths"]) == 0
        if rows == 0:
            continue
        assert int(stats["species_count"]) == 1, (
            f"{path} contains {stats['species_count']} species"
        )
        species = str(stats["species"])
        assert species not in observed_species
        observed_species.add(species)
        expected_rows += rows

    assert expected_rows > 0
    merge_parquets_streaming(input_paths, output_path)
    actual_rows = (
        pl.scan_parquet(output_path).select(pl.len()).collect(engine="streaming").item()
    )
    assert actual_rows == expected_rows


def write_dataset_split_files(
    combined_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    selection_path: str | Path,
    species_counts_path: str | Path,
    summary_path: str | Path,
    *,
    region_label: str,
    species_scope: str = "all",
    add_rc: bool,
    validation_chrom: str,
    max_validation_rows: int,
    seed: int,
) -> None:
    assert species_scope in {"all", "mammals_only"}
    original = pl.scan_parquet(combined_path)
    if region_label != "all":
        original = original.filter(pl.col("region_label") == region_label)
    if species_scope == "mammals_only":
        original = original.filter(
            pl.col("alignment_source").is_in(["human_reference", "zoonomia_cactus"])
        )
    original_rows = original.select(pl.len()).collect(engine="streaming").item()
    assert original_rows > 0, (
        f"empty dataset cohort: region={region_label}, scope={species_scope}"
    )

    train_original = original.filter(pl.col("source_chrom") != validation_chrom)
    if add_rc:
        train = pl.concat(
            [
                train_original.with_columns(pl.lit("+").alias("augmentation")),
                train_original.with_columns(
                    reverse_complement_col(pl.col("sequence")).alias("sequence"),
                    pl.lit("-").alias("augmentation"),
                ),
            ],
            how="vertical",
        )
    else:
        train = train_original.with_columns(pl.lit("+").alias("augmentation"))

    for path in [train_path, validation_path, selection_path, species_counts_path]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    train.sink_parquet(train_path)

    candidates = (
        original.filter(pl.col("source_chrom") == validation_chrom)
        .collect(engine="streaming")
        .with_columns(pl.lit("+").alias("augmentation"))
    )
    result = assign_train_validation_splits(
        candidates,
        validation_chrom=validation_chrom,
        max_validation_rows=max_validation_rows,
        seed=seed,
    )
    assert result.train.is_empty()
    result.validation.drop("row_id").write_parquet(validation_path)
    result.selection_manifest.write_csv(selection_path, separator="\t")
    result.species_counts.write_csv(species_counts_path, separator="\t")
    train_rows = (
        pl.scan_parquet(train_path).select(pl.len()).collect(engine="streaming").item()
    )
    Path(summary_path).write_text(
        json.dumps(
            {
                "region_label": region_label,
                "species_scope": species_scope,
                "seed": seed,
                "validation_chrom": validation_chrom,
                "train_rows": train_rows,
                "eligible_validation_rows": candidates.height,
                "validation_rows": result.validation.height,
                "realized_token_count": result.realized_token_count,
            },
            indent=2,
        )
        + "\n"
    )


def write_qc_files(
    anchors_path: str | Path,
    accepted_path: str | Path,
    rejected_paths: list[str],
    manifest_path: str | Path,
    per_anchor_path: str | Path,
    per_scope_path: str | Path,
    rejections_path: str | Path,
    aggregates_path: str | Path,
    *,
    validation_chrom: str,
) -> None:
    anchors = read_anchor_catalog(anchors_path).with_columns(
        pl.when(pl.col("source_chrom") == validation_chrom)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("train"))
        .alias("split")
    )
    write_projection_qc_tables_streaming(
        anchors,
        accepted_path,
        rejected_paths,
        read_species_manifest(str(manifest_path)),
        per_anchor_path,
        per_scope_path,
        rejections_path,
        aggregates_path,
    )


def write_inspection_files(
    sequences_path: str | Path,
    rejected_paths: list[str],
    sample_path: str | Path,
    rejected_sample_path: str | Path,
    report_path: str | Path,
    *,
    seed: int,
    rows_per_region: int,
    fragmented_rows: int,
    rejected_rows_per_reason: int,
    require_zrs: bool = True,
) -> None:
    """Write bounded-memory deterministic samples and a pending review report."""
    assert rows_per_region > 0
    assert fragmented_rows >= 0
    assert rejected_rows_per_reason > 0
    sequences = pl.scan_parquet(sequences_path)
    identity_hash = pl.concat_str(
        [
            pl.col("query_name"),
            pl.col("species"),
            pl.col("alignment_source"),
            pl.col("t_chrom"),
            pl.col("t_start").cast(pl.String),
        ],
        separator="\t",
    ).hash(seed=seed)

    candidate_query_names: set[str] = set()
    zrs_rows = sequences.filter(
        pl.col("query_name").str.to_lowercase().str.starts_with("zrs_")
    ).collect(engine="streaming")
    if require_zrs:
        assert_zrs_broad_recovery(zrs_rows)
    candidate_query_names.update(zrs_rows["query_name"].unique().to_list())

    # Select a small set of candidate anchors with Polars' bounded top-k, then
    # materialize every species for only those anchors. The existing pure
    # sampler makes the final row choices and computes complete clade counts.
    candidate_multiplier = 4
    for region_label in ["cds", "ccre_non_promoter"]:
        candidates = (
            sequences.filter(pl.col("region_label") == region_label)
            .select("query_name", identity_hash.alias("_sample_hash"))
            .bottom_k(
                rows_per_region * candidate_multiplier,
                by="_sample_hash",
            )
            .collect(engine="streaming")
        )
        assert candidates.height > 0, (
            f"inspection requires recovered {region_label} rows"
        )
        candidate_query_names.update(candidates["query_name"].to_list())

    if fragmented_rows:
        fragmented_candidates = (
            sequences.filter(pl.col("fragment_count") > 1)
            .select("query_name", identity_hash.alias("_sample_hash"))
            .bottom_k(
                fragmented_rows * candidate_multiplier,
                by="_sample_hash",
            )
            .collect(engine="streaming")
        )
        candidate_query_names.update(fragmented_candidates["query_name"].to_list())

    assert candidate_query_names
    inspection_rows = sequences.filter(
        pl.col("query_name").is_in(candidate_query_names)
    ).collect(engine="streaming")
    sample = build_inspection_sample(
        inspection_rows,
        seed=seed,
        rows_per_region=rows_per_region,
        fragmented_rows=fragmented_rows,
    )
    assert rejected_paths
    rejected = pl.concat(
        [pl.scan_parquet(path) for path in rejected_paths], how="vertical"
    )
    rejection_reasons = (
        rejected.select("rejection_reason")
        .unique()
        .collect(engine="streaming")["rejection_reason"]
        .sort()
        .to_list()
    )
    rejected_samples: list[pl.DataFrame] = []
    rejected_hash = pl.concat_str(
        [
            pl.col("query_name"),
            pl.col("species"),
            pl.col("alignment_source"),
            pl.col("source_chrom"),
            pl.col("source_start").cast(pl.String),
        ],
        separator="\t",
    ).hash(seed=seed)
    for rejection_reason in rejection_reasons:
        rejected_samples.append(
            rejected.filter(pl.col("rejection_reason") == rejection_reason)
            .with_columns(rejected_hash.alias("_sample_hash"))
            .sort(
                "fragment_count",
                "_sample_hash",
                descending=[True, False],
            )
            .limit(rejected_rows_per_reason)
            .drop("_sample_hash")
            .collect(engine="streaming")
        )
    rejected_candidates = (
        pl.concat(rejected_samples, how="vertical")
        if rejected_samples
        else pl.DataFrame(schema=pl.read_parquet_schema(rejected_paths[0]))
    )
    rejected_sample = build_rejection_inspection_sample(
        rejected_candidates,
        seed=seed,
        rows_per_reason=rejected_rows_per_reason,
    )
    for path in [sample_path, rejected_sample_path, report_path]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    sample.write_csv(sample_path, separator="\t")
    rejected_sample.write_csv(rejected_sample_path, separator="\t")
    Path(report_path).write_text(
        render_inspection_report(
            sample, rejected_sample, seed=seed, require_zrs=require_zrs
        )
    )


def write_dataset_card(
    train_path: str | Path,
    validation_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    pipeline_commit: str,
    hf_repo: str,
    region_label: str,
    species_scope: str,
) -> None:
    """Write the reviewable HF README required before any upload."""
    assert len(pipeline_commit) == 40, "dataset cards require a commit-pinned SHA"
    train_rows = (
        pl.scan_parquet(train_path).select(pl.len()).collect(engine="streaming").item()
    )
    validation_rows = (
        pl.scan_parquet(validation_path)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    train_schema = pl.read_parquet_schema(train_path)
    assert train_schema == pl.read_parquet_schema(validation_path)
    assert species_scope in {"all", "mammals_only"}
    selected = read_species_manifest(str(manifest_path)).filter(pl.col("selected"))
    if species_scope == "mammals_only":
        selected = selected.filter(pl.col("backend") == "zoonomia_cactus")
    species_counts = (
        selected.group_by("backend", "clade")
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
    source_description = (
        "the Zoonomia 447-mammal Cactus alignment"
        if species_scope == "mammals_only"
        else (
            "the Zoonomia 447-mammal Cactus alignment and UCSC hg38 "
            "MultiZ 100-way alignment"
        )
    )
    text = f"""---
tags:
- biology
- genomics
- dna
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/*.jsonl.zst
  - split: validation
    path: data/validation/*.jsonl.zst
---

# `{hf_repo}`

Human-anchored 255 bp vertebrate sequences from {source_description}. This
draft covers the `{region_label}` region cohort with `{species_scope}` species
scope and preserves source FASTA/2bit letter case.

Anchor eligibility uses the pipeline's pinned phyloP conservation filter.
Sequence case is independent of that filter: lowercase bases preserve source
repeat masking, uppercase bases preserve source non-repeat-masked sequence, and
conservation scores never rewrite emitted characters or case.

Produced by the [commit-pinned vertebrate projection pipeline]({pipeline_url}).

## Splits

- `train`: {train_rows:,} rows; no chromosome-18 source anchors.
- `validation`: {validation_rows:,} original-orientation chromosome-18 rows
  ({validation_rows * 256:,} tokens including BOS).

The selected target manifest contains {selected.height:,} family-deduplicated
projection targets; human reference rows are added separately once per anchor.

| Projection backend | Clade | Selected species |
|---|---|---:|
{species_lines}

## Schema

{schema_lines}
"""
    Path(output_path).write_text(text)
