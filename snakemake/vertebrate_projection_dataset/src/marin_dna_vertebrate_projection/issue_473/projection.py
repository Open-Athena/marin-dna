"""Issue #473 adapters that leave the established projection path untouched."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from marin_dna_vertebrate_projection.issue_473.contract import (
    apply_projection_contract,
)
from marin_dna_vertebrate_projection.issue_473.policy import (
    ANCHOR_COLUMNS,
    PROJECTION_REQUEST_COLUMNS,
)
from marin_dna_vertebrate_projection.maf import (
    FRAGMENT_SCHEMA,
    iter_projected_anchor_fragments,
)
from marin_dna_vertebrate_projection.manifest import read_species_manifest


def read_projection_requests(
    path: str | Path, *, target_length: int = 255
) -> pl.DataFrame:
    """Read issue-specific requests and validate both coordinate roles."""
    request_path = Path(path)
    frame = (
        pl.read_parquet(request_path)
        if request_path.suffix == ".parquet"
        else pl.read_csv(request_path, separator="\t")
    )
    missing = set(PROJECTION_REQUEST_COLUMNS) - set(frame.columns)
    assert not missing, f"projection requests missing columns: {sorted(missing)}"
    assert frame["query_name"].n_unique() == frame.height
    assert frame["source_chrom"].str.starts_with("chr").all()
    assert (frame["source_start"] >= 0).all()
    assert (frame["source_end"] - frame["source_start"] == target_length).all()
    assert frame["projection_policy"].n_unique() == 1
    assert (frame["landmark_width"] > 0).all()
    assert (frame["landmark_width"] % 2 == 1).all()
    assert (frame["projection_start"] >= frame["source_start"]).all()
    assert (frame["projection_end"] <= frame["source_end"]).all()
    assert (
        frame["projection_end"] - frame["projection_start"] == frame["landmark_width"]
    ).all()
    return frame.select(PROJECTION_REQUEST_COLUMNS).sort(
        "source_chrom", "source_start", "query_name"
    )


def write_hal_request_bed6(requests_path: str | Path, output_path: str | Path) -> None:
    """Write the policy landmark, not the 255 bp identity anchor, to HAL BED6."""
    requests = read_projection_requests(requests_path)
    requests.select(
        pl.col("source_chrom"),
        pl.col("projection_start"),
        pl.col("projection_end"),
        pl.col("query_name"),
        pl.lit(0).alias("score"),
        pl.lit("+").alias("strand"),
    ).write_csv(output_path, separator="\t", include_header=False)


def iter_maf_projection_request_fragments(
    maf_path: str | Path,
    requests: pl.DataFrame,
    species_manifest: pl.DataFrame,
    *,
    source_alignment_name: str = "hg38",
) -> Iterator[dict[str, object]]:
    """Project policy landmarks through MAF while retaining 255 bp identities."""
    missing = set(PROJECTION_REQUEST_COLUMNS) - set(requests.columns)
    assert not missing, f"projection requests missing columns: {sorted(missing)}"
    anchors_by_name = {
        str(row["query_name"]): row
        for row in requests.select(ANCHOR_COLUMNS).to_dicts()
    }
    projection_anchors = requests.select(
        "query_name",
        "source_chrom",
        pl.col("projection_start").alias("source_start"),
        pl.col("projection_end").alias("source_end"),
        "region_label",
    )
    for fragment in iter_projected_anchor_fragments(
        maf_path,
        projection_anchors,
        species_manifest,
        source_alignment_name=source_alignment_name,
    ):
        anchor = anchors_by_name[str(fragment["query_name"])]
        fragment["source_start"] = int(anchor["source_start"])
        fragment["source_end"] = int(anchor["source_end"])
        yield fragment


def project_requests_from_maf(
    maf_path: str | Path,
    requests: pl.DataFrame,
    species_manifest: pl.DataFrame,
    *,
    source_alignment_name: str = "hg38",
) -> pl.DataFrame:
    """Materialize issue #473 MAF candidates for tests and small inputs."""
    fragments = list(
        iter_maf_projection_request_fragments(
            maf_path,
            requests,
            species_manifest,
            source_alignment_name=source_alignment_name,
        )
    )
    if not fragments:
        return pl.DataFrame(schema=FRAGMENT_SCHEMA)
    result = pl.DataFrame(fragments, schema=FRAGMENT_SCHEMA)
    assert (result["source_fragment_start"] >= result["source_start"]).all()
    assert (result["source_fragment_end"] <= result["source_end"]).all()
    return result.sort("query_name", "species", "mapping_id", "fragment_id")


def write_maf_request_candidates(
    maf_path: str | Path,
    requests_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    rows_per_batch: int = 5_000,
) -> None:
    """Stream issue #473 MAF candidates into species-clustered row groups."""
    assert rows_per_batch > 0
    requests = read_projection_requests(requests_path)
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
            for fragment in iter_maf_projection_request_fragments(
                maf_path, requests, manifest
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
    assert all(int(value) == 0 for value in stats.values())


def write_contract_outputs(
    fragments_path: str | Path,
    accepted_path: str | Path,
    rejected_path: str | Path,
    *,
    target_length: int,
    pre_resize_min_length: int,
    pre_resize_max_length: int,
) -> None:
    """Apply the isolated policy contract to one fragment Parquet."""
    schema = pl.read_parquet_schema(fragments_path)
    missing = set(FRAGMENT_SCHEMA) - set(schema)
    assert not missing, f"projection fragments missing columns: {sorted(missing)}"
    result = apply_projection_contract(
        pl.read_parquet(fragments_path),
        target_length=target_length,
        pre_resize_min_length=pre_resize_min_length,
        pre_resize_max_length=pre_resize_max_length,
    )
    Path(accepted_path).parent.mkdir(parents=True, exist_ok=True)
    Path(rejected_path).parent.mkdir(parents=True, exist_ok=True)
    result.accepted.write_parquet(accepted_path)
    result.rejected.write_parquet(rejected_path)


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
    """Apply the isolated policy contract to one clustered MAF species."""
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
