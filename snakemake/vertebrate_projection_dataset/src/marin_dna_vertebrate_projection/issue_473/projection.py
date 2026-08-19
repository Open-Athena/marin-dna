"""Issue #473 adapters that leave the established projection path untouched."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.issue_473.policy import (
    ANCHOR_COLUMNS,
    PROJECTION_REQUEST_COLUMNS,
)
from marin_dna_vertebrate_projection.maf import (
    FRAGMENT_SCHEMA,
    iter_projected_anchor_fragments,
)


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
