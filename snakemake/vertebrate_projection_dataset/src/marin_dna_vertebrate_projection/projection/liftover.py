"""Checksum-pinned UCSC liftOver adapter for center-nucleotide projection.

All BED coordinates at this boundary are 0-based and half-open.  Each input
interval is the one-base center landmark from a 255 bp human anchor.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.maf import FRAGMENT_SCHEMA
from marin_dna_vertebrate_projection.manifest import validate_species_manifest
from marin_dna_vertebrate_projection.projection.center import (
    read_projection_requests,
)

CHECKSUM_SOURCE_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/md5sum.txt"
)
CHAIN_BASE_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver"


@dataclass(frozen=True)
class LiftoverChain:
    """One immutable hg38-to-target UCSC chain object."""

    alignment_name: str
    source_url: str
    checksum_source_url: str
    byte_size: int
    md5: str


def chain_filename(alignment_name: str) -> str:
    """Return UCSC's case-sensitive hg38 chain filename for a target DB."""
    assert alignment_name and alignment_name[0].isalnum()
    target = alignment_name[0].upper() + alignment_name[1:]
    return f"hg38To{target}.over.chain.gz"


def read_liftover_chain_manifest(path: str | Path) -> dict[str, LiftoverChain]:
    """Read and validate the committed UCSC chain manifest."""
    frame = pl.read_csv(path, separator="\t")
    required = {
        "alignment_name",
        "source_url",
        "checksum_source_url",
        "byte_size",
        "md5",
    }
    missing = required - set(frame.columns)
    assert not missing, f"liftOver manifest missing columns: {sorted(missing)}"
    assert frame.height > 0
    assert frame["alignment_name"].n_unique() == frame.height
    assert frame["source_url"].n_unique() == frame.height
    assert (frame["byte_size"] > 0).all()
    assert frame["md5"].str.contains(r"^[0-9a-f]{32}$").all()
    assert (frame["checksum_source_url"] == CHECKSUM_SOURCE_URL).all()
    result: dict[str, LiftoverChain] = {}
    for row in frame.to_dicts():
        alignment_name = str(row["alignment_name"])
        expected_url = f"{CHAIN_BASE_URL}/{chain_filename(alignment_name)}"
        assert row["source_url"] == expected_url, (
            f"unexpected chain URL for {alignment_name}: {row['source_url']}"
        )
        result[alignment_name] = LiftoverChain(
            alignment_name=alignment_name,
            source_url=expected_url,
            checksum_source_url=str(row["checksum_source_url"]),
            byte_size=int(row["byte_size"]),
            md5=str(row["md5"]),
        )
    return result


def validate_liftover_chain_manifest(
    chains: dict[str, LiftoverChain], expected_names: list[str]
) -> None:
    """Require exactly one version-matched chain for every selected target."""
    assert len(expected_names) == len(set(expected_names))
    assert set(chains) == set(expected_names), {
        "missing": sorted(set(expected_names) - set(chains)),
        "unexpected": sorted(set(chains) - set(expected_names)),
    }


def file_md5(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a chain without loading it into memory."""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_liftover_chain(path: str | Path, expected: LiftoverChain) -> None:
    """Reject missing, truncated, or content-mismatched chain inputs."""
    local_path = Path(path)
    assert local_path.is_file(), f"missing liftOver chain: {local_path}"
    assert local_path.stat().st_size == expected.byte_size, (
        f"chain size mismatch for {expected.alignment_name}: "
        f"{local_path.stat().st_size} != {expected.byte_size}"
    )
    observed_md5 = file_md5(local_path)
    assert observed_md5 == expected.md5, (
        f"chain MD5 mismatch for {expected.alignment_name}: "
        f"{observed_md5} != {expected.md5}"
    )


def stage_liftover_chain(expected: LiftoverChain, destination: str | Path) -> None:
    """Download, checksum, and atomically install one public UCSC chain."""
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        try:
            verify_liftover_chain(destination_path, expected)
            return
        except AssertionError:
            destination_path.unlink()

    partial = destination_path.with_name(f".{destination_path.name}.partial")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(
        expected.source_url, headers={"User-Agent": "marin-dna/517"}
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            partial.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        verify_liftover_chain(partial, expected)
        partial.replace(destination_path)
    finally:
        partial.unlink(missing_ok=True)


def run_liftover(
    input_bed: str | Path,
    chain_path: str | Path,
    mapped_bed: str | Path,
    unmapped_bed: str | Path,
    *,
    min_match: float = 0.95,
    multiple: bool = False,
) -> float:
    """Run UCSC liftOver on 0-based half-open BED6 center landmarks."""
    assert 0 < min_match <= 1
    mapped = Path(mapped_bed)
    unmapped = Path(unmapped_bed)
    mapped.parent.mkdir(parents=True, exist_ok=True)
    unmapped.parent.mkdir(parents=True, exist_ok=True)
    command = ["liftOver", f"-minMatch={min_match:g}"]
    if multiple:
        command.append("-multiple")
    command.extend([str(input_bed), str(chain_path), str(mapped), str(unmapped)])
    start = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter() - start


def _read_bed6(path: str | Path) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    with Path(path).open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            assert len(fields) >= 6, f"malformed BED6 line {line_number}: {line!r}"
            rows.append(
                {
                    "t_chrom": fields[0],
                    "t_start": int(fields[1]),
                    "t_end": int(fields[2]),
                    "query_name": fields[3],
                    "score": int(fields[4]),
                    "t_strand": fields[5],
                }
            )
    schema = {
        "t_chrom": pl.String,
        "t_start": pl.Int64,
        "t_end": pl.Int64,
        "query_name": pl.String,
        "score": pl.Int64,
        "t_strand": pl.String,
    }
    return pl.DataFrame(rows, schema=schema)


def validate_liftover_partition(
    input_bed: str | Path,
    mapped_bed: str | Path,
    unmapped_bed: str | Path,
    *,
    multiple: bool,
) -> dict[str, int]:
    """Require every input query to appear in mapped XOR unmapped output."""
    source = _read_bed6(input_bed)
    mapped = _read_bed6(mapped_bed)
    unmapped = _read_bed6(unmapped_bed)
    assert source["query_name"].n_unique() == source.height
    assert unmapped["query_name"].n_unique() == unmapped.height
    if not multiple:
        assert mapped["query_name"].n_unique() == mapped.height
    source_names = set(source["query_name"].to_list())
    mapped_names = set(mapped["query_name"].to_list())
    unmapped_names = set(unmapped["query_name"].to_list())
    assert mapped_names.isdisjoint(unmapped_names)
    assert mapped_names | unmapped_names == source_names, {
        "missing": sorted(source_names - mapped_names - unmapped_names),
        "unexpected": sorted((mapped_names | unmapped_names) - source_names),
    }
    assert (mapped["t_start"] >= 0).all()
    assert (mapped["t_end"] - mapped["t_start"] == 1).all()
    assert set(mapped["t_strand"].to_list()) <= {"+", "-"}
    return {
        "input_queries": source.height,
        "mapped_queries": len(mapped_names),
        "mapped_rows": mapped.height,
        "unmapped_queries": unmapped.height,
    }


def attach_target_sizes(
    records: pl.DataFrame, chrom_sizes_tsv: str | Path
) -> pl.DataFrame:
    """Attach exact target chromosome sizes and reject unknown names."""
    sizes = pl.read_csv(
        chrom_sizes_tsv,
        separator="\t",
        has_header=False,
        new_columns=["t_chrom", "t_src_size"],
        schema_overrides={"t_chrom": pl.String, "t_src_size": pl.Int64},
    )
    joined = records.join(sizes, on="t_chrom", how="left")
    assert joined["t_src_size"].null_count() == 0, (
        "liftOver output contains target chromosomes absent from the pinned 2bit"
    )
    return joined


def liftover_records_to_fragments(
    records: pl.DataFrame,
    requests: pl.DataFrame,
    species_manifest: pl.DataFrame,
    *,
    alignment_name: str,
) -> pl.DataFrame:
    """Convert mapped BED6 rows into the shared projection fragment schema."""
    required_records = {
        "query_name",
        "t_chrom",
        "t_start",
        "t_end",
        "t_strand",
        "t_src_size",
    }
    missing = required_records - set(records.columns)
    assert not missing, f"liftOver records missing columns: {sorted(missing)}"
    validated_requests = read_projection_requests_frame(requests)
    validate_species_manifest(species_manifest)
    metadata = species_manifest.filter(
        (pl.col("backend") == "ucsc_multiz100way")
        & pl.col("selected")
        & (pl.col("alignment_name") == alignment_name)
    ).select(
        "alignment_name",
        "scientific_name",
        "assembly",
        "taxonomy_id",
        "family",
        "clade",
        "phylogenetic_rank",
    )
    assert metadata.height == 1, f"missing selected target metadata: {alignment_name}"
    joined = (
        records.join(
            validated_requests.select(
                "query_name",
                "source_chrom",
                "source_start",
                "source_end",
                "projection_start",
                "projection_end",
                "region_label",
            ),
            on="query_name",
            how="inner",
        )
        .join(
            metadata.with_columns(pl.lit(1, dtype=pl.Int8).alias("_join")),
            how="cross",
        )
        .with_row_index("mapping_number")
    )
    assert joined.height == records.height, (
        "every mapped BED row must match one request"
    )
    if joined.is_empty():
        return pl.DataFrame(schema=FRAGMENT_SCHEMA)
    result = joined.select(
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        "region_label",
        pl.col("projection_start").alias("source_fragment_start"),
        pl.col("projection_end").alias("source_fragment_end"),
        pl.col("scientific_name").alias("species"),
        "alignment_name",
        "assembly",
        "taxonomy_id",
        "family",
        "clade",
        "phylogenetic_rank",
        pl.lit("ucsc_multiz100way").alias("alignment_source"),
        "t_chrom",
        "t_start",
        "t_end",
        "t_strand",
        "t_src_size",
        pl.concat_str(
            pl.lit("chain:"),
            pl.col("query_name"),
            pl.lit(":"),
            pl.col("alignment_name"),
            pl.lit(":"),
            pl.col("mapping_number").cast(pl.String),
        ).alias("mapping_id"),
        pl.concat_str(
            pl.lit("chain:"),
            pl.col("query_name"),
            pl.lit(":"),
            pl.col("alignment_name"),
            pl.lit(":"),
            pl.col("mapping_number").cast(pl.String),
            pl.lit(":0"),
        ).alias("fragment_id"),
        (pl.col("t_end") - pl.col("t_start")).alias("aligned_bases"),
    )
    return result.cast(FRAGMENT_SCHEMA).sort("query_name", "mapping_id")


def read_projection_requests_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate an in-memory request frame through the canonical reader contract."""
    required = {
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        "projection_start",
        "projection_end",
        "landmark_width",
        "projection_policy",
        "region_label",
    }
    missing = required - set(frame.columns)
    assert not missing, f"projection requests missing columns: {sorted(missing)}"
    assert frame["query_name"].n_unique() == frame.height
    assert (frame["source_end"] - frame["source_start"] == 255).all()
    assert (frame["projection_start"] == frame["source_start"] + 127).all()
    assert (frame["projection_end"] == frame["projection_start"] + 1).all()
    assert frame["projection_policy"].unique().to_list() == ["center_1"]
    return frame


def write_liftover_fragments(
    input_bed: str | Path,
    mapped_bed: str | Path,
    unmapped_bed: str | Path,
    requests_path: str | Path,
    manifest_path: str | Path,
    chrom_sizes_tsv: str | Path,
    output_path: str | Path,
    *,
    alignment_name: str,
    multiple: bool,
) -> None:
    """Validate one liftOver partition and write shared-schema fragments."""
    validate_liftover_partition(input_bed, mapped_bed, unmapped_bed, multiple=multiple)
    records = attach_target_sizes(_read_bed6(mapped_bed), chrom_sizes_tsv)
    requests = read_projection_requests(requests_path)
    manifest = pl.read_csv(manifest_path, separator="\t")
    fragments = liftover_records_to_fragments(
        records,
        requests,
        manifest,
        alignment_name=alignment_name,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fragments.write_parquet(output)


def rewrite_functional_liftover_card(path: str | Path) -> None:
    """Make the generated issue #517 card explicit about chain projection."""
    card = Path(path)
    text = card.read_text()
    old = "the Zoonomia 447-mammal Cactus alignment and UCSC hg38 MultiZ 100-way alignment"
    new = (
        "the Zoonomia 447-mammal Cactus alignment and official version-matched "
        "UCSC hg38-to-target liftOver chains"
    )
    assert text.count(old) == 1, "generated card source description changed"
    marker = (
        "Non-human rows project only the central human nucleotide and extract the "
        "255 bp target window centered on its unique mapped locus.\n"
    )
    note = (
        "\nFor the 28 non-mammalian targets, the stable `ucsc_multiz100way` "
        "`alignment_source` value identifies the pinned UCSC assembly cohort; the "
        "actual projection operation uses the matching checksum-pinned pairwise "
        "liftOver chain.\n"
    )
    assert text.count(marker) == 1, "generated card projection description changed"
    card.write_text(text.replace(old, new).replace(marker, marker + note))
