"""Additive UCSC chain adapter for the center-1 projection contract.

Coordinates are 0-based, half-open throughout. Chain ``t`` is the human
source; chain ``q`` is the destination assembly. This reader never opens HAL.
The existing manifest's backend identifies alignment provenance, not the
program used to query it; the audit records the actual chain and operation.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.contract import apply_projection_contract
from marin_dna_vertebrate_projection.maf import FRAGMENT_SCHEMA
from marin_dna_vertebrate_projection.manifest import read_species_manifest
from marin_dna_vertebrate_projection.projection.center import read_projection_requests
from marin_dna_vertebrate_projection.projection.requests import (
    build_projection_requests,
)

ASSET_COLUMNS = {
    "alignment_name",
    "assembly",
    "source_assembly",
    "chain_origin",
    "chain",
    "chain_sha256",
    "source_sizes",
    "source_sizes_sha256",
    "target_sizes",
    "target_sizes_sha256",
}
BED_SCHEMA = {
    "t_chrom": pl.String,
    "t_start": pl.Int64,
    "t_end": pl.Int64,
    "query_name": pl.String,
    "score": pl.String,
    "t_strand": pl.String,
}


def read_chain_assets(
    path: str | Path, species_path: str | Path
) -> dict[str, dict[str, str]]:
    """Require pinned assets for an explicit subset of selected family targets.

    Paths are absolute, project-relative, or explicit S3 URIs. No implicit
    downloads, assembly aliases, or destination selection are performed.
    """
    selected = read_species_manifest(str(species_path)).filter(pl.col("selected"))
    assemblies = dict(selected.select("alignment_name", "assembly").iter_rows())
    result: dict[str, dict[str, str]] = {}
    with Path(path).open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not ASSET_COLUMNS <= set(reader.fieldnames or []):
            raise ValueError("chain asset manifest is missing required columns")
        for row in reader:
            if any(not row.get(key) for key in ASSET_COLUMNS):
                raise ValueError("chain asset manifest contains empty fields")
            name = row["alignment_name"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in result:
                raise ValueError(f"invalid or duplicate alignment name: {name}")
            if assemblies.get(name) != row["assembly"]:
                raise ValueError(
                    f"chain target is not a version-matched selected assembly: {name}"
                )
            if row["source_assembly"] != "hg38":
                raise ValueError(
                    "this adapter requires the pinned hg38 source coordinate system"
                )
            for key in ("chain_sha256", "source_sizes_sha256", "target_sizes_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                    raise ValueError(f"invalid {key} for {name}")
            result[name] = {key: row[key] for key in sorted(ASSET_COLUMNS)}
    if not result:
        raise ValueError("no chain targets selected")
    sources = {
        (row["source_sizes"], row["source_sizes_sha256"]) for row in result.values()
    }
    if len(sources) != 1:
        raise ValueError(
            "all targets must use the same pinned human chromosome dictionary"
        )
    return result


def file_sha256(path: str | Path) -> str:
    """Hash an input without materializing it."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_sizes(path: str | Path) -> dict[str, int]:
    """Read an exact contig dictionary without rewriting sequence names."""
    sizes: dict[str, int] = {}
    with Path(path).open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 2 or fields[0] in sizes or int(fields[1]) <= 0:
                raise ValueError(
                    f"invalid or duplicate chromosome size: {line.strip()}"
                )
            sizes[fields[0]] = int(fields[1])
    if not sizes:
        raise ValueError("empty chromosome dictionary")
    return sizes


def validate_chain_inputs(
    chain: str | Path,
    source_sizes: str | Path,
    target_sizes: str | Path,
    expected: dict[str, str],
    output: str | Path,
) -> None:
    """Check hashes and every chain header's direction, contigs, and sizes.

    This is a structural scan, not a HAL-equivalence test. Chain blocks are
    parsed by liftOver itself. A wrong assembly or swapped direction fails
    before invoking the projector, even for contigs absent from the sample.
    """
    for key, path in (
        ("chain", chain),
        ("source_sizes", source_sizes),
        ("target_sizes", target_sizes),
    ):
        if file_sha256(path) != expected[f"{key}_sha256"]:
            raise ValueError(f"{key} SHA-256 mismatch for {expected['alignment_name']}")
    source = read_sizes(source_sizes)
    target = read_sizes(target_sizes)
    count = 0
    opener = gzip.open if str(chain).endswith(".gz") else open
    with opener(chain, "rt") as handle:
        for line in handle:
            if not line.startswith("chain "):
                continue
            fields = line.split()
            if len(fields) != 13:
                raise ValueError("malformed chain header")
            (
                _,
                score,
                t_name,
                t_size,
                t_strand,
                t_start,
                t_end,
                q_name,
                q_size,
                q_strand,
                q_start,
                q_end,
                chain_id,
            ) = fields
            int(score)
            int(chain_id)
            if source.get(t_name) != int(t_size) or target.get(q_name) != int(q_size):
                raise ValueError(
                    f"chain direction/assembly size mismatch: {t_name} -> {q_name}"
                )
            if t_strand != "+" or q_strand not in {"+", "-"}:
                raise ValueError("invalid chain strand")
            if not (0 <= int(t_start) < int(t_end) <= int(t_size)):
                raise ValueError("invalid source chain bounds")
            if not (0 <= int(q_start) < int(q_end) <= int(q_size)):
                raise ValueError("invalid target chain bounds")
            count += 1
    if not count:
        raise ValueError("chain contains no headers")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(
        json.dumps({"assets": expected, "chain_headers": count}, indent=2) + "\n"
    )


def prepare_chain_requests(
    anchors_path: str | Path,
    source_sizes_path: str | Path,
    requests_path: str | Path,
    bed_path: str | Path,
) -> None:
    """Retain source identities and submit only their central human base."""
    schema = {
        "query_name": pl.String,
        "source_chrom": pl.String,
        "source_start": pl.Int64,
        "source_end": pl.Int64,
        "region_label": pl.String,
    }
    path = Path(anchors_path)
    anchors = (
        pl.read_parquet(path)
        if path.suffix == ".parquet"
        else pl.read_csv(path, separator="\t", schema_overrides=schema)
    ).select(list(schema))
    if any(anchors.schema[key] != dtype for key, dtype in schema.items()):
        raise ValueError(
            "anchor columns must use string identities and Int64 coordinates"
        )
    if not anchors["source_chrom"].str.starts_with("chr").all():
        raise ValueError("source chromosome names must use the pinned hg38 dictionary")
    if anchors.is_empty() or sum(anchors.null_count().row(0)):
        raise ValueError("anchor catalog must be nonempty and contain no nulls")
    if (
        anchors["query_name"].str.contains(r"\s").any()
        or (anchors["query_name"].str.len_chars() == 0).any()
    ):
        raise ValueError("query names must be nonempty BED-safe unique identifiers")
    sizes = read_sizes(source_sizes_path)
    for chrom, end in (
        anchors.group_by("source_chrom").agg(pl.col("source_end").max()).iter_rows()
    ):
        if chrom not in sizes or end > sizes[chrom]:
            raise ValueError(f"anchor exceeds pinned human bounds: {chrom}:{end}")
    requests = build_projection_requests(anchors)
    Path(requests_path).parent.mkdir(parents=True, exist_ok=True)
    Path(bed_path).parent.mkdir(parents=True, exist_ok=True)
    requests.write_parquet(requests_path)
    requests.select(
        "source_chrom",
        "projection_start",
        "projection_end",
        "query_name",
        pl.lit("0").alias("score"),
        pl.lit("+").alias("strand"),
    ).write_csv(bed_path, separator="\t", include_header=False)


def read_bed6(path: str | Path) -> pl.DataFrame:
    """Read mapped BED6 or UCSC's comment-annotated unmapped BED6."""
    rows: list[tuple[str, int, int, str, str, str]] = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 6:
                raise ValueError("expected exactly six BED fields")
            chrom, start, end, name, score, strand = fields
            if int(start) < 0 or int(end) - int(start) != 1 or strand not in {"+", "-"}:
                raise ValueError(f"invalid center-1 BED record: {name}")
            rows.append((chrom, int(start), int(end), name, score, strand))
    return pl.DataFrame(rows, schema=BED_SCHEMA, orient="row")


def write_chain_projections(
    requests_path: str | Path,
    mapped_path: str | Path,
    unmapped_path: str | Path,
    target_sizes_path: str | Path,
    species_path: str | Path,
    alignment_name: str,
    validation_path: str | Path,
    accepted_path: str | Path,
    rejected_path: str | Path,
    audit_path: str | Path,
) -> None:
    """Account for every query, reject multiplicity, and apply the shared contract."""
    requests = read_projection_requests(requests_path)
    mapped, unmapped = read_bed6(mapped_path), read_bed6(unmapped_path)
    source_names = set(requests["query_name"])
    mapped_names, unmapped_names = (
        set(mapped["query_name"]),
        set(unmapped["query_name"]),
    )
    if unmapped.height != len(unmapped_names) or mapped_names & unmapped_names:
        raise ValueError("mapped and unmapped outputs must partition input queries")
    if source_names != mapped_names | unmapped_names:
        raise ValueError("liftOver lost queries or produced unknown query IDs")
    # Unmapped output must contain the original center, not an arbitrary row
    # sharing its name. In particular, never silently rewrite chromosome names.
    expected_unmapped = requests.filter(
        pl.col("query_name").is_in(unmapped_names)
    ).select(
        pl.col("source_chrom").alias("t_chrom"),
        pl.col("projection_start").alias("t_start"),
        pl.col("projection_end").alias("t_end"),
        "query_name",
        pl.lit("0").alias("score"),
        pl.lit("+").alias("t_strand"),
    )
    if not expected_unmapped.sort("query_name").equals(unmapped.sort("query_name")):
        raise ValueError("unmapped coordinates differ from their input centers")
    metadata = read_species_manifest(str(species_path)).filter(
        pl.col("selected") & (pl.col("alignment_name") == alignment_name)
    )
    if metadata.height != 1:
        raise ValueError("expected exactly one selected target")
    metadata = metadata.select(
        pl.col("scientific_name").alias("species"),
        "alignment_name",
        "assembly",
        "taxonomy_id",
        "family",
        "clade",
        "phylogenetic_rank",
        pl.col("backend").alias("alignment_source"),
    )
    validation = json.loads(Path(validation_path).read_text())
    assets = validation["assets"]
    if (
        assets["alignment_name"] != alignment_name
        or assets["assembly"] != metadata["assembly"][0]
    ):
        raise ValueError("validated chain does not match selected target")
    if file_sha256(target_sizes_path) != assets["target_sizes_sha256"]:
        raise ValueError("target dictionary changed after chain validation")
    sizes = read_sizes(target_sizes_path)
    size_frame = pl.DataFrame(
        {"t_chrom": list(sizes), "t_src_size": list(sizes.values())}
    )
    joined = (
        mapped.join(requests, on="query_name", validate="m:1")
        .join(metadata, how="cross")
        .join(size_frame, on="t_chrom", how="left")
    )
    if (
        joined["t_src_size"].null_count()
        or (joined["t_end"] > joined["t_src_size"]).any()
    ):
        raise ValueError("mapped center exceeds pinned target bounds")
    fragments = (
        joined.with_row_index("_id")
        .select(
            "query_name",
            "source_chrom",
            "source_start",
            "source_end",
            "region_label",
            pl.col("projection_start").alias("source_fragment_start"),
            pl.col("projection_end").alias("source_fragment_end"),
            "species",
            "alignment_name",
            "assembly",
            "taxonomy_id",
            "family",
            "clade",
            "phylogenetic_rank",
            "alignment_source",
            "t_chrom",
            "t_start",
            "t_end",
            "t_strand",
            "t_src_size",
            pl.col("_id").cast(pl.String).alias("mapping_id"),
            pl.col("_id").cast(pl.String).alias("fragment_id"),
            pl.lit(1, dtype=pl.Int64).alias("aligned_bases"),
        )
        .cast(FRAGMENT_SCHEMA)
    )
    result = apply_projection_contract(fragments)
    # The historical contract contains only queries with mapping fragments.
    # This additive adapter also emits explicit unmapped query rejections.
    rejected_schema = result.rejected.schema
    missing_rows = (
        requests.filter(pl.col("query_name").is_in(unmapped_names))
        .join(metadata, how="cross")
        .with_columns(
            pl.lit("unmapped").alias("rejection_reason"),
            pl.lit(
                "liftOver reported no mapping; see raw unmapped BED for reason"
            ).alias("detail"),
            pl.lit(0, dtype=pl.Int64).alias("fragment_count"),
        )
        .select(list(rejected_schema))
        .cast(rejected_schema)
    )
    rejected = pl.concat([result.rejected, missing_rows]).sort("query_name")
    accepted = result.accepted.sort("query_name")
    if accepted.height + rejected.height != requests.height:
        raise ValueError(
            "accepted/rejected output does not account for every input query"
        )
    for path in (accepted_path, rejected_path, audit_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    accepted.write_parquet(accepted_path)
    rejected.write_parquet(rejected_path)
    Path(audit_path).write_text(
        json.dumps(
            {
                "projection_policy": "center_1",
                "coordinate_system": "0-based half-open",
                "projector": "UCSC liftOver",
                "options": ["-minMatch=0.95", "-multiple"],
                "assets": assets,
                "requests_sha256": file_sha256(requests_path),
                "species_manifest_sha256": file_sha256(species_path),
                "input_queries": requests.height,
                "mapped_rows": mapped.height,
                "mapped_queries": len(mapped_names),
                "unmapped_queries": len(unmapped_names),
                "accepted_queries": accepted.height,
                "rejected_queries": rejected.height,
                "rejections": dict(
                    rejected.group_by("rejection_reason").len().iter_rows()
                ),
            },
            indent=2,
        )
        + "\n"
    )
