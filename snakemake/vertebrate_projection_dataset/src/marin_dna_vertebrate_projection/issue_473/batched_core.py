"""Batch the two required issue #473 projections without changing old rules.

The established rules issue one alignment pass per policy.  This additive
prefill path namespaces both required request sets, invokes HAL once per target
species and scans each chromosome MAF once, then restores the exact per-run
raw output identities consumed by the unchanged downstream contract rules.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from marin_dna_zoonomia_projection.projection.hal import run_halliftover

from marin_dna_vertebrate_projection.issue_473.projection import (
    iter_maf_projection_request_fragments,
    read_projection_requests,
)
from marin_dna_vertebrate_projection.maf import FRAGMENT_SCHEMA
from marin_dna_vertebrate_projection.manifest import read_species_manifest

CORE_RUNS = ("full_center_1", "full_enhancer_full_window")
_NAME_PREFIX = "issue473batch|"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _ordered_paths(paths: Mapping[str, str | Path]) -> list[tuple[str, Path]]:
    assert set(paths) == set(CORE_RUNS), (
        f"batched core paths must name exactly {list(CORE_RUNS)}"
    )
    return [(run, Path(paths[run])) for run in CORE_RUNS]


def encode_query_name(run: str, query_name: str) -> str:
    """Namespace a BED name while preserving arbitrary separators in the ID."""
    assert run in CORE_RUNS
    assert query_name and "\t" not in query_name and "\n" not in query_name
    return f"{_NAME_PREFIX}{run}|{query_name}"


def decode_query_name(encoded: str) -> tuple[str, str]:
    """Invert :func:`encode_query_name` and reject foreign HAL rows."""
    assert encoded.startswith(_NAME_PREFIX), f"foreign batched query name: {encoded}"
    run, separator, query_name = encoded.removeprefix(_NAME_PREFIX).partition("|")
    assert separator and run in CORE_RUNS and query_name
    return run, query_name


def write_batched_hal_request_bed6(
    request_paths: Mapping[str, str | Path],
    output_path: str | Path,
    manifest_path: str | Path,
) -> None:
    """Concatenate both validated request BEDs with reversible namespacing."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    inputs: dict[str, dict[str, object]] = {}
    try:
        with partial.open("w") as stream:
            for run, request_path in _ordered_paths(request_paths):
                requests = read_projection_requests(request_path)
                inputs[run] = {
                    "path": str(request_path),
                    "rows": requests.height,
                    "sha256": _sha256(request_path),
                }
                requests.select(
                    pl.col("source_chrom"),
                    pl.col("projection_start"),
                    pl.col("projection_end"),
                    pl.concat_str(
                        pl.lit(f"{_NAME_PREFIX}{run}|"), pl.col("query_name")
                    ).alias("query_name"),
                    pl.lit(0).alias("score"),
                    pl.lit("+").alias("strand"),
                ).write_csv(stream, separator="\t", include_header=False)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)

    _write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "kind": "issue_473_batched_core_requests",
            "runs": list(CORE_RUNS),
            "inputs": inputs,
            "output": {
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "rows": sum(int(item["rows"]) for item in inputs.values()),
                "sha256": _sha256(destination),
            },
        },
    )


def split_batched_hal_output(
    combined_path: str | Path,
    output_paths: Mapping[str, str | Path],
) -> dict[str, dict[str, object]]:
    """Split one HAL BED stream and restore the original query names."""
    ordered = _ordered_paths(output_paths)
    partials = {
        run: path.with_name(f".{path.name}.partial.{os.getpid()}")
        for run, path in ordered
    }
    streams: dict[str, object] = {}
    counts = {run: 0 for run, _path in ordered}
    try:
        for run, path in ordered:
            path.parent.mkdir(parents=True, exist_ok=True)
            streams[run] = partials[run].open("w")
        with Path(combined_path).open() as source:
            for line_number, line in enumerate(source, start=1):
                fields = line.rstrip("\n").split("\t")
                assert len(fields) == 6, (
                    f"combined HAL row {line_number} has {len(fields)} columns"
                )
                run, query_name = decode_query_name(fields[3])
                fields[3] = query_name
                streams[run].write("\t".join(fields) + "\n")  # type: ignore[attr-defined]
                counts[run] += 1
        for stream in streams.values():
            stream.close()  # type: ignore[attr-defined]
        streams.clear()
        for run, path in ordered:
            partials[run].replace(path)
    finally:
        for stream in streams.values():
            stream.close()  # type: ignore[attr-defined]
        for partial in partials.values():
            partial.unlink(missing_ok=True)

    return {
        run: {
            "path": str(path),
            "rows": counts[run],
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for run, path in ordered
    }


def run_batched_hal_liftover(
    hal_path: str | Path,
    source_species: str,
    batched_bed_path: str | Path,
    target_species: str,
    output_paths: Mapping[str, str | Path],
    receipt_path: str | Path,
) -> None:
    """Invoke HAL once for one species and atomically split both run outputs."""
    first_output = _ordered_paths(output_paths)[0][1]
    first_output.parent.mkdir(parents=True, exist_ok=True)
    combined = first_output.parent / f".combined.{target_species}.{os.getpid()}.bed"
    try:
        elapsed = run_halliftover(
            hal_path,
            source_species,
            batched_bed_path,
            target_species,
            combined,
            no_dupes=True,
        )
        outputs = split_batched_hal_output(combined, output_paths)
        _write_json_atomic(
            receipt_path,
            {
                "schema_version": 1,
                "kind": "issue_473_batched_core_hal",
                "runs": list(CORE_RUNS),
                "source_species": source_species,
                "target_species": target_species,
                "request_bed_sha256": _sha256(batched_bed_path),
                "combined_output": {
                    "rows": sum(int(item["rows"]) for item in outputs.values()),
                    "bytes": combined.stat().st_size,
                    "sha256": _sha256(combined),
                },
                "elapsed_seconds": elapsed,
                "outputs": outputs,
            },
        )
    finally:
        combined.unlink(missing_ok=True)


def write_batched_maf_request_candidates(
    maf_path: str | Path,
    request_paths: Mapping[str, str | Path],
    manifest_path: str | Path,
    output_paths: Mapping[str, str | Path],
    receipt_path: str | Path,
    *,
    rows_per_batch: int = 5_000,
) -> None:
    """Scan one chromosome MAF once and split fragments into both core runs."""
    assert rows_per_batch > 0
    ordered_requests = _ordered_paths(request_paths)
    ordered_outputs = _ordered_paths(output_paths)
    request_info: dict[str, dict[str, object]] = {}
    request_frames: list[pl.DataFrame] = []
    for run, request_path in ordered_requests:
        requests = read_projection_requests(request_path)
        request_info[run] = {
            "path": str(request_path),
            "rows": requests.height,
            "sha256": _sha256(request_path),
        }
        request_frames.append(
            requests.with_columns(
                pl.concat_str(
                    pl.lit(f"{_NAME_PREFIX}{run}|"), pl.col("query_name")
                ).alias("query_name")
            )
        )
    combined_requests = pl.concat(request_frames, how="vertical")
    manifest = read_species_manifest(str(manifest_path))
    selected = manifest.filter(
        (pl.col("backend") == "ucsc_multiz100way") & pl.col("selected")
    )
    alignment_names = sorted(selected["alignment_name"].to_list())
    assert alignment_names

    output_by_run = dict(ordered_outputs)
    for path in output_by_run.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    counts = {run: 0 for run in CORE_RUNS}
    buffers: dict[tuple[str, str], list[dict[str, object]]] = {
        (run, name): [] for run in CORE_RUNS for name in alignment_names
    }

    with TemporaryDirectory(
        prefix=".batched-maf-", dir=next(iter(output_by_run.values())).parent
    ) as temp_dir:
        temporary = Path(temp_dir)
        writers: dict[tuple[str, str], pq.ParquetWriter] = {}

        def flush(run: str, alignment_name: str) -> None:
            key = (run, alignment_name)
            rows = buffers[key]
            if not rows:
                return
            table = pl.DataFrame(rows, schema=FRAGMENT_SCHEMA).to_arrow()
            writer = writers.get(key)
            if writer is None:
                part = temporary / run / f"{alignment_name}.parquet"
                part.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(
                    part, table.schema, compression="zstd", write_statistics=True
                )
                writers[key] = writer
            writer.write_table(table)
            rows.clear()

        try:
            for fragment in iter_maf_projection_request_fragments(
                maf_path, combined_requests, manifest
            ):
                run, query_name = decode_query_name(str(fragment["query_name"]))
                fragment["query_name"] = query_name
                alignment_name = str(fragment["alignment_name"])
                key = (run, alignment_name)
                assert key in buffers
                buffers[key].append(fragment)
                counts[run] += 1
                if len(buffers[key]) >= rows_per_batch:
                    flush(run, alignment_name)
            for run in CORE_RUNS:
                for alignment_name in alignment_names:
                    flush(run, alignment_name)
        finally:
            for writer in writers.values():
                writer.close()

        for run, output in ordered_outputs:
            partial = output.with_name(f".{output.name}.partial.{os.getpid()}")
            output_writer: pq.ParquetWriter | None = None
            try:
                for alignment_name in alignment_names:
                    part = temporary / run / f"{alignment_name}.parquet"
                    if not part.exists():
                        continue
                    for batch in pq.ParquetFile(part).iter_batches(
                        batch_size=rows_per_batch
                    ):
                        table = pa.Table.from_batches([batch])
                        if output_writer is None:
                            output_writer = pq.ParquetWriter(
                                partial,
                                table.schema,
                                compression="zstd",
                                write_statistics=True,
                            )
                        output_writer.write_table(table)
            finally:
                if output_writer is not None:
                    output_writer.close()
            if not partial.exists():
                pl.DataFrame(schema=FRAGMENT_SCHEMA).write_parquet(partial)
            partial.replace(output)

    outputs = {
        run: {
            "path": str(path),
            "rows": counts[run],
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for run, path in ordered_outputs
    }
    _write_json_atomic(
        receipt_path,
        {
            "schema_version": 1,
            "kind": "issue_473_batched_core_maf",
            "runs": list(CORE_RUNS),
            "maf_path": str(maf_path),
            "maf_bytes": Path(maf_path).stat().st_size,
            "manifest_sha256": _sha256(manifest_path),
            "requests": request_info,
            "outputs": outputs,
        },
    )


def write_batched_prefill_manifest(
    receipt_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    expected_species: Sequence[str],
    expected_chroms: Sequence[str],
) -> None:
    """Close the batched prefill gate only after every exact receipt exists."""
    receipts = [json.loads(Path(path).read_text()) for path in receipt_paths]
    hal = [
        item for item in receipts if item.get("kind") == "issue_473_batched_core_hal"
    ]
    maf = [
        item for item in receipts if item.get("kind") == "issue_473_batched_core_maf"
    ]
    assert {str(item["target_species"]) for item in hal} == set(expected_species)
    assert {
        Path(str(item["maf_path"])).name.removesuffix(".maf.gz") for item in maf
    } == set(expected_chroms)
    assert len(hal) == len(expected_species) and len(maf) == len(expected_chroms)
    for item in receipts:
        assert item["runs"] == list(CORE_RUNS)
        assert set(item["outputs"]) == set(CORE_RUNS)
    _write_json_atomic(
        output_path,
        {
            "schema_version": 1,
            "kind": "issue_473_batched_core_prefill",
            "runs": list(CORE_RUNS),
            "hal_species": sorted(expected_species),
            "multiz_chroms": sorted(expected_chroms),
            "hal_receipts": len(hal),
            "maf_receipts": len(maf),
            "receipt_sha256": {str(path): _sha256(path) for path in receipt_paths},
        },
    )
