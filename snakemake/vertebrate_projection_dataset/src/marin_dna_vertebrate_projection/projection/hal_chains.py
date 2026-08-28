"""Whole-genome HAL-to-UCSC-chain generation and parity benchmarks."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

_GENOME_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_BED_COLUMNS = [
    "t_chrom",
    "t_start",
    "t_end",
    "query_name",
    "score",
    "t_strand",
]
_BED_SCHEMA = {
    "t_chrom": pl.String,
    "t_start": pl.Int64,
    "t_end": pl.Int64,
    "query_name": pl.String,
    "score": pl.Int64,
    "t_strand": pl.String,
}


def _validate_genome_name(name: str) -> str:
    assert _GENOME_NAME.fullmatch(name), f"unsafe HAL genome name: {name!r}"
    return name


def _atomic_json(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    partial.replace(output)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_chrom_sizes(path: str | Path) -> dict[str, int]:
    """Read a two-column chromosome-size file with uniqueness checks."""
    sizes: dict[str, int] = {}
    with Path(path).open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.rstrip("\n").split("\t")
            assert len(fields) == 2, f"invalid chrom sizes line {line_number}: {raw_line!r}"
            name, size_text = fields
            size = int(size_text)
            assert name and name not in sizes and size > 0
            sizes[name] = size
    assert sizes, f"empty chromosome-size file: {path}"
    return sizes


def write_hal_genome_assets(
    hal_path: str | Path,
    genome: str,
    chrom_sizes_path: str | Path,
    whole_genome_bed_path: str | Path,
    twobit_path: str | Path,
) -> None:
    """Stream one HAL genome into the assets required by ``axtChain``."""
    genome = _validate_genome_name(genome)
    chrom_sizes = Path(chrom_sizes_path)
    whole_bed = Path(whole_genome_bed_path)
    twobit = Path(twobit_path)
    for output in (chrom_sizes, whole_bed, twobit):
        output.parent.mkdir(parents=True, exist_ok=True)

    sizes_partial = chrom_sizes.with_name(f".{chrom_sizes.name}.partial")
    bed_partial = whole_bed.with_name(f".{whole_bed.name}.partial")
    twobit_partial = twobit.with_name(f".{twobit.name}.partial")
    for partial in (sizes_partial, bed_partial, twobit_partial):
        partial.unlink(missing_ok=True)

    try:
        with sizes_partial.open("w") as handle:
            subprocess.run(
                ["halStats", "--chromSizes", genome, str(hal_path)],
                check=True,
                stdout=handle,
            )
        sizes = read_chrom_sizes(sizes_partial)
        with bed_partial.open("w") as handle:
            for sequence, size in sizes.items():
                handle.write(f"{sequence}\t0\t{size}\n")

        hal2fasta = subprocess.Popen(
            ["hal2fasta", str(hal_path), genome],
            stdout=subprocess.PIPE,
        )
        assert hal2fasta.stdout is not None
        fatotwobit = subprocess.run(
            ["faToTwoBit", "stdin", str(twobit_partial)],
            stdin=hal2fasta.stdout,
            check=False,
        )
        hal2fasta.stdout.close()
        hal_returncode = hal2fasta.wait()
        assert hal_returncode == 0, f"hal2fasta failed for {genome}: {hal_returncode}"
        assert fatotwobit.returncode == 0, (
            f"faToTwoBit failed for {genome}: {fatotwobit.returncode}"
        )
        assert twobit_partial.stat().st_size > 0

        observed = subprocess.run(
            ["twoBitInfo", str(twobit_partial), "/dev/stdout"],
            check=True,
            capture_output=True,
            text=True,
        )
        observed_path = twobit_partial.with_name(f".{twobit.name}.sizes")
        observed_path.write_text(observed.stdout)
        try:
            assert read_chrom_sizes(observed_path) == sizes, (
                f"HAL and 2bit chromosome sizes differ for {genome}"
            )
        finally:
            observed_path.unlink(missing_ok=True)

        sizes_partial.replace(chrom_sizes)
        bed_partial.replace(whole_bed)
        twobit_partial.replace(twobit)
    finally:
        for partial in (sizes_partial, bed_partial, twobit_partial):
            partial.unlink(missing_ok=True)


def build_hal_to_chain_pipeline(
    *,
    hal_path: str | Path,
    query_genome: str,
    query_bed: str | Path,
    target_genome: str,
    target_twobit: str | Path,
    query_twobit: str | Path,
    output_chain: str | Path,
    no_dupes: bool,
    linear_gap: str,
) -> str:
    """Return the released Cactus HAL→PSL→chain pipeline as safe shell text."""
    query_genome = _validate_genome_name(query_genome)
    target_genome = _validate_genome_name(target_genome)
    assert linear_gap in {"medium", "loose"}
    hal_command = ["halLiftover"]
    if no_dupes:
        hal_command.append("--noDupes")
    hal_command.extend(
        [
            "--outPSL",
            str(hal_path),
            query_genome,
            str(query_bed),
            target_genome,
            "/dev/stdout",
        ]
    )
    commands = [
        shlex.join(hal_command),
        shlex.join(["pslPosTarget", "/dev/stdin", "/dev/stdout"]),
        shlex.join(
            [
                "axtChain",
                "-psl",
                "-verbose=0",
                f"-linearGap={linear_gap}",
                "/dev/stdin",
                str(target_twobit),
                str(query_twobit),
                "/dev/stdout",
            ]
        ),
        shlex.join(["gzip", "-n", "-c"]),
    ]
    return " | ".join(commands) + f" > {shlex.quote(str(output_chain))}"


def _read_meminfo() -> dict[str, int]:
    fields: dict[str, int] = {}
    with Path("/proc/meminfo").open() as handle:
        for raw_line in handle:
            key, value = raw_line.split(":", 1)
            amount = int(value.strip().split()[0]) * 1024
            fields[key] = amount
    return fields


class _SystemMonitor:
    def __init__(self, path: str | Path, *, interval_seconds: float = 5.0) -> None:
        self.path = Path(path)
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.samples: list[dict[str, int | float | str]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        memory = _read_meminfo()
        disk = shutil.disk_usage(self.path.parent)
        self.samples.append(
            {
                "time_utc": datetime.now(UTC).isoformat(),
                "monotonic_seconds": time.monotonic(),
                "mem_available_bytes": memory["MemAvailable"],
                "cached_bytes": memory.get("Cached", 0),
                "dirty_bytes": memory.get("Dirty", 0),
                "disk_free_bytes": disk.free,
            }
        )

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(self.interval_seconds)

    def __enter__(self) -> _SystemMonitor:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join()
        self._sample()

    def summary(self) -> dict[str, int | float]:
        assert self.samples
        return {
            "sample_count": len(self.samples),
            "sample_interval_seconds": self.interval_seconds,
            "minimum_mem_available_bytes": min(
                int(sample["mem_available_bytes"]) for sample in self.samples
            ),
            "maximum_cached_bytes": max(
                int(sample["cached_bytes"]) for sample in self.samples
            ),
            "maximum_dirty_bytes": max(
                int(sample["dirty_bytes"]) for sample in self.samples
            ),
            "minimum_disk_free_bytes": min(
                int(sample["disk_free_bytes"]) for sample in self.samples
            ),
        }


def _parse_gnu_time(path: str | Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or ": " not in line:
                continue
            key, value = line.split(": ", 1)
            metrics[key.strip()] = value.strip()
    return metrics


def _tool_version(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0] if output else "unknown"


def validate_chain_direction(
    chain_path: str | Path,
    *,
    target_chrom_sizes: str | Path,
    query_chrom_sizes: str | Path,
) -> dict[str, int]:
    """Require every chain header to use the requested target/query assemblies."""
    target_sizes = read_chrom_sizes(target_chrom_sizes)
    query_sizes = read_chrom_sizes(query_chrom_sizes)
    chain_count = 0
    aligned_bases = 0
    with gzip.open(chain_path, "rt") as handle:
        for raw_line in handle:
            if not raw_line.startswith("chain "):
                stripped = raw_line.strip()
                if stripped:
                    aligned_bases += int(stripped.split()[0])
                continue
            fields = raw_line.split()
            assert len(fields) == 13, f"invalid chain header: {raw_line!r}"
            assert fields[2] in target_sizes, f"unexpected chain tName: {fields[2]}"
            assert int(fields[3]) == target_sizes[fields[2]]
            assert fields[4] == "+"
            assert fields[7] in query_sizes, f"unexpected chain qName: {fields[7]}"
            assert int(fields[8]) == query_sizes[fields[7]]
            assert fields[9] in {"+", "-"}
            chain_count += 1
    assert chain_count > 0, f"chain contains no headers: {chain_path}"
    assert aligned_bases > 0
    return {"chain_count": chain_count, "aligned_block_bases": aligned_bases}


def run_hal_to_chain(
    *,
    hal_path: str | Path,
    query_genome: str,
    query_bed: str | Path,
    query_chrom_sizes: str | Path,
    query_twobit: str | Path,
    target_genome: str,
    target_chrom_sizes: str | Path,
    target_twobit: str | Path,
    output_chain: str | Path,
    output_metrics: str | Path,
    no_dupes: bool,
    linear_gap: str = "medium",
) -> dict[str, object]:
    """Generate one whole-genome chain atomically and record resource metrics."""
    output = Path(output_chain)
    metrics_path = Path(output_metrics)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    time_file = metrics_path.with_name(f".{metrics_path.name}.time.partial")
    stderr_file = metrics_path.with_name(f".{metrics_path.name}.stderr.partial")
    for path in (partial, time_file, stderr_file):
        path.unlink(missing_ok=True)

    command = build_hal_to_chain_pipeline(
        hal_path=hal_path,
        query_genome=query_genome,
        query_bed=query_bed,
        target_genome=target_genome,
        target_twobit=target_twobit,
        query_twobit=query_twobit,
        output_chain=partial,
        no_dupes=no_dupes,
        linear_gap=linear_gap,
    )
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    try:
        with _SystemMonitor(output) as monitor, stderr_file.open("w") as stderr:
            subprocess.run(
                [
                    "/usr/bin/time",
                    "-v",
                    "-o",
                    str(time_file),
                    "bash",
                    "-o",
                    "pipefail",
                    "-c",
                    command,
                ],
                check=True,
                stderr=stderr,
            )
        wall_seconds = time.perf_counter() - started
        direction = validate_chain_direction(
            partial,
            target_chrom_sizes=target_chrom_sizes,
            query_chrom_sizes=query_chrom_sizes,
        )
        payload: dict[str, object] = {
            "schema_version": 1,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "wall_seconds": wall_seconds,
            "query_genome": query_genome,
            "target_genome": target_genome,
            "chain_direction": f"{target_genome}_to_{query_genome}",
            "no_dupes": no_dupes,
            "linear_gap": linear_gap,
            "command": command.replace(str(partial), str(output)),
            "hal_bytes": Path(hal_path).stat().st_size,
            "chain_bytes": partial.stat().st_size,
            "chain_sha256": _sha256(partial),
            "direction_audit": direction,
            "system_monitor": monitor.summary(),
            "gnu_time": _parse_gnu_time(time_file),
            "tool_versions": {
                "halLiftover": _tool_version(["halLiftover", "--help"]),
                "axtChain": _tool_version(["axtChain"]),
            },
            "stderr_bytes": stderr_file.stat().st_size,
            "stderr_sha256": _sha256(stderr_file),
        }
        partial.replace(output)
        _atomic_json(metrics_path, payload)
        stderr_file.unlink(missing_ok=True)
        return payload
    finally:
        for path in (partial, time_file):
            path.unlink(missing_ok=True)
        if stderr_file.exists() and stderr_file.stat().st_size == 0:
            stderr_file.unlink()


def _count_bed_records(path: str | Path) -> int:
    count = 0
    with Path(path).open() as handle:
        for raw_line in handle:
            if raw_line.strip() and not raw_line.startswith("#"):
                count += 1
    return count


def run_liftover_benchmark(
    *,
    input_bed: str | Path,
    chain_path: str | Path,
    mapped_bed: str | Path,
    unmapped_bed: str | Path,
    metrics_path: str | Path,
    expected_queries: int,
    min_match: float = 0.95,
) -> dict[str, object]:
    """Run UCSC liftOver once and require complete mapped/unmapped accounting."""
    assert 0 < min_match <= 1
    mapped = Path(mapped_bed)
    unmapped = Path(unmapped_bed)
    metrics = Path(metrics_path)
    for output in (mapped, unmapped, metrics):
        output.parent.mkdir(parents=True, exist_ok=True)
    mapped_partial = mapped.with_name(f".{mapped.name}.partial")
    unmapped_partial = unmapped.with_name(f".{unmapped.name}.partial")
    time_file = metrics.with_name(f".{metrics.name}.time.partial")
    for partial in (mapped_partial, unmapped_partial, time_file):
        partial.unlink(missing_ok=True)
    command = [
        "liftOver",
        f"-minMatch={min_match:g}",
        str(input_bed),
        str(chain_path),
        str(mapped_partial),
        str(unmapped_partial),
    ]
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    try:
        with _SystemMonitor(mapped) as monitor:
            subprocess.run(
                ["/usr/bin/time", "-v", "-o", str(time_file), *command],
                check=True,
            )
        wall_seconds = time.perf_counter() - started
        mapped_records = _count_bed_records(mapped_partial)
        unmapped_records = _count_bed_records(unmapped_partial)
        assert mapped_records + unmapped_records == expected_queries, (
            f"liftOver accounting mismatch: {mapped_records} + {unmapped_records} "
            f"!= {expected_queries}"
        )
        payload: dict[str, object] = {
            "schema_version": 1,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "wall_seconds": wall_seconds,
            "expected_queries": expected_queries,
            "mapped_queries": mapped_records,
            "unmapped_queries": unmapped_records,
            "min_match": min_match,
            "chain_sha256": _sha256(chain_path),
            "command": shlex.join(command).replace(
                str(mapped_partial), str(mapped)
            ).replace(str(unmapped_partial), str(unmapped)),
            "system_monitor": monitor.summary(),
            "gnu_time": _parse_gnu_time(time_file),
        }
        mapped_partial.replace(mapped)
        unmapped_partial.replace(unmapped)
        _atomic_json(metrics, payload)
        return payload
    finally:
        for partial in (mapped_partial, unmapped_partial, time_file):
            partial.unlink(missing_ok=True)


def _read_bed6(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if path.stat().st_size == 0:
        return pl.DataFrame(schema=_BED_SCHEMA)
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        comment_prefix="#",
        new_columns=_BED_COLUMNS,
        schema_overrides=_BED_SCHEMA,
    )


def _mapping_sets(path: str | Path, prefix: str) -> pl.DataFrame:
    frame = _read_bed6(path).with_columns(
        pl.concat_str(
            [
                pl.col("t_chrom"),
                pl.col("t_start").cast(pl.String),
                pl.col("t_end").cast(pl.String),
                pl.col("t_strand"),
            ],
            separator=":",
        ).alias("mapping")
    )
    return frame.group_by("query_name").agg(
        pl.col("mapping").sort().alias(f"{prefix}_mappings"),
        pl.len().alias(f"{prefix}_mapping_count"),
    )


def write_chain_parity_audit(
    *,
    input_bed: str | Path,
    direct_bed: str | Path,
    chain_bed: str | Path,
    summary_path: str | Path,
    discrepancies_path: str | Path,
    expected_queries: int,
) -> dict[str, object]:
    """Compare all direct-HAL and chain mapping sets by stable query name."""
    queries = _read_bed6(input_bed).select("query_name")
    assert queries.height == expected_queries
    assert queries["query_name"].n_unique() == expected_queries
    direct = _mapping_sets(direct_bed, "direct")
    chain = _mapping_sets(chain_bed, "chain")
    unknown_direct = direct.join(queries, on="query_name", how="anti").height
    unknown_chain = chain.join(queries, on="query_name", how="anti").height
    assert unknown_direct == 0 and unknown_chain == 0

    audit = (
        queries.join(direct, on="query_name", how="left")
        .join(chain, on="query_name", how="left")
        .with_columns(
            pl.col("direct_mapping_count").fill_null(0),
            pl.col("chain_mapping_count").fill_null(0),
        )
        .with_columns(
            pl.when(
                (pl.col("direct_mapping_count") == 0)
                & (pl.col("chain_mapping_count") == 0)
            )
            .then(pl.lit("exact_unmapped"))
            .when(
                (pl.col("direct_mapping_count") > 0)
                & (pl.col("chain_mapping_count") > 0)
                & (pl.col("direct_mappings") == pl.col("chain_mappings"))
            )
            .then(pl.lit("exact_mapped"))
            .when(
                (pl.col("direct_mapping_count") == 0)
                & (pl.col("chain_mapping_count") > 0)
            )
            .then(pl.lit("chain_only"))
            .when(
                (pl.col("direct_mapping_count") > 0)
                & (pl.col("chain_mapping_count") == 0)
            )
            .then(pl.lit("direct_only"))
            .otherwise(pl.lit("mapping_conflict"))
            .alias("parity_class")
        )
    )
    counts = {
        row["parity_class"]: row["len"]
        for row in audit.group_by("parity_class").len().to_dicts()
    }
    exact_queries = counts.get("exact_unmapped", 0) + counts.get("exact_mapped", 0)
    direct_mapped = int((audit["direct_mapping_count"] > 0).sum())
    payload: dict[str, object] = {
        "schema_version": 1,
        "expected_queries": expected_queries,
        "parity_counts": counts,
        "exact_queries": exact_queries,
        "exact_fraction": exact_queries / expected_queries,
        "direct_mapped_queries": direct_mapped,
        "exact_mapped_fraction_of_direct_mapped": (
            counts.get("exact_mapped", 0) / direct_mapped if direct_mapped else 1.0
        ),
        "direct_multiple_mapping_queries": int(
            (audit["direct_mapping_count"] > 1).sum()
        ),
        "chain_multiple_mapping_queries": int(
            (audit["chain_mapping_count"] > 1).sum()
        ),
    }
    discrepancies = audit.filter(~pl.col("parity_class").str.starts_with("exact_"))
    output = Path(discrepancies_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    discrepancies.write_parquet(output)
    _atomic_json(summary_path, payload)
    return payload


def write_uniform_grid_center_bed(
    *,
    chrom_sizes_path: str | Path,
    undefined_bed_path: str | Path,
    output_bed_path: str | Path,
    standard_chroms: list[str],
    window_size: int,
    step_size: int,
    expected_queries: int,
) -> None:
    """Write the exact 0-based center-1 requests for the uniform human grid."""
    assert window_size > 0 and window_size % 2 == 1 and step_size > 0
    sizes = read_chrom_sizes(chrom_sizes_path)
    assert set(standard_chroms) == set(sizes)
    undefined: dict[str, list[tuple[int, int]]] = {chrom: [] for chrom in sizes}
    with Path(undefined_bed_path).open() as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.startswith("#"):
                continue
            chrom, start_text, end_text, *_ = raw_line.rstrip("\n").split("\t")
            if chrom in undefined:
                start, end = int(start_text), int(end_text)
                assert 0 <= start < end <= sizes[chrom]
                undefined[chrom].append((start, end))
    for intervals in undefined.values():
        intervals.sort()

    output = Path(output_bed_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    partial.unlink(missing_ok=True)
    count = 0
    center_offset = window_size // 2
    try:
        with partial.open("w") as handle:
            for chrom in standard_chroms:
                intervals = undefined[chrom]
                interval_index = 0
                for window_index, start in enumerate(
                    range(0, sizes[chrom] - window_size + 1, step_size),
                    start=1,
                ):
                    end = start + window_size
                    while (
                        interval_index < len(intervals)
                        and intervals[interval_index][1] <= start
                    ):
                        interval_index += 1
                    overlaps_undefined = (
                        interval_index < len(intervals)
                        and intervals[interval_index][0] < end
                    )
                    if overlaps_undefined:
                        continue
                    center = start + center_offset
                    name = f"win_{chrom}_{window_index:09d}"
                    handle.write(f"{chrom}\t{center}\t{center + 1}\t{name}\t0\t+\n")
                    count += 1
        assert count == expected_queries, f"uniform-grid count {count} != {expected_queries}"
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
