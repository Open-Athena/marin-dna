"""Adaptive whole-genome HAL-to-chain generation across mammal targets."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from marin_dna_vertebrate_projection.mirror import s3_object_size, verify_hal_object
from marin_dna_vertebrate_projection.projection.hal_chains import (
    run_direction_matched_hal_to_chain,
    write_hal_genome_assets,
)
from marin_dna_vertebrate_projection.provenance import (
    hash_pipeline_config,
    write_producer_manifest,
)


@dataclass(frozen=True)
class RampThresholds:
    minimum_mem_available_bytes: int
    minimum_disk_free_bytes: int
    maximum_cpu_busy_fraction: float
    maximum_cpu_iowait_fraction: float
    maximum_load_per_cpu: float


def read_target_species(path: str | Path, source_genome: str) -> list[str]:
    """Read the family-deduplicated cohort, preserving manifest order."""
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    species = [str(row["species"]) for row in rows]
    assert species and len(species) == len(set(species))
    assert source_genome in species
    targets = [name for name in species if name != source_genome]
    assert len(targets) == 107
    return targets


def validate_smoke_gate_payloads(
    payloads: dict[str, dict[str, Any]], expected_queries: int = 9_374
) -> dict[str, dict[str, int | float]]:
    """Require exact regional parity before any ramp worker is launched."""
    assert set(payloads) == {
        "Papio_anubis",
        "Mus_musculus",
        "Loxodonta_africana",
    }
    validated: dict[str, dict[str, int | float]] = {}
    for species, payload in payloads.items():
        assert int(payload["expected_queries"]) == expected_queries
        assert int(payload["exact_queries"]) == expected_queries
        assert float(payload["exact_fraction"]) == 1.0
        assert int(payload["chain_multiple_mapping_queries"]) == 0
        counts = payload["parity_counts"]
        assert isinstance(counts, dict)
        exact_mapped = int(counts["exact_mapped"])
        exact_unmapped = int(counts["exact_unmapped"])
        assert exact_mapped + exact_unmapped == expected_queries
        validated[species] = {
            "exact_queries": expected_queries,
            "exact_mapped": exact_mapped,
            "exact_unmapped": exact_unmapped,
            "exact_fraction": 1.0,
        }
    return validated


def next_concurrency(
    current: int,
    maximum: int,
    snapshot: dict[str, int | float],
    thresholds: RampThresholds,
) -> int:
    """Double concurrency only when every measured safety gate passes."""
    assert 1 <= current <= maximum
    safe = (
        int(snapshot["mem_available_bytes"])
        >= thresholds.minimum_mem_available_bytes
        and int(snapshot["disk_free_bytes"])
        >= thresholds.minimum_disk_free_bytes
        and float(snapshot["cpu_busy_fraction"])
        <= thresholds.maximum_cpu_busy_fraction
        and float(snapshot["cpu_iowait_fraction"])
        <= thresholds.maximum_cpu_iowait_fraction
        and float(snapshot["load_per_cpu"])
        <= thresholds.maximum_load_per_cpu
    )
    return min(maximum, current * 2) if safe else current


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    assert isinstance(payload, dict)
    assert payload["pipeline_version"] == "hal-chains-directional-ramp-v1"
    assert payload["tier"] == "full"
    assert payload["cactus_version"] == "3.3.0"
    assert int(payload["kent_version"]) == 482
    assert payload["source_genome"] == "Homo_sapiens"
    assert payload["linear_gap"] == "medium"
    assert int(payload["chain_min_score"]) == -1_000_000
    return payload


def _s3_bucket_key(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    assert parsed.scheme == "s3" and parsed.netloc and parsed.path.lstrip("/")
    return parsed.netloc, parsed.path.lstrip("/")


def _s3_size(uri: str) -> int | None:
    bucket, key = _s3_bucket_key(uri)
    result = subprocess.run(
        [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--query",
            "ContentLength",
            "--output",
            "text",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return int(result.stdout.strip()) if result.returncode == 0 else None


def _upload(path: Path, uri: str) -> None:
    subprocess.run(
        ["aws", "s3", "cp", str(path), uri, "--no-progress"], check=True
    )
    assert _s3_size(uri) == path.stat().st_size


def _download_json(uri: str, directory: Path) -> dict[str, Any]:
    output = directory / f"{len(list(directory.iterdir())):03d}.json"
    subprocess.run(
        ["aws", "s3", "cp", uri, str(output), "--no-progress"], check=True
    )
    payload = json.loads(output.read_text())
    assert isinstance(payload, dict)
    return payload


def _result_identity(
    config: dict[str, Any], pipeline_commit: str
) -> tuple[str, str, str]:
    assert len(pipeline_commit) == 40
    config_sha256 = hash_pipeline_config(config)
    relative = "/".join(
        [
            str(config["pipeline_version"]),
            pipeline_commit,
            config_sha256,
            str(config["tier"]),
        ]
    )
    return config_sha256, relative, (
        f"{str(config['s3_results_root']).rstrip('/')}/{relative}"
    )


def _asset_paths(asset_root: Path, genome: str) -> tuple[Path, Path, Path]:
    directory = asset_root / genome
    return (
        directory / "chrom.sizes",
        directory / "whole_genome.bed",
        directory / "genome.2bit",
    )


def _bootstrap_source_assets(config: dict[str, Any], asset_root: Path) -> None:
    source = str(config["source_genome"])
    destinations = _asset_paths(asset_root, source)
    bootstrap = Path(str(config["bootstrap_source_asset_root"]))
    sources = (
        bootstrap / f"{source}.chrom.sizes",
        bootstrap / f"{source}.whole_genome.bed",
        bootstrap / f"{source}.2bit",
    )
    for source_path, destination in zip(sources, destinations, strict=True):
        if destination.exists():
            continue
        if not source_path.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_path, destination)
        except OSError:
            shutil.copy2(source_path, destination)


def _ensure_assets(
    hal_path: Path, asset_root: Path, genome: str
) -> tuple[Path, Path, Path]:
    paths = _asset_paths(asset_root, genome)
    lock = asset_root / f".{genome}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
            write_hal_genome_assets(hal_path, genome, *paths)
    return paths


def _species_uris(s3_base: str, species: str) -> tuple[str, str]:
    root = f"{s3_base}/chains/{species}"
    return f"{root}/human_to_species.chain.gz", f"{root}/chain_generation.json"


def _external_species_status(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for species, raw_uris in dict(config["external_species"]).items():
        uris = dict(raw_uris)
        chain_bytes = _s3_size(str(uris["chain_s3_uri"]))
        metrics_bytes = _s3_size(str(uris["metrics_s3_uri"]))
        statuses[str(species)] = {
            "complete": (chain_bytes or 0) > 0 and (metrics_bytes or 0) > 0,
            "chain_bytes": chain_bytes,
            "metrics_bytes": metrics_bytes,
        }
    return statuses


def run_worker(config_path: str | Path, pipeline_commit: str, species: str) -> None:
    """Generate, verify, and durably upload one target chain."""
    config = _load_config(config_path)
    targets = read_target_species(config["target_manifest"], config["source_genome"])
    assert species in targets
    config_sha256, relative, s3_base = _result_identity(config, pipeline_commit)
    chain_uri, metrics_uri = _species_uris(s3_base, species)
    if (_s3_size(chain_uri) or 0) > 0 and (_s3_size(metrics_uri) or 0) > 0:
        return

    hal_path = Path(str(config["hal_stage_path"]))
    assert hal_path.is_file()
    asset_root = Path(str(config["local_asset_root"]))
    source = str(config["source_genome"])
    source_sizes, source_bed, source_twobit = _ensure_assets(
        hal_path, asset_root, source
    )
    destination_sizes, _, destination_twobit = _ensure_assets(
        hal_path, asset_root, species
    )

    local_root = Path(str(config["local_work_root"])) / relative / "chains" / species
    chain = local_root / "human_to_species.chain.gz"
    metrics = local_root / "chain_generation.json"
    chain.unlink(missing_ok=True)
    metrics.unlink(missing_ok=True)
    payload = run_direction_matched_hal_to_chain(
        hal_path=hal_path,
        source_genome=source,
        source_bed=source_bed,
        source_chrom_sizes=source_sizes,
        source_twobit=source_twobit,
        destination_genome=species,
        destination_chrom_sizes=destination_sizes,
        destination_twobit=destination_twobit,
        output_chain=chain,
        output_metrics=metrics,
        min_score=int(config["chain_min_score"]),
        linear_gap=str(config["linear_gap"]),
    )
    payload["pipeline_commit"] = pipeline_commit
    payload["config_sha256"] = config_sha256
    _atomic_json(metrics, payload)
    _upload(chain, chain_uri)
    _upload(metrics, metrics_uri)
    chain.unlink()
    metrics.unlink()


def _read_cpu_counters() -> tuple[int, int, int]:
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    total = sum(values)
    idle = values[3]
    iowait = values[4] if len(values) > 4 else 0
    return total, idle, iowait


def _node_snapshot(
    work_root: Path,
    previous_cpu: tuple[int, int, int],
) -> tuple[dict[str, int | float], tuple[int, int, int]]:
    current_cpu = _read_cpu_counters()
    total_delta = max(1, current_cpu[0] - previous_cpu[0])
    idle_delta = current_cpu[1] - previous_cpu[1]
    iowait_delta = current_cpu[2] - previous_cpu[2]
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0]) * 1024
    cpu_count = os.cpu_count() or 1
    snapshot: dict[str, int | float] = {
        "time_utc": datetime.now(UTC).timestamp(),
        "mem_available_bytes": meminfo["MemAvailable"],
        "disk_free_bytes": shutil.disk_usage(work_root).free,
        "cpu_busy_fraction": max(0.0, (total_delta - idle_delta) / total_delta),
        "cpu_iowait_fraction": max(0.0, iowait_delta / total_delta),
        "load_per_cpu": os.getloadavg()[0] / cpu_count,
    }
    return snapshot, current_cpu


def _write_controller_metadata(
    config: dict[str, Any],
    pipeline_commit: str,
    config_sha256: str,
    relative: str,
    s3_base: str,
    targets: list[str],
    validated_smoke: dict[str, dict[str, int | float]],
) -> None:
    root = Path(str(config["local_work_root"])) / relative / "metadata"
    producer = root / "producer.json"
    write_producer_manifest(
        producer,
        pipeline_commit=pipeline_commit,
        config_sha256=config_sha256,
        pipeline_version=str(config["pipeline_version"]),
        tier=str(config["tier"]),
    )
    cohort = root / "cohort.json"
    _atomic_json(
        cohort,
        {
            "schema_version": 1,
            "source_genome": config["source_genome"],
            "target_count": len(targets),
            "targets": targets,
            "external_species": config["external_species"],
        },
    )
    smoke = root / "smoke-gate.json"
    _atomic_json(smoke, {"schema_version": 1, "species": validated_smoke})
    for path in (producer, cohort, smoke):
        _upload(path, f"{s3_base}/metadata/{path.name}")


def run_controller(config_path: str | Path, pipeline_commit: str) -> int:
    """Run the bounded adaptive-concurrency controller until the cohort finishes."""
    config = _load_config(config_path)
    targets = read_target_species(config["target_manifest"], config["source_genome"])
    external = set(config["external_species"])
    assert external <= set(targets)
    scheduled_targets = [species for species in targets if species not in external]
    priority = [str(species) for species in config["priority_species"]]
    assert set(priority) <= set(scheduled_targets)
    scheduled_targets = priority + [
        species for species in scheduled_targets if species not in set(priority)
    ]

    config_sha256, relative, s3_base = _result_identity(config, pipeline_commit)
    work_root = Path(str(config["local_work_root"]))
    work_root.mkdir(parents=True, exist_ok=True)
    state_root = work_root / relative / "controller"
    state_root.mkdir(parents=True, exist_ok=True)
    hal_path = Path(str(config["hal_stage_path"]))
    verify_hal_object(hal_path, expected_size=s3_object_size(config["hal_s3_uri"]))
    asset_root = Path(str(config["local_asset_root"]))
    _bootstrap_source_assets(config, asset_root)
    _ensure_assets(hal_path, asset_root, str(config["source_genome"]))

    with tempfile.TemporaryDirectory(dir=state_root) as temp_directory:
        temp = Path(temp_directory)
        smoke_payloads = {
            species: _download_json(str(uri), temp)
            for species, uri in dict(config["smoke_gate_uris"]).items()
        }
    validated_smoke = validate_smoke_gate_payloads(smoke_payloads)
    _write_controller_metadata(
        config,
        pipeline_commit,
        config_sha256,
        relative,
        s3_base,
        targets,
        validated_smoke,
    )

    completed: list[str] = []
    remaining: deque[str] = deque()
    for species in scheduled_targets:
        chain_uri, metrics_uri = _species_uris(s3_base, species)
        if (_s3_size(chain_uri) or 0) > 0 and (_s3_size(metrics_uri) or 0) > 0:
            completed.append(species)
        else:
            remaining.append(species)

    thresholds = RampThresholds(
        minimum_mem_available_bytes=int(config["minimum_mem_available_bytes"]),
        minimum_disk_free_bytes=int(config["minimum_disk_free_bytes"]),
        maximum_cpu_busy_fraction=float(config["maximum_cpu_busy_fraction"]),
        maximum_cpu_iowait_fraction=float(config["maximum_cpu_iowait_fraction"]),
        maximum_load_per_cpu=float(config["maximum_load_per_cpu"]),
    )
    target_concurrency = int(config["initial_concurrency"])
    maximum_concurrency = int(config["maximum_concurrency"])
    assert 1 <= target_concurrency <= maximum_concurrency <= (os.cpu_count() or 1)
    ramp_interval = float(config["ramp_interval_seconds"])
    poll_interval = float(config["poll_interval_seconds"])
    upload_interval = float(config["state_upload_interval_seconds"])
    maximum_attempts = int(config["maximum_attempts"])
    attempts: dict[str, int] = {}
    active: dict[str, tuple[subprocess.Popen[bytes], Any]] = {}
    failures: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    previous_cpu = _read_cpu_counters()
    last_ramp = time.monotonic()
    last_upload = 0.0
    state_path = state_root / "controller-state.json"
    state_uri = f"{s3_base}/metadata/controller-state.json"

    def record(event: str, **details: Any) -> None:
        events.append(
            {"time_utc": datetime.now(UTC).isoformat(), "event": event, **details}
        )

    def write_state(snapshot: dict[str, int | float], status: str) -> None:
        external_status = _external_species_status(config)
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": status,
                "pipeline_commit": pipeline_commit,
                "config_sha256": config_sha256,
                "target_species": len(targets),
                "externally_managed_species": sorted(external),
                "external_species_status": external_status,
                "scheduled_species": len(scheduled_targets),
                "target_concurrency": target_concurrency,
                "active_species": sorted(active),
                "completed_species": sorted(completed),
                "failed_species": failures,
                "remaining_species": list(remaining),
                "node": snapshot,
                "events": events,
            },
        )
        _upload(state_path, state_uri)

    def launch(species: str) -> None:
        attempts[species] = attempts.get(species, 0) + 1
        log_path = state_root / "logs" / f"{species}.attempt-{attempts[species]}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("wb")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "marin_dna_vertebrate_projection.cli.run_hal_chain_ramp",
                "worker",
                "--config",
                str(Path(config_path).resolve()),
                "--pipeline-commit",
                pipeline_commit,
                "--species",
                species,
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        active[species] = (process, handle)
        record("worker_started", species=species, attempt=attempts[species])

    record("controller_started", initial_concurrency=target_concurrency)
    snapshot: dict[str, int | float] = {
        "time_utc": datetime.now(UTC).timestamp(),
        "mem_available_bytes": 0,
        "disk_free_bytes": 0,
        "cpu_busy_fraction": 0.0,
        "cpu_iowait_fraction": 0.0,
        "load_per_cpu": 0.0,
    }
    try:
        while remaining or active:
            for species, (process, handle) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                handle.close()
                del active[species]
                if returncode == 0:
                    completed.append(species)
                    record("worker_completed", species=species)
                elif attempts[species] < maximum_attempts:
                    remaining.appendleft(species)
                    record(
                        "worker_retry_queued", species=species, returncode=returncode
                    )
                else:
                    failures[species] = returncode
                    record("worker_failed", species=species, returncode=returncode)

            while remaining and len(active) < target_concurrency:
                launch(remaining.popleft())

            time.sleep(poll_interval)
            snapshot, previous_cpu = _node_snapshot(work_root, previous_cpu)
            now = time.monotonic()
            if now - last_ramp >= ramp_interval and remaining:
                proposed = next_concurrency(
                    target_concurrency,
                    maximum_concurrency,
                    snapshot,
                    thresholds,
                )
                if proposed > target_concurrency:
                    record(
                        "concurrency_increased",
                        previous=target_concurrency,
                        current=proposed,
                        node=snapshot,
                    )
                    target_concurrency = proposed
                else:
                    record(
                        "concurrency_held", current=target_concurrency, node=snapshot
                    )
                last_ramp = now
            if now - last_upload >= upload_interval:
                write_state(snapshot, "running")
                last_upload = now
    except BaseException:
        record("controller_interrupted")
        for process, _ in active.values():
            os.killpg(process.pid, signal.SIGTERM)
        for process, handle in active.values():
            process.wait()
            handle.close()
        write_state(snapshot, "interrupted")
        raise

    external_status = _external_species_status(config)
    incomplete_external = [
        species
        for species, species_status in external_status.items()
        if not species_status["complete"]
    ]
    if incomplete_external:
        record("external_species_incomplete", species=incomplete_external)
    status = "failed" if failures or incomplete_external else "succeeded"
    record("controller_finished", status=status)
    write_state(snapshot, status)
    done = state_root / "controller.done"
    done.write_text("1\n" if status == "failed" else "0\n")
    _upload(done, f"{s3_base}/metadata/controller.done")
    return 1 if status == "failed" else 0
