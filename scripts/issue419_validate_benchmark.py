"""Validate and summarize the issue #419 one-scaffold benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyBigWig

from marin_dna.data.dna import NUCLEOTIDES
from marin_dna.pipelines.chinchilla_logo import (
    load_score_shard,
    logo_from_log_probabilities,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--gpu-hourly-cost", type=float, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--max-sampled-shards", type=int, default=25)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ucsc-hub-check-kent-version", type=int)
    parser.add_argument(
        "--update-release-manifest",
        action="store_true",
        help="Record benchmark and round-trip validation results in release.json.",
    )
    return parser.parse_args()


def _numeric_field(value: str) -> float:
    return float(value.strip().split()[0])


def summarize_gpu_csv(path: Path) -> dict[str, Any]:
    rows: list[list[str]] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if row and row[0].strip() != "timestamp":
                assert len(row) == 9, row
                rows.append(row)
    assert rows, f"no GPU samples in {path}"
    utilization = np.array([_numeric_field(row[2]) for row in rows])
    memory_mib = np.array([_numeric_field(row[4]) for row in rows])
    power_watts = np.array([_numeric_field(row[6]) for row in rows])
    temperature_c = np.array([_numeric_field(row[7]) for row in rows])
    timestamps = [
        datetime.strptime(
            row[0].strip(),
            "%Y/%m/%d %H:%M:%S.%f" if "." in row[0] else "%Y/%m/%d %H:%M:%S",
        )
        for row in rows
    ]
    assert timestamps == sorted(timestamps), "GPU timestamps are not monotonic"
    active = utilization > 0
    assert active.any(), "GPU monitor contains no active samples"
    return {
        "sample_count": len(rows),
        "sampling_interval_seconds": 5,
        "monitoring_started_at": timestamps[0].isoformat(),
        "monitoring_ended_at": timestamps[-1].isoformat(),
        "monitoring_wall_seconds": (timestamps[-1] - timestamps[0]).total_seconds(),
        "gpu_name": rows[0][1].strip(),
        "active_sample_count": int(active.sum()),
        "active_utilization_mean_percent": float(utilization[active].mean()),
        "active_utilization_p50_percent": float(np.quantile(utilization[active], 0.5)),
        "active_utilization_p95_percent": float(np.quantile(utilization[active], 0.95)),
        "peak_memory_mib": float(memory_mib.max()),
        "peak_power_watts": float(power_watts.max()),
        "peak_temperature_c": float(temperature_c.max()),
    }


def update_release_manifest(
    path: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    ucsc_hub_check_kent_version: int | None = None,
) -> None:
    """Record measured cost and external round-trip checks in the release manifest."""
    scoring = summary["full_scaffold_scoring"]
    manifest["benchmark"] = {
        "full_scaffold_scoring": {
            key: scoring[key]
            for key in (
                "chrom",
                "window_count",
                "logical_scored_sequence_count",
                "scored_base_count",
                "model_inference_seconds",
                "wall_seconds_this_invocation",
                "bases_per_second",
                "batch_size",
                "num_workers",
                "bf16_full_eval",
                "torch_compile",
                "peak_vram_bytes",
                "gpu_hourly_cost_usd",
                "model_inference_cost_usd",
                "model_inference_usd_per_billion_scored_bases",
            )
        },
        "batch_sweep": summary["batch_sweep"],
        "gpu_monitor": summary["gpu_monitor"],
        "bigwig_construction": summary["bigwig_construction"],
    }
    manifest["validation"]["bigwig_round_trip"] = {
        "status": "passed",
        **summary["validation"],
    }
    hub_check: dict[str, Any] = {"status": "pending"}
    if ucsc_hub_check_kent_version is not None:
        assert ucsc_hub_check_kent_version > 0
        hub_check = {
            "status": "passed",
            "kent_source_version": ucsc_hub_check_kent_version,
            "checked_remote_bigwig_count": 8,
        }
    manifest["validation"]["ucsc_rendering"] = {
        "hub_check": hub_check,
        "manual_browser_rendering": "pending user review",
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _flat_index_to_position(shard: Any, flat_index: int) -> int:
    assert 0 <= flat_index < len(shard.log_probabilities)
    window_index = int(
        np.searchsorted(shard.score_offsets, flat_index, side="right") - 1
    )
    within_window = flat_index - int(shard.score_offsets[window_index])
    position = int(shard.emit_start[window_index]) + within_window
    assert position < int(shard.emit_end[window_index])
    return position


def validate_artifacts(
    results_root: Path,
    manifest: dict[str, Any],
    max_sampled_shards: int,
) -> dict[str, Any]:
    release_root = results_root / "release"
    for relative_path, identity in manifest["files"].items():
        path = release_root / relative_path
        assert path.stat().st_size == identity["bytes"]
        assert sha256_file(path) == identity["sha256"]

    expected_covered_bases = int(manifest["coverage"]["scored_bases"])
    bigwigs = {
        (kind, base): pyBigWig.open(str(release_root / "bigwig" / kind / f"{base}.bw"))
        for kind in ("logprob", "logo")
        for base in NUCLEOTIDES
    }
    try:
        for handle in bigwigs.values():
            assert handle.header()["nBasesCovered"] == expected_covered_bases

        shard_paths = sorted((results_root / "shards").glob("*/part-*.npz"))
        assert shard_paths
        selected_indices = np.linspace(
            0,
            len(shard_paths) - 1,
            min(max_sampled_shards, len(shard_paths)),
            dtype=int,
        )
        sampled_positions = 0
        max_logprob_absolute_error = 0.0
        max_logo_absolute_error = 0.0
        for shard_index in np.unique(selected_indices):
            shard = load_score_shard(shard_paths[int(shard_index)])
            flat_indices = np.unique(
                np.linspace(
                    0,
                    len(shard.log_probabilities) - 1,
                    min(5, len(shard.log_probabilities)),
                    dtype=int,
                )
            )
            for flat_index in flat_indices:
                position = _flat_index_to_position(shard, int(flat_index))
                expected_logp = shard.log_probabilities[int(flat_index)]
                expected_logo = logo_from_log_probabilities(expected_logp[None, :])
                expected_heights = expected_logo.glyph_heights_bits[0]
                observed_logp = np.array(
                    [
                        bigwigs[("logprob", base)].values(
                            shard.chrom, position, position + 1
                        )[0]
                        for base in NUCLEOTIDES
                    ],
                    dtype=np.float32,
                )
                observed_heights = np.array(
                    [
                        bigwigs[("logo", base)].values(
                            shard.chrom, position, position + 1
                        )[0]
                        for base in NUCLEOTIDES
                    ],
                    dtype=np.float32,
                )
                assert np.isfinite(observed_logp).all()
                assert np.isfinite(observed_heights).all()
                logp_error = float(np.max(np.abs(observed_logp - expected_logp)))
                logo_error = float(np.max(np.abs(observed_heights - expected_heights)))
                max_logprob_absolute_error = max(max_logprob_absolute_error, logp_error)
                max_logo_absolute_error = max(max_logo_absolute_error, logo_error)
                np.testing.assert_allclose(observed_logp, expected_logp, atol=1e-6)
                np.testing.assert_allclose(
                    observed_heights, expected_heights, atol=1e-6
                )
                sampled_positions += 1
    finally:
        for handle in bigwigs.values():
            handle.close()

    return {
        "manifest_file_count": len(manifest["files"]),
        "manifest_checksums_verified": True,
        "bigwig_covered_bases": expected_covered_bases,
        "sampled_shard_count": len(np.unique(selected_indices)),
        "sampled_position_count": sampled_positions,
        "max_logprob_absolute_error": max_logprob_absolute_error,
        "max_logo_absolute_error": max_logo_absolute_error,
    }


def main() -> None:
    args = parse_args()
    assert args.gpu_hourly_cost > 0
    assert len(args.expected_commit) == 40
    assert args.max_sampled_shards > 0
    results_root = args.results_root
    manifest = json.loads(
        (results_root / "release" / "manifest" / "release.json").read_text()
    )
    assert manifest["application"]["commit"] == args.expected_commit
    scoring = manifest["runtime"]["scoring"]
    assert len(scoring) == 1
    runtime = scoring[0]
    scored_bases = int(runtime["scored_base_count"])
    inference_seconds = float(runtime["model_inference_seconds"])
    scoring_wall_seconds = float(runtime["wall_seconds_this_invocation"])
    assert 0 < inference_seconds <= scoring_wall_seconds
    bases_per_second = scored_bases / inference_seconds

    gpu_monitor = summarize_gpu_csv(results_root / "benchmark" / "gpu.csv")
    gpu_monitor["gpu_hourly_cost_usd"] = args.gpu_hourly_cost
    gpu_monitor["monitored_cost_usd"] = (
        gpu_monitor["monitoring_wall_seconds"] / 3_600 * args.gpu_hourly_cost
    )
    summary = {
        "application_commit": args.expected_commit,
        "coverage": manifest["coverage"],
        "full_scaffold_scoring": {
            **runtime,
            "scoring_io_and_orchestration_seconds": (
                scoring_wall_seconds - inference_seconds
            ),
            "gpu_hourly_cost_usd": args.gpu_hourly_cost,
            "model_inference_cost_usd": (
                inference_seconds / 3_600 * args.gpu_hourly_cost
            ),
            "model_inference_usd_per_billion_scored_bases": (
                1_000_000_000 / bases_per_second / 3_600 * args.gpu_hourly_cost
            ),
        },
        "batch_sweep": json.loads(
            (results_root / "benchmark" / "batch_sweep" / "summary.json").read_text()
        ),
        "bigwig_construction": manifest["runtime"]["artifact_construction"][0],
        "gpu_monitor": gpu_monitor,
        "validation": validate_artifacts(
            results_root, manifest, args.max_sampled_shards
        ),
    }
    output = args.output or results_root / "benchmark" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.update_release_manifest:
        update_release_manifest(
            results_root / "release" / "manifest" / "release.json",
            manifest,
            summary,
            args.ucsc_hub_check_kent_version,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
