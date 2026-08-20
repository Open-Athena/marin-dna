"""Run the #473 report with rejection evidence scoped to fixed anchors.

The reused #417 rejection inventory covers a larger anchor catalog than issue
#473.  This additive entry point preserves the original report implementation
and filters its explicit, rejected-only out-of-scope rows before writing the
fixed-catalog outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.issue_473 import report as base_report


def build_fixed_catalog_outcome_counts(
    anchors: pl.DataFrame,
    species_manifest: pl.DataFrame,
    accepted_by_policy: Mapping[str, pl.LazyFrame],
    rejected_by_policy: Mapping[str, pl.LazyFrame],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Drop only rejection rows outside the declared fixed anchor catalog."""
    outcomes, summary = base_report.build_outcome_counts(
        anchors,
        species_manifest,
        accepted_by_policy,
        rejected_by_policy,
    )
    out_of_scope = outcomes.filter(pl.col("region_label").is_null())
    if not out_of_scope.is_empty():
        assert out_of_scope["projection_policy"].n_unique() == 1
        assert out_of_scope["projection_policy"][0] == "full_window"
        assert out_of_scope["outcome"].str.starts_with("rejected:").all()
        assert out_of_scope["count"].sum() > 0

    scoped_outcomes = outcomes.filter(pl.col("region_label").is_not_null())
    scoped_summary = summary.filter(pl.col("region_label").is_not_null())
    assert scoped_outcomes["region_label"].null_count() == 0
    assert scoped_summary["region_label"].null_count() == 0
    assert scoped_summary["requested_rows"].null_count() == 0
    return scoped_outcomes, scoped_summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_report(
    *,
    anchors_path: Path,
    species_manifest_path: Path,
    accepted_paths: Mapping[str, Path],
    rejection_paths: Mapping[str, Sequence[Path]],
    output_dir: Path,
    producer_commit: str,
    producer_config_sha256: str,
) -> None:
    """Run the bounded report while preserving a separate code boundary."""
    assert len(producer_commit) == 40 and len(producer_config_sha256) == 64
    anchors = pl.read_parquet(anchors_path)
    species_manifest = pl.read_csv(species_manifest_path, separator="\t")
    accepted = {
        policy: pl.scan_parquet(path) for policy, path in accepted_paths.items()
    }
    rejected = {
        policy: pl.concat(
            [
                pl.scan_parquet(path).select(
                    "query_name", "species", "rejection_reason"
                )
                for path in paths
            ],
            how="vertical",
        )
        for policy, paths in rejection_paths.items()
    }
    assert all(rejection_paths.values()), "each policy needs rejection evidence"
    accepted_metrics = pl.concat(
        [
            base_report.summarize_accepted_rows(
                accepted[policy],
                policy=policy,
                landmark_width=base_report.POLICY_WIDTHS[policy],
            )
            for policy in sorted(base_report.POLICY_WIDTHS)
        ],
        how="vertical",
    )
    outcomes, summary = build_fixed_catalog_outcome_counts(
        anchors,
        species_manifest,
        accepted,
        rejected,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "accepted_metrics": output_dir / "accepted_metrics.parquet",
        "outcome_counts": output_dir / "outcome_counts.parquet",
        "region_policy_summary": output_dir / "region_policy_summary.parquet",
        "report": output_dir / "report.md",
    }
    accepted_metrics.write_parquet(outputs["accepted_metrics"])
    outcomes.write_parquet(outputs["outcome_counts"])
    summary.write_parquet(outputs["region_policy_summary"])
    base_report._write_report(summary, outputs["report"])
    manifest = {
        "producer_commit": producer_commit,
        "producer_config_sha256": producer_config_sha256,
        "coordinate_system": "0-based half-open",
        "policies": base_report.POLICY_WIDTHS,
        "accepted_paths": {key: str(value) for key, value in accepted_paths.items()},
        "rejection_path_counts": {
            key: len(value) for key, value in rejection_paths.items()
        },
        "fixed_catalog_scope": True,
        "outputs": {
            key: {"path": path.name, "sha256": _sha256(path)}
            for key, path in outputs.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--species-manifest", type=Path, required=True)
    parser.add_argument("--full-accepted", type=Path, required=True)
    parser.add_argument("--center-accepted", type=Path, required=True)
    parser.add_argument(
        "--full-rejection", action="append", type=Path, default=[], required=True
    )
    parser.add_argument(
        "--center-rejection", action="append", type=Path, default=[], required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--producer-config-sha256", required=True)
    args = parser.parse_args()
    run_report(
        anchors_path=args.anchors,
        species_manifest_path=args.species_manifest,
        accepted_paths={
            "full_window": args.full_accepted,
            "center_1": args.center_accepted,
        },
        rejection_paths={
            "full_window": args.full_rejection,
            "center_1": args.center_rejection,
        },
        output_dir=args.output_dir,
        producer_commit=args.producer_commit,
        producer_config_sha256=args.producer_config_sha256,
    )


if __name__ == "__main__":
    main()
