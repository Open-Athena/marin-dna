"""Check frozen comparisons, resource receipts, and final genomic outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from window_conservation.prepare import sha256


def audit_bed(path: Path, lengths: dict[str, int], expected_bins: int) -> int:
    bins_seen, stretches = 0, 0
    previous_chrom, previous_end = None, -1
    finished: set[str] = set()
    with path.open() as handle:
        for line in handle:
            chrom, start, end, name, bed_score, strand, score, bins = (
                line.rstrip().split("\t")
            )
            start, end, bins = int(start), int(end), int(bins)
            assert 0 <= start < end <= lengths[chrom]
            assert start % 100 == end % 100 == 0 and end - start == 100 * bins
            assert name.startswith("candidate_") and strand == "."
            assert 0 <= int(bed_score) <= 1000 and 0 <= float(score) <= 1
            if chrom != previous_chrom:
                assert chrom not in finished
                finished.add(chrom)
                previous_chrom, previous_end = chrom, -1
            assert start > previous_end, "overlap or unmerged adjacent selections"
            previous_end = end
            bins_seen += bins
            stretches += 1
    assert bins_seen == expected_bins
    return stretches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment-dir", default="extension")
    parser.add_argument("--protocol", type=Path, default=Path("config/extension.json"))
    args = parser.parse_args()
    root = args.root / args.experiment_dir
    report = root / "report"
    validation = json.loads((report / "validation.json").read_text())
    selection = json.loads((report / "selection.json").read_text())
    assert selection == validation["selection"]
    assert len(validation["freeze_sha"]) == 40
    assert selection["manifest_sha256"] == sha256(root / "data/manifest.json")
    assert selection["protocol_sha256"] == sha256(args.protocol)
    assert len(validation["cells"]) == 15
    assert {(cell["panel"], cell["sample"]) for cell in validation["cells"]} == {
        (panel, sample)
        for panel in [3, 6, 10]
        for sample in ["rate4", "rate8", "rate16", "bottom16", "bottom32"]
    }
    for cell in validation["cells"]:
        metrics = cell["validation_metrics"]
        assert 0 <= metrics["selected_annotated_fraction"] <= 1
        assert 0 <= metrics["annotated_base_recall"] <= 1
        lo, hi = cell["validation_intervals"]["selected_annotated_fraction"]
        assert 0 <= lo <= hi <= 1
        assert metrics["selected_bases"] == 100 * metrics["selected_windows"]
    manifest = json.loads((args.root / "data/manifest.json").read_text())
    for name in ["primary", "baseline"]:
        for result in validation[name]:
            audit_bed(
                report / f"{name}-{result['budget']}.bed",
                manifest["sources"][0]["chromosomes"],
                result["selected_windows"],
            )
    labels = np.load(root / f"data/labels-{selection['validation_chromosome']}.npz")
    assert np.all(
        (0 <= labels["conserved_bases"])
        & (labels["conserved_bases"] <= labels["label_covered_bases"])
    )
    assert np.all(labels["label_covered_bases"] <= 100)
    assert np.all(labels["ends"] - labels["starts"] == 100)
    memory = json.loads((args.root / "extension/memory/measurements.json").read_text())
    assert len(memory) == 60
    assert (
        len(
            {
                (
                    r["mode"],
                    r["bits"],
                    r["species"],
                    r["windows_per_species"],
                    r["repeat"],
                )
                for r in memory
            }
        )
        == 60
    )
    assert all(
        r["baseline_parity_rows"] == r["species"] * r["windows_per_species"]
        for r in memory
        if r["bits"] == 2
    )
    global_result = json.loads((report / "global3.json").read_text())
    expected_query_rows = sum(
        manifest["sources"][0]["chromosomes"][c] // 100
        for c in (
            {"chr1", "chr4"}
            if args.experiment_dir == "repeatfree"
            else {"chr1", "chr2"}
        )
    )
    assert global_result["query_parity_rows"] == expected_query_rows
    global_selection = json.loads(
        (root / "global3/selection/selection.json").read_text()
    )
    for species, plan in global_selection["species"].items():
        assert plan["selected_windows"] == max(1, int(0.05 * plan["eligible_windows"]))
        assert (
            audit_bed(
                root / f"global3/selection/species-{species}.bed",
                manifest["sources"][int(species) - 1]["chromosomes"],
                plan["selected_windows"],
            )
            == plan["stretches"]
        )
    spatial = json.loads((args.root / "extension/report/spatial.json").read_text())
    assert len(spatial["summary"]) == 960
    assert all(row["replicates"] == 20 for row in spatial["summary"])
    receipts = (
        list((root / "logs").glob("*.receipt.json"))
        + list((root / "memory").rglob("*.receipt.json"))
        + list((root / "profiles").glob("*.receipt.json"))
    )
    assert all(json.loads(path.read_text())["returncode"] == 0 for path in receipts)
    development = json.loads((root / "logs/development.receipt.json").read_text())
    heldout = json.loads((root / "logs/validation.receipt.json").read_text())
    assert development["end_epoch"] < heldout["start_epoch"]
    output = {
        "status": "passed",
        "fresh_validation_cells": 15,
        "resource_runs": 60,
        "global_query_parity_rows": expected_query_rows,
        "global_intervals": global_selection["input_rows"],
        "resource_receipts": len(receipts),
        "freeze_sha": validation["freeze_sha"],
    }
    (report / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
