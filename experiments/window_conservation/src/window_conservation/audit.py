"""Validate resource and spatial-output contracts without selecting new settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    held = json.loads((root / "report/heldout.json").read_text())
    for prefix, result in [("", held), ("density-", held["density_alternative"])]:
        for fraction, metrics in result["budgets"].items():
            assert metrics["selected_bases"] == 100 * metrics["selected_windows"]
            assert 0 <= metrics["selected_annotated_fraction"] <= 1
            assert 0 <= metrics["annotated_base_recall"] <= 1
            lengths = []
            last_end = -1
            for line in (
                (root / "report" / f"stretches-{prefix}{fraction}.bed")
                .read_text()
                .splitlines()
            ):
                chrom, start, end, name, bed_score, strand, score, bins = line.split(
                    "\t"
                )
                start, end, bins = int(start), int(end), int(bins)
                assert chrom == "chr2" and 0 <= start < end <= 242193529
                assert (
                    name.startswith("candidate_")
                    and strand == "."
                    and 0 <= int(bed_score) <= 1000
                )
                assert start > last_end  # touching intervals must already have merged
                assert start % 100 == end % 100 == 0
                assert end - start == bins * 100 and 0 <= float(score) <= 1
                last_end = end
                lengths.append(end - start)
            assert sum(lengths) == metrics["selected_bases"]
    raw = json.loads((root / "scaling/measurements.json").read_text())
    assert len(raw) == 21 and all(
        row["build"]["measurements"][0]["bases"] == row["windows"] * 100 for row in raw
    )
    for chrom in ["chr1", "chr2"]:
        data = np.load(root / "data" / f"labels-{chrom}.npz")
        assert np.all(
            (0 <= data["conserved_bases"])
            & (data["conserved_bases"] <= data["label_covered_bases"])
        )
        assert np.all(data["label_covered_bases"] <= 100)
        assert np.all(data["ends"] - data["starts"] == 100)
    print(
        json.dumps(
            {
                "status": "passed",
                "scaling_runs": len(raw),
                "budgets": list(held["budgets"]),
                "label_chromosomes": ["chr1", "chr2"],
            }
        )
    )


if __name__ == "__main__":
    main()
