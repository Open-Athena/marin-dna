"""Select on chr1, then evaluate the frozen extension once on fresh chr3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from window_conservation.evaluate import (
    intervals,
    load_chromosome,
    metrics,
    write_stretches,
)
from window_conservation.prepare import sha256


def variants(root: Path) -> list[dict]:
    output = []
    for panel in [3, 6, 10]:
        for sample, mode, filename, seeds in [
            ("rate4", "rate", "human.tsv", 25),
            ("rate8", "rate", "bits3.tsv", 12.5),
            ("rate16", "rate", "bits4.tsv", 6.25),
            ("bottom16", "bottom", "bottom16.tsv", 16),
            ("bottom32", "bottom", "human.tsv", 32),
        ]:
            output.append(
                {
                    "panel": panel,
                    "sample": sample,
                    "nominal_seeds": seeds,
                    "path": str(root / "scores" / mode / str(panel) / filename),
                }
            )
    return output


def load(root: Path, chrom: str, protocol: dict, choice: dict) -> dict[str, np.ndarray]:
    return load_chromosome(root, 25, chrom, protocol, Path(choice["path"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "validation"], required=True)
    parser.add_argument("--freeze-sha")
    args = parser.parse_args()
    root = args.root / "extension"
    report = root / "report"
    report.mkdir(exist_ok=True)
    protocol = json.loads(Path("config/protocol.json").read_text())
    extension = json.loads(Path("config/extension.json").read_text())
    label_path = root / "data/phyloP_447m.bw"
    if not label_path.exists():
        label_path.symlink_to(args.root / "data/phyloP_447m.bw")
    if args.split == "dev":
        rows, winners = [], []
        for variant in variants(root):
            values = load(root, "chr1", protocol, variant)
            cell = []
            for score in extension["scores"]:
                result = metrics(values, values[score], 0.05)
                cell.append({**variant, "score": score, **result})
            # Memory proxy is the measured distinct key count below; no labels
            # enter tie breaking except the explicitly declared density endpoint.
            winner = min(
                cell,
                key=lambda row: (-row["selected_annotated_fraction"], row["score"]),
            )
            winners.append(winner)
            rows.extend(cell)
            print(
                "development",
                variant["panel"],
                variant["sample"],
                winner["score"],
                winner["selected_annotated_fraction"],
                flush=True,
            )
            del values
        primary = min(
            winners,
            key=lambda row: (
                -row["selected_annotated_fraction"],
                row["nominal_seeds"],
                row["panel"],
                row["score"],
            ),
        )
        frozen = {
            "primary": primary,
            "cell_winners": winners,
            "baseline": {**variants(root)[0], "score": "any_copy4"},
            "protocol_sha256": sha256(Path("config/extension.json")),
            "manifest_sha256": sha256(root / "data/manifest.json"),
            "development_chromosome": "chr1",
            "validation_chromosome": "chr3",
        }
        (report / "development.json").write_text(json.dumps(rows, indent=2) + "\n")
        (report / "selection.json").write_text(json.dumps(frozen, indent=2) + "\n")
        print(
            "primary frozen",
            primary["panel"],
            primary["sample"],
            primary["score"],
            flush=True,
        )
    else:
        assert args.freeze_sha and len(args.freeze_sha) == 40
        assert not (report / "validation.json").exists(), "Validation already evaluated"
        frozen = json.loads((report / "selection.json").read_text())
        assert frozen["protocol_sha256"] == sha256(Path("config/extension.json"))
        assert frozen["manifest_sha256"] == sha256(root / "data/manifest.json")
        output = {
            "freeze_sha": args.freeze_sha,
            "selection": frozen,
            "cells": [],
            "primary": [],
            "baseline": [],
        }
        for choice in frozen["cell_winners"]:
            values = load(root, "chr3", protocol, choice)
            output["cells"].append(
                {
                    **choice,
                    "validation_metrics": metrics(
                        values, values[choice["score"]], 0.05
                    ),
                }
            )
            print("validation cell", choice["panel"], choice["sample"], flush=True)
            del values
        for label in ["primary", "baseline"]:
            choice = frozen[label]
            values = load(root, "chr3", protocol, choice)
            for budget in [0.01, 0.05, 0.10]:
                result = metrics(values, values[choice["score"]], budget)
                if budget == 0.05:
                    result["intervals"] = intervals(
                        values, values[choice["score"]], budget, 200
                    )
                output[label].append({"budget": budget, **result})
                write_stretches(
                    report / f"{label}-{budget}.bed",
                    "chr3",
                    values,
                    values[choice["score"]],
                    budget,
                )
            print("validated", label, flush=True)
            del values
        # Paired block resampling of the declared main endpoint.
        primary = load(root, "chr3", protocol, frozen["primary"])
        baseline = load(root, "chr3", protocol, frozen["baseline"])
        assert np.array_equal(primary["start"], baseline["start"])
        unique, inverse = np.unique(primary["blocks"], return_inverse=True)
        rng = np.random.default_rng(577)
        diffs = []
        order_p = np.lexsort((primary["tie"], -primary[frozen["primary"]["score"]]))
        order_b = np.lexsort((baseline["tie"], -baseline[frozen["baseline"]["score"]]))
        for _ in range(200):
            weights = np.bincount(
                rng.integers(0, len(unique), size=len(unique)), minlength=len(unique)
            )[inverse].astype(float)
            p = metrics(
                primary, primary[frozen["primary"]["score"]], 0.05, weights, order_p
            )
            b = metrics(
                baseline, baseline[frozen["baseline"]["score"]], 0.05, weights, order_b
            )
            diffs.append(
                p["selected_annotated_fraction"] - b["selected_annotated_fraction"]
            )
        output["paired_density_difference_interval"] = np.quantile(
            diffs, [0.025, 0.975]
        ).tolist()
        (report / "validation.json").write_text(json.dumps(output, indent=2) + "\n")
        print("fresh validation complete", flush=True)


if __name__ == "__main__":
    main()
