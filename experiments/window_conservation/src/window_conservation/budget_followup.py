"""User-requested budget sensitivity for frozen scores; no parameter selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from window_conservation.evaluate import (
    load_chromosome,
    metrics,
    select_indices,
    write_stretches,
)
from window_conservation.extension_audit import audit_bed
from window_conservation.prepare import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment-dir", default="repeatfree")
    args = parser.parse_args()
    root = args.root / args.experiment_dir
    report = root / "report"
    validation_path = report / "validation.json"
    validation = json.loads(validation_path.read_text())
    selection = json.loads((report / "selection.json").read_text())
    assert selection == validation["selection"]
    assert selection["manifest_sha256"] == sha256(root / "data/manifest.json")
    protocol = json.loads((root / "data/protocol.json").read_text())
    manifest = json.loads((root / "data/manifest.json").read_text())
    assert manifest["protocol_sha256"] == sha256(root / "data/protocol.json")
    chrom = selection["validation_chromosome"]
    lengths = json.loads((args.root / "data/manifest.json").read_text())["sources"][0][
        "chromosomes"
    ]
    output = {
        "freeze_sha": validation["freeze_sha"],
        "validation_sha256": sha256(validation_path),
        "selection_sha256": sha256(report / "selection.json"),
        "chromosome": chrom,
        "interpretation": "Descriptive user-requested budget sensitivity. The 20% cutoff was requested after the original validation; scores and model choices remain frozen.",
        "primary": [],
        "baseline": [],
    }
    for name in ["primary", "baseline"]:
        choice = selection[name]
        values = load_chromosome(root, 25, chrom, protocol, Path(choice["path"]))
        previous = {row["budget"]: row for row in validation[name]}
        for budget in [0.01, 0.05, 0.10, 0.20]:
            result = metrics(values, values[choice["score"]], budget)
            if budget in previous:
                # The original held-out output is immutable and must reproduce.
                assert all(result[key] == previous[budget][key] for key in result)
            path = report / f"budget-{name}-{budget}.bed"
            write_stretches(path, chrom, values, values[choice["score"]], budget)
            stretches = audit_bed(path, lengths, result["selected_windows"])
            chosen = select_indices(
                values[choice["score"]], values["tie"], result["selected_windows"]
            )
            selected_scores = values[choice["score"]][chosen]
            output[name].append(
                {
                    "budget": budget,
                    "stretches": stretches,
                    "selection_threshold": float(selected_scores.min()),
                    "selected_zero_score_windows": int((selected_scores == 0).sum()),
                    **result,
                }
            )
        print("budget follow-up", name, flush=True)
    assert sha256(validation_path) == output["validation_sha256"]
    (report / "budget-followup.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
