"""Reuse pinned inputs for the repeat-excluded cohort without reading labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from window_conservation.local_run import extract
from window_conservation.prepare import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    target = root / "repeatfree"
    target.mkdir(exist_ok=False)
    for name in ["data", "logs", "scores", "report"]:
        (target / name).mkdir()
    data = target / "data"
    settings = json.loads(Path("config/repeatfree.json").read_text())
    protocol = json.loads(Path("config/protocol.json").read_text())
    protocol.update(
        maximum_repeat_fraction=settings["maximum_repeat_fraction"],
        exclude_lowercase_twobit=str(root / "data/human.2bit"),
        lowercase_label_policy=settings["lowercase_label_policy"],
        development_chromosome=settings["dev_chromosome"],
        heldout_chromosome=settings["validation_chromosome"],
    )
    (data / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for name in ["genomes.list", "Homo_sapiens.GRCh38.115.gtf.gz", "ccre.bare.parquet"]:
        (data / name).symlink_to(root / "extension/data" / name)
    (data / "phyloP_447m.bw").symlink_to(root / "data/phyloP_447m.bw")
    extract(root / "data/human.fa", data / "query.fa", {
        settings["dev_chromosome"], settings["validation_chromosome"]
    })
    manifest = {
        "pilot_input_manifest_sha256": sha256(root / "data/manifest.json"),
        "extension_input_manifest_sha256": sha256(root / "extension/data/manifest.json"),
        "query_sha256": sha256(data / "query.fa"),
        "protocol_sha256": sha256(data / "protocol.json"),
        "experiment_protocol_sha256": sha256(Path("config/repeatfree.json")),
        "inputs": "Reuse genomes, lowercase masks and annotations pinned in the two parent manifests.",
    }
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Repeat-excluded inputs prepared; no validation labels accessed", flush=True)


if __name__ == "__main__":
    main()
