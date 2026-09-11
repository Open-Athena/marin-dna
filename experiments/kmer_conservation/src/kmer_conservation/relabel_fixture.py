"""Version the physical-locus metadata without changing sequences, intervals, or split."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from kmer_conservation.fixture import assign_components, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.root / "data/contexts.jsonl.gz", "rt") as handle:
        records = [json.loads(line) for line in handle]
    anchors = [r for r in records if r["kind"] == "anchor"]
    before = [
        (r["id"], r["sequence"], r["start"], r["end"], r["split"]) for r in records
    ]
    assign_components(anchors, 568, 0.4)
    after = [
        (r["id"], r["sequence"], r["start"], r["end"], r["split"]) for r in records
    ]
    assert before == after
    out = args.output / "data"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "contexts.jsonl.gz"
    with gzip.open(path, "wt") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    manifest = json.loads((args.root / "data/fixture_manifest.json").read_text())
    manifest["schema_version"] = 2
    manifest["source_contexts_sha256"] = manifest["contexts_sha256"]
    manifest["contexts_sha256"] = sha256(path)
    manifest["split_components"] = len({r["split_component"] for r in anchors})
    manifest["physical_loci_by_species"] = {
        s: len({r["component"] for r in anchors if r["species"] == s})
        for s in ["human", "mouse", "armadillo"]
    }
    manifest["components"] = sum(manifest["physical_loci_by_species"].values())
    (out / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for source in (args.root / "data").iterdir():
        if source.suffix in [".parquet", ".2bit", ".fasta", ".tsv"]:
            (out / source.name).symlink_to(source)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in [
                    "schema_version",
                    "split_components",
                    "physical_loci_by_species",
                    "contexts_sha256",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
