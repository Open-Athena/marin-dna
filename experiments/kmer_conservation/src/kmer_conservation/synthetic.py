"""Known-boundary tract controls with independent flanks, divergence, and indels."""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

from kmer_conservation.fixture import features, sha256, stable_hash


def mutate(
    sequence: str, divergence: float, indel_rate: float, rng: random.Random
) -> str:
    output = []
    for base in sequence:
        if rng.random() < indel_rate / 2:
            continue
        output.append(
            rng.choice([b for b in "ACGT" if b != base])
            if rng.random() < divergence
            else base
        )
        if rng.random() < indel_rate / 2:
            output.append(rng.choice("ACGT"))
    return "".join(output)


def planted_records(seed: int = 568) -> list[dict]:
    records = []
    for span in [32, 64, 128, 255, 511, 1024]:
        for divergence in [0.0, 0.15, 0.30]:
            for indel in [0.0, 0.03]:
                for replicate in [0, 1]:
                    name = f"plant:{span}:{divergence}:{indel}:{replicate}"
                    rng = random.Random(stable_hash(f"{seed}:{name}"))
                    tract = "".join(rng.choices("ACGT", k=span))
                    for species in ["human", "mouse", "armadillo"]:
                        changed = (
                            tract
                            if species == "human"
                            else mutate(tract, divergence, indel, rng)
                        )
                        left = rng.randrange(1024, 2048)
                        seq = (
                            "".join(rng.choices("ACGT", k=left))
                            + changed
                            + "".join(rng.choices("ACGT", k=4096 - left - len(changed)))
                        )
                        gc, repeat, complexity = features(seq)
                        records.append(
                            {
                                "id": f"{stable_hash(name):016x}_{species}",
                                "group": name,
                                "component": name,
                                "species": species,
                                "assembly": "synthetic",
                                "chrom": name,
                                "start": rng.randrange(0, 10000),
                                "end": 0,
                                "kind": "anchor",
                                "split": "dev",
                                "sequence": seq,
                                "gc": gc,
                                "repeat": repeat,
                                "complexity": complexity,
                                "tract_start": left,
                                "tract_end": left + len(changed),
                                "ancestral_span": span,
                                "divergence": divergence,
                                "indel_rate": indel,
                                "replicate": replicate,
                            }
                        )
                        records[-1]["end"] = records[-1]["start"] + 4096
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--duplicates", type=int, default=0)
    args = parser.parse_args()
    records = planted_records()
    duplicates = []
    for parent in [r for r in records if r["species"] == "human"]:
        tract = parent["sequence"][parent["tract_start"] : parent["tract_end"]]
        for species in ["mouse", "armadillo"]:
            for copy in range(args.duplicates):
                name = f"paralog:{parent['component']}:{species}:{copy}"
                rng = random.Random(stable_hash(name))
                changed = mutate(tract, parent["divergence"], parent["indel_rate"], rng)
                left = rng.randrange(1024, 2048)
                seq = (
                    "".join(rng.choices("ACGT", k=left))
                    + changed
                    + "".join(rng.choices("ACGT", k=4096 - left - len(changed)))
                )
                gc, repeat, complexity = features(seq)
                duplicate = dict(parent)
                duplicate.update(
                    id=f"{stable_hash(name):016x}_{species}",
                    group=name,
                    component=name,
                    species=species,
                    chrom=name,
                    kind="planted_paralog",
                    parent=parent["component"],
                    sequence=seq,
                    gc=gc,
                    repeat=repeat,
                    complexity=complexity,
                    tract_start=left,
                    tract_end=left + len(changed),
                )
                duplicates.append(duplicate)
    records.extend(duplicates)
    with gzip.open(args.root / "data/contexts.jsonl.gz", "rt") as handle:
        real = [json.loads(line) for line in handle]
    for species in ["human", "mouse", "armadillo"]:
        records.extend(
            [r for r in real if r["species"] == species and r["kind"] == "background"][
                :128
            ]
        )
    out = args.root / (
        f"synthetic-paralogs{args.duplicates}/data"
        if args.duplicates
        else "synthetic/data"
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / "contexts.jsonl.gz"
    with gzip.open(path, "wt") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    (out / "fixture_manifest.json").write_text(
        json.dumps(
            {
                "contexts_sha256": sha256(path),
                "records": len(records),
                "planted_loci": 72,
                "seed": 568,
                "duplicates_per_target_locus": args.duplicates,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
