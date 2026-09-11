"""Matched Linclust clusters as candidates, followed by full-set Jaccard ranking."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from kmer_conservation.core import make_windows
from kmer_conservation.sketch import prediction, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--divisor", type=int, default=2)
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--mmseqs", type=Path, required=True)
    parser.add_argument("--frozen-selection", type=Path)
    args = parser.parse_args()
    if args.split == "heldout":
        assert args.frozen_selection is not None
        assert [args.width, args.k, args.divisor, False] in json.loads(
            args.frozen_selection.read_text()
        )["allowed_settings"]
    with gzip.open(args.root / "data/contexts.jsonl.gz", "rt") as handle:
        records = [json.loads(line) for line in handle]
    started = time.time()
    windows = {
        s: make_windows(
            [r for r in records if r["species"] == s], args.width, args.k, args.divisor
        )
        for s in ["human", "mouse", "armadillo"]
    }
    directory = args.root / f"linclust-w{args.width}-d{args.divisor}"
    directory.mkdir(parents=True, exist_ok=True)
    cluster_path = directory / "assignments.tsv"
    resource_path = directory / "resources.json"
    if not cluster_path.exists():
        fasta = directory / "windows.fasta"
        with fasta.open("w") as handle:
            for species, win in windows.items():
                for wi, owner in enumerate(win.owners):
                    seq = win.records[owner]["sequence"][
                        win.starts[wi] : win.starts[wi] + args.width
                    ]
                    handle.write(f">{species}_{wi}\n{seq}\n")
        commands = [
            [
                str(args.mmseqs),
                "createdb",
                str(fasta),
                str(directory / "db"),
                "--dbtype",
                "2",
                "--shuffle",
                "0",
            ],
            [
                str(args.mmseqs),
                "linclust",
                str(directory / "db"),
                str(directory / "clu"),
                str(directory / "tmp"),
                "--threads",
                "8",
                "--min-seq-id",
                "0.4",
                "-c",
                "0.7",
                "--cov-mode",
                "0",
                "--kmer-per-seq",
                "148",
                "--kmer-per-seq-scale",
                "0",
                "--spaced-kmer-mode",
                "1",
                "--mask",
                "1",
                "--mask-lower-case",
                "1",
                "--cluster-mode",
                "0",
                "-e",
                "0.001",
            ],
            [
                str(args.mmseqs),
                "createtsv",
                str(directory / "db"),
                str(directory / "db"),
                str(directory / "clu"),
                str(cluster_path),
            ],
        ]
        receipts = []
        for number, command in enumerate(commands):
            start = time.time()
            with (directory / f"stage{number}.log").open("w") as log:
                subprocess.run(
                    [
                        "/usr/bin/time",
                        "-v",
                        "-o",
                        str(directory / f"stage{number}.time"),
                        *command,
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            receipts.append({"command": command, "seconds": time.time() - start})
        resource_path.write_text(json.dumps(receipts, indent=2) + "\n")
    clusters: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    member_cluster = {}
    with cluster_path.open() as handle:
        for line in handle:
            representative, member = line.strip().split("\t")
            species, index = member.rsplit("_", 1)
            assert member not in member_cluster
            member_cluster[member] = representative
            clusters[representative][species].append(int(index))
    assert len(member_cluster) == sum(len(w.features) for w in windows.values())
    predictions = []
    for target_species, sources in [
        ("mouse", ["human"]),
        ("armadillo", ["human", "mouse"]),
    ]:
        target = windows[target_species]
        for source_species in sources:
            source = windows[source_species]
            components = sorted(
                {
                    r["component"]
                    for r in source.records
                    if r["kind"] == "anchor" and r["split"] == args.split
                }
            )
            for component in components:
                indices = [
                    wi
                    for wi, owner in enumerate(source.owners)
                    if source.records[owner]["kind"] == "anchor"
                    and source.records[owner]["component"] == component
                ]
                start = time.time()
                representatives = {
                    member_cluster[f"{source_species}_{wi}"] for wi in indices
                }
                candidates = {
                    wi
                    for rep in representatives
                    for wi in clusters[rep][target_species]
                }
                scores = np.zeros(len(target.features))
                for wi in candidates:
                    b = target.features[wi]
                    for qi in indices:
                        a = source.features[qi]
                        overlap = len(np.intersect1d(a, b, assume_unique=True))
                        if overlap:
                            scores[wi] = max(
                                scores[wi], overlap / (len(a) + len(b) - overlap)
                            )
                row = prediction(
                    component,
                    source_species,
                    target_species,
                    scores,
                    target,
                    0,
                    len(candidates) * len(indices),
                )
                row["seconds"] = time.time() - start
                row["linclust_candidate_windows"] = len(candidates)
                predictions.append(row)
    result = summarize(predictions)
    result.update(
        width=args.width,
        k=args.k,
        divisor=args.divisor,
        split=args.split,
        method="linclust",
        cluster_count=len(clusters),
        input_windows=len(member_cluster),
        clustering_stages=json.loads(resource_path.read_text()),
        wall_seconds=time.time() - started,
    )
    prefix = (
        args.root
        / "results"
        / f"{args.split}-w{args.width}-k{args.k}-d{args.divisor}-linclust"
    )
    with gzip.open(str(prefix) + ".predictions.jsonl.gz", "wt") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    Path(str(prefix) + ".summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
