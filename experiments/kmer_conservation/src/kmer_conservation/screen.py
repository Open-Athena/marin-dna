"""Measure exact full-set Jaccard locus retrieval on a fixed target universe."""

from __future__ import annotations

import argparse
import gzip
import json
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from kmer_conservation.core import (
    exact_index,
    exact_scores,
    make_windows,
    rank_loci,
    rank_truth,
    recall,
    truth_loci,
)
from kmer_conservation.fixture import sha256


def run_setting(
    records: list[dict],
    out: Path,
    width: int,
    k: int,
    divisor: int,
    split: str,
    mask: bool,
) -> dict:
    started = time.time()
    by_species = {
        s: [r for r in records if r["species"] == s]
        for s in ["human", "mouse", "armadillo"]
    }
    windows = {
        s: make_windows(rows, width, k, divisor, mask) for s, rows in by_species.items()
    }
    build_seconds = time.time() - started
    resources = {}
    predictions = []
    # Three ordered directions cover each species pair once, with a fixed target species.
    for target_species, sources in [
        ("mouse", ["human"]),
        ("armadillo", ["human", "mouse"]),
    ]:
        start = time.time()
        target = windows[target_species]
        matrix, vocab, lengths = exact_index(target)
        resources[target_species] = {
            "index_seconds": time.time() - start,
            "index_bytes": matrix.data.nbytes
            + matrix.indices.nbytes
            + matrix.indptr.nbytes
            + vocab.nbytes,
            "windows": len(lengths),
            "edge_windows": target.edge_count,
            "features": len(vocab),
            "postings": int(matrix.nnz),
            "maximum_posting": int(np.diff(matrix.indptr).max()),
        }
        for source_species in sources:
            source = windows[source_species]
            queries: dict[str, list[int]] = defaultdict(list)
            for wi, owner in enumerate(source.owners):
                row = source.records[owner]
                if row["kind"] == "anchor" and row["split"] == split:
                    queries[row["component"]].append(wi)
            for number, (component, indices) in enumerate(sorted(queries.items())):
                features = [source.features[i] for i in indices]
                start = time.time()
                scores, positive_pairs, posting_work = exact_scores(
                    features, matrix, vocab, lengths
                )
                all_ranked = rank_loci(scores, target, limit=len(target.records))
                ranked = all_ranked[:100]
                elapsed = time.time() - start
                truth = truth_loci(source.records, target.records, component)
                ranks = rank_truth(ranked, truth)
                query_rows = [
                    r
                    for r in source.records
                    if r["component"] == component and r["kind"] == "anchor"
                ]
                predictions.append(
                    {
                        "query": component,
                        "source": source_species,
                        "target": target_species,
                        "ranks": ranks,
                        "split_component": query_rows[0].get(
                            "split_component", component
                        ),
                        "seconds": elapsed,
                        "raw_positive_window_pairs": positive_pairs,
                        "posting_work": posting_work,
                        "query_windows": len(indices),
                        "candidate_windows": int(np.count_nonzero(scores)),
                        "candidate_loci": len(all_ranked),
                        "query_gc": float(np.mean([r["gc"] for r in query_rows])),
                        "query_repeat": float(
                            np.mean([r["repeat"] for r in query_rows])
                        ),
                        "prior128": any(r.get("prior128", False) for r in query_rows),
                        "hits": ranked,
                    }
                )
                if number % 32 == 0:
                    print(
                        width,
                        k,
                        divisor,
                        source_species,
                        target_species,
                        number,
                        len(queries),
                        flush=True,
                    )
        del matrix, vocab, lengths
    summary = {
        "width": width,
        "k": k,
        "divisor": divisor,
        "mask": mask,
        "split": split,
        "n": sum(len(p["ranks"]) for p in predictions),
        "n_queries": len(predictions),
        "build_windows_seconds": build_seconds,
        "wall_seconds": time.time() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "resources": resources,
        "raw_windows": sum(len(w.features) for w in windows.values()),
    }
    for budget in [1, 10, 100]:
        summary[f"recall_at_{budget}"] = recall(predictions, budget)
        selected = [hit for p in predictions for hit in p["hits"][:budget]]
        summary[f"injected_decoy_fraction_at_{budget}"] = sum(
            h["kind"] == "shuffled_decoy" for h in selected
        ) / max(1, len(selected))
    summary["query_seconds"] = sum(p["seconds"] for p in predictions)
    summary["positive_window_pairs"] = sum(
        p["raw_positive_window_pairs"] for p in predictions
    )
    summary["posting_work"] = sum(p["posting_work"] for p in predictions)
    prefix = out / f"{split}-w{width}-k{k}-d{divisor}-mask{int(mask)}"
    with gzip.open(str(prefix) + ".predictions.jsonl.gz", "wt") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction) + "\n")
    Path(str(prefix) + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--divisor", type=int, default=2)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--frozen-selection", type=Path)
    args = parser.parse_args()
    cfg = json.loads(Path("config/screen.json").read_text())
    if args.width is None or args.k is None:
        # ru_maxrss is process-lifetime accounting: isolate every parameter arm.
        for width in [args.width] if args.width else cfg["window_lengths"]:
            for k in [args.k] if args.k else cfg["kmer_lengths"]:
                command = [
                    sys.executable,
                    "-m",
                    "kmer_conservation.screen",
                    "--root",
                    str(args.root),
                    "--split",
                    args.split,
                    "--width",
                    str(width),
                    "--k",
                    str(k),
                    "--divisor",
                    str(args.divisor),
                ]
                if args.mask:
                    command.append("--mask")
                if args.frozen_selection:
                    command.extend(["--frozen-selection", str(args.frozen_selection)])
                subprocess.run(command, check=True)
        return
    if args.split == "heldout":
        if args.frozen_selection is None:
            parser.error(
                "held-out scoring requires a frozen development selection artifact"
            )
        selection = json.loads(args.frozen_selection.read_text())
        assert [args.width, args.k, args.divisor, args.mask] in selection[
            "allowed_settings"
        ]
    data = args.root / "data/contexts.jsonl.gz"
    manifest = json.loads((args.root / "data/fixture_manifest.json").read_text())
    assert sha256(data) == manifest["contexts_sha256"]
    with gzip.open(data, "rt") as handle:
        records = [json.loads(line) for line in handle]
    out = args.root / "results"
    out.mkdir(parents=True, exist_ok=True)
    for width in [args.width] if args.width else cfg["window_lengths"]:
        for k in [args.k] if args.k else cfg["kmer_lengths"]:
            run_setting(records, out, width, k, args.divisor, args.split, args.mask)


if __name__ == "__main__":
    main()
