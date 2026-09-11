"""Budget-preserving scale unions and bounded alignment-verification diagnostics."""

from __future__ import annotations

import argparse
import gzip
import json
import resource
import time
from pathlib import Path

import edlib
import numpy as np

from kmer_conservation.core import make_windows, rank_truth
from kmer_conservation.fixture import stable_hash
from kmer_conservation.sketch import summarize


def read_predictions(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def union_predictions(inputs: list[list[dict]]) -> list[dict]:
    """Reciprocal rank fusion of frozen top-100 lists, then one common locus budget."""
    maps = [
        {(r["query"], r["source"], r["target"]): r for r in rows} for rows in inputs
    ]
    assert all(set(m) == set(maps[0]) for m in maps)
    output = []
    for key in sorted(maps[0]):
        started = time.time()
        scores, hits = {}, {}
        for rows in maps:
            for rank, hit in enumerate(rows[key]["hits"], 1):
                component = hit["component"]
                scores[component] = scores.get(component, 0.0) + 1 / (60 + rank)
                hits[component] = hit
        ordered = sorted(hits, key=lambda x: (-scores[x], stable_hash(x)))[:100]
        ranked = [{**hits[c], "score": scores[c]} for c in ordered]
        truth = set(maps[0][key]["ranks"])
        assert all(set(m[key]["ranks"]) == truth for m in maps)
        ranks = rank_truth(ranked, truth)
        output.append(
            {
                "query": key[0],
                "source": key[1],
                "target": key[2],
                "ranks": ranks,
                "split_component": maps[0][key].get("split_component", key[0]),
                "seconds": sum(m[key].get("seconds", 0) for m in maps)
                + time.time()
                - started,
                "work": sum(len(m[key]["hits"]) for m in maps),
                "upstream_positive_window_pairs": sum(
                    m[key].get("raw_positive_window_pairs", 0) for m in maps
                ),
                "upstream_posting_work": sum(
                    m[key].get("posting_work", 0) for m in maps
                ),
                "candidate_loci": len(hits),
                "hits": ranked,
            }
        )
    return output


def verify(
    root: Path, predictions: list[dict], width: int, k: int, divisor: int
) -> dict:
    with gzip.open(root / "data/contexts.jsonl.gz", "rt") as handle:
        records = [json.loads(line) for line in handle]
    preparation_started = time.time()
    windows = {
        s: make_windows([r for r in records if r["species"] == s], width, k, divisor)
        for s in ["human", "mouse", "armadillo"]
    }
    started = time.time()
    preparation_seconds = started - preparation_started
    work = 0
    for row in predictions:
        source, target = windows[row["source"]], windows[row["target"]]
        indices = [
            wi
            for wi, owner in enumerate(source.owners)
            if source.records[owner]["component"] == row["query"]
            and source.records[owner]["kind"] == "anchor"
        ]
        for hit in row["hits"]:
            wi = hit["window"]
            values = target.features[wi]
            qi = max(
                indices,
                key=lambda i: (
                    len(np.intersect1d(values, source.features[i], assume_unique=True))
                    / max(
                        1,
                        len(values)
                        + len(source.features[i])
                        - len(
                            np.intersect1d(
                                values, source.features[i], assume_unique=True
                            )
                        ),
                    )
                ),
            )
            arow, brow = (
                source.records[source.owners[qi]],
                target.records[target.owners[wi]],
            )
            a = arow["sequence"][source.starts[qi] : source.starts[qi] + width].upper()
            b = brow["sequence"][target.starts[wi] : target.starts[wi] + width].upper()
            reverse = b.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            # Full-window global edit distance is deliberately a separate strict gate.
            distances = [
                edlib.align(a, t, mode="NW", task="distance", k=int(0.3 * width))[
                    "editDistance"
                ]
                for t in [b, reverse]
            ]
            hit["verified"] = any(d >= 0 for d in distances)
            hit["edit_distance"] = min((d for d in distances if d >= 0), default=None)
            work += 2
    result = {
        "width": width,
        "k": k,
        "divisor": divisor,
        "n": sum(len(r["ranks"]) for r in predictions),
        "n_queries": len(predictions),
        "profiled_budget": 100,
        "build_windows_seconds": preparation_seconds,
        "seconds": time.time() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "alignment_calls": work,
        "threshold": "global edit distance <= floor(0.30 W), best of both strands, no backfill",
    }
    for budget in [1, 10, 100]:
        result[f"recall_at_{budget}"] = (
            sum(
                len(
                    set(r["ranks"])
                    & {h["component"] for h in r["hits"][:budget] if h["verified"]}
                )
                for r in predictions
            )
            / result["n"]
        )
        result[f"verified_candidates_at_{budget}"] = sum(
            h["verified"] for r in predictions for h in r["hits"][:budget]
        )
        result[f"verified_decoys_at_{budget}"] = sum(
            h["verified"] and h["kind"] == "shuffled_decoy"
            for r in predictions
            for h in r["hits"][:budget]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--mode", choices=["union", "verify"], required=True)
    parser.add_argument("--width", type=int, default=255)
    parser.add_argument("--k", type=int, default=9)
    args = parser.parse_args()
    directory = args.root / "results"
    if args.mode == "union":
        inputs = [
            directory / f"{args.split}-w{w}-k{k}-d2-mask0.predictions.jsonl.gz"
            for w, k in [(255, 9), (1024, 13)]
        ]
        rows = union_predictions([read_predictions(p) for p in inputs])
        result = summarize(rows)
        result["inputs"] = [str(p) for p in inputs]
        stages = [
            json.loads(
                Path(
                    str(p).replace(".predictions.jsonl.gz", ".summary.json")
                ).read_text()
            )
            for p in inputs
        ]
        result["build_windows_seconds"] = sum(
            s["build_windows_seconds"] for s in stages
        )
        result["index_seconds"] = sum(
            r["index_seconds"] for s in stages for r in s["resources"].values()
        )
        result["index_bytes"] = sum(
            r["index_bytes"] for s in stages for r in s["resources"].values()
        )
        result["upstream_positive_window_pairs"] = sum(
            r["upstream_positive_window_pairs"] for r in rows
        )
        result["upstream_posting_work"] = sum(r["upstream_posting_work"] for r in rows)
        result["work_unit"] = (
            "top-100 list entries merged; upstream work reported separately"
        )
        prefix = directory / f"{args.split}-union-255k9-1024k13"
    else:
        rows = read_predictions(
            directory
            / f"{args.split}-w{args.width}-k{args.k}-d2-mask0.predictions.jsonl.gz"
        )
        result = verify(args.root, rows, args.width, args.k, 2)
        prefix = directory / f"{args.split}-w{args.width}-k{args.k}-verify"
    with gzip.open(str(prefix) + ".predictions.jsonl.gz", "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    Path(str(prefix) + ".summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
