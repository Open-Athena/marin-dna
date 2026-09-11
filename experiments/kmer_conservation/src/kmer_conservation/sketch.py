"""Separate independent-permutation sketch error from compatible LSH loss."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
from datasketch import MinHash

from kmer_conservation.core import Windows, make_windows, rank_loci
from kmer_conservation.fixture import sha256


def signatures(features: list[np.ndarray], size: int) -> np.ndarray:
    # Lossless canonical integer keys enter the library's documented affine64 mixer.
    template = MinHash(num_perm=size, seed=568, scheme="affine64", hashfunc=int)
    output = np.empty((len(features), size), dtype=np.uint64)
    for i, values in enumerate(features):
        sketch = template.copy()
        sketch.update_batch(values)
        output[i] = sketch.hashvalues
    return output


def mix(values: np.ndarray) -> np.ndarray:
    values = values.copy()
    values ^= values >> np.uint64(33)
    values *= np.uint64(0xFF51AFD7ED558CCD)
    values ^= values >> np.uint64(33)
    values *= np.uint64(0xC4CEB9FE1A85EC53)
    return values ^ (values >> np.uint64(33))


def band_keys(signature: np.ndarray, rows: int) -> list[np.ndarray]:
    assert signature.shape[1] % rows == 0
    keys = []
    for start in range(0, signature.shape[1], rows):
        key = signature[:, start].copy()
        for position in range(1, rows):
            key = mix(mix(key) ^ signature[:, start + position])
        keys.append(key)
    return keys


def build_lsh(signature: np.ndarray, rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    output = []
    for keys in band_keys(signature, rows):
        order = np.argsort(keys, kind="stable").astype(np.int32)
        output.append((keys[order], order))
    return output


def lsh_scores(
    query: np.ndarray,
    target: np.ndarray,
    index: list[tuple[np.ndarray, np.ndarray]],
    rows: int,
    valid_target: np.ndarray,
    valid_query: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    keys = band_keys(query, rows)
    score = np.zeros(len(target), dtype=np.float64)
    emitted = unique_pairs = 0
    for qi in np.flatnonzero(valid_query):
        chunks = []
        for band, (sorted_keys, order) in enumerate(index):
            left = np.searchsorted(sorted_keys, keys[band][qi], side="left")
            right = np.searchsorted(sorted_keys, keys[band][qi], side="right")
            emitted += int(right - left)
            chunks.append(order[left:right])
        ids = np.unique(np.concatenate(chunks))
        ids = ids[valid_target[ids]]
        unique_pairs += len(ids)
        # Re-score the sketch for every returned window; no exact-set oracle rerank.
        if len(ids):
            values = np.mean(target[ids] == query[qi], axis=1)
            np.maximum.at(score, ids, values)
    return score, emitted, unique_pairs


def scan_library() -> ctypes.CDLL:
    src = Path(__file__).with_name("scan.cpp")
    library = src.with_name("scan.so")
    if not library.exists() or src.stat().st_mtime > library.stat().st_mtime:
        subprocess.run(
            [
                "g++",
                "-O3",
                "-march=native",
                "-fopenmp",
                "-shared",
                "-fPIC",
                str(src),
                "-o",
                str(library),
            ],
            check=True,
        )
    lib = ctypes.CDLL(str(library))
    array64 = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
    array32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    lib.signature_scan.argtypes = [
        array64,
        array64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        array32,
    ]
    lib.signature_scan.restype = None
    return lib


def scan_scores(
    query: np.ndarray, target: np.ndarray, hashes: int, threads: int = 8
) -> np.ndarray:
    output = np.zeros(len(target), dtype=np.int32)
    lib = scan_library()
    lib.signature_scan(
        query, target, len(query), len(target), target.shape[1], hashes, threads, output
    )
    return output / hashes


def prediction(
    component: str,
    source: str,
    target: str,
    scores: np.ndarray,
    windows: Windows,
    seconds: float,
    work: int,
) -> dict:
    all_hits = rank_loci(scores, windows, limit=len(windows.records))
    hits = all_hits[:100]
    ranks = [i + 1 for i, hit in enumerate(hits) if hit["component"] == component]
    return {
        "query": component,
        "source": source,
        "target": target,
        "rank": ranks[0] if ranks else None,
        "seconds": seconds,
        "work": work,
        "candidate_loci": len(all_hits),
        "candidate_windows": int(np.count_nonzero(scores)),
        "hits": hits,
    }


def summarize(rows: list[dict]) -> dict:
    summary = {
        "n": len(rows),
        "query_seconds": sum(r["seconds"] for r in rows),
        "work": sum(r["work"] for r in rows),
    }
    for budget in [1, 10, 100]:
        summary[f"recall_at_{budget}"] = float(
            np.mean([r["rank"] is not None and r["rank"] <= budget for r in rows])
        )
        hits = [h for r in rows for h in r["hits"][:budget]]
        summary[f"injected_decoy_fraction_at_{budget}"] = sum(
            h["kind"] == "shuffled_decoy" for h in hits
        ) / max(1, len(hits))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--divisor", type=int, default=2)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--hashes", type=int, default=128)
    parser.add_argument("--method", choices=["scan", "lsh"], required=True)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--frozen-selection", type=Path)
    args = parser.parse_args()
    if args.split == "heldout":
        assert args.frozen_selection is not None
        allowed = json.loads(args.frozen_selection.read_text())["allowed_sketches"]
        assert [
            args.width,
            args.k,
            args.divisor,
            args.mask,
            args.hashes,
            args.method,
            args.rows,
        ] in allowed
    path = args.root / "data/contexts.jsonl.gz"
    assert (
        sha256(path)
        == json.loads((args.root / "data/fixture_manifest.json").read_text())[
            "contexts_sha256"
        ]
    )
    with gzip.open(path, "rt") as handle:
        records = [json.loads(line) for line in handle]
    started = time.time()
    windows = {
        s: make_windows(
            [r for r in records if r["species"] == s],
            args.width,
            args.k,
            args.divisor,
            args.mask,
        )
        for s in ["human", "mouse", "armadillo"]
    }
    prep = time.time() - started
    cache = args.root / "sketches"
    cache.mkdir(parents=True, exist_ok=True)
    sigs, times = {}, {}
    for species, win in windows.items():
        path = (
            cache
            / f"{species}-w{args.width}-k{args.k}-d{args.divisor}-m{int(args.mask)}-h{args.hashes}.npz"
        )
        if path.exists():
            data = np.load(path)
            assert str(data["fixture_sha"]) == sha256(
                args.root / "data/contexts.jsonl.gz"
            )
            sigs[species] = data["signature"]
            times[species] = float(data["seconds"])
        else:
            start = time.time()
            sigs[species] = signatures(win.features, args.hashes)
            times[species] = time.time() - start
            np.savez(
                path,
                signature=sigs[species],
                seconds=times[species],
                fixture_sha=sha256(args.root / "data/contexts.jsonl.gz"),
            )
        print("sketch", species, sigs[species].shape, times[species], flush=True)
    predictions, resources = [], {}
    for target_species, sources in [
        ("mouse", ["human"]),
        ("armadillo", ["human", "mouse"]),
    ]:
        target = windows[target_species]
        target_sig = sigs[target_species]
        valid_target = np.array([len(f) > 0 for f in target.features])
        start = time.time()
        index = build_lsh(target_sig, args.rows) if args.method == "lsh" else None
        resources[target_species] = {
            "index_seconds": time.time() - start,
            "index_bytes": sum(a.nbytes + b.nbytes for a, b in index) if index else 0,
            "signature_bytes": target_sig.nbytes,
            "largest_bucket": max(
                int(np.diff(np.r_[0, np.flatnonzero(np.diff(a)) + 1, len(a)]).max())
                for a, b in index
            )
            if index
            else 0,
        }
        for source_species in sources:
            source = windows[source_species]
            components = sorted(
                {
                    r["component"]
                    for r in source.records
                    if r["kind"] == "anchor" and r["split"] == args.split
                }
            )
            for number, component in enumerate(components):
                ids = np.array(
                    [
                        wi
                        for wi, owner in enumerate(source.owners)
                        if source.records[owner]["component"] == component
                        and source.records[owner]["kind"] == "anchor"
                    ]
                )
                query = np.ascontiguousarray(sigs[source_species][ids])
                valid_query = np.array([len(source.features[i]) > 0 for i in ids])
                start = time.time()
                if args.method == "scan":
                    score = scan_scores(
                        np.ascontiguousarray(query[valid_query]),
                        target_sig,
                        args.hashes,
                    )
                    score[~valid_target] = 0
                    work = int(valid_query.sum()) * len(target_sig) * args.hashes
                    unique_pairs = int(valid_query.sum()) * len(target_sig)
                else:
                    score, work, unique_pairs = lsh_scores(
                        query, target_sig, index, args.rows, valid_target, valid_query
                    )
                elapsed = time.time() - start
                result = prediction(
                    component,
                    source_species,
                    target_species,
                    score,
                    target,
                    elapsed,
                    work,
                )
                result["seconds"] = time.time() - start
                result["evaluated_window_pairs"] = unique_pairs
                predictions.append(result)
                if number % 32 == 0:
                    print(
                        args.method,
                        args.hashes,
                        args.rows,
                        source_species,
                        target_species,
                        number,
                        len(components),
                        flush=True,
                    )
        del index
    result = summarize(predictions)
    result.update(
        width=args.width,
        k=args.k,
        divisor=args.divisor,
        mask=args.mask,
        split=args.split,
        hashes=args.hashes,
        method=args.method,
        rows=args.rows,
        resources=resources,
        feature_seconds=prep,
        sketch_seconds=sum(times.values()),
        sketch_seconds_by_species=times,
        wall_seconds=time.time() - started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    prefix = (
        args.root
        / "results"
        / f"{args.split}-w{args.width}-k{args.k}-d{args.divisor}-mask{int(args.mask)}-h{args.hashes}-{args.method}-r{args.rows}"
    )
    with gzip.open(str(prefix) + ".predictions.jsonl.gz", "wt") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    Path(str(prefix) + ".summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
