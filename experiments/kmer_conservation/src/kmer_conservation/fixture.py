"""Regenerate pinned homology controls and fixed genomic retrieval contexts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import boto3
import polars as pl
import py2bit
import yaml


def stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def features(seq: str) -> tuple[float, float, float]:
    upper = seq.upper()
    valid = sum(upper.count(b) for b in "ACGT")
    gc = (upper.count("G") + upper.count("C")) / max(valid, 1)
    masked = sum(b.islower() for b in seq) / len(seq)
    complexity = len({upper[i : i + 3] for i in range(len(seq) - 2)}) / 64
    return gc, masked, complexity


def assign_components(records: list[dict], seed: int, heldout: float) -> None:
    """Union homologs and every overlapping genomic context before splitting."""
    parents = {str(r["group"]): str(r["group"]) for r in records}

    def find(a: str) -> str:
        while parents[a] != a:
            parents[a] = parents[parents[a]]
            a = parents[a]
        return a

    def union(a: str, b: str) -> None:
        a, b = sorted((find(a), find(b)))
        parents[b] = a

    intervals: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        intervals[row["species"], row["chrom"]].append(row)
    for rows in intervals.values():
        active: list[dict] = []
        for row in sorted(rows, key=lambda x: x["start"]):
            active = [r for r in active if r["end"] > row["start"]]
            for other in active:
                union(row["group"], other["group"])
            active.append(row)
    for row in records:
        row["split_component"] = find(row["group"])
        row["split"] = (
            "heldout"
            if stable_hash(f"split:{seed}:{row['split_component']}") / 2**64 < heldout
            else "dev"
        )
    # Candidate budgets count connected intervals in one genome. Homology links
    # establish split isolation but must never merge disjoint target loci.
    for (species, chrom), rows in intervals.items():
        groups: list[list[dict]] = []
        right = -1
        for row in sorted(rows, key=lambda x: x["start"]):
            if not groups or row["start"] >= right:
                groups.append([])
            groups[-1].append(row)
            right = max(right, row["end"])
        for group in groups:
            start = min(r["start"] for r in group)
            end = max(r["end"] for r in group)
            for row in group:
                row["component"] = f"locus:{species}:{chrom}:{start}:{end}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cfg = json.loads(Path("config/screen.json").read_text())
    prior_cfg = yaml.safe_load(Path("config/prior_fixture.yaml").read_text())
    sources = prior_cfg["homology_fixture"]["sources"]
    client = boto3.client("s3", region_name="us-east-2")
    manifest: dict = {"sources": [], "config": cfg, "started": started}
    for source in sources:
        for field, suffix in [("uri", "parquet"), ("genome_uri", "2bit")]:
            bucket, key = source[field].removeprefix("s3://").split("/", 1)
            head = client.head_object(Bucket=bucket, Key=key)
            prefix = "genome_" if field.startswith("genome") else ""
            assert head["ETag"].strip('"') == source[f"{prefix}etag"]
            assert head["ContentLength"] == source[f"{prefix}size_bytes"]
            path = data / f"{source['label']}.{suffix}"
            if not path.exists():
                client.download_file(bucket, key, str(path))
            assert path.stat().st_size == head["ContentLength"]
            manifest["sources"].append(
                {
                    "uri": source[field],
                    "etag": head["ETag"],
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
            print("input", path.name, flush=True)

    # Run the original function, unchanged, from the commit-pinned source archive.
    prior_src = args.prior / "snakemake/analysis/linclust_conservation/src"
    sys.path.insert(0, str(prior_src))
    from marin_dna_linclust_conservation.homology_fixture import (
        ProjectionSource,
        build_projection_fixture,
    )

    manifest["original_fixture"] = build_projection_fixture(
        sources=[ProjectionSource.from_dict(s) for s in sources],
        paths=[data / f"{s['label']}.parquet" for s in sources],
        max_anchors=128,
        candidate_anchors=1024,
        window_length=255,
        fasta_path=data / "prior128.fasta",
        truth_path=data / "prior128.truth.tsv",
    )
    columns = [
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        "region_label",
        "species",
        "assembly",
        "sequence",
        "t_chrom",
        "t_strand",
        "t_src_size",
        "pre_resize_t_start",
        "pre_resize_t_end",
    ]
    frames = {
        s["label"]: pl.read_parquet(data / f"{s['label']}.parquet", columns=columns)
        for s in sources
    }
    common = set.intersection(*(set(df["query_name"]) for df in frames.values()))
    original = sorted(
        set(pl.read_csv(data / "prior128.truth.tsv", separator="\t")["query_name"])
    )
    shuffled = sorted(
        common - set(original), key=lambda x: stable_hash(f"{cfg['seed']}:{x}")
    )
    # Include a surplus so boundary and sequence-quality rejection cannot shrink the screen.
    candidates = original + shuffled[: cfg["additional_anchors"] * 8]
    by_species = {
        label: {
            r["query_name"]: r
            for r in df.filter(pl.col("query_name").is_in(candidates)).to_dicts()
        }
        for label, df in frames.items()
    }
    del frames
    handles = {
        s["label"]: py2bit.open(str(data / f"{s['label']}.2bit"), True) for s in sources
    }

    def extract(label: str, chrom: str, start: int, end: int) -> str:
        handle = handles[label]
        assert 0 <= start < end <= handle.chroms()[chrom]
        # storeMasked=True preserves lowercase directly in py2bit 1.0.1.
        return handle.sequence(chrom, start, end)

    length = cfg["context_length"]
    records: list[dict] = []
    rejected: dict[str, int] = defaultdict(int)
    extra = 0
    for name in candidates:
        rows = [by_species[s["label"]][name] for s in sources]
        assert (
            len({(r["source_chrom"], r["source_start"], r["source_end"]) for r in rows})
            == 1
        )
        if name not in original and any(
            len(r["sequence"]) != 255
            or set(r["sequence"].upper()) - set("ACGT")
            or features(r["sequence"])[1] > 0.5
            for r in rows
        ):
            rejected["extra_anchor_original_quality"] += 1
            continue
        new = []
        for source, row in zip(sources, rows, strict=True):
            label = source["label"]
            assert (
                row["species"] == source["species"]
                and row["assembly"] == source["assembly"]
            )
            chrom = row["t_chrom"]
            assert row["t_src_size"] == handles[label].chroms()[chrom]
            center = (row["pre_resize_t_start"] + row["pre_resize_t_end"]) // 2
            start, end = center - length // 2, center + length // 2
            if start < 0 or end > row["t_src_size"]:
                break
            seq = extract(label, chrom, start, end)
            assert len(seq) == length
            new.append(
                {
                    "id": f"{stable_hash(name):016x}_{label}",
                    "group": name,
                    "species": label,
                    "assembly": source["assembly"],
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "strand": "+",
                    "projection_strand": row["t_strand"],
                    "kind": "anchor",
                    "prior128": name in original,
                    "region_label": row["region_label"],
                    "sequence": seq,
                    "gc": features(seq)[0],
                    "repeat": features(seq)[1],
                    "complexity": features(seq)[2],
                }
            )
        if len(new) != len(sources):
            rejected["context_out_of_bounds"] += 1
            continue
        records.extend(new)
        extra += name not in original
        if extra == cfg["additional_anchors"]:
            break
    assert extra == cfg["additional_anchors"]
    assign_components(records, cfg["seed"], cfg["heldout_fraction"])
    occupied: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in records:
        occupied[row["species"], row["chrom"]].append((row["start"], row["end"]))

    for source in sources:
        label = source["label"]
        rng = random.Random(stable_hash(f"{cfg['seed']}:{label}:background"))
        chrom_sizes = {
            chrom: size
            for chrom, size in handles[label].chroms().items()
            if size >= length * 4
        }
        chroms, weights = zip(*chrom_sizes.items(), strict=True)
        pool = []
        while len(pool) < cfg["backgrounds_per_species"] + 2000:
            chrom = rng.choices(chroms, weights=weights)[0]
            start = rng.randrange(0, chrom_sizes[chrom] - length + 1)
            end = start + length
            if any(start < b and a < end for a, b in occupied[label, chrom]):
                continue
            seq = extract(label, chrom, start, end)
            if sum(b in "ACGTacgt" for b in seq) < 0.95 * length:
                continue
            occupied[label, chrom].append((start, end))
            gc, repeat, complexity = features(seq)
            name = f"background:{label}:{chrom}:{start}"
            pool.append(
                {
                    "id": f"{stable_hash(name):016x}_{label}",
                    "group": name,
                    "component": name,
                    "species": label,
                    "assembly": source["assembly"],
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "strand": "+",
                    "kind": "background",
                    "split": "background",
                    "sequence": seq,
                    "gc": gc,
                    "repeat": repeat,
                    "complexity": complexity,
                }
            )
        chosen = pool[: cfg["backgrounds_per_species"]]
        remaining = pool[cfg["backgrounds_per_species"] :]
        # One nearest unused real context per anchor, in scaled GC/repeat/complexity space.
        for anchor in [
            r for r in records if r["species"] == label and r["kind"] == "anchor"
        ]:
            match = min(
                remaining,
                key=lambda r: sum(
                    ((r[f] - anchor[f]) / scale) ** 2
                    for f, scale in [("gc", 0.05), ("repeat", 0.1), ("complexity", 0.1)]
                ),
            )
            remaining.remove(match)
            match["kind"] = "matched_background"
            match["matched_to"] = anchor["id"]
            chosen.append(match)
        # Additional high-repeat contexts expose repetitive-posting behavior.
        for match in sorted(remaining, key=lambda r: -r["repeat"])[:64]:
            match["kind"] = "repeat_challenge"
            chosen.append(match)
        records.extend(chosen)
        print("background", label, len(chosen), flush=True)

    # Preserve mononucleotide composition and the exact lowercase mask at each base.
    # These are injected decoys, not a claim about the correctness of real-genome hits.
    decoys = []
    for anchor in [r for r in records if r["kind"] == "anchor"]:
        rng = random.Random(stable_hash(f"shuffle:{anchor['id']}"))
        letters = list(anchor["sequence"].upper())
        rng.shuffle(letters)
        seq = "".join(
            b.lower() if a.islower() else b
            for a, b in zip(anchor["sequence"], letters, strict=True)
        )
        decoy = dict(anchor)
        name = f"shuffle:{anchor['id']}"
        decoy.update(
            id=f"{stable_hash(name):016x}_{anchor['species']}",
            group=name,
            component=name,
            kind="shuffled_decoy",
            sequence=seq,
            parent=anchor["id"],
        )
        decoys.append(decoy)
    records.extend(decoys)
    assert len({r["id"] for r in records}) == len(records)
    output = data / "contexts.jsonl.gz"
    with gzip.open(output, "wt") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    manifest.update(
        record_count=len(records),
        anchor_count=len({r["group"] for r in records if r["kind"] == "anchor"}),
        components=len({r["component"] for r in records if r["kind"] == "anchor"}),
        rejection_counts=dict(rejected),
        contexts_sha256=sha256(output),
        ended=time.time(),
    )
    (data / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}), flush=True)


if __name__ == "__main__":
    main()
