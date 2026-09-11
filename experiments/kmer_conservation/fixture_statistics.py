import collections
import gzip
import json
import statistics
from pathlib import Path

root = Path("/data/issue568/v2")
with gzip.open(root / "data/contexts.jsonl.gz", "rt") as f:
    data = [json.loads(x) for x in f]
stats = {}
for kind in sorted({r["kind"] for r in data}):
    rows = [r for r in data if r["kind"] == kind]
    stats[kind] = {"contexts": len(rows)}
    for feature in ["gc", "repeat", "complexity"]:
        v = sorted(r[feature] for r in rows)
        stats[kind][feature] = {
            "min": min(v),
            "mean": statistics.mean(v),
            "median": statistics.median(v),
            "max": max(v),
        }
    stats[kind]["ambiguous_bases"] = sum(
        sum(
            r["sequence"].upper().count(b)
            for b in set(r["sequence"].upper()) - set("ACGT")
        )
        for r in rows
    )
coverage = {}
for species in ["human", "mouse", "armadillo"]:
    chroms = collections.defaultdict(list)
    for r in data:
        if r["species"] == species and r["kind"] != "shuffled_decoy":
            chroms[r["chrom"]].append((r["start"], r["end"]))
    total = 0
    for ranges in chroms.values():
        end = -1
        for a, b in sorted(ranges):
            total += max(0, b - max(a, end))
            end = max(end, b)
    coverage[species] = total
(root / "fixture_statistics.json").write_text(
    json.dumps({"by_context_kind": stats, "unique_genomic_bp": coverage}, indent=2)
    + "\n"
)
print(json.dumps({"unique_genomic_bp": coverage, "by_context_kind": stats}, indent=2))
