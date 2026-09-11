import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import polars as pl

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/issue568/data")
with gzip.open(root / "contexts.jsonl.gz", "rt") as handle:
    rows = [json.loads(line) for line in handle]
anchors = [r for r in rows if r["kind"] == "anchor"]
report = {
    "component_splits": dict(
        Counter(
            split
            for split, component in {
                (r["split"], r["split_component"]) for r in anchors
            }
        )
    ),
    "record_kinds": dict(Counter(r["kind"] for r in rows)),
    "central255_matches": {},
}
for label in ["human", "mouse", "armadillo"]:
    sub = [r for r in anchors if r["species"] == label]
    names = [r["group"] for r in sub]
    projected = {
        r["query_name"]: r
        for r in pl.read_parquet(root / f"{label}.parquet")
        .filter(pl.col("query_name").is_in(names))
        .to_dicts()
    }
    exact = 0
    for row in sub:
        seq = row["sequence"][2048 - 127 : 2048 + 128].upper()
        expected = projected[row["group"]]["sequence"].upper()
        rc = expected.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        exact += seq in [expected, rc]
    report["central255_matches"][label] = [exact, len(sub)]
assert all(a == b for a, b in report["central255_matches"].values()), report
assert not {r["split_component"] for r in anchors if r["split"] == "dev"} & {
    r["split_component"] for r in anchors if r["split"] == "heldout"
}
for component in {r["component"] for r in anchors}:
    sub = sorted(
        [r for r in anchors if r["component"] == component], key=lambda r: r["start"]
    )
    assert len({(r["species"], r["chrom"]) for r in sub}) == 1
    right = sub[0]["end"]
    for row in sub[1:]:
        assert row["start"] < right
        right = max(right, row["end"])
assert (
    0.25
    <= report["component_splits"]["heldout"] / sum(report["component_splits"].values())
    <= 0.55
)
assert (
    len(
        {
            (r["species"], r["chrom"], r["start"], r["end"])
            for r in rows
            if r["kind"] in ["background", "matched_background", "repeat_challenge"]
        }
    )
    == 3960
)
print(json.dumps(report, indent=2))
(root / "fixture_audit.json").write_text(json.dumps(report, indent=2) + "\n")
