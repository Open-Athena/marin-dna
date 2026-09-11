"""Audit complete published prediction files independently of the scorer."""

import argparse
import gzip
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    reference: dict[str, dict] = {}
    checked = 0
    for path in sorted((args.root / "results").glob("*.predictions.jsonl.gz")):
        if "verify" in path.name:
            continue
        split = path.name.split("-")[0]
        with gzip.open(path, "rt") as handle:
            rows = [json.loads(line) for line in handle]
        identity = {
            (r["source"], r["target"], r["query"]): sorted(r["ranks"]) for r in rows
        }
        assert len(identity) == len(rows)
        if split not in reference:
            reference[split] = identity
        assert identity == reference[split], path
        summary = json.loads(
            Path(
                str(path).replace(".predictions.jsonl.gz", ".summary.json")
            ).read_text()
        )
        assert sum(len(r["ranks"]) for r in rows) == summary["n"]
        assert len(rows) == summary["n_queries"]
        for row in rows:
            hits = [h["component"] for h in row["hits"]]
            assert len(set(hits)) == len(hits) <= 100
            assert len(hits) <= row["candidate_loci"]
            assert row["seconds"] >= 0 and math.isfinite(row["seconds"])
            for truth, rank in row["ranks"].items():
                assert rank == (hits.index(truth) + 1 if truth in hits else None)
        for budget in [1, 10, 100]:
            count = sum(
                rank is not None and rank <= budget
                for row in rows
                for rank in row["ranks"].values()
            )
            assert abs(count / summary["n"] - summary[f"recall_at_{budget}"]) < 1e-12
        checked += 1
    assert checked > 0
    print(
        json.dumps(
            {
                "checked_prediction_files": checked,
                "query_counts": {s: len(rows) for s, rows in reference.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
