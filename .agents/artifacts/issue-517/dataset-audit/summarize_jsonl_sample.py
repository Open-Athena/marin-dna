#!/usr/bin/env python3
"""Summarize a bounded stream of projected-sequence JSONL rows."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from typing import Any


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-rows", type=int, default=100_000)
    args = parser.parse_args()

    counters: dict[str, Counter[str]] = {
        "alignment_source": Counter(),
        "augmentation": Counter(),
        "clade": Counter(),
        "source_chrom": Counter(),
        "species": Counter(),
    }
    query_names: set[str] = set()
    source_loci: set[tuple[str, int, int]] = set()
    row_count = 0
    sequence_bases = 0
    ambiguous_bases = 0
    lowercase_bases = 0
    gc_bases = 0
    high_ambiguity_rows = 0
    all_ambiguous_rows = 0
    fragment_count_sum = 0
    aligned_bases_sum = 0

    for line in sys.stdin:
        if row_count >= args.max_rows:
            break
        row: dict[str, Any] = json.loads(line)
        sequence = str(row["sequence"])
        upper = sequence.upper()
        ambiguous = sum(base not in "ACGT" for base in upper)

        row_count += 1
        sequence_bases += len(sequence)
        ambiguous_bases += ambiguous
        lowercase_bases += sum(base.islower() for base in sequence)
        gc_bases += upper.count("G") + upper.count("C")
        high_ambiguity_rows += ambiguous * 2 >= len(sequence)
        all_ambiguous_rows += ambiguous == len(sequence)
        fragment_count_sum += int(row["fragment_count"])
        aligned_bases_sum += int(row["aligned_bases"])

        query_names.add(str(row["query_name"]))
        source_loci.add(
            (
                str(row["source_chrom"]),
                int(row["source_start"]),
                int(row["source_end"]),
            )
        )
        for column in counters:
            counters[column][str(row[column])] += 1

    summary = {
        "dataset": args.dataset,
        "sample_contract": "first rows of immutable shuffled train shard_0000",
        "rows": row_count,
        "distinct_query_names": len(query_names),
        "distinct_source_loci": len(source_loci),
        "mean_sequence_length": _fraction(sequence_bases, row_count),
        "mean_ambiguous_fraction": _fraction(ambiguous_bases, sequence_bases),
        "mean_lowercase_fraction": _fraction(lowercase_bases, sequence_bases),
        "mean_gc_fraction": _fraction(gc_bases, sequence_bases - ambiguous_bases),
        "high_ambiguity_row_fraction": _fraction(high_ambiguity_rows, row_count),
        "all_ambiguous_row_fraction": _fraction(all_ambiguous_rows, row_count),
        "mean_fragment_count": _fraction(fragment_count_sum, row_count),
        "mean_aligned_bases": _fraction(aligned_bases_sum, row_count),
        "counts": {
            name: dict(counter.most_common()) for name, counter in counters.items()
        },
    }
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
