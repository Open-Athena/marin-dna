#!/usr/bin/env python3
"""Compute paired matched-VEP AUPRC deltas on exact shared benchmark rows."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import polars as pl

from marin_dna_evals.metrics import paired_metric_delta_bootstrap


KEYS = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]
SCORE_COLUMNS = [*KEYS, "llr_fwd", "llr_rc"]


def _scores(
    path: str, subset: str, score_name: str, score_protocol: str
) -> pl.DataFrame:
    mean_llr = (pl.col("llr_fwd") + pl.col("llr_rc")) / 2
    score = -mean_llr if score_protocol == "minus_llr" else mean_llr.abs()
    return (
        pl.read_parquet(path, columns=SCORE_COLUMNS)
        .filter(pl.col("subset") == subset)
        .with_columns(score.alias(score_name))
        .select(*KEYS, score_name)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        action="append",
        nargs=4,
        metavar=("LABEL", "CURRENT", "BASELINE", "SUBSET"),
        required=True,
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=517)
    parser.add_argument(
        "--score-protocol",
        choices=("minus_llr", "abs_llr"),
        default="minus_llr",
    )
    args = parser.parse_args()

    results: dict[str, dict[str, Any]] = {}
    for label, current_path, baseline_path, subset in args.comparison:
        current = _scores(
            current_path, subset, "score_current", args.score_protocol
        )
        baseline = _scores(
            baseline_path, subset, "score_baseline", args.score_protocol
        )
        paired = current.join(baseline, on=KEYS, how="inner", validate="1:1")
        assert current.height == baseline.height == paired.height, (
            label,
            current.height,
            baseline.height,
            paired.height,
        )
        frame = paired.to_pandas()
        result = paired_metric_delta_bootstrap(
            frame["label"],
            frame["score_current"],
            frame["score_baseline"],
            frame["match_group"],
            n_bootstrap=args.n_bootstrap,
            rng=args.seed,
        )
        results[label] = {
            "subset": subset,
            "current": current_path,
            "baseline": baseline_path,
            "score_protocol": args.score_protocol,
            **result,
        }

    json.dump(results, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
