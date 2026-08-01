"""Compare aligned issue #430 VEP scores with a paired group bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from marin_dna.pipelines.evals.inference_benchmark import VARIANT_KEY_COLUMNS
from marin_dna.pipelines.evals.metrics import paired_auprc_degradation_metrics


SCORE_COLUMN = "minus_llr_avg"
METADATA_COLUMNS = ["target", "subset", "match_group"]


def _top_tail_overlap(
    baseline_score: np.ndarray,
    candidate_score: np.ndarray,
    fraction: float,
) -> dict[str, float | int]:
    assert 0 < fraction <= 1
    assert len(baseline_score) == len(candidate_score)
    k = max(1, int(np.ceil(len(baseline_score) * fraction)))
    baseline_top = np.argpartition(baseline_score, -k)[-k:]
    candidate_top = np.argpartition(candidate_score, -k)[-k:]
    overlap = len(np.intersect1d(baseline_top, candidate_top, assume_unique=True))
    return {"k": k, "overlap": overlap, "fraction": overlap / k}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=430)
    parser.add_argument("--degradation-threshold", type=float, default=0.01)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    baseline = pd.read_parquet(args.baseline)
    candidate = pd.read_parquet(args.candidate)
    required = [*VARIANT_KEY_COLUMNS, *METADATA_COLUMNS, SCORE_COLUMN]
    for name, frame in (("baseline", baseline), ("candidate", candidate)):
        missing = [column for column in required if column not in frame.columns]
        assert not missing, f"{name} missing columns: {missing}"
        assert not frame[required].isna().any().any(), (
            f"{name} contains missing values in comparison columns"
        )
        assert not frame.duplicated(VARIANT_KEY_COLUMNS).any(), (
            f"{name} contains duplicate variant keys"
        )
        assert np.isfinite(frame[SCORE_COLUMN]).all(), (
            f"{name} contains non-finite scores"
        )

    joined = baseline[required].merge(
        candidate[required],
        on=VARIANT_KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_candidate"),
    )
    assert len(joined) == len(baseline) == len(candidate), (
        f"unaligned score rows: joined={len(joined)} baseline={len(baseline)} "
        f"candidate={len(candidate)}"
    )
    for column in METADATA_COLUMNS:
        assert (joined[f"{column}_baseline"] == joined[f"{column}_candidate"]).all(), (
            f"baseline/candidate disagree on {column!r}"
        )

    comparison_dataset = pd.DataFrame(
        {
            "label": joined["target_baseline"].astype(int),
            "subset": joined["subset_baseline"],
            "match_group": joined["match_group_baseline"],
        }
    )
    baseline_score = joined[f"{SCORE_COLUMN}_baseline"].to_numpy(dtype=float)
    candidate_score = joined[f"{SCORE_COLUMN}_candidate"].to_numpy(dtype=float)
    degradation = paired_auprc_degradation_metrics(
        comparison_dataset,
        baseline_score,
        candidate_score,
        n_bootstrap=args.n_bootstrap,
        rng=args.seed,
        degradation_threshold=args.degradation_threshold,
    )

    score_delta = candidate_score - baseline_score
    abs_delta = np.abs(score_delta)
    summary = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "n_variants": len(joined),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "degradation_threshold": args.degradation_threshold,
        "score_delta_definition": "candidate_minus_baseline",
        "score_delta": {
            "exact_equal_fraction": float(np.mean(score_delta == 0)),
            "mean": float(np.mean(score_delta)),
            "mean_abs": float(np.mean(abs_delta)),
            "median_abs": float(np.median(abs_delta)),
            "p95_abs": float(np.percentile(abs_delta, 95)),
            "p99_abs": float(np.percentile(abs_delta, 99)),
            "max_abs": float(np.max(abs_delta)),
        },
        "rank_diagnostics": {
            "spearman": float(spearmanr(baseline_score, candidate_score).statistic),
            "top_1_percent_overlap": _top_tail_overlap(
                baseline_score, candidate_score, 0.01
            ),
            "top_5_percent_overlap": _top_tail_overlap(
                baseline_score, candidate_score, 0.05
            ),
        },
        "degradation": degradation.to_dict(orient="records"),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    degradation.to_csv(out_dir / "degradation.csv", index=False)
    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
