"""Recompute PairwiseAccuracy metrics from an existing scores parquet.

Use when you have a scores parquet from a prior run (e.g. 1B/7B/40B
mendelian results that were scored before JSD was added to the metrics
step) and want to add new score columns without re-running inference.

Reads ``--scores`` and writes a fresh ``--output`` metrics parquet,
scoring every score-column that exists in the input among
``minus_llr`` / ``abs_llr`` / ``next_token_jsd_mean`` × ``{"", "_fwd", "_rev"}``.

Tags rows with ``--model``, ``--dataset-name``, ``--split`` so the
output matches what the inference entry would have produced.
"""

import argparse
from pathlib import Path

import pandas as pd

from marin_dna.pipelines.evals.metrics import compute_pairwise_metrics


REQUIRED_VARIANT_COLUMNS = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "subset",
    "match_group",
)
SCORE_COLUMN_BASES = ("minus_llr", "abs_llr", "next_token_jsd_mean")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", required=True, help="Input scores parquet.")
    p.add_argument("--output", required=True, help="Output metrics parquet.")
    p.add_argument("--model", required=True, help="Model name to tag rows with.")
    p.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset short name (e.g. 'mendelian_traits', 'complex_traits').",
    )
    p.add_argument("--split", default="train")
    args = p.parse_args()

    df = pd.read_parquet(args.scores)
    missing = [c for c in REQUIRED_VARIANT_COLUMNS if c not in df.columns]
    assert not missing, f"scores parquet missing required cols: {missing}"

    score_cols: list[str] = []
    for base in SCORE_COLUMN_BASES:
        for suffix in ("", "_fwd", "_rev"):
            col = f"{base}{suffix}"
            if col in df.columns:
                score_cols.append(col)
    assert score_cols, (
        f"no scored columns found in {args.scores}; expected one of "
        f"{SCORE_COLUMN_BASES} (× '' / _fwd / _rev)"
    )
    print(f"[compute_metrics] scoring {len(score_cols)} columns: {score_cols}")

    metrics = compute_pairwise_metrics(
        dataset=df[list(REQUIRED_VARIANT_COLUMNS)],
        scores=df[score_cols],
        score_columns=score_cols,
    )
    metrics["model"] = args.model
    metrics["dataset"] = args.dataset_name
    metrics["split"] = args.split

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(out_path, index=False)
    print(f"[compute_metrics] wrote {out_path} ({len(metrics)} rows)")


if __name__ == "__main__":
    main()
