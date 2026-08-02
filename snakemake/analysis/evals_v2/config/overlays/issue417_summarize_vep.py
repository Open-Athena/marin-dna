"""Create the frozen paired Mendelian and SGE report for issue #417.

Run after the four evals_v2 score and metric cells have completed:

    uv run --group genome-s3 python scripts/issue417_summarize_vep.py \
        --experiment-commit COMMIT_SHA

Inputs are read directly from the canonical evals_v2 S3 prefix. The JSON and
Markdown comparison are written beneath its issue-specific comparisons path.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import pandas as pd
import polars as pl

from marin_dna_evals.issue417_summary import (
    ARMS,
    COMBINED_ARM,
    DATASETS,
    MAMMALS_ARM,
    SCORE_TYPE,
    SPLIT,
    build_issue417_comparison,
)

DEFAULT_RESULTS_PREFIX = "s3://oa-bolinas/snakemake/analysis/evals_v2"
DEFAULT_OUTPUT_PREFIX = f"{DEFAULT_RESULTS_PREFIX}/results/comparisons/issue417"
MODEL_NAMES = {
    MAMMALS_ARM: "exp417-cds-mammals-only-step-4999",
    COMBINED_ARM: "exp417-cds-combined-vertebrates-step-4999",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX)
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-commit", required=True)
    return parser.parse_args()


def _read_parquet(path: str) -> pd.DataFrame:
    frame = pl.read_parquet(path)
    assert frame.height > 0, f"empty parquet: {path}"
    return frame.to_pandas()


def _write_text(path: str, content: str) -> None:
    if path.startswith("s3://"):
        with fsspec.open(path, "wt") as handle:
            handle.write(content)
        return
    local = Path(path)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(content)


def _markdown(
    rows: list[dict[str, object]],
    *,
    revision: str,
    generated_at: str,
) -> str:
    lines = [
        "# Issue #417 terminal-checkpoint VEP comparison",
        "",
        f"- Experiment commit: {revision}",
        f"- Generated: {generated_at}",
        f"- Split: {SPLIT}",
        f"- Score: {SCORE_TYPE}",
        "- Reported scopes: missense, splicing, and synonymous where available; "
        "no global row",
        "- Delta: combined vertebrates minus mammals only",
        "- Uncertainty: 1,000 paired bootstrap iterations, seed 0",
        "",
        "| Dataset | Scope | Mammals AUPRC +/- SE | Combined AUPRC +/- SE | "
        "Delta (95% CI) | paired p | Units / rows |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {scope} | {mammals_value:.4f} +/- {mammals_se:.4f} | "
            "{combined_value:.4f} +/- {combined_se:.4f} | "
            "{delta:+.4f} ({ci_low:+.4f}, {ci_high:+.4f}) | "
            "{p_two_sided:.4g} | {n_units:,} / {n_rows:,} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Mendelian units are matched match_group clusters. SGE units are "
            "qualifying MaveDB accessions; its bootstrap resamples shared rows "
            "within each accession before macro-averaging over the fixed "
            "qualifying accession set.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    assert args.n_bootstrap == 1000, "issue #417 protocol freezes n_bootstrap=1000"
    assert args.seed == 0, "issue #417 protocol freezes seed=0"
    assert len(args.experiment_commit) == 40 and all(
        character in "0123456789abcdef" for character in args.experiment_commit
    ), "experiment commit must be a full lowercase Git SHA"
    root = args.results_prefix.rstrip("/")
    output = args.output_prefix.rstrip("/")

    scores = {}
    metrics = {}
    source_paths: dict[str, str] = {}
    for arm in ARMS:
        model = MODEL_NAMES[arm]
        for dataset in DATASETS:
            score_path = f"{root}/results/scores/{model}/{dataset}.parquet"
            metric_path = f"{root}/results/metrics/{model}/{dataset}.parquet"
            scores[(arm, dataset)] = _read_parquet(score_path)
            metrics[(arm, dataset)] = _read_parquet(metric_path)
            source_paths[f"scores/{arm}/{dataset}"] = score_path
            source_paths[f"metrics/{arm}/{dataset}"] = metric_path

    comparison = build_issue417_comparison(
        scores,
        metrics,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    rows = comparison.to_dict(orient="records")
    generated_at = datetime.now(UTC).isoformat()
    revision = args.experiment_commit
    payload = {
        "schema_version": 1,
        "issue": 417,
        "experiment_commit": revision,
        "generated_at": generated_at,
        "split": SPLIT,
        "score_type": SCORE_TYPE,
        "delta": "combined_vertebrates - mammals_only",
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.seed,
        "source_paths": source_paths,
        "results": rows,
    }
    json_text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown = _markdown(rows, revision=revision, generated_at=generated_at)

    json_path = f"{output}/summary.json"
    markdown_path = f"{output}/summary.md"
    _write_text(json_path, json_text)
    _write_text(markdown_path, markdown)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    print(markdown)


if __name__ == "__main__":
    main()
