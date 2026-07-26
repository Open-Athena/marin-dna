#!/usr/bin/env python3
"""Audit issue #402 optimizer and throughput dynamics from exact W&B runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import wandb


RUNS = {
    ("46M", "131k"): "gonzalobenegas/marin/dna-exp402-rag-h640-p46M-30K-scratch",
    ("46M", "2.10M"): ("gonzalobenegas/marin/dna-exp402-rag-h640-p46M-B2M-30K-scratch"),
    ("104M", "131k"): "gonzalobenegas/marin/dna-exp402-rag-h768-p104M-30K-scratch",
    ("104M", "2.10M"): (
        "gonzalobenegas/marin/dna-exp402-rag-h768-p104M-B2M-30K-scratch"
    ),
}
METRIC_KEYS = (
    "train/loss",
    "grad/norm/total",
    "params/norm/total",
    "throughput/duration",
    "throughput/loading_time",
    "throughput/mfu",
    "optim/learning_rate",
    "optim/adam_lr",
)
GRADIENT_CLIP_NORM = 0.1
EARLY_WINDOW = (1_000, 3_600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_training_dynamics"),
    )
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help="Accept the available prefix of the two large-batch runs.",
    )
    return parser.parse_args()


def load_history(api: wandb.Api, *, allow_running: bool) -> pl.DataFrame:
    """Load complete finite training-metric records from the four exact runs."""
    rows: list[dict[str, object]] = []
    for (model, batch), run_path in RUNS.items():
        run = api.run(run_path)
        expected_states = {"finished", "running"} if allow_running else {"finished"}
        assert run.state in expected_states, (model, batch, run.state)
        for record in run.scan_history(
            keys=["global_step", *METRIC_KEYS], page_size=1_000
        ):
            if record.get("global_step") is None:
                continue
            if any(record.get(key) is None for key in METRIC_KEYS):
                continue
            rows.append(
                {
                    "model": model,
                    "batch": batch,
                    "run_state": run.state,
                    "step": int(record["global_step"]),
                    **{key: float(record[key]) for key in METRIC_KEYS},
                }
            )
    history = pl.DataFrame(rows).sort("model", "batch", "step")
    assert history.height > 0
    grouped_steps = history.group_by("model", "batch", "step").agg(
        pl.len().alias("n_records"),
        *[pl.col(key).n_unique().alias(f"{key}_n_unique") for key in METRIC_KEYS],
    )
    duplicate_steps = grouped_steps.filter(pl.col("n_records") > 1)
    n_unique_columns = [f"{key}_n_unique" for key in METRIC_KEYS]
    invalid_duplicates = duplicate_steps.filter(
        (pl.col("n_records") != 2)
        | (pl.col("step") % 1_000 != 0)
        | pl.any_horizontal(pl.col(column) != 1 for column in n_unique_columns)
    )
    assert invalid_duplicates.is_empty(), invalid_duplicates
    history = history.unique(
        subset=["model", "batch", "step"], keep="last", maintain_order=True
    )
    for key in METRIC_KEYS:
        assert history.filter(~pl.col(key).is_finite()).is_empty(), key
    assert history.filter(
        (pl.col("train/loss") <= 0)
        | (pl.col("grad/norm/total") <= 0)
        | (pl.col("params/norm/total") <= 0)
        | (pl.col("throughput/duration") <= 0)
        | (pl.col("throughput/loading_time") < 0)
        | (pl.col("throughput/mfu") <= 0)
        | (pl.col("optim/learning_rate") < 0)
        | (pl.col("optim/adam_lr") < 0)
    ).is_empty()
    if not allow_running:
        completed = history.group_by("model", "batch").agg(pl.col("step").max())
        assert completed.filter(pl.col("step") < 29_990).is_empty(), completed
    return history


def summarize(history: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Summarize steady training and a matched early-update clipping window."""
    after_1k = history.filter(pl.col("step") >= EARLY_WINDOW[0])
    summary = (
        after_1k.group_by("model", "batch", "run_state")
        .agg(
            pl.len().alias("n_logged_steps"),
            pl.col("step").min().alias("first_step"),
            pl.col("step").max().alias("last_step"),
            pl.col("grad/norm/total").median().alias("grad_norm_p50"),
            pl.col("grad/norm/total").quantile(0.95).alias("grad_norm_p95"),
            pl.col("grad/norm/total").max().alias("grad_norm_max"),
            (pl.col("grad/norm/total") > GRADIENT_CLIP_NORM)
            .mean()
            .alias("fraction_preclip_norm_gt_0_1"),
            pl.col("params/norm/total")
            .sort_by("step")
            .first()
            .alias("parameter_norm_first"),
            pl.col("params/norm/total")
            .sort_by("step")
            .last()
            .alias("parameter_norm_last"),
            pl.col("throughput/duration").median().alias("step_seconds_p50"),
            pl.col("throughput/duration").quantile(0.99).alias("step_seconds_p99"),
            pl.col("throughput/loading_time").median().alias("loader_seconds_p50"),
            pl.col("throughput/loading_time")
            .quantile(0.99)
            .alias("loader_seconds_p99"),
            pl.corr("throughput/loading_time", "throughput/duration").alias(
                "loader_step_time_pearson"
            ),
            pl.col("throughput/mfu").median().alias("mfu_p50"),
        )
        .with_columns(
            (pl.col("parameter_norm_last") / pl.col("parameter_norm_first") - 1).alias(
                "parameter_norm_fractional_change"
            )
        )
        .sort("model", "batch")
    )
    assert summary.height == len(RUNS)
    assert summary.filter(
        ~pl.col("grad_norm_p50").is_finite()
        | ~pl.col("parameter_norm_fractional_change").is_finite()
        | ~pl.col("loader_step_time_pearson").is_finite()
    ).is_empty()

    common_early_end = (
        history.group_by("model", "batch")
        .agg(pl.col("step").max().alias("available_last_step"))
        .group_by("model")
        .agg(
            pl.col("available_last_step")
            .min()
            .clip(upper_bound=EARLY_WINDOW[1])
            .alias("common_last_step")
        )
    )
    early = (
        history.join(common_early_end, on="model", how="inner")
        .filter(pl.col("step").is_between(EARLY_WINDOW[0], pl.col("common_last_step")))
        .group_by("model", "batch")
        .agg(
            pl.len().alias("n_logged_steps"),
            pl.col("step").min().alias("first_step"),
            pl.col("step").max().alias("last_step"),
            pl.col("grad/norm/total").median().alias("grad_norm_p50"),
            pl.col("grad/norm/total").quantile(0.95).alias("grad_norm_p95"),
            (pl.col("grad/norm/total") > GRADIENT_CLIP_NORM)
            .mean()
            .alias("fraction_preclip_norm_gt_0_1"),
        )
        .sort("model", "batch")
    )
    assert early.height == len(RUNS)
    assert early.filter(pl.col("n_logged_steps") < 100).is_empty(), early
    assert (
        early.group_by("model")
        .agg(pl.col("last_step").n_unique().alias("n_last_steps"))
        .filter(pl.col("n_last_steps") != 1)
        .is_empty()
    ), early
    return summary, early


def main() -> None:
    args = parse_args()
    history = load_history(wandb.Api(), allow_running=args.allow_running)
    summary, early = summarize(history)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history.write_parquet(args.output_dir / "history.parquet", compression="zstd")
    summary.write_parquet(args.output_dir / "summary.parquet", compression="zstd")
    early.write_parquet(args.output_dir / "early_matched.parquet", compression="zstd")
    with pl.Config(tbl_rows=20, tbl_cols=30, tbl_width_chars=220):
        print("STEADY/LIVE SUMMARY")
        print(summary)
        print(
            "\nCOMMON EARLY WINDOW (steps 1,000–per-model shared end, capped at 3,600)"
        )
        print(early)


if __name__ == "__main__":
    main()
