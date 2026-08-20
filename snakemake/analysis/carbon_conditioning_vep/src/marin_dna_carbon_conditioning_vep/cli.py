"""Command-line entry points used by the isolated Snakemake workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from marin_dna_carbon_conditioning_vep.pipeline import (
    build_validated_windows,
    label_blind_smoke_sample,
    stage_analysis_windows,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict)
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _run_preflight(args: argparse.Namespace) -> None:
    from marin_dna_carbon_conditioning_vep.preflight import run_prompt_preflight

    config = _load_yaml(args.config)
    prompt = config["prompt_preflight"]
    results, summary = run_prompt_preflight(
        model_repo=str(config["model"]["repo"]),
        model_revision=str(config["model"]["revision"]),
        dtype_name=str(config["inference"]["dtype"]),
        dataset_repo=str(prompt["dataset_repo"]),
        dataset_revision=str(prompt["dataset_revision"]),
        dataset_config=str(prompt["dataset_config"]),
        split=str(prompt["split"]),
        target_species=[str(value) for value in prompt["target_species"]],
        rows_per_species=int(prompt["rows_per_species"]),
        max_sequence_bp=int(prompt["max_sequence_bp"]),
        generated_tokens=int(prompt["generated_tokens"]),
        batch_size=int(prompt["batch_size"]),
        grammar_templates={
            str(name): str(template) for name, template in prompt["grammars"].items()
        },
        conditions={
            str(name): None if value is None else str(value)
            for name, value in config["conditions"].items()
        },
        selection_tolerance=float(prompt["selection_tolerance"]),
    )
    results.to_parquet(args.output_parquet, index=False)
    _write_json(args.output_json, summary)


def _build_windows(args: argparse.Namespace) -> None:
    config = _load_yaml(args.config)
    preflight = _load_json(args.preflight)
    assert preflight.get("status") == "selected", (
        "prompt preflight has not selected a grammar; Mendelian labels remain gated"
    )
    windows, exclusions = build_validated_windows(
        dataset_config=config["dataset"],
        reference_config=config["reference"],
        analysis_config=config["analysis"],
        window_size=int(config["inference"]["window_size_bp"]),
    )
    windows.to_parquet(args.output_windows, index=False)
    exclusions.to_parquet(args.output_exclusions, index=False)


def _build_smoke_sample(args: argparse.Namespace) -> None:
    config = _load_yaml(args.config)
    windows = pd.read_parquet(args.windows)
    sample = label_blind_smoke_sample(
        windows,
        n_rows=int(config["smoke"]["rows"]),
        seed=int(config["smoke"]["seed"]),
    )
    sample.to_parquet(args.output, index=False)


def _stage_windows(args: argparse.Namespace) -> None:
    config = _load_yaml(args.config)
    windows = stage_analysis_windows(
        args.source,
        dataset_config=config["dataset"],
        reference_config=config["reference"],
        analysis_config=config["analysis"],
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(destination, index=False)


def _score(args: argparse.Namespace) -> None:
    from marin_dna_carbon_conditioning_vep.scoring import score_condition_dataframe

    config = _load_yaml(args.config)
    preflight = _load_json(args.preflight)
    assert preflight.get("status") == "selected"
    condition = str(args.condition)
    assert condition in config["conditions"], f"unknown condition {condition!r}"
    windows = pd.read_parquet(args.windows)
    scores, runtime = score_condition_dataframe(
        windows,
        condition=condition,
        prompt_grammar=str(preflight["selected_grammar"]),
        prefix=str(preflight["selected_prefixes"][condition]),
        model_repo=str(config["model"]["repo"]),
        model_revision=str(config["model"]["revision"]),
        dtype_name=str(config["inference"]["dtype"]),
        batch_size=int(config["inference"]["batch_size"]),
        kmer_size=int(config["inference"]["kmer_size"]),
    )
    scores.to_parquet(args.output, index=False)
    _write_json(args.runtime, runtime)


def _absolute_metrics(args: argparse.Namespace) -> None:
    from marin_dna_carbon_conditioning_vep.metrics import compute_absolute_auprc

    config = _load_yaml(args.config)
    metric_config = config["metrics"]
    scores = pd.read_parquet(args.scores)
    metrics = compute_absolute_auprc(
        scores,
        condition=str(args.condition),
        score_column=str(metric_config["score_column"]),
        n_bootstrap=int(metric_config["n_bootstrap"]),
        bootstrap_seed=int(metric_config["bootstrap_seed"]),
        min_groups_for_macro=int(metric_config["min_groups_for_macro"]),
    )
    metrics.to_parquet(args.output, index=False)


def _paired_deltas(args: argparse.Namespace) -> None:
    from marin_dna_carbon_conditioning_vep.metrics import compute_paired_auprc_deltas

    config = _load_yaml(args.config)
    metric_config = config["metrics"]
    condition_a, condition_b = [
        str(value) for value in metric_config["comparisons"][args.comparison]
    ]
    score_a = pd.read_parquet(args.score_a)
    score_b = pd.read_parquet(args.score_b)
    deltas = compute_paired_auprc_deltas(
        score_a,
        score_b,
        comparison=str(args.comparison),
        condition_a=condition_a,
        condition_b=condition_b,
        score_column=str(metric_config["score_column"]),
        n_bootstrap=int(metric_config["n_bootstrap"]),
        bootstrap_seed=int(metric_config["bootstrap_seed"]),
        min_groups_for_macro=int(metric_config["min_groups_for_macro"]),
    )
    deltas.to_parquet(args.output, index=False)


def _report(args: argparse.Namespace) -> None:
    from marin_dna_carbon_conditioning_vep.report import render_summary

    config = _load_yaml(args.config)
    preflight = _load_json(args.preflight)
    absolute = pd.concat(
        [pd.read_parquet(path) for path in args.absolute_metrics],
        ignore_index=True,
    )
    deltas = pd.concat(
        [pd.read_parquet(path) for path in args.paired_deltas],
        ignore_index=True,
    )
    exclusions = pd.read_parquet(args.exclusions)
    runtimes = [_load_json(path) for path in args.runtimes]
    summary = render_summary(
        config=config,
        preflight=preflight,
        absolute_metrics=absolute,
        paired_deltas=deltas,
        exclusions=exclusions,
        runtimes=runtimes,
    )
    Path(args.output).write_text(summary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--output-json", required=True)
    preflight.add_argument("--output-parquet", required=True)
    preflight.set_defaults(function=_run_preflight)

    windows = subparsers.add_parser("build-windows")
    windows.add_argument("--config", required=True)
    windows.add_argument("--preflight", required=True)
    windows.add_argument("--output-windows", required=True)
    windows.add_argument("--output-exclusions", required=True)
    windows.set_defaults(function=_build_windows)

    smoke = subparsers.add_parser("smoke-sample")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--windows", required=True)
    smoke.add_argument("--output", required=True)
    smoke.set_defaults(function=_build_smoke_sample)

    stage = subparsers.add_parser("stage-windows")
    stage.add_argument("--config", required=True)
    stage.add_argument("--source", required=True)
    stage.add_argument("--output", required=True)
    stage.set_defaults(function=_stage_windows)

    score = subparsers.add_parser("score")
    score.add_argument("--config", required=True)
    score.add_argument("--preflight", required=True)
    score.add_argument("--windows", required=True)
    score.add_argument("--condition", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--runtime", required=True)
    score.set_defaults(function=_score)

    absolute = subparsers.add_parser("absolute-metrics")
    absolute.add_argument("--config", required=True)
    absolute.add_argument("--scores", required=True)
    absolute.add_argument("--condition", required=True)
    absolute.add_argument("--output", required=True)
    absolute.set_defaults(function=_absolute_metrics)

    paired = subparsers.add_parser("paired-deltas")
    paired.add_argument("--config", required=True)
    paired.add_argument("--comparison", required=True)
    paired.add_argument("--score-a", required=True)
    paired.add_argument("--score-b", required=True)
    paired.add_argument("--output", required=True)
    paired.set_defaults(function=_paired_deltas)

    report = subparsers.add_parser("report")
    report.add_argument("--config", required=True)
    report.add_argument("--preflight", required=True)
    report.add_argument("--absolute-metrics", nargs="+", required=True)
    report.add_argument("--paired-deltas", nargs="+", required=True)
    report.add_argument("--exclusions", required=True)
    report.add_argument("--runtimes", nargs="+", required=True)
    report.add_argument("--output", required=True)
    report.set_defaults(function=_report)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
