#!/usr/bin/env python3
"""Summarize issue #402's official final-checkpoint Complex Traits/SGE metrics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import polars as pl

RESULTS_ROOT = "s3://oa-bolinas/snakemake/analysis/evals_v2/results"
MODELS = {
    "Ortholog-RAG 46M (30k)": "exp402-rag-h640-p46m-b2m-step-29999",
    "Ortholog-RAG 104M (30k)": "exp402-rag-h768-p104m-b2m-step-29999",
}
COMPLEX_SCORE = "abs_llr_avg"
SGE_SCORE = "minus_llr_avg"
EXPECTED_ROWS = {"complex_traits": 11_630, "sge": 23_853}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read_metrics(
    results_root: str, model: str, dataset: str, *, probe: bool
) -> pl.DataFrame:
    result_type = "probe_metrics" if probe else "metrics"
    path = f"{results_root.rstrip('/')}/{result_type}/{model}/{dataset}.parquet"
    metrics = pl.read_parquet(path)
    assert metrics.select(pl.col("split").unique()).to_series().to_list() == ["train"]
    assert metrics.select(pl.col("dataset").unique()).to_series().to_list() == [dataset]
    assert metrics.select(pl.col("model").unique()).to_series().to_list() == [model]
    score_type = (
        "probe_score"
        if probe
        else (COMPLEX_SCORE if dataset == "complex_traits" else SGE_SCORE)
    )
    selected = metrics.filter(pl.col("score_type") == score_type)
    assert selected.height > 0
    return selected


def _audit_scores(results_root: str, model: str, dataset: str) -> None:
    score_type = COMPLEX_SCORE if dataset == "complex_traits" else SGE_SCORE
    columns = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "label",
        "subset",
        "llr_fwd",
        "llr_rc",
        "llr_avg",
        score_type,
    ]
    path = f"{results_root.rstrip('/')}/scores/{model}/{dataset}.parquet"
    scores = pl.read_parquet(path, columns=columns)
    expected = EXPECTED_ROWS[dataset]
    assert scores.height == expected
    assert (
        scores.select(pl.struct("chrom", "pos", "ref", "alt").n_unique()).item()
        == expected
    )
    assert scores.null_count().sum_horizontal().item() == 0
    for column in ("llr_fwd", "llr_rc", "llr_avg", score_type):
        assert scores.filter(~pl.col(column).is_finite()).is_empty()
    average_error = scores.select(
        (pl.col("llr_avg") - (pl.col("llr_fwd") + pl.col("llr_rc")) / 2).abs().max()
    ).item()
    assert average_error < 1e-4
    expected_score = (
        pl.col("llr_avg").abs() if dataset == "complex_traits" else -pl.col("llr_avg")
    )
    transform_error = scores.select(
        (pl.col(score_type) - expected_score).abs().max()
    ).item()
    assert transform_error < 1e-6


def _row(frame: pl.DataFrame, **keys: str) -> dict[str, object]:
    selected = frame
    for column, value in keys.items():
        selected = selected.filter(pl.col(column) == value)
    assert selected.height == 1, (keys, selected)
    return selected.row(0, named=True)


def _value(frame: pl.DataFrame, **keys: str) -> str:
    row = _row(frame, **keys)
    value = float(row["value"])
    se = float(row["se"])
    assert math.isfinite(value) and math.isfinite(se)
    return f"{value:.6f} ± {se:.6f}"


def _assert_equal_support(frames: dict[str, pl.DataFrame], columns: list[str]) -> None:
    reference_label = next(iter(frames))
    reference = frames[reference_label].select(columns).sort(columns)
    for label, frame in frames.items():
        support = frame.select(columns).sort(columns)
        assert support.equals(reference), (reference_label, label)


def summarize(results_root: str) -> str:
    for model in MODELS.values():
        for dataset in EXPECTED_ROWS:
            _audit_scores(results_root, model, dataset)

    complex_zero = {
        label: _read_metrics(results_root, model, "complex_traits", probe=False)
        for label, model in MODELS.items()
    }
    complex_probe = {
        label: _read_metrics(results_root, model, "complex_traits", probe=True)
        for label, model in MODELS.items()
    }
    sge_zero = {
        label: _read_metrics(results_root, model, "sge", probe=False)
        for label, model in MODELS.items()
    }
    sge_probe = {
        label: _read_metrics(results_root, model, "sge", probe=True)
        for label, model in MODELS.items()
    }

    _assert_equal_support(complex_zero, ["subset", "n_groups", "n_rows"])
    _assert_equal_support(complex_probe, ["subset", "n_pos", "n"])
    sge_support = ["subset", "accession", "gene", "n", "n_pos"]
    _assert_equal_support(sge_zero, sge_support)
    _assert_equal_support(sge_probe, sge_support)

    complex_subsets = sorted(
        set(
            complex_zero[next(iter(MODELS))]
            .filter(
                ~pl.col("subset").str.starts_with("_")
                & (pl.col("n_groups") >= 30)
                & pl.col("value").is_finite()
            )
            .get_column("subset")
        )
    )
    probe_subsets = set(
        complex_probe[next(iter(MODELS))]
        .filter(
            ~pl.col("subset").str.starts_with("_")
            & (pl.col("n_pos") >= 30)
            & pl.col("value").is_finite()
        )
        .get_column("subset")
    )
    assert probe_subsets == set(complex_subsets)

    lines = [
        "| Benchmark | Model | Official zero-shot AUPRC | Official frozen-probe AUPRC |",
        "|:--|:--|--:|--:|",
    ]
    for label in MODELS:
        lines.append(
            f"| Complex Traits macro | {label} | "
            f"{_value(complex_zero[label], subset='_macro_avg_')} | "
            f"{_value(complex_probe[label], subset='_macro_avg_')} |"
        )
        lines.append(
            f"| SGE subset × accession macro | {label} | "
            f"{_value(sge_zero[label], subset='_macro_avg_', accession='_macro_avg_')} | "
            f"{_value(sge_probe[label], subset='_macro_avg_', accession='_macro_avg_')} |"
        )

    lines.extend(
        [
            "",
            "<details>",
            "<summary>Official Complex Traits per-subset results</summary>",
            "",
            "| Subset | 46M zero-shot | 46M probe | 104M zero-shot | 104M probe |",
            "|:--|--:|--:|--:|--:|",
        ]
    )
    labels = list(MODELS)
    for subset in complex_subsets:
        lines.append(
            f"| {subset} | {_value(complex_zero[labels[0]], subset=subset)} | "
            f"{_value(complex_probe[labels[0]], subset=subset)} | "
            f"{_value(complex_zero[labels[1]], subset=subset)} | "
            f"{_value(complex_probe[labels[1]], subset=subset)} |"
        )
    lines.extend(["", "</details>", ""])

    lines.extend(
        [
            "<details>",
            "<summary>Official SGE results macro-averaged over accessions</summary>",
            "",
            "| Subset | 46M zero-shot | 46M probe | 104M zero-shot | 104M probe |",
            "|:--|--:|--:|--:|--:|",
        ]
    )
    sge_display_subsets = ("missense_variant", "splicing", "_macro_avg_")
    for protocol, frames in (("zero-shot", sge_zero), ("probe", sge_probe)):
        for label, frame in frames.items():
            available = set(
                frame.filter(pl.col("accession") == "_macro_avg_").get_column("subset")
            )
            assert set(sge_display_subsets) <= available, (protocol, label)
    for subset in sge_display_subsets:
        lines.append(
            f"| {subset} | "
            f"{_value(sge_zero[labels[0]], subset=subset, accession='_macro_avg_')} | "
            f"{_value(sge_probe[labels[0]], subset=subset, accession='_macro_avg_')} | "
            f"{_value(sge_zero[labels[1]], subset=subset, accession='_macro_avg_')} | "
            f"{_value(sge_probe[labels[1]], subset=subset, accession='_macro_avg_')} |"
        )
    lines.extend(
        [
            "",
            "</details>",
            "",
            (
                "All rows are standard `evals_v2` **train** metrics. Complex Traits "
                "zero-shot uses `abs((LLR_fwd + LLR_rc)/2)`. SGE zero-shot uses "
                "`-(LLR_fwd + LLR_rc)/2` and the official per-accession aggregation. "
                "The SGE table shows the two consequence subsets and their official "
                "macro; the probe protocol does not emit the separate pooled `both` row. "
                "The two model sizes have identical metric support within every "
                "reported protocol."
            ),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    markdown = summarize(args.results_root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
