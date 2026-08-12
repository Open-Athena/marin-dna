#!/usr/bin/env python3
"""Audit and summarize issue #402's official Complex/SGE baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

RESULTS_ROOT = "s3://oa-bolinas/snakemake/analysis/evals_v2/results"
CONSERVATION_ROOT = "s3://oa-bolinas/snakemake/conservation_eval/results"
PHYLOP_SCORE = "phyloP_447m"
M51_MODEL = "mix-v0.9-p1B-i24-exp135-m5.1-step-59158"
S3_STORAGE_OPTIONS = {"aws_region": "us-east-2"}
MODELS = {
    "Ortholog-RAG 46M": "exp402-rag-h640-p46m-b2m-step-29999",
    "Ortholog-RAG 104M": "exp402-rag-h768-p104m-b2m-step-29999",
    "MarinDNA m5.1": M51_MODEL,
}
EXPECTED_ROWS = {"complex_traits": 11_630, "sge": 23_853}
KEY_COLUMNS = {
    "complex_traits": [
        "chrom",
        "pos",
        "ref",
        "alt",
        "label",
        "subset",
        "match_group",
    ],
    "sge": [
        "chrom",
        "pos",
        "ref",
        "alt",
        "label",
        "subset",
        "gene",
        "mavedb_urn",
    ],
}
ZERO_SCORE = {"complex_traits": "abs_llr_avg", "sge": "minus_llr_avg"}
COMPLEX_SUBSET_LABELS = {
    "3_prime_UTR_variant": "3′ UTR",
    "5_prime_UTR_variant": "5′ UTR",
    "distal": "Distal",
    "missense_variant": "Missense",
    "non_coding_transcript_exon_variant": "Non-coding exon",
    "tss_proximal": "TSS proximal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--conservation-root", default=CONSERVATION_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _model_path(root: str, kind: str, model: str, dataset: str) -> str:
    return f"{root.rstrip('/')}/{kind}/{model}/{dataset}.parquet"


def _phylop_path(root: str, dataset: str, metrics: bool) -> str:
    filename = "metrics_train.parquet" if metrics else f"{PHYLOP_SCORE}_train.parquet"
    return f"{root.rstrip('/')}/{dataset}/{filename}"


def _read_keys(path: str, dataset: str) -> pl.DataFrame:
    columns = KEY_COLUMNS[dataset]
    frame = pl.read_parquet(
        path, columns=columns, storage_options=S3_STORAGE_OPTIONS
    ).cast(
        {
            "chrom": pl.String,
            "pos": pl.Int64,
            "ref": pl.String,
            "alt": pl.String,
            "label": pl.Boolean,
            "subset": pl.String,
            **(
                {"match_group": pl.Int64}
                if dataset == "complex_traits"
                else {"gene": pl.String, "mavedb_urn": pl.String}
            ),
        }
    )
    assert frame.height == EXPECTED_ROWS[dataset], (path, frame.height)
    assert frame.unique().height == frame.height, path
    assert frame.null_count().sum_horizontal().item() == 0, path
    return frame.sort(columns)


def _assert_exact_variant_sets(
    results_root: str, conservation_root: str, dataset: str
) -> dict[str, int]:
    frames = {
        label: _read_keys(_model_path(results_root, "scores", model, dataset), dataset)
        for label, model in MODELS.items()
    }
    frames["phyloP 447 mammals"] = _read_keys(
        _phylop_path(conservation_root, dataset, metrics=False), dataset
    )
    reference = frames["Ortholog-RAG 46M"]
    for label, frame in frames.items():
        assert frame.equals(reference), f"{dataset}: row mismatch for {label}"

    phylop_scores = pl.read_parquet(
        _phylop_path(conservation_root, dataset, metrics=False),
        columns=["score"],
        storage_options=S3_STORAGE_OPTIONS,
    )
    assert phylop_scores.height == EXPECTED_ROWS[dataset]
    missing = phylop_scores.select(
        (pl.col("score").is_null() | pl.col("score").is_nan()).sum()
    ).item()
    return {"rows": reference.height, "phylop_missing": int(missing)}


def _read_model_metrics(
    results_root: str, dataset: str, model: str, probe: bool
) -> pl.DataFrame:
    kind = "probe_metrics" if probe else "metrics"
    frame = pl.read_parquet(
        _model_path(results_root, kind, model, dataset),
        storage_options=S3_STORAGE_OPTIONS,
    )
    assert frame.select(pl.col("split").unique()).to_series().to_list() == ["train"]
    assert frame.select(pl.col("dataset").unique()).to_series().to_list() == [dataset]
    score_type = "probe_score" if probe else ZERO_SCORE[dataset]
    selected = frame.filter(pl.col("score_type") == score_type)
    assert selected.height > 0
    return selected


def _read_phylop_metrics(conservation_root: str, dataset: str) -> pl.DataFrame:
    frame = pl.read_parquet(
        _phylop_path(conservation_root, dataset, metrics=True),
        storage_options=S3_STORAGE_OPTIONS,
    )
    assert frame.select(pl.col("split").unique()).to_series().to_list() == ["train"]
    assert frame.select(pl.col("dataset").unique()).to_series().to_list() == [dataset]
    selected = frame.filter(
        (pl.col("score_name") == PHYLOP_SCORE) & (pl.col("score_type") == "score")
    )
    assert selected.height > 0
    return selected


def _assert_metric_support(
    dataset: str,
    zero: dict[str, pl.DataFrame],
    probes: dict[str, pl.DataFrame],
    phylop: pl.DataFrame,
) -> None:
    if dataset == "complex_traits":
        zero_support_columns = ["subset", "n_groups", "n_rows"]
        probe_support_columns = ["subset", "n_pos", "n"]
    else:
        zero_support_columns = ["subset", "accession", "gene", "n", "n_pos"]
        probe_support_columns = zero_support_columns

    reference = (
        zero["Ortholog-RAG 46M"].select(zero_support_columns).sort(zero_support_columns)
    )
    for label, frame in {**zero, "phyloP 447 mammals": phylop}.items():
        support = frame.select(zero_support_columns).sort(zero_support_columns)
        assert support.equals(reference), (
            f"{dataset}: zero-shot support mismatch for {label}"
        )

    probe_reference = (
        probes["Ortholog-RAG 46M"]
        .select(probe_support_columns)
        .sort(probe_support_columns)
    )
    for label, frame in probes.items():
        support = frame.select(probe_support_columns).sort(probe_support_columns)
        assert support.equals(probe_reference), (
            f"{dataset}: probe support mismatch for {label}"
        )

    if dataset == "complex_traits":
        probe_leaves = probe_reference.filter(~pl.col("subset").str.starts_with("_"))
        zero_leaves = (
            reference.filter(pl.col("subset").is_in(probe_leaves["subset"].implode()))
            .rename({"n_groups": "n_pos", "n_rows": "n"})
            .sort(probe_support_columns)
        )
        assert probe_leaves.equals(zero_leaves), (
            "complex_traits: zero-shot/probe eligible-row support mismatch"
        )


def _one(frame: pl.DataFrame, **filters: str) -> dict[str, object]:
    selected = frame
    for column, value in filters.items():
        selected = selected.filter(pl.col(column) == value)
    assert selected.height == 1, (filters, selected)
    return selected.row(0, named=True)


def _value(frame: pl.DataFrame, **filters: str) -> str:
    row = _one(frame, **filters)
    return f"{float(row['value']):.6f} ± {float(row['se']):.6f}"


def _comparison_frames(
    results_root: str, conservation_root: str, dataset: str
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame], pl.DataFrame]:
    zero = {
        label: _read_model_metrics(results_root, dataset, model, probe=False)
        for label, model in MODELS.items()
    }
    probes = {
        label: _read_model_metrics(results_root, dataset, model, probe=True)
        for label, model in MODELS.items()
    }
    phylop = _read_phylop_metrics(conservation_root, dataset)
    _assert_metric_support(dataset, zero, probes, phylop)
    return zero, probes, phylop


def _complex_markdown(
    zero: dict[str, pl.DataFrame],
    probes: dict[str, pl.DataFrame],
    phylop: pl.DataFrame,
) -> list[str]:
    lines = [
        "### Complex Traits — exact-row official train comparison",
        "",
        "| Method | Zero-shot macro AUPRC | Frozen-probe macro AUPRC |",
        "|:--|--:|--:|",
    ]
    for label in MODELS:
        lines.append(
            f"| {label} | {_value(zero[label], subset='_macro_avg_')} | "
            f"{_value(probes[label], subset='_macro_avg_')} |"
        )
    lines.append(
        f"| phyloP 447 mammals | {_value(phylop, subset='_macro_avg_')} | N/A |"
    )
    lines.extend(
        [
            "",
            "<details>",
            "<summary>Complex Traits per-subset AUPRC</summary>",
            "",
            "| Subset | 46M zero | 46M probe | 104M zero | 104M probe | m5.1 zero | m5.1 probe | phyloP 447m |",
            "|:--|--:|--:|--:|--:|--:|--:|--:|",
        ]
    )
    for subset, display in COMPLEX_SUBSET_LABELS.items():
        lines.append(
            f"| {display} | {_value(zero['Ortholog-RAG 46M'], subset=subset)} | "
            f"{_value(probes['Ortholog-RAG 46M'], subset=subset)} | "
            f"{_value(zero['Ortholog-RAG 104M'], subset=subset)} | "
            f"{_value(probes['Ortholog-RAG 104M'], subset=subset)} | "
            f"{_value(zero['MarinDNA m5.1'], subset=subset)} | "
            f"{_value(probes['MarinDNA m5.1'], subset=subset)} | "
            f"{_value(phylop, subset=subset)} |"
        )
    lines.extend(["", "</details>"])
    return lines


def _sge_markdown(
    zero: dict[str, pl.DataFrame],
    probes: dict[str, pl.DataFrame],
    phylop: pl.DataFrame,
) -> list[str]:
    headline = {"subset": "_macro_avg_", "accession": "_macro_avg_"}
    lines = [
        "### SGE — exact-row official train comparison",
        "",
        "| Method | Zero-shot subset × accession macro AUPRC | Frozen-probe macro AUPRC |",
        "|:--|--:|--:|",
    ]
    for label in MODELS:
        lines.append(
            f"| {label} | {_value(zero[label], **headline)} | "
            f"{_value(probes[label], **headline)} |"
        )
    lines.append(f"| phyloP 447 mammals | {_value(phylop, **headline)} | N/A |")
    lines.extend(
        [
            "",
            "<details>",
            "<summary>SGE per-subset macro AUPRC over accessions</summary>",
            "",
            "| Subset | 46M zero | 46M probe | 104M zero | 104M probe | m5.1 zero | m5.1 probe | phyloP 447m |",
            "|:--|--:|--:|--:|--:|--:|--:|--:|",
        ]
    )
    for subset, display in (
        ("missense_variant", "Missense"),
        ("splicing", "Splicing"),
        ("_macro_avg_", "Macro"),
    ):
        filt = {"subset": subset, "accession": "_macro_avg_"}
        lines.append(
            f"| {display} | {_value(zero['Ortholog-RAG 46M'], **filt)} | "
            f"{_value(probes['Ortholog-RAG 46M'], **filt)} | "
            f"{_value(zero['Ortholog-RAG 104M'], **filt)} | "
            f"{_value(probes['Ortholog-RAG 104M'], **filt)} | "
            f"{_value(zero['MarinDNA m5.1'], **filt)} | "
            f"{_value(probes['MarinDNA m5.1'], **filt)} | "
            f"{_value(phylop, **filt)} |"
        )
    lines.extend(["", "</details>"])
    return lines


def summarize(results_root: str, conservation_root: str) -> str:
    audits = {
        dataset: _assert_exact_variant_sets(results_root, conservation_root, dataset)
        for dataset in EXPECTED_ROWS
    }
    complex_zero, complex_probes, complex_phylop = _comparison_frames(
        results_root, conservation_root, "complex_traits"
    )
    sge_zero, sge_probes, sge_phylop = _comparison_frames(
        results_root, conservation_root, "sge"
    )
    lines = [
        *_complex_markdown(complex_zero, complex_probes, complex_phylop),
        "",
        *_sge_markdown(sge_zero, sge_probes, sge_phylop),
        "",
        (
            "Exact-row audit: all four methods have identical variant and metric-membership "
            f"fields on {audits['complex_traits']['rows']:,} Complex and "
            f"{audits['sge']['rows']:,} SGE train variants. phyloP has "
            f"{audits['complex_traits']['phylop_missing']} missing Complex scores and "
            f"{audits['sge']['phylop_missing']} missing SGE scores; the official conservation "
            "pipeline retains those rows and fills missing scores with 0 before metrics."
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    markdown = summarize(args.results_root, args.conservation_root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
