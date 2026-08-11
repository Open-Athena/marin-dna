#!/usr/bin/env python3
"""Summarize issue #402's official train-split Mendelian comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

RESULTS_ROOT = "s3://oa-bolinas/snakemake/analysis/evals_v2/results"
CONSERVATION_METRICS = (
    "s3://oa-bolinas/snakemake/conservation_eval/results/"
    "mendelian_traits/metrics_train.parquet"
)
DATASET = "mendelian_traits"
SCORE_TYPE = "minus_llr_avg"
PHYLOP_SCORE = "phyloP_447m"
M51_MODEL = "mix-v0.9-p1B-i24-exp135-m5.1-step-59158"
RAG_MODELS = {
    "Ortholog-RAG 46M (30k)": "exp402-rag-h640-p46m-b2m-step-29999",
    "Ortholog-RAG 104M (30k)": "exp402-rag-h768-p104m-b2m-step-29999",
}
EXPECTED_SUBSETS = {
    "3_prime_UTR_variant",
    "5_prime_UTR_variant",
    "distal",
    "missense_variant",
    "non_coding_transcript_exon_variant",
    "splicing",
    "synonymous_variant",
    "tss_proximal",
}
SUBSET_LABELS = {
    "3_prime_UTR_variant": "3′ UTR",
    "5_prime_UTR_variant": "5′ UTR",
    "distal": "Distal",
    "missense_variant": "Missense",
    "non_coding_transcript_exon_variant": "Non-coding exon",
    "splicing": "Splicing",
    "synonymous_variant": "Synonymous",
    "tss_proximal": "TSS proximal",
}
SUBSET_ORDER = list(SUBSET_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--conservation-metrics", default=CONSERVATION_METRICS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read_standard(results_root: str, model: str) -> pl.DataFrame:
    path = f"{results_root.rstrip('/')}/metrics/{model}/{DATASET}.parquet"
    metrics = pl.read_parquet(path)
    assert metrics.select(pl.col("split").unique()).to_series().to_list() == ["train"]
    assert metrics.select(pl.col("dataset").unique()).to_series().to_list() == [DATASET]
    selected = metrics.filter(pl.col("score_type") == SCORE_TYPE)
    assert selected.filter(pl.col("subset") == "_macro_avg_").height == 1
    eligible = selected.filter(
        ~pl.col("subset").str.starts_with("_") & (pl.col("n_groups") >= 30)
    )
    assert set(eligible["subset"]) == EXPECTED_SUBSETS
    assert "mature_miRNA_variant" not in eligible["subset"]
    return selected


def _read_probe(results_root: str, model: str) -> pl.DataFrame:
    path = f"{results_root.rstrip('/')}/probe_metrics/{model}/{DATASET}.parquet"
    metrics = pl.read_parquet(path)
    assert metrics.select(pl.col("split").unique()).to_series().to_list() == ["train"]
    assert metrics.select(pl.col("dataset").unique()).to_series().to_list() == [DATASET]
    selected = metrics.filter(pl.col("score_type") == "probe_score")
    assert selected.filter(pl.col("subset") == "_macro_avg_").height == 1
    selected = selected.filter(
        pl.col("subset").str.starts_with("_") | (pl.col("n_pos") >= 30)
    )
    assert (
        set(selected.filter(~pl.col("subset").str.starts_with("_"))["subset"])
        == EXPECTED_SUBSETS
    )
    return selected


def _read_phylop(path: str) -> pl.DataFrame:
    metrics = pl.read_parquet(path)
    assert metrics.select(pl.col("split").unique()).to_series().to_list() == ["train"]
    assert metrics.select(pl.col("dataset").unique()).to_series().to_list() == [DATASET]
    selected = metrics.filter(
        (pl.col("score_type") == "score") & (pl.col("score_name") == PHYLOP_SCORE)
    )
    assert selected.filter(pl.col("subset") == "_macro_avg_").height == 1
    eligible = selected.filter(
        ~pl.col("subset").str.starts_with("_") & (pl.col("n_groups") >= 30)
    )
    assert set(eligible["subset"]) == EXPECTED_SUBSETS
    return selected


def _row(frame: pl.DataFrame, subset: str) -> dict[str, object]:
    selected = frame.filter(pl.col("subset") == subset)
    assert selected.height == 1, (subset, selected)
    return selected.row(0, named=True)


def _value(frame: pl.DataFrame, subset: str) -> str:
    row = _row(frame, subset)
    return f"{float(row['value']):.6f} ± {float(row['se']):.6f}"


def _assert_exact_support(
    standard: dict[str, pl.DataFrame],
    probes: dict[str, pl.DataFrame],
    phylop: pl.DataFrame,
) -> None:
    support_columns = ["subset", "n_groups", "n_rows"]
    reference = (
        standard["MarinDNA exp135-1B-m5.1"].select(support_columns).sort("subset")
    )
    for label, metrics in standard.items():
        support = metrics.select(support_columns).sort("subset")
        assert support.equals(reference), label

    eligible_reference = reference.filter(pl.col("subset").is_in(EXPECTED_SUBSETS))
    for label, metrics in probes.items():
        support = (
            metrics.filter(pl.col("subset").is_in(EXPECTED_SUBSETS))
            .select("subset", "n_pos", "n")
            .rename({"n_pos": "n_groups", "n": "n_rows"})
            .sort("subset")
        )
        assert support.equals(eligible_reference), label

    phylop_support = (
        phylop.select("subset", "n_groups", "n_total")
        .rename({"n_total": "n_rows"})
        .sort("subset")
    )
    assert phylop_support.equals(reference)
    assert _row(phylop, "_macro_avg_")["n_nan"] == 8


def summarize(results_root: str, conservation_metrics: str) -> str:
    standard = {
        **{
            label: _read_standard(results_root, model)
            for label, model in RAG_MODELS.items()
        },
        "MarinDNA exp135-1B-m5.1": _read_standard(results_root, M51_MODEL),
    }
    probes = {
        **{
            label: _read_probe(results_root, model)
            for label, model in RAG_MODELS.items()
        },
        "MarinDNA exp135-1B-m5.1": _read_probe(results_root, M51_MODEL),
    }
    phylop = _read_phylop(conservation_metrics)
    _assert_exact_support(standard, probes, phylop)

    rag_candidates: list[tuple[float, str, str]] = []
    for label in RAG_MODELS:
        zero_shot = float(_row(standard[label], "_macro_avg_")["value"])
        probe = float(_row(probes[label], "_macro_avg_")["value"])
        rag_candidates.extend(
            [(zero_shot, label, "zero-shot"), (probe, label, "probe")]
        )
    best_value, best_rag, best_protocol = max(rag_candidates)

    lines = [
        f"Best ortholog-RAG result: **{best_value:.6f}** ({best_rag}, {best_protocol}).",
        "",
        "| Model | Official zero-shot macro AUPRC | Official frozen-probe macro AUPRC |",
        "|:--|--:|--:|",
        f"| {best_rag} | {_value(standard[best_rag], '_macro_avg_')} | {_value(probes[best_rag], '_macro_avg_')} |",
        f"| MarinDNA exp135-1B-m5.1 | {_value(standard['MarinDNA exp135-1B-m5.1'], '_macro_avg_')} | {_value(probes['MarinDNA exp135-1B-m5.1'], '_macro_avg_')} |",
        f"| phyloP 447 mammals | {_value(phylop, '_macro_avg_')} | N/A |",
        "",
        "<details>",
        "<summary>Official per-subset comparison on the same train variants</summary>",
        "",
        "| Subset | RAG zero-shot | RAG probe | m5.1 zero-shot | m5.1 probe | phyloP 447m |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for subset in SUBSET_ORDER:
        lines.append(
            f"| {SUBSET_LABELS[subset]} | {_value(standard[best_rag], subset)} | "
            f"{_value(probes[best_rag], subset)} | "
            f"{_value(standard['MarinDNA exp135-1B-m5.1'], subset)} | "
            f"{_value(probes['MarinDNA exp135-1B-m5.1'], subset)} | {_value(phylop, subset)} |"
        )
    lines.extend(
        [
            "",
            "</details>",
            "",
            (
                "All rows use the official `train` split and metric functions. The macro "
                "averages include exactly the eight subsets with at least 30 positive groups; "
                "`mature_miRNA_variant` is excluded by that gate. phyloP was scored on the "
                "same variant rows and is missing 8/16,100 eligible-row scores (reported as "
                "coverage, not silently replaced)."
            ),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    markdown = summarize(args.results_root, args.conservation_metrics)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
