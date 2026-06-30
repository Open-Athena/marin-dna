"""Accessibility-QTL leaderboard assembly for the dashboard (issues #311/#312).

The **formatter half** of the supervised accessibility-QTL benchmark: the model /
dataset registries and the tidy-row assembly the dashboard's Accessibility QTL page
consumes. It reads the per-model metric frames already computed and **frozen on S3**
(``s3://oa-bolinas/qtl_benchmark/metrics/{model}/{dataset}.parquet``) and reshapes them
into the page's long-form table; it computes no metrics itself.

The benchmark *scorer* (``compute_qtl_split_metrics`` + the ChromBPNet-coupled
``compute_supervised_qtl_metrics``) that produced those frozen frames was removed with
the ChromBPNet island in #332; only this model-agnostic formatter remains.
"""

from __future__ import annotations

import polars as pl

# Dashboard benchmark model registry (#312): maps each scored model's key (its
# ``metrics/{model}/…`` S3 prefix) to a display name + a coarse group. The group
# separates the external supervised baselines from our (future) fine-tuned gLMs (#243)
# — the comparison the page exists to make — and drives the row swatch. A model
# discovered on S3 without an entry here falls back to its raw key + ``"other"``.
QTL_BENCHMARK_MODELS: dict[str, dict[str, str]] = {
    "alphagenome": {"display": "AlphaGenome", "group": "supervised"},
    "chrombpnet": {"display": "ChromBPNet", "group": "supervised"},
    "enformer": {"display": "Enformer", "group": "supervised"},
}

# The two accessibility-QTL assays the dashboard's "macro" scope averages over.
QTL_BENCHMARK_DATASETS: tuple[str, ...] = ("caqtl", "dsqtl")

# Metric / count columns surfaced on the page (AUROC/Spearman stay in the per-model
# parquets but aren't shown).
_DASH_METRIC_COLS: tuple[str, ...] = (
    "causality_auPRC",
    "causality_se",
    "direction_pearson",
    "direction_pearson_se",
)
_DASH_COUNT_COLS: tuple[str, ...] = ("n_rows", "n_pos")


def macro_average_metrics(
    metrics: pl.DataFrame, datasets: tuple[str, ...] = QTL_BENCHMARK_DATASETS
) -> pl.DataFrame:
    """Equal-weight macro-average of each model's per-assay metrics → one row per (model, split).

    The dashboard's "macro" scope (#312): for each ``(model, split)`` the metric value is
    the unweighted mean over ``datasets`` and the SE is ``sqrt(Σ se_i²) / k`` — the SE of
    a mean of ``k`` *independent* estimates, which these are (caQTL and dsQTL are disjoint
    variant sets; contrast the conservative independent-*sum* used for matched-pair model
    deltas). ``n_rows`` / ``n_pos`` are summed (pooled counts, for the tooltip). Asserts
    every ``(model, split)`` group carries exactly ``datasets`` so a missing assay fails
    loud rather than silently averaging one. Returns the standard per-(model, split) metric
    columns plus ``dataset="macro"``.
    """
    k = len(datasets)
    assert k >= 1, "need at least one dataset to average"
    sub = metrics.filter(pl.col("dataset").is_in(datasets))
    # One pass: aggregate the metrics and the per-group dataset count together, then
    # assert completeness before returning (an incomplete group's wrong mean never escapes).
    macro = sub.group_by("model", "split").agg(
        pl.col("dataset").n_unique().alias("_nd"),
        pl.col("causality_auPRC").mean().alias("causality_auPRC"),
        (pl.col("causality_se").pow(2).sum().sqrt() / k).alias("causality_se"),
        pl.col("direction_pearson").mean().alias("direction_pearson"),
        (pl.col("direction_pearson_se").pow(2).sum().sqrt() / k).alias(
            "direction_pearson_se"
        ),
        pl.col("n_rows").sum().alias("n_rows"),
        pl.col("n_pos").sum().alias("n_pos"),
    )
    incomplete = macro.filter(pl.col("_nd") != k)
    assert incomplete.height == 0, (
        f"macro-average needs all of {sorted(datasets)} per (model, split); incomplete "
        f"groups: {incomplete.select('model', 'split').rows()}"
    )
    return macro.drop("_nd").with_columns(pl.lit("macro").alias("dataset"))


def assemble_benchmark_rows(metrics: pl.DataFrame) -> pl.DataFrame:
    """Tidy long-form rows for the Accessibility QTL dashboard page (#312).

    ``metrics`` is the concatenation of the per-model ``metrics/{model}/{dataset}.parquet``
    frames (computed offline; the benchmark scorer was removed in #332), already filtered
    to the split(s) the page shows (the dashboard passes ``split == "train"`` only).
    Appends a ``macro`` scope (:func:`macro_average_metrics`) to the per-assay rows,
    attaches each model's display name + group from :data:`QTL_BENCHMARK_MODELS` (unknown
    models fall back to their key + ``"other"`` — opaque passthrough, so a newly-dropped
    model still appears), and returns ``[model, display, group, scope, split, *metric cols,
    n_rows, n_pos]`` with ``scope ∈ {caqtl, dsqtl, macro}``.
    """
    needed = ("dataset", "model", "split", *_DASH_METRIC_COLS, *_DASH_COUNT_COLS)
    for col in needed:
        assert col in metrics.columns, f"metrics frame missing column {col!r}"
    per_assay = metrics.select(needed)
    macro = macro_average_metrics(metrics).select(needed)
    combined = pl.concat([per_assay, macro], how="vertical").rename(
        {"dataset": "scope"}
    )

    registry = pl.DataFrame(
        {
            "model": list(QTL_BENCHMARK_MODELS),
            "display": [m["display"] for m in QTL_BENCHMARK_MODELS.values()],
            "group": [m["group"] for m in QTL_BENCHMARK_MODELS.values()],
        }
    )
    combined = combined.join(registry, on="model", how="left").with_columns(
        pl.col("display").fill_null(pl.col("model")),
        pl.col("group").fill_null(pl.lit("other")),
    )
    return combined.select(
        "model",
        "display",
        "group",
        "scope",
        "split",
        *_DASH_METRIC_COLS,
        *_DASH_COUNT_COLS,
    )
