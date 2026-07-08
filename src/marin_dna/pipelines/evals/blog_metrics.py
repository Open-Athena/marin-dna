"""Direct-S3 evals_v2 metrics reader, keyed by a raw run-name/step model id.

The blog figure family (#361) plots AUPRC for scaling-ladder and mixture-lineage
*intermediate* checkpoints. Those checkpoints are scored by the evals_v2 pipeline
(parquets on S3) but are **not** registered in ``dashboard/models.yaml`` — so the
``Model``-keyed readers in ``leaderboard.py`` (``fetch_method_metrics`` /
``*_normalized_rows``) cannot reach them. This module mirrors ``leaderboard.py``'s
S3 path templates and tidy transform, but keyed by the raw evals_v2 model id (the
``{id}`` path segment), so a figure recipe can pull metrics for any
``(checkpoint, dataset, world)``.

Worlds (columns of the blog figure matrix, per checkpoint):
  * ``LLR``   zero-shot   → ``results/metrics/{id}/{dataset}.parquet``
  * ``probe`` supervised  → ``results/probe_metrics/{id}/{dataset}.parquet``
  * SGE re-uses the ``metrics/{id}/sge.parquet`` grid (add here when the ladder/
    lineage SGE sweep lands — #364).

Kept deliberately separate from ``leaderboard.py``: the small path-template
duplication is intentional (the dashboard's registered-model path stays
untouched), while the ``score_type`` map is *reused* via ``score_type_for`` so the
two never drift. Output schema matches the leaderboard tidy shape:
``[model_id, subset, value, se, n, n_positives]`` (one row per subset plus the
``_macro_avg_`` aggregate; on the macro row ``n`` / ``n_positives`` carry
K = qualifying subsets, mirroring ``fetch_method_metrics``).
"""

from __future__ import annotations

import functools
import os

import polars as pl

from marin_dna.pipelines.evals.leaderboard import score_type_for
from marin_dna.pipelines.evals.metrics import MACRO_AVG_SUBSET

S3 = "s3://oa-bolinas"
SPLIT = "train"
_RESULTS = f"{S3}/snakemake/analysis/evals_v2/results"

_TIDY_COLUMNS = ["model_id", "subset", "value", "se", "n", "n_positives"]


def _storage_options() -> dict[str, str] | None:
    """Anonymous S3 reads via ``BOLINAS_S3_ANON=1`` (mirrors ``leaderboard``)."""
    if os.environ.get("BOLINAS_S3_ANON") in ("1", "true"):
        return {"aws_skip_signature": "true", "aws_region": "us-east-2"}
    return None


@functools.lru_cache(maxsize=None)
def _read_parquet(path: str) -> pl.DataFrame:
    """Cached S3 read (one fetch per (path, process); tests monkeypatch this)."""
    return pl.read_parquet(path, storage_options=_storage_options())


def llr_metrics_path(model_id: str, dataset: str) -> str:
    """S3 path to one checkpoint's zero-shot metrics parquet."""
    return f"{_RESULTS}/metrics/{model_id}/{dataset}.parquet"


def probe_metrics_path(model_id: str, dataset: str) -> str:
    """S3 path to one checkpoint's frozen-embedding probe-metrics parquet."""
    return f"{_RESULTS}/probe_metrics/{model_id}/{dataset}.parquet"


def read_llr_metrics(
    model_id: str,
    dataset: str,
    *,
    protocol: str = "LLR",
    split: str = SPLIT,
) -> pl.DataFrame:
    """Tidy zero-shot AUPRC rows for one checkpoint id (unregistered checkpoints).

    ``score_type`` is resolved from ``leaderboard.score_type_for("marin_dna",
    protocol, dataset)`` — ``minus_llr_avg`` for mendelian_traits, ``abs_llr_avg``
    for complex_traits. Raises ``LookupError`` if the parquet has no rows for that
    ``(score_type, split)`` (e.g. the checkpoint hasn't been scored with this
    protocol yet), matching ``fetch_method_metrics``' posture.
    """
    score_type = score_type_for("marin_dna", protocol, dataset)
    path = llr_metrics_path(model_id, dataset)
    df = _read_parquet(path).filter(
        (pl.col("score_type") == score_type) & (pl.col("split") == split)
    )
    if df.height == 0:
        raise LookupError(
            f"no {protocol}/{score_type} rows for {model_id!r} on {dataset!r} "
            f"(split={split!r}) in {path}"
        )
    return df.with_columns(
        pl.when(pl.col("subset") == MACRO_AVG_SUBSET)
        .then(pl.col("n_groups"))
        .otherwise(pl.col("n_rows"))
        .alias("n"),
        pl.col("n_groups").alias("n_positives"),
        pl.lit(model_id).alias("model_id"),
    ).select(_TIDY_COLUMNS)


def read_probe_metrics(
    model_id: str,
    dataset: str,
    *,
    split: str = SPLIT,
) -> pl.DataFrame:
    """Tidy frozen-embedding linear-probe AUPRC rows for one checkpoint id.

    Keeps the ``probe_score`` rows on ``split`` (the ``minus_llr_avg`` baseline
    that the probe parquet also carries is dropped). On the ``_macro_avg_`` row,
    ``n`` / ``n_positives`` are overloaded to K = qualifying subsets (finite,
    non-macro), mirroring ``probe_normalized_rows``; per-subset ``n`` /
    ``n_positives`` are the parquet's ``n`` / ``n_pos``. Below-gate subsets (null
    probe value) are preserved as NaN rows for the caller to drop.
    """
    path = probe_metrics_path(model_id, dataset)
    df = _read_parquet(path).filter(
        (pl.col("score_type") == "probe_score") & (pl.col("split") == split)
    )
    if df.height == 0:
        raise LookupError(
            f"no probe_score rows for {model_id!r} on {dataset!r} "
            f"(split={split!r}) in {path}"
        )
    k = df.filter(
        (pl.col("subset") != MACRO_AVG_SUBSET) & pl.col("value").is_not_null()
    ).height
    # `se` is absent in the leaner per-subset-only probe schema (the #347 chrom-cluster
    # SE + `_macro_avg_` aggregate isn't emitted by every probe run — e.g. the blog
    # ladder/lineage cells). Fill NaN to keep the tidy schema stable; when there's no
    # `_macro_avg_` row the `when` branches below never fire, so per-subset `n` /
    # `n_positives` pass through unchanged.
    se_expr = pl.col("se") if "se" in df.columns else pl.lit(float("nan"))
    return df.with_columns(
        se_expr.alias("se"),
        pl.when(pl.col("subset") == MACRO_AVG_SUBSET)
        .then(k)
        .otherwise(pl.col("n"))
        .alias("n"),
        pl.when(pl.col("subset") == MACRO_AVG_SUBSET)
        .then(k)
        .otherwise(pl.col("n_pos"))
        .alias("n_positives"),
        pl.lit(model_id).alias("model_id"),
    ).select(_TIDY_COLUMNS)
