"""Observable Framework data loader: official caQTL/dsQTL benchmark metrics → one parquet.

Reads the per-model metric frames written by the #311 benchmark
(``s3://oa-bolinas/qtl_benchmark/metrics/{model}/{caqtl,dsqtl}.parquet``), keeps the
``train`` (odd-chroms) dev split — even-chrom ``test`` is held out for final eval, matching
the other leaderboards — and assembles the tidy long-form rows (adding the macro-across-
assays scope) via ``marin_dna_evals.qtl_scoring.assemble_benchmark_rows``. Feeds
the Accessibility QTL page (issue #312).

Models come from the ``QTL_BENCHMARK_MODELS`` registry, **not** S3 enumeration: the
dashboard CI role has ``GetObject`` but no ``ListBucket`` (same constraint as
``nuc_dep.zip.py``). So adding a model is one registry line + its S3 parquet. A registry
model whose parquet isn't on S3 yet is skipped with a warning; we assert ≥1 model loaded so
an IAM/path misconfiguration fails the build loudly instead of shipping an empty page.
"""

from __future__ import annotations

import io
import os
import sys
from typing import Any

import boto3
import polars as pl
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from marin_dna_evals.qtl_scoring import (
    QTL_BENCHMARK_DATASETS,
    QTL_BENCHMARK_MODELS,
    assemble_benchmark_rows,
)

BUCKET = "oa-bolinas"
PREFIX = "qtl_benchmark/metrics"
SPLIT = "train"  # odd chroms — the dev split; even-chrom test held out for final eval

# Codes meaning "not a readable present key" → not built yet, skip. Without ListBucket,
# S3 answers a missing key with 403 (not 404), so accept both (mirrors nuc_dep.zip.py).
_SKIP_CODES = {"NoSuchKey", "404", "403", "AccessDenied"}


def _s3_client() -> Any:
    """S3 client on the bucket's region; ``BOLINAS_S3_ANON=1`` → unsigned reads
    (mirrors ``leaderboard._storage_options`` / ``nuc_dep.zip.py``)."""
    if os.environ.get("BOLINAS_S3_ANON") in ("1", "true"):
        return boto3.client(
            "s3", region_name="us-east-2", config=Config(signature_version=UNSIGNED)
        )
    return boto3.client("s3", region_name="us-east-2")


def _read_model(s3: Any, model: str) -> pl.DataFrame | None:
    """Read + concat a model's per-assay metric parquets, or None if any is missing."""
    parts: list[pl.DataFrame] = []
    for ds in QTL_BENCHMARK_DATASETS:
        key = f"{PREFIX}/{model}/{ds}.parquet"
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in _SKIP_CODES:
                print(
                    f"  ! accessibility_qtl skip (not on S3): {key} [{code}]",
                    file=sys.stderr,
                )
                return None
            raise
        parts.append(pl.read_parquet(io.BytesIO(obj["Body"].read())))
    return pl.concat(parts, how="vertical")


def main() -> None:
    s3 = _s3_client()
    frames = [
        df
        for model in QTL_BENCHMARK_MODELS
        if (df := _read_model(s3, model)) is not None
    ]
    assert frames, (
        f"loaded 0 benchmark models from s3://{BUCKET}/{PREFIX}/ — expected ≥1. Check the "
        f"QTL_BENCHMARK_MODELS registry, the S3 parquets, and the dashboard IAM read perms."
    )
    metrics = pl.concat(frames, how="vertical").filter(pl.col("split") == SPLIT)
    rows = assemble_benchmark_rows(metrics)
    print(
        f"[accessibility_qtl] {len(frames)} model(s) → {rows.height} rows "
        f"({rows['scope'].n_unique()} scopes)",
        file=sys.stderr,
    )
    rows.write_parquet(sys.stdout.buffer)


if __name__ == "__main__":
    main()
