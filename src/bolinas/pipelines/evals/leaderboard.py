"""S3 → tidy long-form parquet for the dashboard data loader.

Reads per-(method, dataset) metrics parquets emitted by the eval snakemake
pipelines, filters by protocol / score-type, and emits one row per
``(method, protocol, subset)`` for the dashboard. The bolinas and
conservation families emit AUPRC + cluster-bootstrap SE under the AUPRC
migration (PRs #194/#195 for bolinas; the conservation_eval mirror for
the seven phyloP / phastCons tracks); alphagenome, gpn_star, and evo2
still emit PairwiseAccuracy + Wald-binomial SE.

  - ``snakemake/analysis/evals_v2/``  → one parquet per ``(model, dataset)``,
    filter by ``score_type`` + ``split``.
  - ``snakemake/conservation_eval/``  → one parquet per ``(dataset, split)``,
    filter by ``score_name`` (the track).
  - ``snakemake/alphagenome_eval/``   → one parquet per dataset, filter by
    ``score_type`` + ``split``.
  - ``snakemake/gpn_star_eval/``      → one parquet per dataset, filter by
    ``score_type`` + ``split`` + ``model``.

Model registry (display name, family, training metadata, etc.) lives in
``dashboard/models.yaml`` and is loaded via ``models.load_models``.
"""

from __future__ import annotations

import functools
import os
import sys

import polars as pl

from bolinas.pipelines.evals.models import ALL_DATASETS, Model, models_for_dataset

S3 = "s3://oa-bolinas"
SPLIT = "train"

# `family: evo2` metrics aren't in S3 — they live on a gist commit (pinned
# below). Polars `read_parquet` handles https URLs natively via fsspec, so
# the path resolver just hands back a stable URL string. Bump GIST_COMMIT
# when re-uploading new Evo2 parquets to the same gist.
EVO2_GIST_OWNER = "gonzalobenegas"
EVO2_GIST_ID = "3649e68fb63ca1f3443e4486078eb4d8"
EVO2_GIST_COMMIT = "362efe18b5d7f6e436e405db18932e8d149d3d07"
EVO2_GIST_BASE = (
    f"https://gist.githubusercontent.com/{EVO2_GIST_OWNER}/{EVO2_GIST_ID}/raw/"
    f"{EVO2_GIST_COMMIT}"
)
# Filename pattern uses short dataset names matching the file we uploaded.
EVO2_DATASET_SHORT: dict[str, str] = {
    "mendelian_traits": "mendelian",
    "complex_traits": "complex",
}

# Per-family scoring protocols. Each protocol maps a dataset → the parquet
# `score_type` column to filter on. The dashboard exposes the non-default
# protocols (where present) as per-family toggle options.
#
# All protocols here are expected to be in the precomputed metrics parquet
# on S3. Adding a new protocol means: (1) extend `metrics.smk` to compute
# pairwise PA for that score column, (2) re-run `compute_metrics`, (3) add
# the protocol entry here.
PROTOCOLS: dict[str, dict[str, dict[str, str]]] = {
    "bolinas": {
        # Default LLR / JSD pick the FWD+RC `_avg` columns. The per-strand
        # `_fwd` / `_rc` columns are also in the parquet for diagnostics
        # but aren't exposed as separate protocols here.
        "LLR": {
            "mendelian_traits": "minus_llr_avg",
            "complex_traits": "abs_llr_avg",
        },
        "JSD": {
            "mendelian_traits": "jsd_avg",
            "complex_traits": "jsd_avg",
        },
    },
    "conservation": {
        "score": {d: "score" for d in ALL_DATASETS},
    },
    "alphagenome": {
        "L2": {d: "alphagenome_max_l2" for d in ALL_DATASETS},
    },
    "gpn_star": {
        "cLLR": {
            "mendelian_traits": "minus_llr_calibrated",
            "complex_traits": "abs_llr_calibrated",
            "eqtl": "abs_llr_calibrated",
        },
        "LLR": {
            "mendelian_traits": "minus_llr",
            "complex_traits": "abs_llr",
            "eqtl": "abs_llr",
        },
    },
    "evo2": {
        # Evo2 metrics live on a pinned gist (legacy PA schema, pre-AUPRC).
        # When the gist is re-emitted under the AUPRC pipeline, bump the
        # commit SHA above and rename these to `_avg` variants.
        "LLR": {
            "mendelian_traits": "minus_llr",
            "complex_traits": "abs_llr",
        },
        "JSD": {
            "mendelian_traits": "next_token_jsd_mean",
            "complex_traits": "next_token_jsd_mean",
        },
    },
}

DEFAULT_PROTOCOL: dict[str, str] = {
    "bolinas": "LLR",
    "conservation": "score",
    "alphagenome": "L2",
    "gpn_star": "cLLR",
    "evo2": "LLR",
}


def score_type_for(family: str, protocol: str, dataset: str) -> str:
    """Score-column name for one (family, protocol, dataset) combination."""
    return PROTOCOLS[family][protocol][dataset]


def _storage_options() -> dict[str, str] | None:
    """Toggle anonymous S3 reads via ``BOLINAS_S3_ANON=1``.

    Lets a build target a public-read bucket prefix without AWS credentials.
    With anything else (default), polars walks the standard credential
    chain (env vars → ``~/.aws`` → IMDS); the dashboard CI uses GitHub OIDC
    via that chain."""
    if os.environ.get("BOLINAS_S3_ANON") in ("1", "true"):
        return {"aws_skip_signature": "true", "aws_region": "us-east-2"}
    return None


@functools.lru_cache(maxsize=None)
def _read_parquet(path: str) -> pl.DataFrame:
    """Cached S3 read so families that share a per-dataset parquet
    (conservation, alphagenome, gpn_star) only fetch once per process."""
    return pl.read_parquet(path, storage_options=_storage_options())


def _parquet_path(method: Model, dataset: str) -> str:
    match method.family:
        case "bolinas":
            return (
                f"{S3}/snakemake/analysis/evals_v2/results/metrics/"
                f"{method.id}/{dataset}.parquet"
            )
        case "conservation":
            return (
                f"{S3}/snakemake/conservation_eval/results/"
                f"{dataset}/metrics_{SPLIT}.parquet"
            )
        case "alphagenome":
            return f"{S3}/snakemake/alphagenome_eval/results/metrics/{dataset}.parquet"
        case "gpn_star":
            return f"{S3}/snakemake/gpn_star_eval/results/metrics/{dataset}.parquet"
        case "evo2":
            short = EVO2_DATASET_SHORT[dataset]
            return f"{EVO2_GIST_BASE}/{short}_{method.id}_train_metrics.parquet"
        case _:
            raise ValueError(f"unknown family {method.family!r}")


def fetch_method_metrics(
    method: Model, dataset: str, protocol: str | None = None
) -> pl.DataFrame:
    """Return rows ``[subset, value, se, n_pairs, n_ties]`` for one
    ``(method, dataset, protocol)`` — including the ``_global_`` and
    ``_macro_avg_`` aggregate rows.

    When ``protocol`` is ``None``, defaults to ``DEFAULT_PROTOCOL[family]``.
    """
    assert dataset in method.datasets, (
        f"{method.id!r} is not registered for dataset {dataset!r}"
    )
    protocol = protocol or DEFAULT_PROTOCOL[method.family]
    assert protocol in PROTOCOLS[method.family], (
        f"unknown protocol {protocol!r} for family {method.family!r}; "
        f"options: {list(PROTOCOLS[method.family])}"
    )
    score_type = PROTOCOLS[method.family][protocol][dataset]
    path = _parquet_path(method, dataset)
    df = _read_parquet(path)
    match method.family:
        case "bolinas" | "alphagenome" | "evo2":
            df = df.filter(pl.col("score_type") == score_type).filter(
                pl.col("split") == SPLIT
            )
        case "conservation":
            df = df.filter(pl.col("score_name") == method.id)
        case "gpn_star":
            df = (
                df.filter(pl.col("score_type") == score_type)
                .filter(pl.col("split") == SPLIT)
                .filter(pl.col("model") == method.id)
            )
        case _:
            raise ValueError(f"unknown family {method.family!r}")
    if df.height == 0:
        raise LookupError(
            f"no metrics rows for {method.id!r} on {dataset!r} with protocol "
            f"{protocol!r} (score_type={score_type!r}) in {path}. The pipeline "
            f"may need to be re-run with this protocol included."
        )
    # Schema bridge: bolinas + conservation migrated from PairwiseAccuracy
    # (n_pairs/n_ties) to AUPRC (n_groups/n_rows). Map n_groups → n_pairs
    # (semantically the bootstrap unit count for AUPRC, the pair count
    # for PA), and fill n_ties with 0 — AUPRC has no ties. Alphagenome,
    # gpn_star, and evo2 still emit the legacy PA schema.
    if method.family in ("bolinas", "conservation"):
        df = df.rename({"n_groups": "n_pairs"}).with_columns(
            pl.lit(0, dtype=pl.Int64).alias("n_ties")
        )
    return df.select(["subset", "value", "se", "n_pairs", "n_ties"])


def normalized_rows(dataset: str) -> pl.DataFrame:
    """Long-form table of all metrics for one dataset.

    Emits one row per ``(method, protocol, subset)`` — so each method
    contributes one row block per protocol registered for its family.

    Protocols whose metrics aren't in the parquet yet (e.g. the JSD path
    before ``compute_metrics`` has been re-run with the new score column)
    log a warning and are skipped rather than failing the build.

    Columns:
      - ``method_id``       — ``Model.id`` (primary key for a row's method)
      - ``method_display``  — ``Model.display``
      - ``family``          — ``Model.family``
      - ``protocol``        — protocol name (e.g. ``LLR``, ``JSD``, ``cLLR``)
      - ``subset``          — consequence subset OR ``_global_`` / ``_macro_avg_``
      - ``value``           — metric value (AUPRC for bolinas + conservation; PairwiseAccuracy for other families)
      - ``se``              — SE (cluster bootstrap for bolinas + conservation; Wald binomial elsewhere)
      - ``n_pairs``         — bootstrap unit count (match groups for AUPRC, pairs for PA; or K = qualifying subsets for ``_macro_avg_``)
      - ``n_ties``          — tied-pair count (always 0 for AUPRC rows)
    """
    # Soft-fail surface, intentionally narrow: only the two legitimate
    # "no data for this protocol yet" exception types.
    #   * `LookupError` — `fetch_method_metrics` raises this when the
    #     parquet exists but has no rows for the requested protocol's
    #     `score_type` (e.g. bolinas JSD before `metrics.smk` was rerun
    #     with the new score column).
    #   * `pl.exceptions.ComputeError` — `pl.read_parquet` raises this
    #     when the parquet file isn't on S3 yet (e.g. a freshly added
    #     external-family entry that the eval pipeline hasn't produced
    #     output for). `FileNotFoundError` covers the local-path case in
    #     tests.
    # Everything else (`AssertionError`, `KeyError` from a malformed
    # registry, `ValueError`, etc.) propagates so config bugs fail loud
    # instead of yielding a silently-empty dashboard.
    soft_fail = (LookupError, pl.exceptions.ComputeError, FileNotFoundError)
    rows: list[dict] = []
    for method in models_for_dataset(dataset):
        for protocol in PROTOCOLS[method.family]:
            try:
                df = fetch_method_metrics(method, dataset, protocol)
            except soft_fail as exc:
                print(
                    f"  ! {method.family}/{protocol} skip for {method.id} "
                    f"({dataset}): {exc}",
                    file=sys.stderr,
                )
                continue
            for row in df.iter_rows(named=True):
                rows.append(
                    {
                        "method_id": method.id,
                        "method_display": method.display,
                        "family": method.family,
                        "protocol": protocol,
                        "subset": row["subset"],
                        "value": float(row["value"]),
                        "se": float(row["se"]),
                        "n_pairs": int(row["n_pairs"]),
                        "n_ties": int(row["n_ties"]),
                    }
                )
    return pl.DataFrame(rows)
