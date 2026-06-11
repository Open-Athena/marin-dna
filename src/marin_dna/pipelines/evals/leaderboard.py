"""S3/gist → tidy long-form parquet for the dashboard data loader.

Reads per-(method, dataset) metrics parquets emitted by the eval snakemake
pipelines (or pinned gist commits, for families without an S3 pipeline),
filters by protocol / score-type, and emits one row per
``(method, protocol, subset)`` for the dashboard. All families now emit
AUPRC + cluster-bootstrap SE on 1:9 matched negatives.

  - ``snakemake/analysis/evals_v2/``  → one parquet per ``(model, dataset)``,
    filter by ``score_type`` + ``split``.
  - ``snakemake/conservation_eval/``  → one parquet per ``(dataset, split)``,
    filter by ``score_name`` (the track).
  - ``snakemake/alphagenome_eval/``   → one parquet per dataset, filter by
    ``score_type`` + ``split``.
  - ``snakemake/gpn_star_eval/``      → one parquet per dataset with V/M/P
    stacked, filter by ``score_type`` + ``model``.

Model registry (display name, family, training metadata, etc.) lives in
``dashboard/models.yaml`` and is loaded via ``models.load_models``.
"""

from __future__ import annotations

import functools
import os
import sys

import polars as pl

from marin_dna.pipelines.evals.metrics import MACRO_AVG_SUBSET
from marin_dna.pipelines.evals.models import ALL_DATASETS, Model, models_for_dataset

S3 = "s3://oa-bolinas"
SPLIT = "train"

# `family: gpn_star` AUPRC metrics now come from the S3 pipeline
# (`snakemake/gpn_star_eval`, refreshed in #278), same as the conservation /
# alphagenome / marin_dna families — one parquet per dataset with V/M/P stacked
# and a `model` column to filter on. The pipeline is the single source of truth;
# the old metrics gist (`3649e68f@cba23a7`) is kept only as the #145 provenance
# record, no longer read here. (Distinct from `gpn_star.GPN_STAR_GIST_BASE`,
# the per-variant *prediction* gist, which the pipeline still consumes as input.)

# `family: evo2` AUPRC metrics gist. Same gist as gpn_star, different
# pinned commit. Bump `EVO2_METRICS_GIST_COMMIT` when re-uploading; see
# `scripts/evo2_eval/README.md` for the upload recipe.
EVO2_METRICS_GIST_OWNER = "gonzalobenegas"
EVO2_METRICS_GIST_ID = "3649e68fb63ca1f3443e4486078eb4d8"
EVO2_METRICS_GIST_COMMIT = "1bce02fe0d831382d24ecbac305d401f153c65fc"
EVO2_METRICS_GIST_BASE = (
    f"https://gist.githubusercontent.com/{EVO2_METRICS_GIST_OWNER}/"
    f"{EVO2_METRICS_GIST_ID}/raw/{EVO2_METRICS_GIST_COMMIT}"
)
# Dataset → metric-parquet filename prefix. Extend when adding complex_traits.
EVO2_DATASET_SHORT: dict[str, str] = {
    "mendelian_traits": "mendelian",
}

# Per-family scoring protocols. Each protocol maps a dataset → the parquet
# `score_type` column to filter on. The dashboard exposes the non-default
# protocols (where present) as per-family toggle options.
#
# All protocols here are expected to be in the precomputed metrics parquet
# on S3 / gist. Adding a new protocol means: (1) extend `metrics.smk` to
# compute AUPRC for that score column, (2) re-run `compute_metrics`,
# (3) add the protocol entry here.
PROTOCOLS: dict[str, dict[str, dict[str, str]]] = {
    "marin_dna": {
        # Default LLR / JSD pick the FWD+RC `_avg` columns. The per-strand
        # `_fwd` variants are surfaced on the dashboard's Protocols page
        # for AVG-vs-FWD exploration; they're not exposed in the
        # leaderboards' protocol toggle (see `PROTOCOL_OPTIONS` in
        # `dashboard/src/components/controls.js`). The `_rc` columns are
        # still in the parquet for diagnostics but aren't surfaced.
        "LLR": {
            "mendelian_traits": "minus_llr_avg",
            "complex_traits": "abs_llr_avg",
        },
        "JSD": {
            "mendelian_traits": "jsd_avg",
            "complex_traits": "jsd_avg",
        },
        "LLR-FWD": {
            "mendelian_traits": "minus_llr_fwd",
            "complex_traits": "abs_llr_fwd",
        },
        "JSD-FWD": {
            "mendelian_traits": "jsd_fwd",
            "complex_traits": "jsd_fwd",
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
        },
        "LLR": {
            "mendelian_traits": "minus_llr",
            "complex_traits": "abs_llr",
        },
    },
    # evo2 mirrors marin_dna; mendelian_traits only (extend when complex lands).
    "evo2": {
        "LLR": {
            "mendelian_traits": "minus_llr_avg",
        },
        "JSD": {
            "mendelian_traits": "jsd_avg",
        },
        "LLR-FWD": {
            "mendelian_traits": "minus_llr_fwd",
        },
        "JSD-FWD": {
            "mendelian_traits": "jsd_fwd",
        },
    },
}

DEFAULT_PROTOCOL: dict[str, str] = {
    "marin_dna": "LLR",
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
    """Cached S3/gist read so families that share a per-dataset parquet
    (conservation, alphagenome, gpn_star) only fetch once per process."""
    return pl.read_parquet(path, storage_options=_storage_options())


def _parquet_path(method: Model, dataset: str) -> str:
    match method.family:
        case "marin_dna":
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
            return f"{EVO2_METRICS_GIST_BASE}/{short}_{method.id}_train_metrics.parquet"
        case _:
            raise ValueError(f"unknown family {method.family!r}")


def fetch_method_metrics(
    method: Model, dataset: str, protocol: str | None = None
) -> pl.DataFrame:
    """Return rows ``[subset, value, se, n, n_positives]`` for one
    ``(method, dataset, protocol)``.

    Matched-pair datasets (mendelian_traits, complex_traits) emit one row per
    consequence ``subset`` plus the ``_global_`` / ``_macro_avg_`` aggregates;
    ``subset`` carries the consequence name and ``n`` is total variants (or K
    on the macro row). See ``normalized_rows`` for the column semantics.

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
        case "marin_dna" | "alphagenome" | "evo2":
            df = df.filter(pl.col("score_type") == score_type).filter(
                pl.col("split") == SPLIT
            )
        case "conservation":
            df = df.filter(pl.col("score_name") == method.id)
        case "gpn_star":
            df = df.filter(pl.col("score_type") == score_type).filter(
                pl.col("model") == method.id
            )
        case _:
            raise ValueError(f"unknown family {method.family!r}")
    if df.height == 0:
        raise LookupError(
            f"no metrics rows for {method.id!r} on {dataset!r} with protocol "
            f"{protocol!r} (score_type={score_type!r}) in {path}. The pipeline "
            f"may need to be re-run with this protocol included."
        )
    df = df.with_columns(
        pl.when(pl.col("subset") == MACRO_AVG_SUBSET)
        .then(pl.col("n_groups"))
        .otherwise(pl.col("n_rows"))
        .alias("n"),
        pl.col("n_groups").alias("n_positives"),
    )
    return df.select(["subset", "value", "se", "n", "n_positives"])


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
      - ``subset``          — consequence subset OR ``_global_`` /
        ``_macro_avg_`` (matched-pair).
      - ``value``           — AUPRC.
      - ``se``              — cluster-bootstrap SE.
      - ``n``               — total variants in the subset (positives +
        matched negatives), except on the ``_macro_avg_`` row where it
        carries K = number of qualifying subsets.
      - ``n_positives``     — positives in the subset (= ``n_groups`` from
        the source AUPRC parquet); K on the macro row. Drives the dashboard's
        ≥30-positives display threshold; never rendered.
    """
    # Soft-fail surface, intentionally narrow: only the two legitimate
    # "no data for this protocol yet" exception types.
    #   * `LookupError` — `fetch_method_metrics` raises this when the
    #     parquet exists but has no rows for the requested protocol's
    #     `score_type` (e.g. marin_dna JSD before `metrics.smk` was rerun
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
                        "n": int(row["n"]),
                        "n_positives": int(row["n_positives"]),
                    }
                )
    return pl.DataFrame(rows)


# Columns kept from each method's SGE metrics parquet, in output order.
_SGE_ROW_COLUMNS = (
    "metric",
    "subset",
    "accession",
    "gene",
    "score_type",
    "value",
    "se",
    "n",
    "n_pos",
)


def _nan_float(x: object) -> float:
    """Map a polars null back to NaN. A pandas float-NaN written to parquet
    round-trips as a *null*, so Spearman rows (``n_pos`` = NaN, by design) and
    any degenerate-bootstrap ``se`` come back as ``None``; keep them as NaN
    rather than crashing on ``float(None)``."""
    return float("nan") if x is None else float(x)


def sge_normalized_rows(dataset: str = "sge") -> pl.DataFrame:
    """Long-form SGE metrics across all registered methods (the dedicated SGE
    leaderboard loader; ``dashboard/src/data/sge.parquet.py`` writes its output).

    SGE keeps the full per-accession × per-subset × per-metric grid from
    ``compute_sge_metrics`` — too rich for ``normalized_rows``' single
    ``[subset, value, se, …]`` row shape — so this emits one row per
    ``(method, score_type, metric, subset, accession)`` and lets the dashboard
    page do the gene-scope / subset / metric / protocol selection.

    Reuses ``_parquet_path``: marin_dna methods have a per-method parquet (every
    ``score_type`` — ``minus_llr_*`` / ``jsd_*`` — kept, so the page's LLR/JSD
    toggle works); conservation methods share one per-dataset parquet, filtered
    to the track by ``score_name == method.id`` (``score_type`` is always
    ``"score"`` there). Methods whose parquet isn't on S3 yet are skipped with a
    warning (same soft-fail posture as ``normalized_rows``).

    Columns: ``[method_id, method_display, family, score_type, metric, subset,
    accession, gene, value, se, n, n_pos]``. ``n_pos`` is the abnormal count for
    AUPRC rows and NaN for Spearman rows.
    """
    # A method registered for SGE may not have its metrics parquet on S3 yet
    # (e.g. the gLMs before their scoring run is done) — polars surfaces an S3
    # 404 as OSError (FileNotFoundError covers the local-path case). Skip + warn
    # rather than fail the whole loader; the parquet appears once the run lands.
    soft_fail = (LookupError, pl.exceptions.ComputeError, FileNotFoundError, OSError)
    rows: list[dict] = []
    for method in models_for_dataset(dataset):
        path = _parquet_path(method, dataset)
        try:
            df = _read_parquet(path)
        except soft_fail as exc:
            print(f"  ! sge skip for {method.id} ({path}): {exc}", file=sys.stderr)
            continue
        if method.family == "conservation":
            # conservation's parquet path is split-specific (…/metrics_{SPLIT}.parquet),
            # so it only needs the per-track score_name select, no split filter.
            df = df.filter(pl.col("score_name") == method.id)
        elif method.family == "gpn_star":
            # All three GPN-Star variants (V/M/P) share ONE per-dataset parquet
            # (`_parquet_path` is model-independent for this family), stacked with a
            # `model` column. Filter to this method's model — without it each of the
            # 3 registered methods would re-emit all 3 models' rows (9× duplication,
            # mislabeled). Train-only parquet, so no split filter (mirrors
            # fetch_method_metrics' gpn_star branch). `model` == `Model.id`.
            df = df.filter(pl.col("model") == method.id)
        elif method.family == "marin_dna":
            # The per-model parquet path is NOT split-specific and carries a
            # `split` column for whichever split was scored — guard it like
            # fetch_method_metrics so a future test-split run can't leak in.
            df = df.filter(pl.col("split") == SPLIT)
        missing = [c for c in _SGE_ROW_COLUMNS if c not in df.columns]
        assert not missing, (
            f"SGE parquet for {method.id!r} missing columns {missing}; got {df.columns}"
        )
        if df.height == 0:
            print(f"  ! sge no rows for {method.id} in {path}", file=sys.stderr)
            continue
        for row in df.iter_rows(named=True):
            rows.append(
                {
                    "method_id": method.id,
                    "method_display": method.display,
                    "family": method.family,
                    "score_type": row["score_type"],
                    "metric": row["metric"],
                    "subset": row["subset"],
                    "accession": row["accession"],
                    "gene": row["gene"],
                    "value": _nan_float(row["value"]),
                    "se": _nan_float(row["se"]),
                    "n": int(row["n"]),
                    "n_pos": _nan_float(row["n_pos"]),
                }
            )
    return pl.DataFrame(rows)
