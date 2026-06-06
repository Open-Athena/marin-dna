"""Tests for ``marin_dna.pipelines.evals.leaderboard``.

The library's surface is `fetch_method_metrics` (one method × protocol →
parquet rows) and `normalized_rows` (one dataset → flat polars DataFrame
the dashboard data loader writes to parquet). Most tests bypass the
cached S3 reader by monkeypatching `_read_parquet` to return synthetic
DataFrames keyed by parquet path.
"""

from __future__ import annotations

import polars as pl
import pytest

from marin_dna.pipelines.evals import leaderboard
from marin_dna.pipelines.evals.leaderboard import (
    DEFAULT_PROTOCOL,
    PROTOCOLS,
    fetch_method_metrics,
    normalized_rows,
    score_type_for,
)
from marin_dna.pipelines.evals.models import Model
from marin_dna.pipelines.evals.metrics import GLOBAL_SUBSET, MACRO_AVG_SUBSET


def _mk_method(
    id: str = "x",
    display: str = "x",
    family: str = "marin_dna",
    description: str = "desc",
    datasets: tuple[str, ...] = ("mendelian_traits",),
    **extra,
) -> Model:
    return Model(
        id=id,
        display=display,
        family=family,  # type: ignore[arg-type]
        description=description,
        datasets=datasets,
        checkpoint=extra.get("checkpoint"),
    )


def _patch_read_parquet(
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[str, pl.DataFrame],
) -> None:
    """Monkeypatch ``_read_parquet`` to return synthetic data keyed by path.

    Flushes the existing lru_cache so cached real-S3 results from prior
    tests don't leak through.
    """
    leaderboard._read_parquet.cache_clear()

    def fake(path: str) -> pl.DataFrame:
        if path not in responses:
            raise FileNotFoundError(f"no synthetic response for {path!r}")
        return responses[path]

    monkeypatch.setattr(leaderboard, "_read_parquet", fake)


def _patch_methods(
    monkeypatch: pytest.MonkeyPatch,
    methods: tuple[Model, ...],
) -> None:
    """Bypass the real models.yaml so tests operate on a small fixture."""
    monkeypatch.setattr(
        "marin_dna.pipelines.evals.models.load_models",
        lambda: methods,
    )


# ---- Protocols --------------------------------------------------------------


def test_default_protocol_keys_match_protocols():
    assert set(DEFAULT_PROTOCOL) == set(PROTOCOLS)
    for fam, default in DEFAULT_PROTOCOL.items():
        assert default in PROTOCOLS[fam], (
            f"family {fam!r} default {default!r} not in PROTOCOLS[{fam!r}]"
        )


def test_score_type_for_returns_dataset_specific_column():
    # MarinDNA family migrated to per-strand atoms + derived AVG under
    # the AUPRC pipeline; default LLR/JSD pick the _avg variants. The
    # per-strand _fwd variants are surfaced as LLR-FWD / JSD-FWD on the
    # dashboard's Protocols pages (not in the leaderboards' protocol
    # toggle — see PROTOCOL_OPTIONS in dashboard/src/components/controls.js).
    assert score_type_for("marin_dna", "LLR", "mendelian_traits") == "minus_llr_avg"
    assert score_type_for("marin_dna", "LLR", "complex_traits") == "abs_llr_avg"
    assert score_type_for("marin_dna", "JSD", "mendelian_traits") == "jsd_avg"
    assert score_type_for("marin_dna", "LLR-FWD", "mendelian_traits") == "minus_llr_fwd"
    assert score_type_for("marin_dna", "LLR-FWD", "complex_traits") == "abs_llr_fwd"
    assert score_type_for("marin_dna", "JSD-FWD", "mendelian_traits") == "jsd_fwd"
    assert score_type_for("evo2", "LLR-FWD", "mendelian_traits") == "minus_llr_fwd"
    assert score_type_for("evo2", "JSD-FWD", "mendelian_traits") == "jsd_fwd"
    assert (
        score_type_for("gpn_star", "cLLR", "mendelian_traits") == "minus_llr_calibrated"
    )
    assert score_type_for("gpn_star", "LLR", "mendelian_traits") == "minus_llr"


def test_gpn_star_parquet_path_resolves_to_s3():
    """GPN-Star metrics now come from the S3 pipeline output (snakemake/
    gpn_star_eval), not the gist — same source pattern as the other S3 families."""
    from marin_dna.pipelines.evals.leaderboard import _parquet_path

    method = _mk_method(
        id="GPN-Star-M",
        display="GPN-Star (M)",
        family="gpn_star",
        description="mammal",
        datasets=("mendelian_traits", "complex_traits"),
    )
    mendelian = _parquet_path(method, "mendelian_traits")
    complex_ = _parquet_path(method, "complex_traits")
    assert mendelian == (
        "s3://oa-bolinas/snakemake/gpn_star_eval/results/metrics/mendelian_traits.parquet"
    )
    assert complex_ == (
        "s3://oa-bolinas/snakemake/gpn_star_eval/results/metrics/complex_traits.parquet"
    )


def test_fetch_method_metrics_unknown_protocol_raises(monkeypatch: pytest.MonkeyPatch):
    methods = (
        _mk_method(
            id="exp55-mammals-step-16999",
            display="exp55-mammals",
            family="marin_dna",
            description="promoters, mammals",
            datasets=("mendelian_traits",),
            checkpoint=None,
        ),
    )
    _patch_methods(monkeypatch, methods)
    _patch_read_parquet(monkeypatch, {})
    with pytest.raises(AssertionError, match="unknown protocol"):
        fetch_method_metrics(methods[0], "mendelian_traits", protocol="not_a_protocol")


# ---- normalized_rows --------------------------------------------------------


def test_normalized_rows_emits_one_block_per_protocol(monkeypatch: pytest.MonkeyPatch):
    """gpn_star has cLLR + LLR protocols; both must appear in normalized_rows.

    Source is the S3 pipeline output (``snakemake/gpn_star_eval``). The dashboard
    sees a single `n` column derived from `n_rows` (per-subset / global) or
    `n_groups` (macro_avg).
    """
    methods = (
        _mk_method(
            id="GPN-Star-M",
            display="GPN-Star-M",
            family="gpn_star",
            description="mammal",
            datasets=("mendelian_traits",),
        ),
    )
    _patch_methods(monkeypatch, methods)

    # No `split` column — the gpn_star S3 parquet is train-only and the read
    # path filters by model + score_type, not split.
    def gpn_rows(score_type, value):
        return [
            {
                "score_type": score_type,
                "model": "GPN-Star-M",
                "subset": "missense_variant",
                "value": value,
                "se": 0.02,
                "n_groups": 100,
                "n_rows": 1000,
            },
            {
                "score_type": score_type,
                "model": "GPN-Star-M",
                "subset": GLOBAL_SUBSET,
                "value": value,
                "se": 0.02,
                "n_groups": 100,
                "n_rows": 1000,
            },
            {
                "score_type": score_type,
                "model": "GPN-Star-M",
                "subset": MACRO_AVG_SUBSET,
                "value": value,
                "se": 0.02,
                "n_groups": 1,
                "n_rows": 1000,
            },
        ]

    gpn_df = pl.DataFrame(
        gpn_rows("minus_llr_calibrated", 0.85) + gpn_rows("minus_llr", 0.80)
    )
    _patch_read_parquet(
        monkeypatch,
        {
            "s3://oa-bolinas/snakemake/gpn_star_eval/results/metrics/"
            "mendelian_traits.parquet": gpn_df
        },
    )
    df = normalized_rows("mendelian_traits")
    assert set(df["protocol"].unique().to_list()) == {"cLLR", "LLR"}
    by_protocol = df.group_by("protocol").agg(pl.len())
    counts = dict(by_protocol.iter_rows())
    assert counts["cLLR"] == counts["LLR"]


def test_normalized_rows_skips_missing_protocol_gracefully(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """When a parquet doesn't have a protocol's score_type rows yet (e.g.
    marin_dna JSD before the pipeline rerun), normalized_rows logs + skips."""
    methods = (
        _mk_method(
            id="exp55-mammals-step-16999",
            display="exp55-mammals",
            family="marin_dna",
            description="promoters, mammals",
            datasets=("mendelian_traits",),
        ),
    )
    _patch_methods(monkeypatch, methods)
    marin_dna_df = pl.DataFrame(
        [
            {
                "score_type": "minus_llr_avg",
                "split": "train",
                "subset": "missense_variant",
                "value": 0.75,
                "se": 0.02,
                "n_groups": 100,
                "n_rows": 1000,
            },
            {
                "score_type": "minus_llr_avg",
                "split": "train",
                "subset": GLOBAL_SUBSET,
                "value": 0.74,
                "se": 0.02,
                "n_groups": 100,
                "n_rows": 1000,
            },
            {
                "score_type": "minus_llr_avg",
                "split": "train",
                "subset": MACRO_AVG_SUBSET,
                "value": 0.75,
                "se": 0.02,
                "n_groups": 1,
                "n_rows": 1000,
            },
        ]
    )
    _patch_read_parquet(
        monkeypatch,
        {
            "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/"
            "exp55-mammals-step-16999/mendelian_traits.parquet": marin_dna_df,
        },
    )
    df = normalized_rows("mendelian_traits")
    assert df["protocol"].unique().to_list() == ["LLR"]
    captured = capsys.readouterr()
    assert "marin_dna/JSD skip" in captured.err
    # `n` = n_rows for per-subset/global, K for macro. `n_positives` =
    # n_groups everywhere. So with these inputs we expect {1000, 1} for n
    # and {100, 1} for n_positives.
    marin_dna_rows = df.filter(pl.col("family") == "marin_dna")
    assert set(marin_dna_rows["n"].to_list()) == {1, 1000}
    assert set(marin_dna_rows["n_positives"].to_list()) == {1, 100}
    macro = marin_dna_rows.filter(pl.col("subset") == MACRO_AVG_SUBSET)
    assert macro["n"][0] == 1, f"macro `n` should carry K, got {macro['n'][0]}"
    assert macro["n_positives"][0] == 1, (
        f"macro `n_positives` should also carry K (= n_groups from source), "
        f"got {macro['n_positives'][0]}"
    )


def test_normalized_rows_propagates_unexpected_exceptions(
    monkeypatch: pytest.MonkeyPatch,
):
    """The soft-fail in `normalized_rows` is intentionally narrow: only
    `LookupError` / `ComputeError` / `FileNotFoundError` (the "protocol not
    yet in parquet" cases) are swallowed. Everything else — config bugs,
    runtime errors, programmer mistakes — must propagate so a broken
    registry doesn't silently yield an empty dashboard."""
    methods = (
        _mk_method(
            id="exp55-mammals-step-16999",
            display="exp55-mammals",
            family="marin_dna",
            description="promoters, mammals",
            datasets=("mendelian_traits",),
        ),
    )
    _patch_methods(monkeypatch, methods)

    def boom(*_a, **_k):
        raise RuntimeError("simulated config bug")

    monkeypatch.setattr(leaderboard, "fetch_method_metrics", boom)
    with pytest.raises(RuntimeError, match="simulated config bug"):
        normalized_rows("mendelian_traits")


def test_normalized_rows_includes_aggregates_and_per_subset(
    monkeypatch: pytest.MonkeyPatch,
):
    methods = (
        _mk_method(
            id="phyloP_241m",
            display="phyloP_241m",
            family="conservation",
            description="",
            datasets=("mendelian_traits",),
        ),
    )
    _patch_methods(monkeypatch, methods)
    # Conservation migrated to AUPRC schema (PR #196) — n_groups / n_rows.
    cons_df = pl.DataFrame(
        [
            {
                "score_name": "phyloP_241m",
                "subset": "missense_variant",
                "value": 0.75,
                "se": 0.02,
                "n_groups": 100,
                "n_rows": 1000,
            },
            {
                "score_name": "phyloP_241m",
                "subset": "splicing",
                "value": 0.65,
                "se": 0.05,
                "n_groups": 40,
                "n_rows": 400,
            },
            {
                "score_name": "phyloP_241m",
                "subset": GLOBAL_SUBSET,
                "value": 0.72,
                "se": 0.018,
                "n_groups": 140,
                "n_rows": 1400,
            },
            {
                "score_name": "phyloP_241m",
                "subset": MACRO_AVG_SUBSET,
                "value": 0.70,
                "se": 0.027,
                "n_groups": 2,
                "n_rows": 1400,
            },
        ]
    )
    _patch_read_parquet(
        monkeypatch,
        {
            "s3://oa-bolinas/snakemake/conservation_eval/results/"
            "mendelian_traits/metrics_train.parquet": cons_df,
        },
    )
    df = normalized_rows("mendelian_traits")
    assert set(df["subset"].to_list()) == {
        "missense_variant",
        "splicing",
        GLOBAL_SUBSET,
        MACRO_AVG_SUBSET,
    }
    assert df["method_id"].unique().to_list() == ["phyloP_241m"]


# ---- QTL (caqtl / dsqtl) ----------------------------------------------------
#
# QTL parquets have a different schema from the matched-pair eval: keyed by
# `metric` (AUPRC / pearson / spearman) × score_type, with NO `subset` and NO
# `n_groups`. `fetch_method_metrics` must branch on `QTL_DATASETS`, overload
# `subset` to carry the metric name, and NOT reach the matched-pair shaping
# (which would `ColumnNotFoundError` on the absent `subset`/`n_groups`).


def _qtl_marin_df() -> pl.DataFrame:
    """marin_dna QTL metrics parquet (caqtl): 3 metrics × 4 referenced
    score_types. n_rows = all variants for AUPRC, positives only for the
    correlations (mirrors the real schema)."""

    def rows(score_type: str) -> list[dict]:
        return [
            {
                "metric": "AUPRC",
                "score_type": score_type,
                "value": 0.089,
                "se": 0.0026,
                "n_rows": 41382,
                "n_pos": 3173,
                "model": "exp136",
                "dataset": "caqtl",
                "split": "train",
            },
            {
                "metric": "pearson",
                "score_type": score_type,
                "value": 0.013,
                "se": 0.0154,
                "n_rows": 3173,
                "n_pos": 3173,
                "model": "exp136",
                "dataset": "caqtl",
                "split": "train",
            },
            {
                "metric": "spearman",
                "score_type": score_type,
                "value": 0.009,
                "se": 0.0177,
                "n_rows": 3173,
                "n_pos": 3173,
                "model": "exp136",
                "dataset": "caqtl",
                "split": "train",
            },
        ]

    return pl.DataFrame(
        rows("abs_llr_avg") + rows("jsd_avg") + rows("abs_llr_fwd") + rows("jsd_fwd")
    )


def test_score_type_for_qtl_datasets():
    # QTL `score_protocol` is abs_llr (unsigned magnitude — no direction), so
    # LLR picks abs_llr_avg, not minus_llr_avg. conservation / alphagenome
    # auto-extend via their `{d: ... for d in ALL_DATASETS}` comprehensions.
    assert score_type_for("marin_dna", "LLR", "caqtl") == "abs_llr_avg"
    assert score_type_for("marin_dna", "JSD", "dsqtl") == "jsd_avg"
    assert score_type_for("marin_dna", "LLR-FWD", "caqtl") == "abs_llr_fwd"
    assert score_type_for("marin_dna", "JSD-FWD", "dsqtl") == "jsd_fwd"
    assert score_type_for("conservation", "score", "caqtl") == "score"
    assert score_type_for("conservation", "score", "dsqtl") == "score"
    assert score_type_for("alphagenome", "L2", "caqtl") == "alphagenome_max_l2"
    assert score_type_for("alphagenome", "L2", "dsqtl") == "alphagenome_max_l2"


def test_fetch_method_metrics_qtl_marin(monkeypatch: pytest.MonkeyPatch):
    method = _mk_method(id="exp136", family="marin_dna", datasets=("caqtl",))
    _patch_methods(monkeypatch, (method,))
    _patch_read_parquet(
        monkeypatch,
        {
            "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/"
            "exp136/caqtl.parquet": _qtl_marin_df(),
        },
    )
    df = fetch_method_metrics(method, "caqtl", protocol="LLR")  # → abs_llr_avg
    assert set(df["subset"].to_list()) == {"AUPRC", "pearson", "spearman"}
    # AUPRC uses all variants; the correlations use positives only.
    auprc = df.filter(pl.col("subset") == "AUPRC")
    assert auprc["n"][0] == 41382
    assert auprc["n_positives"][0] == 3173
    pear = df.filter(pl.col("subset") == "pearson")
    assert pear["n"][0] == 3173
    assert abs(pear["value"][0] - 0.013) < 1e-9
    # Guard: the source schema genuinely lacks the matched-pair columns, so a
    # regression that drops the QTL branch would crash here (not silently pass).
    assert "n_groups" not in _qtl_marin_df().columns
    assert "subset" not in _qtl_marin_df().columns


def test_normalized_rows_qtl_marin_emits_metric_rows_per_protocol(
    monkeypatch: pytest.MonkeyPatch,
):
    method = _mk_method(id="exp136", family="marin_dna", datasets=("caqtl",))
    _patch_methods(monkeypatch, (method,))
    _patch_read_parquet(
        monkeypatch,
        {
            "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/"
            "exp136/caqtl.parquet": _qtl_marin_df(),
        },
    )
    df = normalized_rows("caqtl")
    assert set(df["subset"].unique().to_list()) == {"AUPRC", "pearson", "spearman"}
    # 4 marin_dna protocols (LLR / JSD / LLR-FWD / JSD-FWD) × 3 metrics. The
    # dashboard later filters to LLR/JSD via PROTOCOL_OPTIONS.
    assert df.height == 12
    for proto in df["protocol"].unique().to_list():
        assert df.filter(pl.col("protocol") == proto).height == 3


def test_fetch_method_metrics_qtl_conservation_filters_by_score_name(
    monkeypatch: pytest.MonkeyPatch,
):
    # The QTL conservation parquet carries every track (incl. phyloP_241m,
    # which is NOT on the leaderboard); the `score_name == method.id` filter
    # keeps only the registered track's 3 metric rows.
    method = _mk_method(id="phyloP_447m", family="conservation", datasets=("caqtl",))
    _patch_methods(monkeypatch, (method,))

    def cons_rows(score_name: str, auprc: float) -> list[dict]:
        return [
            {
                "metric": "AUPRC",
                "score_type": "score",
                "value": auprc,
                "se": 0.0019,
                "n_rows": 41382,
                "n_pos": 3173,
                "score_name": score_name,
                "n_nan": 0,
                "n_total": 41382,
                "split": "train",
                "dataset": "caqtl",
            },
            {
                "metric": "pearson",
                "score_type": "score",
                "value": 0.02,
                "se": 0.016,
                "n_rows": 3173,
                "n_pos": 3173,
                "score_name": score_name,
                "n_nan": 0,
                "n_total": 41382,
                "split": "train",
                "dataset": "caqtl",
            },
            {
                "metric": "spearman",
                "score_type": "score",
                "value": 0.03,
                "se": 0.017,
                "n_rows": 3173,
                "n_pos": 3173,
                "score_name": score_name,
                "n_nan": 0,
                "n_total": 41382,
                "split": "train",
                "dataset": "caqtl",
            },
        ]

    cons = pl.DataFrame(
        cons_rows("phyloP_447m", 0.079) + cons_rows("phyloP_241m", 0.076)
    )
    _patch_read_parquet(
        monkeypatch,
        {
            "s3://oa-bolinas/snakemake/conservation_eval/results/"
            "caqtl/metrics_train.parquet": cons,
        },
    )
    df = fetch_method_metrics(method, "caqtl")
    assert df.height == 3  # only phyloP_447m; phyloP_241m dropped
    assert set(df["subset"].to_list()) == {"AUPRC", "pearson", "spearman"}
    assert df.filter(pl.col("subset") == "AUPRC")["value"][0] == 0.079


def test_fetch_method_metrics_qtl_alphagenome(monkeypatch: pytest.MonkeyPatch):
    method = _mk_method(id="AlphaGenome", family="alphagenome", datasets=("caqtl",))
    _patch_methods(monkeypatch, (method,))
    ag = pl.DataFrame(
        [
            {
                "metric": "AUPRC",
                "score_type": "alphagenome_max_l2",
                "value": 0.282,
                "se": 0.0082,
                "n_rows": 41382,
                "n_pos": 3173,
                "dataset": "caqtl",
                "split": "train",
            },
            {
                "metric": "pearson",
                "score_type": "alphagenome_max_l2",
                "value": 0.270,
                "se": 0.0199,
                "n_rows": 3173,
                "n_pos": 3173,
                "dataset": "caqtl",
                "split": "train",
            },
            {
                "metric": "spearman",
                "score_type": "alphagenome_max_l2",
                "value": 0.220,
                "se": 0.0168,
                "n_rows": 3173,
                "n_pos": 3173,
                "dataset": "caqtl",
                "split": "train",
            },
        ]
    )
    _patch_read_parquet(
        monkeypatch,
        {
            "s3://oa-bolinas/snakemake/alphagenome_eval/results/metrics/"
            "caqtl.parquet": ag,
        },
    )
    df = fetch_method_metrics(method, "caqtl")
    assert set(df["subset"].to_list()) == {"AUPRC", "pearson", "spearman"}
    assert df.filter(pl.col("subset") == "AUPRC")["value"][0] == 0.282


# ---- BOLINAS_S3_ANON env toggle --------------------------------------------


def test_storage_options_off_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BOLINAS_S3_ANON", raising=False)
    from marin_dna.pipelines.evals.leaderboard import _storage_options

    assert _storage_options() is None


def test_storage_options_anonymous_when_env_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BOLINAS_S3_ANON", "1")
    from marin_dna.pipelines.evals.leaderboard import _storage_options

    opts = _storage_options()
    assert opts is not None
    assert opts["aws_skip_signature"] == "true"
    assert opts["aws_region"] == "us-east-2"


def test_storage_options_anonymous_accepts_true(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BOLINAS_S3_ANON", "true")
    from marin_dna.pipelines.evals.leaderboard import _storage_options

    assert _storage_options() is not None


def test_storage_options_ignores_other_values(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BOLINAS_S3_ANON", "no")
    from marin_dna.pipelines.evals.leaderboard import _storage_options

    assert _storage_options() is None
