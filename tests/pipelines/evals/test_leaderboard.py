"""Tests for ``bolinas.pipelines.evals.leaderboard``.

The library's surface is `fetch_method_metrics` (one method × protocol →
parquet rows) and `normalized_rows` (one dataset → flat polars DataFrame
the dashboard data loader writes to parquet). Most tests bypass the
cached S3 reader by monkeypatching `_read_parquet` to return synthetic
DataFrames keyed by parquet path.
"""

from __future__ import annotations

import polars as pl
import pytest

from bolinas.pipelines.evals import leaderboard
from bolinas.pipelines.evals.leaderboard import (
    DEFAULT_PROTOCOL,
    PROTOCOLS,
    fetch_method_metrics,
    normalized_rows,
    score_type_for,
)
from bolinas.pipelines.evals.models import Model
from bolinas.pipelines.evals.metrics import GLOBAL_SUBSET, MACRO_AVG_SUBSET


def _mk_method(
    id: str = "x",
    display: str = "x",
    family: str = "bolinas",
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
        "bolinas.pipelines.evals.models.load_models",
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
    # Bolinas family migrated to per-strand atoms + derived AVG under
    # the AUPRC pipeline; default LLR/JSD pick the _avg variants.
    assert score_type_for("bolinas", "LLR", "mendelian_traits") == "minus_llr_avg"
    assert score_type_for("bolinas", "LLR", "complex_traits") == "abs_llr_avg"
    assert score_type_for("bolinas", "JSD", "mendelian_traits") == "jsd_avg"
    assert (
        score_type_for("gpn_star", "cLLR", "mendelian_traits") == "minus_llr_calibrated"
    )
    assert score_type_for("gpn_star", "LLR", "mendelian_traits") == "minus_llr"


def test_gpn_star_parquet_path_resolves_to_pinned_gist():
    """The gist URL has the pinned commit + the dataset-stacked filename."""
    from bolinas.pipelines.evals.leaderboard import (
        GPN_STAR_METRICS_GIST_BASE,
        GPN_STAR_METRICS_GIST_COMMIT,
        _parquet_path,
    )

    method = _mk_method(
        id="GPN-Star-M",
        display="GPN-Star (M)",
        family="gpn_star",
        description="mammal",
        datasets=("mendelian_traits", "complex_traits"),
    )
    mendelian = _parquet_path(method, "mendelian_traits")
    complex_ = _parquet_path(method, "complex_traits")
    assert mendelian.startswith(GPN_STAR_METRICS_GIST_BASE), mendelian
    assert GPN_STAR_METRICS_GIST_COMMIT in mendelian
    assert mendelian.endswith("/mendelian_traits.GPN-Star.parquet")
    assert complex_.endswith("/complex_traits.GPN-Star.parquet")


def test_fetch_method_metrics_unknown_protocol_raises(monkeypatch: pytest.MonkeyPatch):
    methods = (
        _mk_method(
            id="exp55-mammals-step-16999",
            display="exp55-mammals",
            family="bolinas",
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

    Source is the AUPRC-schema gist (issue #145 last comment). The dashboard
    sees a single `n` column derived from `n_rows` (per-subset / global) or
    `n_groups` (macro_avg).
    """
    from bolinas.pipelines.evals.leaderboard import GPN_STAR_METRICS_GIST_BASE

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

    def gpn_rows(score_type, value):
        return [
            {
                "score_type": score_type,
                "split": "train",
                "model": "GPN-Star-M",
                "subset": "missense_variant",
                "value": value,
                "se": 0.02,
                "n_groups": 100,
                "n_rows": 1000,
            },
            {
                "score_type": score_type,
                "split": "train",
                "model": "GPN-Star-M",
                "subset": GLOBAL_SUBSET,
                "value": value,
                "se": 0.02,
                "n_groups": 100,
                "n_rows": 1000,
            },
            {
                "score_type": score_type,
                "split": "train",
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
        {f"{GPN_STAR_METRICS_GIST_BASE}/mendelian_traits.GPN-Star.parquet": gpn_df},
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
    bolinas JSD before the pipeline rerun), normalized_rows logs + skips."""
    methods = (
        _mk_method(
            id="exp55-mammals-step-16999",
            display="exp55-mammals",
            family="bolinas",
            description="promoters, mammals",
            datasets=("mendelian_traits",),
        ),
    )
    _patch_methods(monkeypatch, methods)
    bolinas_df = pl.DataFrame(
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
            "exp55-mammals-step-16999/mendelian_traits.parquet": bolinas_df,
        },
    )
    df = normalized_rows("mendelian_traits")
    assert df["protocol"].unique().to_list() == ["LLR"]
    captured = capsys.readouterr()
    assert "bolinas/JSD skip" in captured.err
    # `n` = n_rows for per-subset/global, K for macro. `n_positives` =
    # n_groups everywhere. So with these inputs we expect {1000, 1} for n
    # and {100, 1} for n_positives.
    bolinas_rows = df.filter(pl.col("family") == "bolinas")
    assert set(bolinas_rows["n"].to_list()) == {1, 1000}
    assert set(bolinas_rows["n_positives"].to_list()) == {1, 100}
    macro = bolinas_rows.filter(pl.col("subset") == MACRO_AVG_SUBSET)
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
            family="bolinas",
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


# ---- BOLINAS_S3_ANON env toggle --------------------------------------------


def test_storage_options_off_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BOLINAS_S3_ANON", raising=False)
    from bolinas.pipelines.evals.leaderboard import _storage_options

    assert _storage_options() is None


def test_storage_options_anonymous_when_env_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BOLINAS_S3_ANON", "1")
    from bolinas.pipelines.evals.leaderboard import _storage_options

    opts = _storage_options()
    assert opts is not None
    assert opts["aws_skip_signature"] == "true"
    assert opts["aws_region"] == "us-east-2"


def test_storage_options_anonymous_accepts_true(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BOLINAS_S3_ANON", "true")
    from bolinas.pipelines.evals.leaderboard import _storage_options

    assert _storage_options() is not None


def test_storage_options_ignores_other_values(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BOLINAS_S3_ANON", "no")
    from bolinas.pipelines.evals.leaderboard import _storage_options

    assert _storage_options() is None
