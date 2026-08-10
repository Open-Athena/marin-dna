"""Tests for ``marin_dna.pipelines.evals.leaderboard``.

The library's surface is `fetch_method_metrics` (one method × protocol →
parquet rows) and `normalized_rows` (one dataset → flat polars DataFrame
the dashboard data loader writes to parquet). Most tests bypass the
cached S3 reader by monkeypatching `_read_parquet` to return synthetic
DataFrames keyed by parquet path.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from marin_dna.pipelines.evals import leaderboard
from marin_dna.pipelines.evals.leaderboard import (
    DEFAULT_PROTOCOL,
    PROTOCOLS,
    fetch_method_metrics,
    normalized_rows,
    probe_normalized_rows,
    score_type_for,
)
from marin_dna.pipelines.evals.metrics import GLOBAL_SUBSET, MACRO_AVG_SUBSET
from marin_dna.pipelines.evals.models import Model


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


# ---- sge_normalized_rows ----------------------------------------------------


def _sge_metrics_parquet(
    score_types, *, score_name=None, split="train", model=None
) -> pl.DataFrame:
    """Tiny SGE metrics parquet: one AUPRC row per score_type at the headline
    (across-accession × across-subset macro) cell. ``model`` stamps the gpn_star
    `model` column (V/M/P); ``score_name`` the conservation track column."""
    rows = [
        {
            "metric": "AUPRC",
            "subset": MACRO_AVG_SUBSET,
            "accession": MACRO_AVG_SUBSET,
            "gene": MACRO_AVG_SUBSET,
            "score_type": st,
            "value": 0.3,
            "se": 0.01,
            "n": 5,
            "n_pos": 100.0,
            # The real evals_v2/conservation metrics parquets carry a `split`
            # column; sge_normalized_rows filters marin_dna rows by it.
            "split": split,
        }
        for st in score_types
    ]
    df = pl.DataFrame(rows)
    if score_name is not None:
        df = df.with_columns(pl.lit(score_name).alias("score_name"))
    if model is not None:
        df = df.with_columns(pl.lit(model).alias("model"))
    return df


def test_sge_normalized_rows_marin_and_conservation(monkeypatch: pytest.MonkeyPatch):
    """marin_dna keeps every score_type (LLR/JSD toggle); conservation methods
    share one parquet, each filtered to its own track by score_name."""
    glm = _mk_method(id="glm1", family="marin_dna", datasets=("sge",))
    cons = _mk_method(id="phyloP_100v", family="conservation", datasets=("sge",))
    cons2 = _mk_method(id="phastCons_43p", family="conservation", datasets=("sge",))
    _patch_methods(monkeypatch, (glm, cons, cons2))

    glm_path = leaderboard._parquet_path(glm, "sge")
    cons_path = leaderboard._parquet_path(cons, "sge")  # shared by both tracks
    cons_df = pl.concat(
        [
            _sge_metrics_parquet(["score"], score_name="phyloP_100v"),
            _sge_metrics_parquet(["score"], score_name="phastCons_43p"),
        ]
    )
    # The marin_dna parquet path is not split-specific, so it can carry both
    # splits; only the train rows should survive.
    glm_df = pl.concat(
        [
            _sge_metrics_parquet(["minus_llr_avg", "jsd_avg"], split="train"),
            _sge_metrics_parquet(["minus_llr_avg", "jsd_avg"], split="test"),
        ]
    )
    _patch_read_parquet(
        monkeypatch,
        {
            glm_path: glm_df,
            cons_path: cons_df,
        },
    )

    df = leaderboard.sge_normalized_rows("sge")
    assert set(df.columns) == {
        "method_id",
        "method_display",
        "family",
        "score_type",
        "metric",
        "subset",
        "accession",
        "gene",
        "value",
        "se",
        "n",
        "n_pos",
    }
    # marin_dna: both score_types survive (drives the dashboard LLR/JSD toggle),
    # but only the train split — the 2 test-split rows are filtered out (height 2).
    g = df.filter(pl.col("method_id") == "glm1")
    assert g.height == 2
    assert set(g["score_type"].to_list()) == {"minus_llr_avg", "jsd_avg"}
    # conservation: each method filtered to its own track, no cross-leak.
    c = df.filter(pl.col("method_id") == "phyloP_100v")
    assert c.height == 1 and (c["score_type"] == "score").all()
    assert df.filter(pl.col("method_id") == "phastCons_43p").height == 1
    # SGE v3 is AUPRC-only; every metric row is AUPRC with a finite n_pos.
    assert set(df["metric"].to_list()) == {"AUPRC"}
    assert df["n_pos"].is_finite().all()


def test_sge_normalized_rows_gpn_star_filters_by_model(
    monkeypatch: pytest.MonkeyPatch,
):
    """The 3 GPN-Star variants share ONE model-stacked sge parquet (the path is
    model-independent for this family). sge_normalized_rows must filter each
    method to its own `model` — without the filter every method would re-emit all
    3 models' rows (9× duplication, mislabeled)."""
    v = _mk_method(id="GPN-Star-V", family="gpn_star", datasets=("sge",))
    m = _mk_method(id="GPN-Star-M", family="gpn_star", datasets=("sge",))
    p = _mk_method(id="GPN-Star-P", family="gpn_star", datasets=("sge",))
    _patch_methods(monkeypatch, (v, m, p))

    # All three resolve to the same shared parquet path.
    path = leaderboard._parquet_path(v, "sge")
    assert leaderboard._parquet_path(m, "sge") == path
    assert leaderboard._parquet_path(p, "sge") == path

    # Model-stacked parquet: each model × 2 score_types (the cLLR + LLR columns).
    stacked = pl.concat(
        [
            _sge_metrics_parquet(["minus_llr_calibrated", "minus_llr"], model=mid)
            for mid in ("GPN-Star-V", "GPN-Star-M", "GPN-Star-P")
        ]
    )
    _patch_read_parquet(monkeypatch, {path: stacked})

    df = leaderboard.sge_normalized_rows("sge")
    # Each method gets only its own 2 rows — 6 total, not 18 (no cross-model leak).
    assert df.height == 6
    for mid in ("GPN-Star-V", "GPN-Star-M", "GPN-Star-P"):
        sub = df.filter(pl.col("method_id") == mid)
        assert sub.height == 2
        assert set(sub["score_type"].to_list()) == {"minus_llr_calibrated", "minus_llr"}
        assert (sub["family"] == "gpn_star").all()


def test_sge_normalized_rows_skips_missing_parquet(monkeypatch: pytest.MonkeyPatch):
    """A method whose metrics parquet isn't on S3 yet (polars surfaces an S3 404
    as OSError) is skipped, not fatal — e.g. the gLMs before their scoring run."""
    glm = _mk_method(id="glm_pending", family="marin_dna", datasets=("sge",))
    cons = _mk_method(id="phyloP_100v", family="conservation", datasets=("sge",))
    _patch_methods(monkeypatch, (glm, cons))
    glm_path = leaderboard._parquet_path(glm, "sge")
    cons_path = leaderboard._parquet_path(cons, "sge")
    leaderboard._read_parquet.cache_clear()

    def fake(path: str) -> pl.DataFrame:
        if path == glm_path:
            raise OSError("object-store error: 404 Not Found")
        if path == cons_path:
            return _sge_metrics_parquet(["score"], score_name="phyloP_100v")
        raise FileNotFoundError(path)

    monkeypatch.setattr(leaderboard, "_read_parquet", fake)
    df = leaderboard.sge_normalized_rows("sge")
    assert set(df["method_id"].to_list()) == {"phyloP_100v"}


# ---- probe_normalized_rows (supervised linear-probe view, #348) --------------


def _probe_path(method_id: str, dataset: str = "mendelian_traits") -> str:
    return (
        f"{leaderboard.S3}/snakemake/analysis/evals_v2/results/probe_metrics/"
        f"{method_id}/{dataset}.parquet"
    )


def _probe_parquet(
    *, model: str = "m1", with_se: bool = True, with_macro: bool = True
) -> pl.DataFrame:
    """Synthetic ``compute_probe_metrics`` output: two qualifying subsets (missense,
    splicing) + one below-gate subset (mature_miRNA, null probe value) for both the
    ``probe_score`` and its ``minus_llr_avg`` baseline, plus the pipeline's ``_macro_avg_``
    row. ``with_se=False`` / ``with_macro=False`` reproduce a pre-#347 (stale) parquet."""
    # (subset, n, n_pos, probe_value, baseline_value)
    subs = [
        ("missense_variant", 500, 50, 0.55, 0.50),
        ("splicing", 400, 40, 0.60, 0.55),
        ("mature_miRNA_variant", 40, 4, None, 0.80),  # below gate; probe skipped (null)
    ]
    value_idx = {"probe_score": 3, "minus_llr_avg": 4}
    rows: list[dict] = []
    for score_type, vi in value_idx.items():
        for s in subs:
            rows.append(
                {
                    "score_type": score_type,
                    "subset": s[0],
                    "value": s[vi],
                    "se": 0.03,
                    "n": s[1],
                    "n_pos": s[2],
                    "n_chrom": 10,
                    "model": model,
                    "dataset": "mendelian_traits",
                    "split": "train",
                }
            )
        if with_macro:
            qual = [s for s in subs if s[vi] is not None and s[2] >= 30]
            rows.append(
                {
                    "score_type": score_type,
                    "subset": MACRO_AVG_SUBSET,
                    "value": sum(s[vi] for s in qual) / len(qual),
                    "se": 0.015,
                    "n": sum(s[1] for s in qual),
                    "n_pos": sum(s[2] for s in qual),
                    "n_chrom": 12,
                    "model": model,
                    "dataset": "mendelian_traits",
                    "split": "train",
                }
            )
    df = pl.DataFrame(rows)
    return df.drop("se") if not with_se else df


_PROBE_COLS = {
    "method_id",
    "method_display",
    "family",
    "protocol",
    "subset",
    "value",
    "se",
    "n",
    "n_positives",
}


def test_nan_float_accepts_only_supported_numeric_values():
    assert math.isnan(leaderboard._nan_float(None))
    assert leaderboard._nan_float(1) == 1.0
    assert leaderboard._nan_float(1.5) == 1.5
    with pytest.raises(AssertionError, match="expected a numeric value or None"):
        leaderboard._nan_float("1")


def test_probe_normalized_rows_maps_supervised_schema(monkeypatch: pytest.MonkeyPatch):
    """A marin_dna model with a #347-schema probe parquet → probe_score per-subset + macro
    rows, protocol='probe', no _global_, macro n/n_positives overloaded to K (qualifying
    subsets), and the below-gate subset kept as a NaN row."""
    m = _mk_method(id="m1", display="M1", family="marin_dna")
    _patch_methods(monkeypatch, (m,))
    _patch_read_parquet(monkeypatch, {_probe_path("m1"): _probe_parquet(model="m1")})

    df = probe_normalized_rows("mendelian_traits")
    assert set(df.columns) == _PROBE_COLS
    assert df["protocol"].unique().to_list() == ["probe"]
    # Only probe_score survives (the baseline is dropped); no _global_ row.
    assert GLOBAL_SUBSET not in df["subset"].to_list()
    assert MACRO_AVG_SUBSET in df["subset"].to_list()

    macro = df.filter(pl.col("subset") == MACRO_AVG_SUBSET)
    assert macro["n"].to_list() == [2] and macro["n_positives"].to_list() == [2]  # K
    assert macro["value"][0] == pytest.approx((0.55 + 0.60) / 2)

    mis = df.filter(pl.col("subset") == "missense_variant")
    assert mis["n"][0] == 500 and mis["n_positives"][0] == 50  # per-subset totals
    mir = df.filter(pl.col("subset") == "mature_miRNA_variant")
    assert mir.height == 1 and mir["value"][0] != mir["value"][0]  # NaN, still emitted


def test_probe_parquet_path_dispatches_by_family():
    """marin_dna probe metrics come from the evals_v2 S3 tree; evo2 from the pinned gist; a
    zero-shot-only family has no probe source (ValueError)."""
    marin = _mk_method(id="m1", family="marin_dna")
    evo = _mk_method(id="evo2_7b", family="evo2")
    cons = _mk_method(id="phyloP_100v", family="conservation")

    assert leaderboard._probe_parquet_path(marin, "mendelian_traits") == (
        f"{leaderboard.S3}/snakemake/analysis/evals_v2/results/probe_metrics/"
        f"m1/mendelian_traits.parquet"
    )
    assert leaderboard._probe_parquet_path(evo, "mendelian_traits") == (
        f"{leaderboard.EVO2_PROBE_METRICS_GIST_BASE}/"
        f"mendelian_evo2_7b_train_probe_metrics.parquet"
    )
    with pytest.raises(ValueError, match="no probe-metrics source"):
        leaderboard._probe_parquet_path(cons, "mendelian_traits")


def test_probe_normalized_rows_probe_families(monkeypatch: pytest.MonkeyPatch):
    """Both probe-capable families contribute supervised rows — marin_dna from S3 and evo2
    from the gist — while a zero-shot-only family (conservation) is skipped before any read
    (it has no per-allele embedding to probe)."""
    marin = _mk_method(id="m1", family="marin_dna")
    evo = _mk_method(id="evo2_1b_base", family="evo2")
    cons = _mk_method(id="phyloP_100v", family="conservation")
    _patch_methods(monkeypatch, (marin, evo, cons))
    _patch_read_parquet(
        monkeypatch,
        {
            leaderboard._probe_parquet_path(marin, "mendelian_traits"): _probe_parquet(
                model="m1"
            ),
            leaderboard._probe_parquet_path(evo, "mendelian_traits"): _probe_parquet(
                model="evo2_1b_base"
            ),
        },
    )

    df = probe_normalized_rows("mendelian_traits")
    assert sorted(df["family"].unique().to_list()) == ["evo2", "marin_dna"]
    assert "conservation" not in df["family"].to_list()
    # evo2 rows are the gist-sourced probe rows (macro n/n_positives overloaded to K=2).
    evo_macro = df.filter(
        (pl.col("family") == "evo2") & (pl.col("subset") == MACRO_AVG_SUBSET)
    )
    assert evo_macro.height == 1
    assert evo_macro["n"][0] == 2 and evo_macro["n_positives"][0] == 2


def test_probe_normalized_rows_skips_stale_schema(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """A pre-#347 parquet (no `se` column, no `_macro_avg_` row) is skipped with a warning
    rather than shown with blank error bars / no aggregate."""
    fresh = _mk_method(id="fresh", family="marin_dna")
    stale = _mk_method(id="stale", family="marin_dna")
    _patch_methods(monkeypatch, (fresh, stale))
    _patch_read_parquet(
        monkeypatch,
        {
            _probe_path("fresh"): _probe_parquet(model="fresh"),
            _probe_path("stale"): _probe_parquet(
                model="stale", with_se=False, with_macro=False
            ),
        },
    )

    df = probe_normalized_rows("mendelian_traits")
    assert df["method_id"].unique().to_list() == ["fresh"]
    assert "stale-schema" in capsys.readouterr().err


def test_probe_normalized_rows_skips_missing_parquet(monkeypatch: pytest.MonkeyPatch):
    """A marin_dna model whose probe parquet isn't on S3 yet is skipped (soft-fail), and
    the empty result still carries the explicit schema so it concatenates downstream."""
    m = _mk_method(id="pending", family="marin_dna")
    _patch_methods(monkeypatch, (m,))
    _patch_read_parquet(monkeypatch, {})  # every read raises FileNotFoundError

    df = probe_normalized_rows("mendelian_traits")
    assert df.height == 0
    assert set(df.columns) == _PROBE_COLS


def test_probe_normalized_rows_soft_fails_unknown_dataset_short(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """A probe-family model registered for a dataset with no `EVO2_DATASET_SHORT` entry
    soft-fails that one model (the path-construction `LookupError` is caught) rather than
    aborting the whole build — the `_probe_parquet_path` call sits inside the try, symmetric
    with the zero-shot loader."""
    # evo2 has no `complex_traits` entry in EVO2_DATASET_SHORT, so path construction raises
    # KeyError (⊂ LookupError). The read is never reached.
    evo = _mk_method(id="evo2_1b_base", family="evo2", datasets=("complex_traits",))
    _patch_methods(monkeypatch, (evo,))
    _patch_read_parquet(monkeypatch, {})

    df = probe_normalized_rows("complex_traits")  # must not raise
    assert df.height == 0
    assert set(df.columns) == _PROBE_COLS
    assert "probe skip" in capsys.readouterr().err
