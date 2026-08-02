"""Frozen paired VEP comparison for issue #417.

This module compares the mammals-only and combined-vertebrates terminal models
on the same held-out Mendelian and SGE rows. It intentionally reports only the
pre-result cells frozen in the issue #417 execution record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from marin_dna_evals.metrics import paired_metric_delta_bootstrap

MAMMALS_ARM = "mammals_only"
COMBINED_ARM = "combined_vertebrates"
ARMS = (MAMMALS_ARM, COMBINED_ARM)
DATASETS = ("mendelian_traits", "sge")
SCORE_TYPE = "minus_llr_avg"
SPLIT = "test"
SGE_SCOPES = ("both", "missense_variant", "splicing")
MACRO = "_macro_avg_"

MENDELIAN_IDENTITY_COLUMNS = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "subset",
    "match_group",
)
SGE_IDENTITY_COLUMNS = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "mavedb_urn",
    "gene",
    "subset",
    "label",
)


def _assert_paired_rows(
    mammals: pd.DataFrame,
    combined: pd.DataFrame,
    identity_columns: Sequence[str],
    *,
    dataset: str,
) -> None:
    """Fail if two score bundles are not row-for-row evaluations of one dataset."""
    for arm, frame in ((MAMMALS_ARM, mammals), (COMBINED_ARM, combined)):
        missing = [column for column in identity_columns if column not in frame.columns]
        assert not missing, f"{dataset} {arm}: missing identity columns {missing}"
    assert len(mammals) == len(combined), (
        f"{dataset}: arm row-count mismatch "
        f"{MAMMALS_ARM}={len(mammals)} {COMBINED_ARM}={len(combined)}"
    )
    try:
        pd.testing.assert_frame_equal(
            mammals[list(identity_columns)].reset_index(drop=True),
            combined[list(identity_columns)].reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise AssertionError(
            f"{dataset}: arm variant identity/order differs; paired inference is invalid"
        ) from error


def _minus_llr_avg(frame: pd.DataFrame, *, label: str) -> np.ndarray:
    for column in ("llr_fwd", "llr_rc"):
        assert column in frame.columns, f"{label}: missing {column!r}"
        assert frame[column].notna().all(), f"{label}: {column!r} contains nulls"
    score = (
        -(
            frame["llr_fwd"].to_numpy(dtype=float)
            + frame["llr_rc"].to_numpy(dtype=float)
        )
        / 2
    )
    assert np.isfinite(score).all(), f"{label}: derived {SCORE_TYPE} is non-finite"
    return score


def _bootstrap_delta_summary(
    point: float,
    bootstrap: np.ndarray,
    *,
    n_bootstrap: int,
) -> dict[str, float]:
    bootstrap = bootstrap[np.isfinite(bootstrap)]
    assert len(bootstrap) > 1, "fewer than two finite paired bootstrap iterations"
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    p_value = min(
        2
        * min(
            float((bootstrap <= 0).mean()),
            float((bootstrap >= 0).mean()),
        ),
        1.0,
    )
    p_value = max(p_value, 1.0 / n_bootstrap)
    return {
        "delta": float(point),
        "se": float(np.std(bootstrap, ddof=1)),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_two_sided": float(p_value),
    }


def paired_sge_macro_delta_bootstrap(
    labels: pd.Series,
    score_combined: pd.Series | np.ndarray,
    score_mammals: pd.Series | np.ndarray,
    accessions: pd.Series,
    subsets: pd.Series,
    *,
    scope: str,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | int | None = 0,
    n_min_per_class: int = 30,
) -> dict[str, float | int]:
    """Paired row-bootstrap of accession-macro SGE AUPRC(combined minus mammals).

    The same row indices are sampled once within each qualifying accession and
    applied to both arms. Every bootstrap macro uses the same fixed accession
    set. An accession qualifies when the selected scope has at least
    n_min_per_class rows in each label class.
    """
    assert scope in SGE_SCOPES, f"unsupported SGE scope {scope!r}"
    lengths = {
        len(labels),
        len(score_combined),
        len(score_mammals),
        len(accessions),
        len(subsets),
    }
    assert len(lengths) == 1, f"SGE paired inputs have unequal lengths: {lengths}"
    assert n_bootstrap > 1, "n_bootstrap must exceed 1"
    assert n_min_per_class > 0, "n_min_per_class must be positive"

    y = np.asarray(labels).astype(int)
    combined = np.asarray(score_combined, dtype=float)
    mammals = np.asarray(score_mammals, dtype=float)
    urn = np.asarray(accessions)
    subset = np.asarray(subsets)
    assert np.isin(y, [0, 1]).all(), "SGE label must be binary"
    assert np.isfinite(combined).all() and np.isfinite(mammals).all(), (
        "SGE paired scores must be finite"
    )
    assert pd.Series(urn).notna().all(), "SGE accession contains nulls"
    assert pd.Series(subset).notna().all(), "SGE subset contains nulls"

    qualifying: list[np.ndarray] = []
    for accession in pd.unique(urn):
        mask = urn == accession
        if scope != "both":
            mask &= subset == scope
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        n_positive = int(y[indices].sum())
        n_negative = int(len(indices) - n_positive)
        if n_positive >= n_min_per_class and n_negative >= n_min_per_class:
            qualifying.append(indices)
    assert qualifying, f"SGE {scope}: no accession passed the class-count gate"

    combined_values = [
        average_precision_score(y[idx], combined[idx]) for idx in qualifying
    ]
    mammals_values = [
        average_precision_score(y[idx], mammals[idx]) for idx in qualifying
    ]
    combined_point = float(np.mean(combined_values))
    mammals_point = float(np.mean(mammals_values))
    point = combined_point - mammals_point

    generator = np.random.default_rng(rng)
    bootstrap = np.full(n_bootstrap, np.nan, dtype=float)
    for iteration in range(n_bootstrap):
        accession_deltas: list[float] = []
        for indices in qualifying:
            sampled = indices[generator.integers(0, len(indices), size=len(indices))]
            sampled_labels = y[sampled]
            if sampled_labels.min() == sampled_labels.max():
                break
            accession_deltas.append(
                float(
                    average_precision_score(sampled_labels, combined[sampled])
                    - average_precision_score(sampled_labels, mammals[sampled])
                )
            )
        if len(accession_deltas) == len(qualifying):
            bootstrap[iteration] = float(np.mean(accession_deltas))

    summary = _bootstrap_delta_summary(point, bootstrap, n_bootstrap=n_bootstrap)
    return {
        "combined_value": combined_point,
        "mammals_value": mammals_point,
        **summary,
        "n_accessions": len(qualifying),
        "n_rows": int(sum(len(indices) for indices in qualifying)),
    }


def _metric_cell(
    metrics: pd.DataFrame,
    *,
    dataset: str,
    scope: str,
) -> pd.Series:
    for column in ("score_type", "subset", "value", "se", "dataset", "split"):
        assert column in metrics.columns, f"{dataset} metrics missing {column!r}"
    selected = metrics[
        (metrics["score_type"] == SCORE_TYPE)
        & (metrics["subset"] == scope)
        & (metrics["dataset"] == dataset)
        & (metrics["split"] == SPLIT)
    ]
    if dataset == "sge":
        for column in ("metric", "accession"):
            assert column in metrics.columns, f"SGE metrics missing {column!r}"
        selected = selected[
            (selected["metric"] == "AUPRC") & (selected["accession"] == MACRO)
        ]
    assert len(selected) == 1, (
        f"{dataset} {scope}: expected one {SCORE_TYPE} metric row, "
        f"found {len(selected)}"
    )
    row = selected.iloc[0]
    assert np.isfinite(row["value"]) and np.isfinite(row["se"]), (
        f"{dataset} {scope}: metric value/SE is non-finite"
    )
    return row


def build_issue417_comparison(
    scores: Mapping[tuple[str, str], pd.DataFrame],
    metrics: Mapping[tuple[str, str], pd.DataFrame],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Validate the four frozen cells and return the paired comparison table."""
    expected = {(arm, dataset) for arm in ARMS for dataset in DATASETS}
    assert set(scores) == expected, f"unexpected score keys: {set(scores) ^ expected}"
    assert set(metrics) == expected, (
        f"unexpected metric keys: {set(metrics) ^ expected}"
    )

    for dataset, identity in (
        ("mendelian_traits", MENDELIAN_IDENTITY_COLUMNS),
        ("sge", SGE_IDENTITY_COLUMNS),
    ):
        _assert_paired_rows(
            scores[(MAMMALS_ARM, dataset)],
            scores[(COMBINED_ARM, dataset)],
            identity,
            dataset=dataset,
        )

    derived: dict[tuple[str, str], np.ndarray] = {}
    for key, frame in scores.items():
        derived[key] = _minus_llr_avg(frame, label=f"{key[0]} {key[1]}")

    rows: list[dict[str, float | int | str]] = []
    mendelian = scores[(MAMMALS_ARM, "mendelian_traits")]
    scopes = ["_global_", *sorted(mendelian["subset"].unique().tolist())]
    for scope in scopes:
        mask = (
            np.ones(len(mendelian), dtype=bool)
            if scope == "_global_"
            else mendelian["subset"].to_numpy() == scope
        )
        mammals_metric = _metric_cell(
            metrics[(MAMMALS_ARM, "mendelian_traits")],
            dataset="mendelian_traits",
            scope=scope,
        )
        combined_metric = _metric_cell(
            metrics[(COMBINED_ARM, "mendelian_traits")],
            dataset="mendelian_traits",
            scope=scope,
        )
        delta = paired_metric_delta_bootstrap(
            mendelian.loc[mask, "label"],
            pd.Series(derived[(COMBINED_ARM, "mendelian_traits")][mask]),
            pd.Series(derived[(MAMMALS_ARM, "mendelian_traits")][mask]),
            mendelian.loc[mask, "match_group"],
            n_bootstrap=n_bootstrap,
            rng=seed,
        )
        assert np.isclose(
            delta["delta"],
            float(combined_metric["value"] - mammals_metric["value"]),
            atol=1e-12,
        ), f"Mendelian {scope}: paired point delta disagrees with metric parquets"
        rows.append(
            {
                "dataset": "mendelian_traits",
                "scope": scope,
                "mammals_value": float(mammals_metric["value"]),
                "mammals_se": float(mammals_metric["se"]),
                "combined_value": float(combined_metric["value"]),
                "combined_se": float(combined_metric["se"]),
                "delta": float(delta["delta"]),
                "delta_se": float(delta["se"]),
                "ci_low": float(delta["ci_low"]),
                "ci_high": float(delta["ci_high"]),
                "p_two_sided": float(delta["p_two_sided"]),
                "n_units": int(delta["n_groups"]),
                "n_rows": int(delta["n_rows"]),
            }
        )

    sge = scores[(MAMMALS_ARM, "sge")]
    for scope in SGE_SCOPES:
        mammals_metric = _metric_cell(
            metrics[(MAMMALS_ARM, "sge")], dataset="sge", scope=scope
        )
        combined_metric = _metric_cell(
            metrics[(COMBINED_ARM, "sge")], dataset="sge", scope=scope
        )
        delta = paired_sge_macro_delta_bootstrap(
            sge["label"],
            derived[(COMBINED_ARM, "sge")],
            derived[(MAMMALS_ARM, "sge")],
            sge["mavedb_urn"],
            sge["subset"],
            scope=scope,
            n_bootstrap=n_bootstrap,
            rng=seed,
        )
        assert np.isclose(
            delta["combined_value"], float(combined_metric["value"]), atol=1e-12
        ), f"SGE {scope}: recomputed combined macro disagrees with metric parquet"
        assert np.isclose(
            delta["mammals_value"], float(mammals_metric["value"]), atol=1e-12
        ), f"SGE {scope}: recomputed mammals macro disagrees with metric parquet"
        rows.append(
            {
                "dataset": "sge",
                "scope": scope,
                "mammals_value": float(mammals_metric["value"]),
                "mammals_se": float(mammals_metric["se"]),
                "combined_value": float(combined_metric["value"]),
                "combined_se": float(combined_metric["se"]),
                "delta": float(delta["delta"]),
                "delta_se": float(delta["se"]),
                "ci_low": float(delta["ci_low"]),
                "ci_high": float(delta["ci_high"]),
                "p_two_sided": float(delta["p_two_sided"]),
                "n_units": int(delta["n_accessions"]),
                "n_rows": int(delta["n_rows"]),
            }
        )

    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy(dtype=float)).all(), (
        "comparison contains non-finite numeric values"
    )
    return result
