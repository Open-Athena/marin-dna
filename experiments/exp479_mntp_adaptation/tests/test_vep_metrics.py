from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from exp479_mntp.vep_metrics import MACRO, matched_metrics, paired_ap_delta, sge_metrics


def test_matched_metrics_and_paired_delta_use_match_groups() -> None:
    rows: list[dict[str, object]] = []
    score: list[float] = []
    baseline: list[float] = []
    for subset in ("coding", "splicing"):
        for group in range(40):
            for label in (True, False):
                rows.append(
                    {
                        "label": label,
                        "subset": subset,
                        "match_group": f"{subset}-{group}",
                    }
                )
                score.append(float(label))
                baseline.append(float(not label))
    variants = pd.DataFrame(rows)
    scores = pd.DataFrame({"candidate": score})
    metrics = matched_metrics(variants, scores, n_bootstrap=20, n_min_groups=30)
    assert metrics.loc[metrics["subset"] == MACRO, "value"].item() == 1.0
    delta = paired_ap_delta(
        variants["label"],
        score,
        baseline,
        variants["match_group"],
        n_bootstrap=20,
    )
    assert delta["delta"] > 0
    assert delta["n_groups"] == 80


def test_paired_delta_weighted_bootstrap_matches_literal_group_resampling() -> None:
    labels = np.array([True, False, True, False, False, True])
    candidate = np.array([0.8, 0.2, 0.6, 0.3, 0.1, 0.7])
    baseline = np.array([0.7, 0.3, 0.4, 0.5, 0.2, 0.6])
    groups = np.array(["b", "b", "a", "a", "c", "c"])
    n_bootstrap = 50
    seed = 7

    actual = paired_ap_delta(
        labels,
        candidate,
        baseline,
        groups,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    group_rows = list(pd.Series(groups).groupby(groups).indices.values())
    rng = np.random.default_rng(seed)
    literal = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sampled = rng.integers(0, len(group_rows), size=len(group_rows))
        rows = np.concatenate([group_rows[value] for value in sampled])
        sampled_labels = labels[rows]
        literal[index] = (
            average_precision_score(sampled_labels, candidate[rows])
            - average_precision_score(sampled_labels, baseline[rows])
            if 0 < sampled_labels.sum() < len(sampled_labels)
            else np.nan
        )

    low, high = np.nanpercentile(literal, [2.5, 97.5])
    assert np.isclose(actual["se"], np.nanstd(literal, ddof=1))
    assert np.isclose(actual["ci_low"], low)
    assert np.isclose(actual["ci_high"], high)


def test_sge_metrics_emit_accession_and_subset_macros() -> None:
    rows: list[dict[str, object]] = []
    values: list[float] = []
    for accession, gene in (("urn:1", "GENE1"), ("urn:2", "GENE2")):
        for subset in ("missense_variant", "splicing"):
            for label in [True] * 30 + [False] * 30:
                rows.append(
                    {
                        "mavedb_urn": accession,
                        "gene": gene,
                        "subset": subset,
                        "label": label,
                    }
                )
                values.append(float(label))
    metrics = sge_metrics(
        pd.DataFrame(rows),
        pd.DataFrame({"candidate": values}),
        n_bootstrap=10,
    )
    headline = metrics[(metrics["subset"] == MACRO) & (metrics["accession"] == MACRO)]
    assert len(headline) == 1
    assert np.isclose(headline["value"].item(), 1.0)
