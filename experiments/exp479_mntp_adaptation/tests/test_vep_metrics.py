from __future__ import annotations

import numpy as np
import pandas as pd

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
