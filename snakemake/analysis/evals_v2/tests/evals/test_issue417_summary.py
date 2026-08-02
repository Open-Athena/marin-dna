"""Tests for the frozen issue #417 paired VEP comparison."""

import numpy as np
import pandas as pd
import pytest

from marin_dna_evals.issue417_summary import (
    COMBINED_ARM,
    MAMMALS_ARM,
    build_issue417_comparison,
    paired_sge_macro_delta_bootstrap,
)
from marin_dna_evals.metrics import (
    compute_auprc_metrics,
    compute_sge_metrics,
)


def _matched_scores() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(4)
    rows: list[dict[str, object]] = []
    for subset_index, subset in enumerate(
        ("missense_variant", "splicing", "synonymous_variant")
    ):
        for group in range(40):
            for offset, label in enumerate((1, 0, 0)):
                rows.append(
                    {
                        "chrom": "18",
                        "pos": subset_index * 10000 + group * 3 + offset,
                        "ref": "A",
                        "alt": "C",
                        "label": label,
                        "subset": subset,
                        "match_group": subset_index * 100 + group,
                    }
                )
    identity = pd.DataFrame(rows)
    mammals_score = rng.normal(size=len(identity))
    combined_score = mammals_score + identity["label"].to_numpy() * 0.7

    result = {}
    for arm, score in (
        (MAMMALS_ARM, mammals_score),
        (COMBINED_ARM, combined_score),
    ):
        frame = identity.copy()
        frame["llr_fwd"] = -score
        frame["llr_rc"] = -score
        result[arm] = frame
    return result


def _sge_scores() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for accession_index, (urn, gene) in enumerate(
        (("urn:1", "GENE1"), ("urn:2", "GENE2"))
    ):
        for subset_index, subset in enumerate(("missense_variant", "splicing")):
            for row in range(80):
                rows.append(
                    {
                        "chrom": "18",
                        "pos": accession_index * 10000 + subset_index * 1000 + row,
                        "ref": "A",
                        "alt": "G",
                        "mavedb_urn": urn,
                        "gene": gene,
                        "subset": subset,
                        "label": row < 40,
                    }
                )
    identity = pd.DataFrame(rows)
    mammals_score = rng.normal(size=len(identity))
    combined_score = mammals_score + identity["label"].to_numpy() * 0.8

    result = {}
    for arm, score in (
        (MAMMALS_ARM, mammals_score),
        (COMBINED_ARM, combined_score),
    ):
        frame = identity.copy()
        frame["llr_fwd"] = -score
        frame["llr_rc"] = -score
        result[arm] = frame
    return result


def _metrics(
    matched: dict[str, pd.DataFrame],
    sge: dict[str, pd.DataFrame],
) -> dict[tuple[str, str], pd.DataFrame]:
    result = {}
    for arm in (MAMMALS_ARM, COMBINED_ARM):
        matched_score = pd.DataFrame(
            {"minus_llr_avg": -(matched[arm]["llr_fwd"] + matched[arm]["llr_rc"]) / 2}
        )
        matched_metrics = compute_auprc_metrics(
            matched[arm][["label", "subset", "match_group"]],
            matched_score,
            n_bootstrap=20,
            rng=0,
        )
        matched_metrics["model"] = arm
        matched_metrics["dataset"] = "mendelian_traits"
        matched_metrics["split"] = "test"
        result[(arm, "mendelian_traits")] = matched_metrics

        sge_score = pd.DataFrame(
            {"minus_llr_avg": -(sge[arm]["llr_fwd"] + sge[arm]["llr_rc"]) / 2}
        )
        sge_metrics = compute_sge_metrics(
            sge[arm][["mavedb_urn", "gene", "subset", "label"]],
            sge_score,
            n_bootstrap=20,
            rng=0,
        )
        sge_metrics["model"] = arm
        sge_metrics["dataset"] = "sge"
        sge_metrics["split"] = "test"
        result[(arm, "sge")] = sge_metrics
    return result


def test_paired_sge_identical_scores_have_zero_delta() -> None:
    sge = _sge_scores()[MAMMALS_ARM]
    score = -(sge["llr_fwd"] + sge["llr_rc"]) / 2
    result = paired_sge_macro_delta_bootstrap(
        sge["label"],
        score,
        score,
        sge["mavedb_urn"],
        sge["subset"],
        scope="missense_variant",
        n_bootstrap=50,
        rng=0,
    )
    assert result["delta"] == pytest.approx(0)
    assert result["se"] == pytest.approx(0)
    assert result["p_two_sided"] == pytest.approx(1)
    assert result["n_accessions"] == 2


def test_build_comparison_reports_only_cds_scopes_and_positive_deltas() -> None:
    matched = _matched_scores()
    sge = _sge_scores()
    scores = {
        (arm, "mendelian_traits"): matched[arm] for arm in (MAMMALS_ARM, COMBINED_ARM)
    }
    scores.update({(arm, "sge"): sge[arm] for arm in (MAMMALS_ARM, COMBINED_ARM)})
    comparison = build_issue417_comparison(
        scores,
        _metrics(matched, sge),
        n_bootstrap=50,
        seed=0,
    )

    assert set(map(tuple, comparison[["dataset", "scope"]].to_numpy())) == {
        ("mendelian_traits", "missense_variant"),
        ("mendelian_traits", "splicing"),
        ("mendelian_traits", "synonymous_variant"),
        ("sge", "missense_variant"),
        ("sge", "splicing"),
    }
    assert (comparison["delta"] > 0).all()
    assert np.isfinite(comparison.select_dtypes(include=[np.number])).all().all()


def test_build_comparison_rejects_row_order_mismatch() -> None:
    matched = _matched_scores()
    sge = _sge_scores()
    combined = matched[COMBINED_ARM].copy()
    combined.loc[[0, 1], "pos"] = combined.loc[[1, 0], "pos"].to_numpy()
    scores = {
        (MAMMALS_ARM, "mendelian_traits"): matched[MAMMALS_ARM],
        (COMBINED_ARM, "mendelian_traits"): combined,
        (MAMMALS_ARM, "sge"): sge[MAMMALS_ARM],
        (COMBINED_ARM, "sge"): sge[COMBINED_ARM],
    }
    with pytest.raises(AssertionError, match="identity/order differs"):
        build_issue417_comparison(
            scores,
            _metrics(matched, sge),
            n_bootstrap=10,
            seed=0,
        )
