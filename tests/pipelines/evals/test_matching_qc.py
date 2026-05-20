"""Tests for the per-subset matching diagnostics helper."""

import numpy as np
import polars as pl
import pytest

from marin_dna.pipelines.evals.matching_qc import compute_matching_qc


def _pre(subsets: list[tuple[str, int, int]]) -> pl.DataFrame:
    """Build a synthetic pre-matching frame.

    Each tuple is ``(consequence_group, n_pos, n_neg)``.
    """
    rows: list[dict] = []
    for grp, n_pos, n_neg in subsets:
        for _ in range(n_pos):
            rows.append(
                {"label": True, "consequence_group": grp, "distance_tss_pc": 1.0}
            )
        for _ in range(n_neg):
            rows.append(
                {"label": False, "consequence_group": grp, "distance_tss_pc": 1.0}
            )
    return pl.DataFrame(rows)


def _post(records: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(records)


class TestComputeMatchingQc:
    def test_no_drops_when_every_positive_matched(self) -> None:
        # 5 positives × 9 negatives = 45 negatives all matched.
        pre = _pre([("missense_variant", 5, 100)])
        post_records: list[dict] = []
        for i in range(5):
            post_records.append(
                {
                    "label": True,
                    "subset": "missense_variant",
                    "distance_tss_pc": float(i),
                }
            )
            for _ in range(9):
                post_records.append(
                    {
                        "label": False,
                        "subset": "missense_variant",
                        "distance_tss_pc": float(i),
                    }
                )
        post = _post(post_records)

        qc = compute_matching_qc(pre, post, ["distance_tss_pc"])

        assert qc.height == 1
        row = qc.row(0, named=True)
        assert row["subset"] == "missense_variant"
        assert row["n_positives_input"] == 5
        assert row["n_positives_kept"] == 5
        assert row["n_dropped"] == 0
        assert row["frac_dropped"] == pytest.approx(0.0)
        assert row["baseline_auprc"] == pytest.approx(0.1)

    def test_n_dropped_matches_subsampling(self) -> None:
        # Subset A: 10 positives, 100 negatives → none dropped, all 10 kept.
        # Subset B: 5 positives, 0 kept (e.g. zero negatives in some category).
        pre = _pre([("A", 10, 100), ("B", 5, 0)])
        post_records = []
        for i in range(10):
            post_records.append(
                {"label": True, "subset": "A", "distance_tss_pc": float(i)}
            )
            for _ in range(9):
                post_records.append(
                    {"label": False, "subset": "A", "distance_tss_pc": float(i)}
                )
        post = _post(post_records)

        qc = compute_matching_qc(pre, post, ["distance_tss_pc"]).sort("subset")
        by_subset = {r["subset"]: r for r in qc.iter_rows(named=True)}

        assert by_subset["A"]["n_positives_input"] == 10
        assert by_subset["A"]["n_positives_kept"] == 10
        assert by_subset["A"]["n_dropped"] == 0

        assert by_subset["B"]["n_positives_input"] == 5
        assert by_subset["B"]["n_positives_kept"] == 0
        assert by_subset["B"]["n_dropped"] == 5
        assert by_subset["B"]["frac_dropped"] == pytest.approx(1.0)

    def test_auprc_near_baseline_when_pos_and_neg_match_on_feature(self) -> None:
        # Pos and neg drawn from the same distribution → AUPRC should be
        # close to baseline (= 0.1 for 1:9).
        rng = np.random.default_rng(0)
        n_pos = 50
        k = 9
        records = []
        for x in rng.normal(size=n_pos):
            records.append({"label": True, "subset": "S", "distance_tss_pc": float(x)})
            for x_neg in rng.normal(size=k):
                records.append(
                    {"label": False, "subset": "S", "distance_tss_pc": float(x_neg)}
                )
        pre = _pre([("S", n_pos, 1000)])
        post = _post(records)

        qc = compute_matching_qc(pre, post, ["distance_tss_pc"])
        row = qc.row(0, named=True)
        baseline = row["baseline_auprc"]
        # Loose upper bound — with 50 positives the sampling noise can put
        # AUPRC well above 0.1; require it stay clearly below 0.5.
        assert abs(row["distance_tss_pc_auprc"] - baseline) < 0.4

    def test_auprc_high_and_sign_detects_leak_direction(self) -> None:
        # Pos all have feature=1.0, neg all have feature=0.0 → perfect
        # positive-leak: +feature predicts label well, AUPRC ≈ 1.0,
        # sign = +1.
        records = []
        for _ in range(10):
            records.append({"label": True, "subset": "S", "distance_tss_pc": 1.0})
            for _ in range(9):
                records.append({"label": False, "subset": "S", "distance_tss_pc": 0.0})
        pre = _pre([("S", 10, 100)])
        post = _post(records)

        qc = compute_matching_qc(pre, post, ["distance_tss_pc"])
        row = qc.row(0, named=True)
        assert row["distance_tss_pc_auprc"] == pytest.approx(1.0)
        assert row["distance_tss_pc_auprc_sign"] == 1

    def test_auprc_sign_negative_when_lower_feature_predicts_positive(self) -> None:
        # Flip the previous test: pos has feature=0, neg has feature=1.
        records = []
        for _ in range(10):
            records.append({"label": True, "subset": "S", "distance_tss_pc": 0.0})
            for _ in range(9):
                records.append({"label": False, "subset": "S", "distance_tss_pc": 1.0})
        pre = _pre([("S", 10, 100)])
        post = _post(records)

        qc = compute_matching_qc(pre, post, ["distance_tss_pc"])
        row = qc.row(0, named=True)
        assert row["distance_tss_pc_auprc"] == pytest.approx(1.0)
        assert row["distance_tss_pc_auprc_sign"] == -1

    def test_auprc_null_when_subset_has_zero_positives_kept(self) -> None:
        # Pre has the subset (so it shows up in input counts), post has only
        # negatives for it → AUPRC undefined.
        pre = _pre([("S", 5, 100)])
        records = [
            {"label": False, "subset": "S", "distance_tss_pc": float(i)}
            for i in range(5)
        ]
        post = _post(records)

        qc = compute_matching_qc(pre, post, ["distance_tss_pc"])
        row = qc.row(0, named=True)
        assert row["n_positives_kept"] == 0
        assert row["distance_tss_pc_auprc"] is None
        assert row["distance_tss_pc_auprc_sign"] is None

    def test_multiple_features_each_get_their_own_columns(self) -> None:
        pre = _pre([("S", 5, 100)])
        # feat_a leaks (pos=1, neg=0); feat_b is matched (pos and neg both 0).
        records = []
        for _ in range(5):
            records.append({"label": True, "subset": "S", "feat_a": 1.0, "feat_b": 0.0})
            for _ in range(9):
                records.append(
                    {"label": False, "subset": "S", "feat_a": 0.0, "feat_b": 0.0}
                )
        post = _post(records)

        qc = compute_matching_qc(pre, post, ["feat_a", "feat_b"])
        row = qc.row(0, named=True)
        assert row["feat_a_auprc"] == pytest.approx(1.0)
        assert row["feat_a_auprc_sign"] == 1
        # feat_b is constant — any AP value is fine, but it shouldn't crash.
        assert row["feat_b_auprc"] is not None
