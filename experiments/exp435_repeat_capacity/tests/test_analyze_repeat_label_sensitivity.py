from __future__ import annotations

import numpy as np
import polars as pl

from analyze_repeat_label_sensitivity import (
    FEATURE_OF_INTEREST,
    Target,
    add_stratum_calls,
    assert_global_reproduction,
    average_precision_both_directions,
    contingency_tables,
    inventory_overlap,
    repeat_free_retention,
    target_definitions,
)


def synthetic_panel() -> pl.DataFrame:
    statuses = [
        "focal_repeat",
        "near_repeat",
        "repeat_free_window",
    ]
    rows = []
    for status in statuses:
        for subset in ("a", "b"):
            for label in (0, 1):
                rows.extend(
                    {
                        "position_status": status,
                        "subset": subset,
                        "label": label,
                    }
                    for _ in range(2)
                )
    return pl.DataFrame(rows)


def test_targets_are_stratified_before_class_size_filtering() -> None:
    targets, counts = target_definitions(synthetic_panel(), minimum_class_size=1)
    assert set(targets) == {
        "all",
        "focal_repeat",
        "near_repeat",
        "repeat_free_window",
    }
    assert all(
        [target.name for target in current] == ["overall", "a", "b"]
        for current in targets.values()
    )
    focal = counts.filter(
        (pl.col("stratum") == "focal_repeat") & (pl.col("target") == "overall")
    ).row(0, named=True)
    assert focal["n"] == 8
    assert focal["n_positive"] == 4
    assert focal["prevalence"] == 0.5
    assert focal["inferential"]


def test_average_precision_handles_ties_and_both_directions() -> None:
    labels = np.asarray([1, 0, 1, 0], dtype=np.uint8)
    scores = np.asarray([[3.0, 0.0], [2.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    raw, negated = average_precision_both_directions(labels, scores, chunk_size=1)
    np.testing.assert_allclose(raw[0], 5 / 6)
    np.testing.assert_allclose(negated[0], 0.5)
    assert np.isfinite(raw).all() and np.isfinite(negated).all()


def test_discovery_requires_both_corrected_tests_and_direction() -> None:
    frame = pl.DataFrame(
        {
            "arm": ["block19-25m"] * 3,
            "block": [19] * 3,
            "budget": [25_000_200] * 3,
            "orientation": ["forward"] * 3,
            "pooling": ["focal"] * 3,
            "response": ["abs_delta"] * 3,
            "response_role": ["primary"] * 3,
            "target_kind": ["overall"] * 3,
            "target": ["overall"] * 3,
            "feature_id": [1, 2, 3],
            "mean_difference": [1.0, 1.0, 1.0],
            "rank_biserial": [0.2, 0.2, -0.2],
            "welch_q": [0.01, 0.01, 0.01],
            "mann_whitney_q": [0.02, 0.20, 0.02],
        }
    )
    called = add_stratum_calls(frame, "repeat_free_window")
    assert called["maximum_q"].to_list() == [0.02, 0.2, 0.02]
    assert called["concordant_discovery"].to_list() == [True, False, False]


def family_rows(stratum: str, discoveries: set[int]) -> pl.DataFrame:
    rows = []
    for feature, effect in ((1, 0.1), (2, 0.2), (FEATURE_OF_INTEREST, 0.3)):
        rows.append(
            {
                "arm": "block19-25m",
                "block": 19,
                "orientation": "forward",
                "response": "abs_delta",
                "repeat_stratum": stratum,
                "target_kind": "overall",
                "target": "overall",
                "feature_id": feature,
                "rank_biserial": effect,
                "concordant_discovery": feature in discoveries,
            }
        )
    return pl.DataFrame(rows)


def test_retention_and_inventory_overlap_use_feature_ids() -> None:
    combined = pl.concat(
        [
            family_rows("all", {1, 2, FEATURE_OF_INTEREST}),
            family_rows("repeat_free_window", {1, FEATURE_OF_INTEREST}),
        ]
    )
    retention = repeat_free_retention(combined).row(0, named=True)
    assert retention["global_discoveries"] == 3
    assert retention["retained_global_discoveries"] == 2
    assert retention["retention_fraction"] == 2 / 3

    overlap = inventory_overlap(
        combined,
        {("block19-25m", "forward"): {1, 9}},
        {("block19-25m", "forward", "abs_delta"): {1, 2}},
    )
    global_row = overlap.filter(pl.col("repeat_stratum") == "all").row(0, named=True)
    assert global_row["label_reference_repeat_overlap"] == 1
    assert global_row["label_paired_repeat_overlap"] == 2
    assert global_row["label_both_repeat_inventories_overlap"] == 1


def test_contingency_tables_report_omnibus_and_pairwise_fdr() -> None:
    omnibus, pairwise = contingency_tables(synthetic_panel())
    assert set(omnibus["target"]) == {"overall", "a", "b"}
    assert pairwise.height == 6
    assert pairwise["contrast"].n_unique() == 2
    assert omnibus.filter(~pl.col("q").is_between(0, 1)).is_empty()
    assert pairwise.filter(~pl.col("q").is_between(0, 1)).is_empty()


def test_global_reproduction_checks_all_columns(tmp_path) -> None:
    expected = pl.DataFrame(
        {
            "target_kind": ["overall"],
            "target": ["overall"],
            "feature_id": pl.Series([1], dtype=pl.UInt32),
            "effect": [0.125],
        }
    )
    path = tmp_path / "expected.parquet"
    expected.write_parquet(path)
    result = assert_global_reproduction(expected, path)
    assert result == {"rows": 1, "maximum_absolute_error": 0.0}


def test_target_dataclass_preserves_indices_and_labels() -> None:
    target = Target(
        stratum="all",
        kind="overall",
        name="overall",
        indices=np.asarray([0, 2]),
        labels=np.asarray([1, 0], dtype=np.uint8),
    )
    assert target.indices.tolist() == [0, 2]
    assert target.labels.tolist() == [1, 0]
