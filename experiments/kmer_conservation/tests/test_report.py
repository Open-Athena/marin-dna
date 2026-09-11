from kmer_conservation.report import group_counts, interval, paired_difference


def test_group_bootstrap_keeps_multi_target_query_pairs_and_shared_homologs():
    rows = [
        {
            "query": "human:1",
            "split_component": "group1",
            "ranks": {"mouse:1": 1, "mouse:2": 11},
        },
        {
            "query": "mouse:1",
            "split_component": "group1",
            "ranks": {"armadillo:1": None},
        },
        {"query": "human:2", "split_component": "group2", "ranks": {"mouse:3": 2}},
    ]
    assert group_counts(rows) == {"group1": (1, 3), "group2": (1, 1)}
    assert interval(rows) == (1 / 3, 1.0)
    assert paired_difference(rows, rows) == {
        "difference": 0.0,
        "ci95": [0.0, 0.0],
        "bootstrap_groups": 2,
        "budget": 10,
    }
