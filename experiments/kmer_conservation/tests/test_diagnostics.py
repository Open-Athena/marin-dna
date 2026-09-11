from kmer_conservation.diagnostics import union_predictions


def test_union_collapses_loci_and_spends_one_final_budget():
    def result(names):
        return [
            {
                "query": "truth",
                "source": "human",
                "target": "mouse",
                "seconds": 2,
                "ranks": {"truth": 1},
                "hits": [
                    {"component": name, "id": name, "kind": "anchor", "score": 1}
                    for name in names
                ],
            }
        ]

    fused = union_predictions(
        [result(["truth", "x", "y"]), result(["truth", "z", "x"])]
    )[0]
    assert len(fused["hits"]) == fused["candidate_loci"] == 4
    assert fused["work"] == 6
    assert fused["seconds"] >= 4
    assert fused["ranks"] == {"truth": 1}
    assert sum(h["component"] == "truth" for h in fused["hits"]) == 1


def test_union_never_invents_missing_truth():
    item = {
        "query": "truth",
        "source": "human",
        "target": "mouse",
        "hits": [],
        "ranks": {"truth": None},
    }
    assert union_predictions([[item], [item]])[0]["ranks"] == {"truth": None}
