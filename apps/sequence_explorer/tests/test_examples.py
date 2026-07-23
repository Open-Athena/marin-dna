from examples import DEFAULT_EXAMPLE, EXAMPLES


def test_recommended_examples_match_issue_387():
    assert [example.name for example in EXAMPLES] == [
        "LDLR",
        "TH",
        "GRIA4",
        "HBA1",
        "tRNA-Arg-TCT-4-1",
    ]
    assert [len(example.sequence) for example in EXAMPLES] == [126, 186, 28, 255, 74]
    assert [example.strand for example in EXAMPLES] == ["+", "-", "+", "+", "-"]
    assert DEFAULT_EXAMPLE.name == "LDLR"
    for example in EXAMPLES:
        assert len(example.sequence) == example.end - example.start
        assert set(example.sequence) <= set("ACGT")
