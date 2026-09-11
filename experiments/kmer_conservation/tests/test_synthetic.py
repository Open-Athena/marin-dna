from kmer_conservation.synthetic import planted_records


def test_planted_boundaries_and_independent_flanks():
    records = planted_records()
    assert len(records) == 216
    for row in records:
        assert len(row["sequence"]) == row["end"] - row["start"] == 4096
        assert 0 <= row["tract_start"] < row["tract_end"] <= 4096
    for i in range(0, len(records), 3):
        trio = records[i : i + 3]
        assert len({r["component"] for r in trio}) == 1
        assert len({r["sequence"][:100] for r in trio}) == 3
        if trio[0]["divergence"] == trio[0]["indel_rate"] == 0:
            assert (
                len({r["sequence"][r["tract_start"] : r["tract_end"]] for r in trio})
                == 1
            )
