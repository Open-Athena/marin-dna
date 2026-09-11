import numpy as np
import pytest

from kmer_conservation.core import (
    INVALID,
    canonical_codes,
    exact_index,
    exact_scores,
    make_windows,
    rank_loci,
    window_starts,
)
from kmer_conservation.fixture import assign_components


def row(seq, name="a", start=0, species="human"):
    return {
        "sequence": seq,
        "id": name,
        "group": name,
        "component": name,
        "chrom": "chr1",
        "start": start,
        "end": start + len(seq),
        "species": species,
        "kind": "anchor",
    }


def test_canonical_reverse_complement_and_ambiguity():
    seq = "ACgTTGCAATNNGATC"
    rc = seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]
    assert np.array_equal(canonical_codes(seq, 5), canonical_codes(rc, 5)[::-1])
    assert np.all(canonical_codes("ACNTA", 3) == INVALID)
    assert len(canonical_codes("A", 5)) == 0


@pytest.mark.parametrize("width", [64, 128, 255, 511, 1024])
@pytest.mark.parametrize("divisor", [1, 2, 4])
def test_tiling_coverage_and_phase(width, divisor):
    stride = max(1, width // divisor)
    starts = window_starts(37, 4096, width, stride, 17)
    assert starts[0] == 0 and starts[-1] + width == 4096
    assert max(np.diff(starts)) <= width
    assert all((s + 37 - 17) % stride == 0 for s in starts[1:-1])


def test_exact_sparse_matches_bruteforce_including_unseen_query_kmers():
    target = make_windows(
        [row("ACGTACGTAAAACCCC", "a"), row("GGGGTTTTAGCTAGCA", "b")], 8, 3, 2
    )
    index, vocab, lengths = exact_index(target)
    query = make_windows([row("CCCCATATNNNGGGGT", "q")], 8, 3, 2)
    scores, count, work = exact_scores(query.features, index, vocab, lengths)
    expected = [
        max(
            (len(set(a) & set(b)) / len(set(a) | set(b)) if len(set(a) | set(b)) else 0)
            for a in query.features
        )
        for b in target.features
    ]
    assert np.allclose(scores, expected)
    assert count == sum(
        bool(set(a) & set(b)) for a in query.features for b in target.features
    )
    assert work == sum(
        len(set(a) & set(b)) for a in query.features for b in target.features
    )


def test_empty_sets_do_not_match_and_loci_collapse_before_budget():
    target = make_windows(
        [row("AAAAAAAAAAAA", "a"), row("AAAAAAAAAAAA", "b", 100)], 6, 3, 2
    )
    scores = np.ones(len(target.features))
    assert len(rank_loci(scores, target, limit=10)) == 2
    index, vocab, lengths = exact_index(target)
    observed, count, work = exact_scores(
        [np.array([], dtype=np.uint64)], index, vocab, lengths
    )
    assert not observed.any() and count == work == 0


def test_overlap_in_either_species_unions_homologs_before_split():
    records = [
        row("A" * 10, "a", 0),
        row("A" * 10, "b", 9),
        row("A" * 10, "a", 500, "mouse"),
        row("A" * 10, "b", 1000, "mouse"),
        row("A" * 10, "c", 19),
    ]
    assign_components(records, 568, 0.4)
    assert records[0]["component"] == records[3]["component"]
    assert records[4]["component"] != records[0]["component"]
    assert len({r["split"] for r in records[:4]}) == 1
