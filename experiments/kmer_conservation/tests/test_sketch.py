import numpy as np

from kmer_conservation.sketch import build_lsh, lsh_scores, scan_scores, signatures


def test_signatures_and_exhaustive_scan_match_direct_comparison():
    sets = [
        np.arange(30, dtype=np.uint64),
        np.arange(10, 45, dtype=np.uint64),
        np.arange(100, 200, dtype=np.uint64),
    ]
    sig = signatures(sets, 128)
    result = scan_scores(sig[:2].copy(), sig, 128, threads=1)
    expected = np.max(np.mean(sig[:2, None, :] == sig[None, :, :], axis=2), axis=0)
    assert np.array_equal(result, expected)
    assert abs(np.mean(sig[0] == sig[1]) - 20 / 45) < 0.15


def test_lsh_band_candidates_and_empty_window_exclusion():
    target = np.array(
        [[1, 2, 3, 4], [1, 2, 9, 9], [1, 8, 3, 8], [0, 0, 0, 0]], dtype=np.uint64
    )
    query = target[:1].copy()
    index = build_lsh(target, rows=2)
    scores, emitted, pairs = lsh_scores(
        query,
        target,
        index,
        2,
        np.array([1, 1, 1, 0], dtype=bool),
        np.array([1], dtype=bool),
    )
    assert np.array_equal(scores, [1, 0.5, 0, 0])
    assert emitted == 3 and pairs == 2
