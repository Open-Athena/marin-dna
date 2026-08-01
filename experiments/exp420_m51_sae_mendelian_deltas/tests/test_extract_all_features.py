from __future__ import annotations

import numpy as np

from analysis import FOCAL_INDEX
from extract_all_features import context_table, sparse_union_table


def test_sparse_union_table_preserves_ref_alt_and_delta() -> None:
    ref = np.asarray([[0, 2, 0], [3, 0, 4]], dtype=np.float32)
    alt = np.asarray([[1, 0, 0], [3, 5, 0]], dtype=np.float32)
    table = sparse_union_table(ref, alt, np.asarray([7, 9], dtype=np.uint32))

    assert table.to_pydict() == {
        "row_index": [7, 7, 9, 9, 9],
        "feature_id": [0, 1, 0, 1, 2],
        "ref_activation": [0.0, 2.0, 3.0, 0.0, 4.0],
        "alt_activation": [1.0, 0.0, 3.0, 5.0, 0.0],
        "delta": [1.0, -2.0, 0.0, 5.0, -4.0],
    }


def test_context_table_excludes_focal_base_from_gc_control() -> None:
    reference = "A" * FOCAL_INDEX + "C" + "G" * FOCAL_INDEX
    alternate = reference[:FOCAL_INDEX] + "T" + reference[FOCAL_INDEX + 1 :]
    table = context_table(
        [
            {
                "row_index": 3,
                "ref_sequence": reference,
                "alt_sequence": alternate,
            }
        ]
    ).to_pydict()

    assert table["row_index"] == [3]
    assert len(table["ref_context"][0]) == 41
    assert table["ref_context"][0][20] == "C"
    assert table["alt_context"][0][20] == "T"
    assert table["flank_gc_count"] == [20]
    assert table["flank_gc_bin"] == [2]
