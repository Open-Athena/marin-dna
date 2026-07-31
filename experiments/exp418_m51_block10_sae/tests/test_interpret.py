from __future__ import annotations

import gzip

import numpy as np
from sklearn.metrics import average_precision_score

from interpret import (
    Intron,
    _f1_threshold,
    _parse_gtf_attributes,
    _sparse_average_precision_all,
    focal_reference_coordinate,
    introns_from_gtf,
    oriented_index_to_reference,
)


def test_gtf_parser_converts_to_zero_based_half_open_introns(tmp_path) -> None:
    path = tmp_path / "test.gtf.gz"
    lines = [
        "#!genome-build GRCh38.p14\n",
        '20\ttest\texon\t101\t150\t.\t+\t.\ttranscript_id "plus"; transcript_biotype "protein_coding";\n',
        '20\ttest\texon\t201\t250\t.\t+\t.\ttranscript_id "plus"; transcript_biotype "protein_coding";\n',
        '20\ttest\texon\t301\t350\t.\t-\t.\ttranscript_id "minus"; transcript_biotype "protein_coding";\n',
        '20\ttest\texon\t401\t450\t.\t-\t.\ttranscript_id "minus"; transcript_biotype "protein_coding";\n',
        '20\ttest\texon\t501\t550\t.\t+\t.\ttranscript_id "ignored"; transcript_biotype "lncRNA";\n',
        '20\ttest\texon\t601\t650\t.\t+\t.\ttranscript_id "ignored"; transcript_biotype "lncRNA";\n',
    ]
    with gzip.open(path, "wt") as stream:
        stream.writelines(lines)

    assert introns_from_gtf(path, {"20"}) == [
        Intron("20", 150, 200, "+"),
        Intron("20", 350, 400, "-"),
    ]


def test_transcript_oriented_focal_coordinates() -> None:
    plus = Intron("20", 100, 200, "+")
    minus = Intron("20", 100, 200, "-")

    assert focal_reference_coordinate(plus, "donor") == 100
    assert focal_reference_coordinate(plus, "acceptor") == 199
    assert focal_reference_coordinate(minus, "donor") == 199
    assert focal_reference_coordinate(minus, "acceptor") == 100
    assert oriented_index_to_reference(plus, 3) == 103
    assert oriented_index_to_reference(minus, 3) == 196


def test_sparse_average_precision_matches_sklearn_with_zero_ties() -> None:
    scores = np.asarray(
        [
            [0.0, 3.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 1, 0, 0, 1, 0], dtype=np.int8)

    expected = np.asarray(
        [average_precision_score(labels, scores[:, index]) for index in range(4)]
    )
    actual = _sparse_average_precision_all(scores, labels)

    np.testing.assert_allclose(actual, expected)


def test_f1_threshold_is_selected_without_test_labels() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.4, 0.35, 0.8])

    assert _f1_threshold(labels, scores) == 0.35


def test_gtf_attributes() -> None:
    assert _parse_gtf_attributes(
        'gene_id "ENSG1"; transcript_id "ENST1"; tag "gencode_primary";'
    ) == {
        "gene_id": "ENSG1",
        "transcript_id": "ENST1",
        "tag": "gencode_primary",
    }
