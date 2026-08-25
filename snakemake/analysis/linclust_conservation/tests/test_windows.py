from marin_dna_linclust_conservation.windows import (
    RejectedWindow,
    RejectionReason,
    Window,
    decode_record_id,
    iter_windows,
)


def test_window_coordinates_length_and_record_id_round_trip() -> None:
    results = list(
        iter_windows(
            "ACGT" * 96,
            accession="GCF_test.1",
            sequence_name="NC_test.1",
        )
    )
    assert [(result.start, result.end) for result in results] == [(0, 255), (128, 383)]
    assert all(isinstance(result, Window) for result in results)
    assert all(
        len(result.sequence) == 255 for result in results if isinstance(result, Window)
    )
    assert decode_record_id(results[0].record_id) == (
        "GCF_test.1",
        "NC_test.1",
        0,
        255,
        "+",
    )


def test_repeat_fraction_boundary_for_odd_window_length() -> None:
    below_half = list(
        iter_windows(
            "a" * 127 + "C" * 128,
            accession="GCF_test.1",
            sequence_name="contig",
        )
    )
    assert isinstance(below_half[0], Window)
    assert below_half[0].repeat_fraction == 127 / 255

    majority = list(
        iter_windows(
            "a" * 128 + "C" * 127,
            accession="GCF_test.1",
            sequence_name="contig",
        )
    )
    assert majority == [
        RejectedWindow(
            "GCF_test.1",
            "contig",
            0,
            255,
            RejectionReason.MAJORITY_SOFT_MASKED,
        )
    ]


def test_ambiguous_base_rejects_window_without_deleting_sequence() -> None:
    result = list(
        iter_windows(
            "A" * 127 + "N" + "A" * 127,
            accession="GCF_test.1",
            sequence_name="contig",
        )
    )
    assert result[0].reason == RejectionReason.AMBIGUOUS_BASE
