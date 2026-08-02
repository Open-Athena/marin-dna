import json
import struct

import numpy as np
import polars as pl

from audit_direction_decoder_geometry import (
    build_pair_table,
    cosine_matrix,
    geometry_summary,
    safetensor_array,
)


def test_safetensor_array_reads_named_f32_tensor(tmp_path) -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f4")
    header = json.dumps(
        {
            "test": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, values.nbytes],
            }
        },
        separators=(",", ":"),
    ).encode()
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + values.tobytes())

    observed = safetensor_array(path, "test")

    np.testing.assert_array_equal(observed, values)


def test_cosine_matrix_and_summary_detect_duplicate_directions() -> None:
    vectors = np.array([[1.0, 0.0], [-2.0, 0.0], [0.0, 3.0]])

    cosine, norms = cosine_matrix(vectors)
    summary = geometry_summary(cosine)

    np.testing.assert_allclose(norms, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        cosine,
        [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert np.isclose(summary["effective_rank"], 1.8)
    assert summary["absolute_cosine_threshold_counts"]["0.9"] == 1


def test_build_pair_table_assigns_giant_and_singleton_categories() -> None:
    feature_ids = list(range(45))
    response_rows = []
    for left in range(45):
        for right in range(left + 1, 45):
            response_rows.append(
                {
                    "feature_id_left": left,
                    "feature_id_right": right,
                    "pearson": 0.5,
                    "absolute_pearson": 0.5,
                    "spearman": 0.4,
                    "absolute_spearman": 0.4,
                }
            )
    components = pl.DataFrame(
        {
            "absolute_pearson_threshold": [0.7] * 45,
            "feature_id": feature_ids,
            "component_index": [1] * 36 + list(range(2, 11)),
            "component_size": [36] * 36 + [1] * 9,
        }
    )
    decoder = np.eye(45)
    encoder = np.eye(45)

    pairs = build_pair_table(
        feature_ids=feature_ids,
        decoder_cosine=decoder,
        encoder_cosine=encoder,
        response_pairs=pl.DataFrame(response_rows),
        components=components,
    )

    assert pairs.height == 990
    assert dict(pairs.group_by("pair_category").len().iter_rows()) == {
        "within_giant_component": 630,
        "giant_to_singleton": 324,
        "between_singletons": 36,
    }
