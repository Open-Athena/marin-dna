"""Real-token boundaries and pooling must survive static inference padding."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.sharding import compact_grug_mesh

from exp586_moe.classification_features import source_slice
from exp586_moe.frozen_features import FrozenFeatures


class ToyCausalModel(eqx.Module):
    def __call__(self, ids, mask):
        hidden = jnp.stack((ids, jnp.cumsum(ids, axis=1)), axis=-1).astype(jnp.float32)
        return hidden, {"capacity_overflow_per_layer": jnp.zeros(1, dtype=jnp.int32)}


def test_last_and_mean_ignore_static_padding():
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    scorer = FrozenFeatures(ToyCausalModel(), mesh, sequence_length=16)
    with jax.set_mesh(mesh):
        values = scorer.evaluate([[2, 3, 4]])
    np.testing.assert_array_equal(values["last"], [[4, 9]])
    np.testing.assert_allclose(values["mean"], [[3, 16 / 3]])
    np.testing.assert_array_equal(values["last"], scorer.evaluate([[2, 3, 4]])["last"])


def test_no_silent_truncation():
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    scorer = FrozenFeatures(ToyCausalModel(), mesh, sequence_length=16)
    with pytest.raises(ValueError):
        scorer.evaluate([[1] * 17])


def test_file_view_keeps_global_feature_ids_and_local_reader_coordinates():
    first = {"rows": 5, "row_start": 0, "row_stop": 5, "path": "first.parquet"}
    second = {"rows": 8, "row_start": 5, "row_stop": 13, "path": "second.parquet"}
    manifest = {
        "inputs": {"task.tsv": {"rows": 13, "files": [first, second]}},
        "samples": "/tmp/example",
    }
    source, view, begin, end = source_slice(
        manifest, {"file": "task.tsv"}, {"start": 7, "stop": 11}
    )
    assert source == second and (begin, end) == (2, 6)
    assert view["inputs"]["task.tsv"] == {"rows": 8, "files": [second]}
    assert manifest["inputs"]["task.tsv"]["rows"] == 13
    with pytest.raises(ValueError):
        source_slice(manifest, {"file": "task.tsv"}, {"start": 4, "stop": 8})
