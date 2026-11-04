"""Real-token boundaries and pooling must survive static inference padding."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.sharding import compact_grug_mesh

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
