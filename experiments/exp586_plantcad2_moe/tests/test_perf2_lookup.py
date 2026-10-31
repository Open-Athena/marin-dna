"""Oracle for the isolated full-vocabulary input lookup optimization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe_hero_ep.model import (
    _compact_embedding_gather,
    _embedding_gather,
)


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("one_hot", [False, True])
def test_lookup_preserves_full_vocabulary_values_and_gradient(fallback, one_hot):
    # Include repeated IDs, unused vocabulary rows, and an optional fallback ID.
    ids = jnp.tile(jnp.asarray([[3, 5, 3, 7], [5, 7, 3, 9 if fallback else 7]]), (4, 1))
    rows = (3, 5, 7)
    table = jnp.arange(17 * 8, dtype=jnp.float32).reshape(17, 8) / 32
    cotangent = jnp.arange(8 * 4 * 8, dtype=jnp.float32).reshape(8, 4, 8) / 16
    with jax.set_mesh(compact_grug_mesh()):
        ids = jax.reshard(ids, P(("replica_dcn", "data", "expert"), None))

        def reference(w):
            return _embedding_gather(w, ids)

        def candidate(w):
            return _compact_embedding_gather(w, ids, rows, one_hot=one_hot)

        actual, backward = jax.jit(lambda w: jax.vjp(candidate, w))(table)
        expected, expected_backward = jax.jit(lambda w: jax.vjp(reference, w))(table)
        gradient = jax.jit(backward)(cotangent)[0]
        expected_gradient = jax.jit(expected_backward)(cotangent)[0]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(gradient, expected_gradient)
    assert gradient.shape == table.shape
    np.testing.assert_array_equal(gradient[0], 0)
