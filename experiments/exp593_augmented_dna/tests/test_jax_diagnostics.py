"""Optional CPU-only check against the already pinned exp586 runtime environment."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from exp593_dna.diagnostics import collect_norms, named_model_arrays


def test_global_norms_on_replicated_and_partitioned_logical_arrays():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    devices = jax.devices("cpu")
    mesh = Mesh(np.asarray(devices), ("test_mesh",))
    host = np.arange(32, dtype=np.float32).reshape(8, 2, 2)
    arrays = {
        ".stacked_blocks.stacked.attn.w_q": jax.device_put(
            host, NamedSharding(mesh, PartitionSpec("test_mesh", None, None))
        ),
        ".token_embed": jax.device_put(
            np.array([3.0, 4.0], dtype=np.float32), NamedSharding(mesh, PartitionSpec())
        ),
    }
    gradients = {k: v * 0.5 for k, v in arrays.items()}
    result = jax.jit(lambda p, g: collect_norms(p, g, xp=jnp))(arrays, gradients)
    expected = np.sqrt(np.square(host).sum() + 25)
    assert np.isclose(float(result["norms/param/overall"]), expected)
    assert np.isclose(float(result["norms/grad/overall"]), expected * 0.5)
    bf16 = named_model_arrays(
        {"weight": jnp.ones((2,), dtype=jnp.bfloat16)}, jax.tree_util
    )
    assert len(bf16) == 1
