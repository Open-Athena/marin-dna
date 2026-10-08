"""Exercise the native selected-position adapter with explicit eight-device sharding."""

import os
import subprocess
import sys


def test_explicit_sharding_in_both_scoring_modes():
    code = """
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.sharding import compact_grug_mesh
from jax.sharding import NamedSharding, PartitionSpec as P
from exp582_moe.eval_runtime import PilotScorer, dna_output_head
class FakeModel(eqx.Module):
    output_proj: jax.Array
    def __call__(self, ids, mask):
        return jnp.broadcast_to(ids[..., None].astype(jnp.float32), (*ids.shape, self.output_proj.shape[0])), {"capacity_overflow_per_layer": jnp.zeros((2,), dtype=jnp.int32)}
mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
with jax.set_mesh(mesh):
    model = FakeModel(jax.device_put(jnp.arange(128, dtype=jnp.float32).reshape(16, 8) / 128, NamedSharding(mesh, P(("data", "expert", "context"), "model"))))
    scorer = PilotScorer(model, mesh, [3,4,5,6])
    np.testing.assert_array_equal(dna_output_head(model, scorer.base_ids), np.asarray(model.output_proj)[:, [3,4,5,6]])
    for af in (False, True):
        for size in (8,3,1):
            ids = [[3,4,5,6]*2048 for _ in range(size)]
            positions = [[0,1,4095,4096,8191] for _ in range(size)]
            features = [4095+i%2 for i in range(size)]
            out = scorer.evaluate(ids, positions, features, allele_frequency=af)
            assert out["probabilities"].shape == (size,10,4)
            np.testing.assert_array_equal(out["probabilities"][:,0], 0)
            np.testing.assert_allclose(out["probabilities"][:,1:5].sum(-1),1,atol=1e-6)
            if af:
                assert out["valid"].all()
                for i in range(size):
                    np.testing.assert_array_equal(out["variant"][i], np.full(16,ids[i][features[i]]))
                np.testing.assert_allclose(out["mean"], 4.5)
    full = scorer.evaluate([[3,4,5,6]*2048], [list(range(8192))], [4096], allele_frequency=False)
    assert full["probabilities"].shape == (1,8192,4)
    np.testing.assert_array_equal(full["probabilities"][:,0], 0)
    np.testing.assert_allclose(full["probabilities"][:,1:].sum(-1), 1, atol=1e-6)
print("eight-device scoring passed")
"""
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
