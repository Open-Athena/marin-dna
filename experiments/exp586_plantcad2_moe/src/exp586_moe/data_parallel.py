"""Full model replication with local MoE kernels and ordinary data parallelism.

The size-one data/expert/model axes disable FSDP, EP, and tensor parallelism;
replica_dcn shards examples while autodiff reduces replicated parameter gradients.
"""

import dataclasses
import hashlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import compact_grug_mesh

from exp586_moe.data import PRETRAINED_DNA_INPUT_TOKEN_IDS
from experiments.grug.moe_hero_ep.model import GrugModelConfig


@dataclasses.dataclass(frozen=True)
class RuntimeModelConfig(GrugModelConfig):
    """Operational kernels that preserve the scientific model contract."""

    compact_input_lookup_ids: tuple[int, ...] = ()
    compact_input_lookup_one_hot: bool = False
    expert_compute_fast_rows: int | tuple[int, ...] = 0


def replicated_mesh() -> jax.sharding.Mesh:
    """Give every GPU a complete model and optimizer replica."""
    return compact_grug_mesh(expert_axis_size=1, replica_axis_size=jax.device_count())


def local_model(
    source: GrugModelConfig, *, implementation: str = "scatter"
) -> RuntimeModelConfig:
    """Change only transport/kernel choices, preserving every learned array."""
    if implementation not in ("scatter", "sonic"):
        raise ValueError("Only H100-compatible local backends are allowed")
    values = {
        field.name: getattr(source, field.name) for field in dataclasses.fields(source)
    }
    values.update(
        moe_implementation=implementation,
        pooled_transport_capacity_factor=None,
        num_expert_waves=1,
        expert_compute_fast_rows=0,
        compact_input_lookup_ids=PRETRAINED_DNA_INPUT_TOKEN_IDS,
        compact_input_lookup_one_hot=True,
    )
    return RuntimeModelConfig(**values)


def assert_replicated_state(state: Any) -> dict[str, int]:
    """Reject hidden weight/optimizer partitioning and count bytes per GPU."""
    sizes = {}
    for name in ("params", "master_params", "opt_state", "pending_qb_betas", "step"):
        arrays = [
            x for x in jax.tree.leaves(getattr(state, name)) if isinstance(x, jax.Array)
        ]
        if any(not x.sharding.is_fully_replicated for x in arrays):
            raise ValueError(f"{name} contains a partitioned array")
        sizes[name] = sum(x.size * x.dtype.itemsize for x in arrays)
    return sizes


def replica_fingerprints(state: Any, mesh: jax.sharding.Mesh) -> np.ndarray:
    """Fingerprint every persistent array on every physical GPU replica."""
    arrays = tuple(x for x in jax.tree.leaves(state) if isinstance(x, jax.Array))
    if not arrays or any(not array.sharding.is_fully_replicated for array in arrays):
        raise ValueError("Physical replica checks require fully replicated arrays")
    if any(array.dtype.itemsize not in (1, 2, 4) for array in arrays):
        raise ValueError("Unsupported persistent-array representation")

    def local(values: tuple[jax.Array, ...]) -> jax.Array:
        fingerprints = []
        for value in values:
            unsigned = {1: jnp.uint8, 2: jnp.uint16, 4: jnp.uint32}[
                value.dtype.itemsize
            ]
            bits = jax.lax.bitcast_convert_type(value, unsigned).reshape(-1)
            bits = bits.astype(jnp.uint32)
            weights = jnp.arange(bits.size, dtype=jnp.uint32) * jnp.uint32(
                2654435761
            ) + jnp.uint32(2246822519)
            fingerprints.append(
                jnp.stack(
                    (
                        jnp.sum(bits, dtype=jnp.uint32),
                        jnp.bitwise_xor.reduce(bits),
                        jnp.sum(bits * weights, dtype=jnp.uint32),
                    )
                )
            )
        return jnp.concatenate(fingerprints)[None, :]

    compute = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(tuple(P() for _ in arrays),),
            out_specs=P("replica_dcn", None),
            check_vma=False,
        )
    )
    result = compute(tuple(jax.sharding.reshard(array, P()) for array in arrays))
    return np.asarray(multihost_utils.process_allgather(result, tiled=True))


def physical_projection_snapshot(
    value: jax.Array, mesh: jax.sharding.Mesh
) -> dict[str, Any]:
    """Independently hash and measure each physical FP32 projection copy."""
    devices = mesh.devices.reshape(-1).tolist()
    ranks, digests, norms = [], [], []
    for shard in value.addressable_shards:
        host = np.asarray(shard.data)
        ranks.append(devices.index(shard.device))
        digests.append(
            np.frombuffer(hashlib.sha256(host.tobytes()).digest(), dtype=np.uint8)
        )
        norms.append(
            np.sqrt(np.sum(np.square(host, dtype=np.float64), dtype=np.float64))
        )
    gather = lambda x: np.asarray(
        multihost_utils.process_allgather(np.asarray(x), tiled=True)
    )
    physical_ranks = gather(ranks)
    order = np.argsort(physical_ranks)
    if not np.array_equal(physical_ranks[order], np.arange(len(devices))):
        raise ValueError("Physical projection check missed or duplicated a GPU")
    return {
        "sha256_by_rank": [bytes(row).hex() for row in gather(digests)[order]],
        "norm_by_rank": gather(norms)[order].tolist(),
    }


def assert_physical_replicas(state: Any, mesh: jax.sharding.Mesh) -> dict[str, Any]:
    """Reject any physical persistent-array or master-projection disagreement."""
    arrays = [
        (jax.tree_util.keystr(path), value)
        for path, value in jax.tree_util.tree_leaves_with_path(state)
        if isinstance(value, jax.Array)
    ]
    values = replica_fingerprints(state, mesh)
    if values.shape != (mesh.shape["replica_dcn"], 3 * len(arrays)):
        raise ValueError(f"Unexpected replica fingerprint layout: {values.shape}")
    mismatches = []
    for index, (path, value) in enumerate(arrays):
        fingerprints = values[:, 3 * index : 3 * (index + 1)]
        differing = np.flatnonzero(np.any(fingerprints != fingerprints[0], axis=1))
        if differing.size:
            mismatches.append(
                {
                    "path": path,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "differing_replicas": differing.tolist(),
                }
            )
    if mismatches:
        raise ValueError(
            f"Persistent arrays differ across physical replicas: {mismatches}"
        )
    projection = physical_projection_snapshot(state.master_params.output_proj, mesh)
    if len(set(projection["sha256_by_rank"])) != 1:
        raise ValueError(
            "FP32 output-projection copies differ across physical replicas"
        )
    norms = projection["norm_by_rank"]
    return {
        "arrays_checked": len(arrays),
        "fingerprints_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "master_projection_sha256": projection["sha256_by_rank"][0],
        "master_projection_norm_min": min(norms),
        "master_projection_norm_max": max(norms),
    }
