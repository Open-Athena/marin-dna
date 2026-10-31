"""Full model replication with local MoE kernels and ordinary data parallelism.

The size-one data/expert/model axes disable FSDP, EP, and tensor parallelism;
replica_dcn shards examples while autodiff reduces replicated parameter gradients.
"""

import dataclasses
from typing import Any

import jax
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
