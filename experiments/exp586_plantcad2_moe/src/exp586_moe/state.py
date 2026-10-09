"""Fresh adaptation state and exact checkpoint metadata."""

import dataclasses
import faulthandler
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import optax
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import CheckpointDebugConfig, load_checkpoint, save_checkpoint
from rigging.filesystem.storage_path import StoragePath

from exp586_moe.schedule import TokenClock
from experiments.grug.moe_hero_ep.model import Transformer
from experiments.grug.moe_hero_ep.train import GrugTrainState, _tree_to_memory_kind


@eqx.filter_jit
def state_all_finite(state: GrugTrainState) -> jax.Array:
    """Check weights, optimizer, and pending router state after the last update."""
    state = _tree_to_memory_kind(state, "device")
    return jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(state)])
    )


@eqx.filter_jit
def state_max_abs_error(a: GrugTrainState, b: GrugTrainState) -> jax.Array:
    """Compare all state arrays, lowering host/device transfers inside JIT.

    The native transfer helper uses abstract shardings and cannot be called
    eagerly on distributed arrays.
    """
    a = _tree_to_memory_kind(a, "device")
    b = _tree_to_memory_kind(b, "device")
    differences = jax.tree.leaves(
        jax.tree.map(
            lambda x, y: jnp.max(
                jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32))
            ),
            a,
            b,
        )
    )
    return jnp.max(jnp.stack(differences))


@eqx.filter_jit
def state_error_by_group(a: GrugTrainState, b: GrugTrainState) -> dict[str, jax.Array]:
    """Locate replay divergence among weights, optimizer state, and router state."""
    result = {}
    for name in ("params", "master_params", "opt_state", "pending_qb_betas", "step"):
        left = _tree_to_memory_kind(getattr(a, name), "device")
        right = _tree_to_memory_kind(getattr(b, name), "device")
        pairs = list(zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True))
        if not pairs:
            continue
        errors = [x.astype(jnp.float32) - y.astype(jnp.float32) for x, y in pairs]
        result[name + "/max_abs"] = jnp.max(
            jnp.stack([jnp.max(jnp.abs(e)) for e in errors])
        )
        result[name + "/relative_l2"] = jnp.sqrt(
            sum(jnp.sum(e * e) for e in errors)
            / jnp.maximum(
                sum(jnp.sum(x.astype(jnp.float32) ** 2) for x, _ in pairs), 1e-30
            )
        )
    return result


def fresh_state(
    weights: Transformer,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    offload: bool,
) -> GrugTrainState:
    """Start a new optimizer while retaining the effective source router biases."""
    master = jmp.get_policy(
        "params=float32,compute=float32,output=float32"
    ).cast_to_param(weights)
    opt_state = optimizer.init(master)
    # The native step always replaces router bias from pending QB, including its
    # first step. Zeroing pending here would erase the restored effective bias.
    pending = -master.stacked_blocks.stacked.mlp.router_bias
    return GrugTrainState(
        step=jax.sharding.reshard(jnp.asarray(0, dtype=jnp.int32), P()),
        params=mp.cast_to_param(master),
        master_params=_tree_to_memory_kind(master, "pinned_host") if offload else None,
        opt_state=_tree_to_memory_kind(opt_state, "pinned_host")
        if offload
        else opt_state,
        ema_params=None,
        pending_qb_betas=pending,
    )


def save_dna_checkpoint(
    state: GrugTrainState,
    clock: TokenClock,
    path: str,
    *,
    data_seed: int,
    tokenizer_sha256: str | None = None,
    scientific_config_sha256: str | None = None,
    runtime_moe_implementation: str | None = None,
    is_temporary: bool = False,
) -> None:
    if int(state.step) != clock.updates:
        raise ValueError("Model step and token/data clock disagree")
    debug = os.environ.get("EXP586_CHECKPOINT_DEBUG") == "1"
    if debug:
        faulthandler.enable(all_threads=True)
    save_checkpoint(
        state,
        clock.updates,
        path,
        is_temporary=is_temporary,
        debug=CheckpointDebugConfig(
            enabled=debug,
            flush_logs=True,
            force_gc_before_serialize=False,
            tracemalloc_frames=None,
        ),
        metadata={
            "exp586_clock": dataclasses.asdict(clock),
            "data_seed": data_seed,
            "exp586_schema": 1,
            "tokenizer_sha256": tokenizer_sha256,
            "scientific_config_sha256": scientific_config_sha256,
            "runtime_moe_implementation": runtime_moe_implementation,
        },
    )


def restore_dna_checkpoint(
    exemplar: GrugTrainState,
    path: str,
    mesh: jax.sharding.Mesh,
    *,
    data_seed: int,
    tokenizer_sha256: str | None = None,
) -> tuple[GrugTrainState, TokenClock]:
    import json

    metadata = json.loads((StoragePath(path) / "metadata.json").read_text())
    if (
        tokenizer_sha256 is not None
        and metadata.get("tokenizer_sha256") != tokenizer_sha256
    ):
        raise ValueError("Checkpoint tokenizer digest mismatch")
    if metadata.get("exp586_schema") != 1 or metadata.get("data_seed") != data_seed:
        raise ValueError("Checkpoint schema or data seed mismatch")
    clock = TokenClock(**metadata["exp586_clock"])
    if metadata["step"] != clock.updates:
        raise ValueError("Checkpoint step and token clock disagree")
    # Upstream restore_template_from deletes the source arrays to release memory.
    # This helper also serves replay comparisons, so construct a non-destructive template.
    template = jax.tree.map(
        lambda x: (
            jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)
            if isinstance(x, jax.Array)
            else x
        ),
        exemplar,
    )
    restored = load_checkpoint(template, path, mesh=mesh, allow_partial=False)
    if int(restored.step) != clock.updates:
        raise ValueError("Restored step and metadata disagree")
    return restored, clock
