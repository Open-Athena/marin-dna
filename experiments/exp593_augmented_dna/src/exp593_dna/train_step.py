"""Optimized FP32-master Grug update with sparse logical-global norm tracking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jmp
import optax
from experiments.grug.moe_hero_ep.model import apply_qb_betas
from experiments.grug.moe_hero_ep.train import GrugTrainState, _loss_and_grads
from jax.sharding import PartitionSpec as P

from exp593_dna.diagnostics import NormPolicy, model_norms

COMPILER_OPTIONS = {"xla_gpu_enable_pdl": False, "xla_gpu_enable_pdl_launch": False}


def make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    norm_policy: NormPolicy | None = None,
) -> Callable[..., Any]:
    """Retain perf5 optimizer placement and BF16 replicas with one reverse pass.

    The Boolean norm switch is dynamic, so the occasional diagnostic does not
    create a second compiled model for each sequence bucket.
    Gradients reach the norm hook before any optimizer transforms.
    """
    policy = norm_policy or NormPolicy()
    fp32 = jmp.get_policy("params=float32,compute=float32,output=float32")
    empty_norms = {
        f"norms/{category}/overall": jnp.float32(0) for category in ("param", "grad")
    }
    empty_norms.update(
        {
            f"norms/{category}/layer_{layer}/{projection}": jnp.float32(0)
            for category in ("param", "grad")
            for layer in policy.layers
            for projection in ("attention_q", "router", "expert_down")
        }
    )

    @partial(jax.jit, donate_argnums=(0,), compiler_options=COMPILER_OPTIONS)
    def step(
        state: GrugTrainState, batch: Any, collect_norms: jax.Array
    ) -> tuple[GrugTrainState, dict[str, Any], dict[str, jax.Array]]:
        if state.master_params is None or state.ema_params is not None:
            raise ValueError("Augmented training requires FP32 masters and no EMA")
        compute_params = apply_qb_betas(state.params, state.pending_qb_betas)
        (loss, routing), gradients = _loss_and_grads(compute_params, batch, mp, 1e-4)
        master_params = apply_qb_betas(state.master_params, state.pending_qb_betas)
        master_gradients = fp32.cast_to_param(gradients)
        norms = jax.lax.cond(
            collect_norms,
            lambda: model_norms(
                master_params, master_gradients, jax.tree_util, xp=jnp, policy=policy
            ),
            lambda: empty_norms,
        )
        updates, optimizer_state = optimizer.update(
            master_gradients, state.opt_state, master_params
        )
        master_params = optax.apply_updates(master_params, updates)
        compute_params = jax.tree.map(
            lambda value: jax.sharding.reshard(value, P()),
            mp.cast_to_param(master_params),
        )
        next_state = replace(
            state,
            step=state.step + jnp.asarray(1, dtype=jnp.int32),
            params=compute_params,
            master_params=master_params,
            opt_state=optimizer_state,
            pending_qb_betas=routing["qb_beta_per_layer"],
        )
        return next_state, {"train/loss": loss, **routing}, norms

    return step
