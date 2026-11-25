"""Optional FP32 router control for the native mixed-precision training step."""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jmp

from experiments.grug.moe_hero_ep.model import Transformer


class RouterFloat32Policy(jmp.Policy):
    """Retain router weights and pending balancing biases through both casts.

    The native model's router einsum promotes its BF16 activations against the
    FP32 weights, so its logits reach top-k without BF16 output rounding.
    All remaining weights and activations follow the inherited BF16 policy.
    """

    @staticmethod
    def _preserve_router(original: Any, cast: Any) -> Any:
        if not isinstance(original, Transformer):
            return cast
        router = original.stacked_blocks.stacked.mlp
        return eqx.tree_at(
            lambda model: (
                model.stacked_blocks.stacked.mlp.router,
                model.stacked_blocks.stacked.mlp.router_bias,
            ),
            cast,
            (router.router.astype(jnp.float32), router.router_bias.astype(jnp.float32)),
        )

    def cast_to_param(self, x: Any) -> Any:
        return self._preserve_router(x, super().cast_to_param(x))

    def cast_to_compute(self, x: Any) -> Any:
        return self._preserve_router(x, super().cast_to_compute(x))
