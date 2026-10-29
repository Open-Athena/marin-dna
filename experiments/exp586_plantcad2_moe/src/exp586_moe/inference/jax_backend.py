"""Native JAX baseline and reduced-head inference paths."""

import dataclasses
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from rigging.filesystem.s3_compat import configure_coreweave_s3
from tokenizers import Tokenizer

from exp586_moe.config import d768_config
from exp586_moe.data import pretrained_dna_tokenizer, tokenizer_digest
from exp586_moe.inference.contract import (
    CHECKPOINT,
    CHECKPOINT_METADATA_DIGEST,
    TOKENIZER_SHA256,
    token_contract,
    validate_sequences,
)
from experiments.grug.moe_hero_ep.model import Transformer
from experiments.grug.moe_hero_ep.weights import restore_weights

JaxVariant = Literal["full_head", "acgt_head"]


def load_model() -> tuple[Transformer, jax.sharding.Mesh, list[int]]:
    """Restore one-device BF16 inference weights and authoritative ACGT IDs."""
    configure_coreweave_s3()
    if jax.device_count() != 1:
        raise ValueError("Single-H100 inference requires exactly one visible device")
    tokenizer = pretrained_dna_tokenizer()
    if tokenizer_digest(tokenizer) != TOKENIZER_SHA256:
        raise ValueError("Tokenizer differs from the trained checkpoint binding")
    ids = token_contract(tokenizer).output_ids
    config = dataclasses.replace(
        d768_config("pretrained"), moe_implementation="sonic", expert_chunks=1
    )
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    policy = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    with jax.set_mesh(mesh):
        model = eqx.filter_jit(policy.cast_to_param)(
            restore_weights(CHECKPOINT, CHECKPOINT_METADATA_DIGEST, config, mesh)
        )
        jax.block_until_ready(model)
    return model, mesh, [ids[base] for base in "ACGT"]


def compile_forward(
    model: Transformer,
    mesh: jax.sharding.Mesh,
    base_ids: list[int],
    *,
    variant: JaxVariant,
) -> Callable[[jax.Array], jax.Array]:
    """Compile either the deliberately naive full head or exact reduced head."""
    selected = np.asarray(base_ids, dtype=np.int32)

    if variant == "full_head":

        @eqx.filter_jit
        def forward(model: Transformer, token_ids: jax.Array) -> jax.Array:
            hidden, _ = model(token_ids, AttentionMask.causal())
            full = jnp.einsum(
                "bsh,hv->bsv",
                hidden,
                model.output_proj,
                preferred_element_type=jnp.float32,
                out_sharding=P(None, None, None),
            )
            return full

    elif variant == "acgt_head":

        @eqx.filter_jit
        def forward(model: Transformer, token_ids: jax.Array) -> jax.Array:
            hidden, _ = model(token_ids, AttentionMask.causal())
            head = model.output_proj[:, selected]
            return jnp.einsum(
                "bsh,hv->bsv",
                hidden,
                head,
                preferred_element_type=jnp.float32,
                out_sharding=P(None, None, None),
            )

    else:
        raise ValueError(f"Unknown JAX variant: {variant}")

    def bound(token_ids: jax.Array) -> jax.Array:
        with jax.set_mesh(mesh):
            return forward(model, token_ids)

    return bound


def encode(tokenizer: Tokenizer, sequences: list[str]) -> np.ndarray:
    """Use the production HF tokenizer contract for complete 8,192-base windows."""
    validate_sequences(sequences)
    encodings = tokenizer.encode_batch(sequences, add_special_tokens=False)
    ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int32)
    if ids.shape != (len(sequences), 8192):
        raise ValueError(f"Unexpected token shape: {ids.shape}")
    return ids
