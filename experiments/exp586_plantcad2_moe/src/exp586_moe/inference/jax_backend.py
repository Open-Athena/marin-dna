"""Native JAX baseline and reduced-head inference paths."""

import dataclasses
import math
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask, GrugAttentionImplementation
from levanter.grug.grug_moe import MoeImplementation
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
JaxOutputDtype = Literal["float32", "bfloat16"]


def configure_sconv_tiles(
    *,
    chunk: int | None = None,
    max_channel_block: int | None = None,
    num_warps: int | None = None,
    rows_per_step: int | None = None,
) -> None:
    """Override Levanter's forward-only short-convolution launch shape.

    The library default was tuned on GB200.  This benchmark targets H100, so keep
    the same kernel and expose only its documented launch-shape parameters.
    """
    if all(
        value is None for value in (chunk, max_channel_block, num_warps, rows_per_step)
    ):
        return

    from levanter.kernels.triton.short_conv import api as short_conv_api
    from levanter.kernels.triton.short_conv import triton_gpu

    current = triton_gpu.FORWARD_TILES
    updated = triton_gpu.TritonShortConvTiles(
        chunk=current.chunk if chunk is None else chunk,
        max_channel_block=(
            current.max_channel_block
            if max_channel_block is None
            else max_channel_block
        ),
        num_warps=current.num_warps if num_warps is None else num_warps,
        rows_per_step=current.rows_per_step if rows_per_step is None else rows_per_step,
    )
    if updated.chunk <= 0 or updated.rows_per_step <= 0:
        raise ValueError(f"Short-convolution tile sizes must be positive: {updated}")
    if updated.chunk % updated.rows_per_step:
        raise ValueError(f"rows_per_step must divide chunk: {updated}")
    triton_gpu.FORWARD_TILES = updated
    sequence_multiple = math.lcm(updated.chunk, triton_gpu.BACKWARD_TILES.chunk)
    triton_gpu.SEQUENCE_MULTIPLE = sequence_multiple
    short_conv_api.TRITON_SEQUENCE_MULTIPLE = sequence_multiple


def load_model(
    *,
    compact_vocab: bool = False,
    moe_implementation: MoeImplementation = "sonic",
    attention_implementation: GrugAttentionImplementation = "gpu_fa4_cute",
    expert_chunks: int = 1,
) -> tuple[Transformer, jax.sharding.Mesh, list[int]]:
    """Restore one-device BF16 inference weights and authoritative ACGT IDs."""
    configure_coreweave_s3()
    if jax.device_count() != 1:
        raise ValueError("Single-H100 inference requires exactly one visible device")
    tokenizer = pretrained_dna_tokenizer()
    if tokenizer_digest(tokenizer) != TOKENIZER_SHA256:
        raise ValueError("Tokenizer differs from the trained checkpoint binding")
    contract = token_contract(tokenizer)
    config = dataclasses.replace(
        d768_config("pretrained"),
        moe_implementation=moe_implementation,
        attention_implementation=attention_implementation,
        expert_chunks=expert_chunks,
    )
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    policy = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    with jax.set_mesh(mesh):
        model = eqx.filter_jit(policy.cast_to_param)(
            restore_weights(CHECKPOINT, CHECKPOINT_METADATA_DIGEST, config, mesh)
        )
        jax.block_until_ready(model)
        if compact_vocab:
            input_ids = jnp.asarray(
                [contract.input_ids[base] for base in "ACGTN"], dtype=jnp.int32
            )
            output_ids = jnp.asarray(
                [contract.output_ids[base] for base in "ACGT"], dtype=jnp.int32
            )
            compact_embed = model.token_embed.at[input_ids].get(
                out_sharding=P(None, None)
            )
            compact_output = model.output_proj.at[:, output_ids].get(
                out_sharding=P(None, None)
            )
            model = eqx.tree_at(
                lambda tree: (tree.token_embed, tree.output_proj),
                model,
                (compact_embed, compact_output),
            )
            jax.block_until_ready(model)
    base_ids = (
        list(range(4))
        if compact_vocab
        else [contract.output_ids[base] for base in "ACGT"]
    )
    return model, mesh, base_ids


def compile_forward(
    model: Transformer,
    mesh: jax.sharding.Mesh,
    base_ids: list[int],
    *,
    variant: JaxVariant,
    output_dtype: JaxOutputDtype = "float32",
) -> Callable[[jax.Array], jax.Array]:
    """Compile either the deliberately naive full head or exact reduced head."""
    selected = np.asarray(base_ids, dtype=np.int32)
    resolved_output_dtype = {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
    }[output_dtype]

    if variant == "full_head":

        @eqx.filter_jit
        def forward(model: Transformer, token_ids: jax.Array) -> jax.Array:
            hidden, _ = model(
                token_ids,
                AttentionMask.causal(),
                collect_router_metrics=False,
            )
            full = jnp.einsum(
                "bsh,hv->bsv",
                hidden,
                model.output_proj,
                preferred_element_type=jnp.float32,
                out_sharding=P(None, None, None),
            )
            return full.astype(resolved_output_dtype)

    elif variant == "acgt_head":

        @eqx.filter_jit
        def forward(model: Transformer, token_ids: jax.Array) -> jax.Array:
            hidden, _ = model(
                token_ids,
                AttentionMask.causal(),
                collect_router_metrics=False,
            )
            head = model.output_proj.at[:, selected].get(out_sharding=P(None, None))
            logits = jnp.einsum(
                "bsh,hv->bsv",
                hidden,
                head,
                preferred_element_type=jnp.float32,
                out_sharding=P(None, None, None),
            )
            return logits.astype(resolved_output_dtype)

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
