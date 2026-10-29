"""Export the pinned exp586 checkpoint once for PyTorch and vLLM consumers."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
from levanter.grug.sharding import compact_grug_mesh
from rigging.filesystem.s3_compat import configure_coreweave_s3

from exp586_moe.config import d768_config
from exp586_moe.data import pretrained_dna_tokenizer, tokenizer_digest
from exp586_moe.inference.contract import (
    CHECKPOINT,
    CHECKPOINT_METADATA_DIGEST,
    TOKENIZER_SHA256,
)
from experiments.grug.moe_hero_ep.weights import restore_weights

DEFAULT_EXPORT = (
    "s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/"
    "inference-optimization/v1/hf-bf16"
)


def export_hf(output: str = DEFAULT_EXPORT) -> None:
    """Write an authoritative BF16 HF/vLLM export from one visible accelerator."""
    configure_coreweave_s3()
    if jax.device_count() != 1:
        raise ValueError("Export requires exactly one visible accelerator")
    tokenizer = pretrained_dna_tokenizer()
    actual_tokenizer_digest = tokenizer_digest(tokenizer)
    if actual_tokenizer_digest != TOKENIZER_SHA256:
        raise ValueError("Pretrained DNA tokenizer differs from the pinned training tokenizer")
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
        converter = (
            config.hf_checkpoint_converter()
            .replaced(tokenizer=tokenizer)
            .with_config_overrides(
                {
                    "dtype": "bfloat16",
                    "exp586_checkpoint": CHECKPOINT,
                    "exp586_checkpoint_metadata_digest": CHECKPOINT_METADATA_DIGEST,
                    "exp586_tokenizer_sha256": TOKENIZER_SHA256,
                }
            )
        )
        converter.save_pretrained(
            model,
            output,
            dtype=jnp.bfloat16,
            save_reference_code=False,
            max_shard_size=1_000_000_000,
            export_host_budget_bytes=8 << 30,
            max_concurrent_shards=2,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(export_hf)
