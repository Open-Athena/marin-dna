from dataclasses import replace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import torch
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.qwen import Qwen3LMHeadModel
from marin_dna_exp550.launch import model_config, training_config
from marin_dna_exp550.recipe import encode_document
from transformers import AutoModelForCausalLM


def test_padding_cannot_change_real_logits_loss_or_gradients_and_hf_roundtrip(tmp_path):
    config = replace(
        model_config(),
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        head_dim=32,
        max_seq_len=512,
        gradient_checkpointing=False,
    )
    model = Qwen3LMHeadModel.init(
        hax.Axis("vocab", 8), config, key=jax.random.PRNGKey(0)
    )
    encoded = encode_document("ACGTN" * 51)

    def example(length, pad_id=0):
        ids = np.asarray(encoded["input_ids"][:length], dtype=np.int32)
        ids[256:] = pad_id
        pos = hax.Axis("position", length)
        return LmExample.causal(
            hax.named(jnp.asarray(ids), pos),
            loss_weight=hax.named(jnp.asarray(encoded["loss_weight"][:length]), pos),
        )

    short, padded, changed = example(256), example(512), example(512, 7)
    logits = eqx.filter_jit(lambda m, e: m(e.tokens, attn_mask=e.attn_mask))
    short_logits = np.asarray(logits(model, short).array)
    padded_logits = np.asarray(logits(model, padded).array)
    np.testing.assert_allclose(short_logits, padded_logits[:256], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(
        short_logits,
        np.asarray(logits(model, changed).array)[:256],
        atol=2e-5,
        rtol=2e-5,
    )
    loss = eqx.filter_jit(lambda m, e: m.compute_next_token_loss(e).array)
    np.testing.assert_allclose(
        loss(model, short), loss(model, padded), atol=2e-5, rtol=2e-5
    )
    gradient = eqx.filter_jit(
        eqx.filter_grad(lambda m, e: m.compute_next_token_loss(e).array)
    )
    first, second = gradient(model, padded), gradient(model, changed)
    for left, right in zip(
        jax.tree.leaves(first), jax.tree.leaves(second), strict=True
    ):
        np.testing.assert_allclose(left, right, atol=2e-5, rtol=2e-5)
    converter = config.hf_checkpoint_converter()
    with jax.set_mesh(jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))):
        converter.save_pretrained(model, str(tmp_path), save_reference_code=False)
    exported = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    with torch.inference_mode():
        actual = (
            exported(
                torch.tensor(np.asarray(short.tokens.array))[None],
                attention_mask=torch.ones(1, 256, dtype=torch.int64),
            )
            .logits[0]
            .float()
            .numpy()
        )
    np.testing.assert_allclose(actual, short_logits, atol=2e-4, rtol=2e-4)


def test_production_recipe_preserves_context_batch_and_schedule():
    config = training_config(
        "gs://marin-us-east1/MarinDNA/test", run_id="test", pilot=False, per_device=5
    )
    assert config.trainer.train_batch_size == 200
    assert config.trainer.num_train_steps == 100000
    assert config.trainer.steps_per_eval == config.hf_save_steps == 10000
    assert config.model.max_seq_len == config.train_seq_len == 10240
    assert config.model.to_hf_config(8).max_position_embeddings == 10240
    assert config.trainer.seed == 0 and config.data_seed == 42
    assert set(config.data.train_weights.values()) == {0.2}
    assert not config.data.auto_build_caches and not config.data.enforce_eos
    assert isinstance(
        LmExample.causal(hax.named(jnp.arange(8), "position")).attn_mask, AttentionMask
    )
