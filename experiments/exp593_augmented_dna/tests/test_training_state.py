from dataclasses import asdict

import pytest

from exp593_dna.schedule import TokenClock
from exp593_dna.state import validate_checkpoint_metadata


def test_checkpoint_metadata_binds_tokenizer_inputs_and_variable_fim_counters():
    bindings = {
        "tokenizer": "tokenizer-sha",
        "inputs": "input-manifest-sha",
        "scientific_config": "objective-sha",
    }
    clock = TokenClock().advance(
        dna_tokens=8000,
        text_tokens=192,
        loss_targets=2,
        examples=1,
        fim_examples=1,
        rc_examples=1,
    )
    metadata = {
        "exp593_schema": 1,
        "exp593_bindings": bindings,
        "exp593_clock": asdict(clock),
        "step": 1,
    }
    assert validate_checkpoint_metadata(metadata, bindings) == clock
    for changed in [
        {**bindings, "tokenizer": "different"},
        {**bindings, "inputs": "different"},
        {**bindings, "scientific_config": "different"},
    ]:
        with pytest.raises(ValueError, match="bindings"):
            validate_checkpoint_metadata(metadata, changed)
    with pytest.raises(ValueError, match="clock"):
        validate_checkpoint_metadata({**metadata, "step": 2}, bindings)
    with pytest.raises(ValueError, match="partition"):
        validate_checkpoint_metadata(
            {**metadata, "exp593_clock": {**asdict(clock), "input_tokens": 9000}},
            bindings,
        )


def test_native_full_state_roundtrip_preserves_optimizer_and_fim_clock(tmp_path):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from exp586_moe.data_parallel import replicated_mesh
    from experiments.grug.moe_hero_ep.train import GrugTrainState
    from jax.sharding import PartitionSpec as P

    from exp593_dna.state import restore_training_checkpoint, save_training_checkpoint

    mesh = replicated_mesh()
    with jax.set_mesh(mesh):
        parameters = {
            "weights": jax.sharding.reshard(
                jnp.arange(12, dtype=jnp.float32).reshape(3, 4), P()
            )
        }
        optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=0.1)
        updates, opt_state = optimizer.update(
            jax.tree.map(jnp.ones_like, parameters),
            optimizer.init(parameters),
            parameters,
        )
        master = optax.apply_updates(parameters, updates)
        state = GrugTrainState(
            step=jax.sharding.reshard(jnp.array(1, dtype=jnp.int32), P()),
            params=jax.tree.map(lambda x: x.astype(jnp.bfloat16), master),
            master_params=master,
            opt_state=opt_state,
            ema_params=None,
            pending_qb_betas=jax.sharding.reshard(
                jnp.ones((8, 2), dtype=jnp.float32), P()
            ),
        )
        clock = TokenClock().advance(
            dna_tokens=8000,
            text_tokens=192,
            loss_targets=2,
            examples=1,
            fim_examples=1,
            rc_examples=1,
        )
        bindings = {"inputs": "frozen-manifest-sha", "tokenizer": "original-tokenizer"}
        path = str(tmp_path / "checkpoint")
        save_training_checkpoint(state, clock, path, bindings)
        with pytest.raises(ValueError, match="overwrite"):
            save_training_checkpoint(state, clock, path, bindings)
        restored, restored_clock = restore_training_checkpoint(
            state, path, mesh, bindings
        )
        assert restored_clock == clock
        for before, after in zip(
            jax.tree.leaves(state), jax.tree.leaves(restored), strict=True
        ):
            assert before.dtype == after.dtype and np.array_equal(
                np.asarray(before), np.asarray(after)
            )


def test_weight_only_import_discards_source_optimizer_and_resets_counters(tmp_path):
    import json

    import jax
    import jax.numpy as jnp
    import jmp
    import numpy as np
    import optax
    from exp586_moe.data_parallel import replicated_mesh
    from exp586_moe.state import fresh_state
    from experiments.grug.moe_hero_ep.model import (
        GrugModelConfig,
        Transformer,
        apply_qb_betas,
    )
    from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
    from experiments.grug.moe_hero_ep.train import GrugTrainState
    from experiments.grug.moe_hero_ep.weights import restore_weights
    from jax.sharding import PartitionSpec as P

    from exp593_dna.state import save_training_checkpoint

    config = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        num_experts=4,
        num_experts_per_token=1,
        max_seq_len=16,
        sliding_window=8,
    )
    mesh = replicated_mesh()
    policy = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=0.1, momentum=0.9)
    with jax.set_mesh(mesh):
        master = Transformer.init(config, key=jax.random.PRNGKey(0))
        opt_state = optimizer.init(master)
        for _ in range(3):
            gradient = jax.tree.map(lambda p: jnp.ones_like(p) * 0.01, master)
            updates, opt_state = optimizer.update(gradient, opt_state, master)
            master = optax.apply_updates(master, updates)
        pending = jax.sharding.reshard(jnp.arange(4, dtype=jnp.float32)[None], P())
        source = GrugTrainState(
            step=jax.sharding.reshard(jnp.array(3, dtype=jnp.int32), P()),
            params=policy.cast_to_param(master),
            master_params=master,
            opt_state=opt_state,
            ema_params=None,
            pending_qb_betas=pending,
        )
        clock = TokenClock(
            input_tokens=3 * 64 * 2048,
            dna_tokens=3 * 64 * 2048,
            loss_targets=3 * 64 * 2047,
            examples=3 * 64,
            updates=3,
        )
        path = tmp_path / "source"
        save_training_checkpoint(source, clock, str(path), {"source": "test"})
        metadata = json.loads((path / "metadata.json").read_text())
        imported = restore_weights(str(path), digest(metadata), config, mesh)
        fresh = fresh_state(
            imported, optimizer, policy, offload=False, retain_master=True
        )
        assert int(source.opt_state.count) == 3
        assert int(fresh.step) == int(fresh.opt_state.count) == 0
        expected_weights = apply_qb_betas(master, pending)
        for expected, actual in zip(
            jax.tree.leaves(expected_weights),
            jax.tree.leaves(fresh.master_params),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
        for expected, actual in zip(
            jax.tree.leaves(optimizer.init(expected_weights)),
            jax.tree.leaves(fresh.opt_state),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
        # Effective router bias is part of the learned model, not momentum.
        effective_bias = expected_weights.stacked_blocks.stacked.mlp.router_bias
        np.testing.assert_array_equal(
            np.asarray(fresh.pending_qb_betas), -np.asarray(effective_bias)
        )
