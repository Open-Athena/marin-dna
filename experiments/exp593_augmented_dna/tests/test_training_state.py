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
