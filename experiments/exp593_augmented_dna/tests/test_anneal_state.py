from dataclasses import replace

import pytest

from exp593_dna.anneal import TRIALS
from exp593_dna.anneal_state import AnnealClock, restore, save, validate_metadata
from exp593_dna.schedule import TokenClock


def test_full_optimizer_roundtrip_preserves_global_history(tmp_path):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from exp586_moe.data_parallel import replicated_mesh
    from experiments.grug.moe_hero_ep.train import GrugTrainState
    from jax.sharding import PartitionSpec as P

    trial = TRIALS["181b"]
    mesh = replicated_mesh()
    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=0.0123)
    with jax.set_mesh(mesh):
        masters = {
            "projection": jax.sharding.reshard(
                jnp.arange(12, dtype=jnp.float32).reshape(3, 4), P()
            )
        }
        gradients = jax.tree.map(lambda x: x + 0.2, masters)
        _, opt_state = optimizer.update(gradients, optimizer.init(masters), masters)
        inner = opt_state.inner_state
        inner = (
            inner[0]._replace(count=jnp.asarray(trial.source_update + 1)),
            *inner[1:],
        )
        opt_state = opt_state._replace(
            count=jnp.asarray(trial.source_update + 1), inner_state=inner
        )
        state = GrugTrainState(
            step=jax.sharding.reshard(
                jnp.asarray(trial.source_update + 1, dtype=jnp.int32), P()
            ),
            params=jax.tree.map(lambda x: x.astype(jnp.bfloat16), masters),
            master_params=masters,
            opt_state=opt_state,
            ema_params=None,
            pending_qb_betas=jax.sharding.reshard(
                jnp.array([[0.1, -0.2]], dtype=jnp.float32), P()
            ),
        )
        stage = TokenClock().advance(
            dna_tokens=131000,
            text_tokens=72,
            loss_targets=64,
            examples=64,
            fim_examples=64,
            rc_examples=31,
        )
        clock = AnnealClock(stage, {"annotation_guided": 64})
        path = str(tmp_path / "full-state")
        bindings = {"trial": "181b", "source": "old1t"}
        save(state, path, trial, clock, bindings, permanent=True)
        template = replace(
            state, step=jnp.array(0, dtype=jnp.int32), opt_state=optimizer.init(masters)
        )
        restored, actual_clock = restore(template, path, trial, mesh, bindings)
        assert actual_clock == clock
        assert (
            int(restored.step)
            == int(restored.opt_state.count)
            == trial.source_update + 1
        )
        for expected, actual in zip(
            jax.tree.leaves(state), jax.tree.leaves(restored), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
        with pytest.raises(ValueError, match="overwrite"):
            save(state, path, trial, clock, bindings, permanent=True)


def test_new_checkpoint_rejects_unguided_fim_and_reset_counters():
    from dataclasses import asdict

    trial = TRIALS["181b"]
    stage = TokenClock().advance(
        dna_tokens=131072,
        text_tokens=0,
        loss_targets=64,
        examples=64,
        fim_examples=64,
        rc_examples=31,
    )
    bindings = {"version": "new"}
    metadata = {
        "exp593_anneal_schema": 1,
        "exp593_bindings": bindings,
        "step": trial.source_update + 1,
        "exp593_anneal_clock": asdict(AnnealClock(stage)),
    }
    with pytest.raises(ValueError, match="not annotation-guided"):
        validate_metadata(metadata, trial, bindings)
    metadata["exp593_anneal_clock"]["mixture"] = {"annotation_guided": 64}
    assert validate_metadata(metadata, trial, bindings).stage == stage
    metadata["step"] = 1
    with pytest.raises(ValueError, match="clocks differ"):
        validate_metadata(metadata, trial, bindings)
