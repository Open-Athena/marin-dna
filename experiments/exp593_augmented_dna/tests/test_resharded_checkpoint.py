"""Exercise native full-state reslicing on virtual devices, without GPU allocation."""

import os
import subprocess
import sys
import textwrap


def test_native_checkpoint_restores_across_replica_mesh_sizes(tmp_path):
    program = textwrap.dedent(
        """
        import json
        import sys
        from pathlib import Path

        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        from experiments.grug.moe_hero_ep.train import GrugTrainState
        from jax.sharding import AxisType, Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        from exp593_dna.schedule import TokenClock
        from exp593_dna.state import checkpoint_template, restore_training_checkpoint, save_training_checkpoint

        devices = np.asarray(jax.devices())
        assert len(devices) == 4
        axes = ('replica_dcn', 'data', 'expert', 'model')
        types = (AxisType.Explicit,) * 4
        one = Mesh(devices.reshape(1, 4, 1, 1), axes, axis_types=types)
        two = Mesh(devices.reshape(2, 2, 1, 1), axes, axis_types=types)
        optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=0.1, momentum=0.9)

        def make_state(mesh, offset):
            with jax.set_mesh(mesh):
                master = {
                    'expert_bank': jax.device_put(
                        np.arange(32, dtype=np.float32).reshape(8, 4) + offset,
                        NamedSharding(mesh, P('replica_dcn', None)),
                    ),
                    'attention': jax.device_put(
                        np.arange(16, dtype=np.float32).reshape(4, 4) - offset,
                        NamedSharding(mesh, P()),
                    ),
                }
                gradient = jax.tree.map(lambda x: x * 0.01 + 1, master)
                update, opt_state = optimizer.update(gradient, optimizer.init(master), master)
                master = optax.apply_updates(master, update)
                # Production keeps BF16 expert weights replicated, but shards
                # their FP32 masters and momentum across replica_dcn.
                parameters = jax.tree.map(
                    lambda x: jax.sharding.reshard(x.astype(jnp.bfloat16), P()), master
                )
                return GrugTrainState(
                    step=jax.device_put(np.array(1, dtype=np.int32), NamedSharding(mesh, P())),
                    params=parameters,
                    master_params=master,
                    opt_state=opt_state,
                    ema_params=None,
                    pending_qb_betas=jax.device_put(
                        np.arange(16, dtype=np.float32).reshape(8, 2) / 100,
                        NamedSharding(mesh, P()),
                    ),
                )

        def assert_same(expected, actual):
            for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
                assert a.dtype == b.dtype
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

        clock = TokenClock().advance(
            dna_tokens=327680, text_tokens=4096, loss_targets=16400,
            examples=64, fim_examples=32, rc_examples=31,
        )
        bindings = {'tokenizer': 'frozen-tokenizer', 'inputs': 'frozen-input-manifest'}
        root = Path(sys.argv[1])
        with jax.set_mesh(one):
            original = make_state(one, 0)
            save_training_checkpoint(original, clock, str(root / 'one'), bindings)
        with jax.set_mesh(two):
            different_template = make_state(two, 1000)
            # The standby frees its disposable state but retains global restore
            # shapes/shardings and the already-compiled update executable.
            freed_template = checkpoint_template(different_template, release=True)
            assert all(x.is_deleted() for x in jax.tree.leaves(different_template) if isinstance(x, jax.Array))
            restored, restored_clock = restore_training_checkpoint(
                freed_template, str(root / 'one'), two, bindings,
            )
            assert_same(original, restored)
            assert restored_clock == clock
            assert restored.master_params['expert_bank'].sharding.mesh == two
            assert restored.master_params['expert_bank'].sharding.spec == P('replica_dcn', None)
            assert restored.params['expert_bank'].sharding.is_equivalent_to(NamedSharding(two, P()), ndim=2)
            assert len({str(s.index) for s in restored.master_params['expert_bank'].addressable_shards}) == 2
            assert len({str(s.index) for s in restored.params['expert_bank'].addressable_shards}) == 1
            save_training_checkpoint(restored, clock, str(root / 'two'), bindings)
        with jax.set_mesh(one):
            roundtrip, roundtrip_clock = restore_training_checkpoint(
                make_state(one, -1000), str(root / 'two'), one, bindings,
            )
            assert_same(original, roundtrip)
            assert roundtrip_clock == clock
            assert roundtrip.master_params['expert_bank'].sharding.mesh == one
            assert len({str(s.index) for s in roundtrip.master_params['expert_bank'].addressable_shards}) == 1
        print(json.dumps({'virtual_devices': 4, 'replica_mesh_sizes': [1, 2, 1], 'all_state_values_equal': True}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"all_state_values_equal": true' in result.stdout
