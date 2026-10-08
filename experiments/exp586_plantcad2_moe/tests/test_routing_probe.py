import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.grug_moe import moe_mlp
from levanter.grug.sharding import compact_grug_mesh

from exp586_moe.routing_probe import (
    RoutingProbeConfig,
    probe_model_config,
    recorded_config_equal,
    verify_replay_clock,
    verify_replay_reference,
)
from exp586_moe.schedule import TokenClock
from experiments.grug.moe_hero_ep.model import QbEstimator


def test_wandb_configuration_serialization_preserves_meaning():
    assert recorded_config_equal(
        {"qb": "HIST", "lr": 0.00010452773507394796},
        {"qb": QbEstimator.HIST, "lr": 0.00010452773507394797},
    )
    assert not recorded_config_equal({"lr": 1e-15}, {"lr": 2e-15})
    assert not recorded_config_equal({"qb": "TOPK"}, {"qb": QbEstimator.HIST})


def test_replay_recovery_requires_matching_endpoint_and_data_cursor():
    checkpoint = (
        "s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp586_plantcad2_moe/"
        "exp586-plantcad2-d768-pretrained-routing-cf32-long-smoke-v1/checkpoints/step-600"
    )
    config = RoutingProbeConfig(
        "pretrained",
        "exp586-recovery-smoke",
        "cw-us-east-02a",
        steps=600,
        replay_control=True,
        resume_replay_checkpoint=checkpoint,
        resume_metadata_digest="a" * 64,
        resume_reference_uri=checkpoint.rsplit("/", 3)[0] + "/replay-reference.json",
        resume_reference_sha256="b" * 64,
    )
    verify_replay_clock(config, TokenClock(updates=600, examples=4800))
    for clock in (
        TokenClock(updates=599, examples=4800),
        TokenClock(updates=600, examples=4792),
    ):
        with pytest.raises(ValueError, match="cursor"):
            verify_replay_clock(config, clock)
    for changes in (
        {"replay_control": False},
        {"condition": "scratch"},
        {"steps": 599},
        {"resume_metadata_digest": None},
        {"resume_replay_checkpoint": None},
        {"resume_reference_sha256": None},
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(config, **changes)

    original = dataclasses.asdict(config) | {
        "run_id": "exp586-plantcad2-d768-pretrained-routing-cf32-long-smoke-v1",
        "model": {"qk_mult": 1.3},
        "optimizer": {"beta2": 0.99},
        "reference_real_tokens_per_update": 30497,
    }
    reference = {
        "configuration": original,
        "source_run_id": original["run_id"],
        "checkpoint": checkpoint,
        "checkpoint_metadata_digest": "a" * 64,
    }
    verify_replay_reference(
        config, reference, original["model"], original["optimizer"], 30497
    )
    with pytest.raises(ValueError, match="capacity_factor"):
        verify_replay_reference(
            dataclasses.replace(config, capacity_factor=32),
            reference,
            original["model"],
            original["optimizer"],
            30497,
        )
    with pytest.raises(ValueError, match="resolved optimizer"):
        verify_replay_reference(
            config, reference, original["model"], {"beta2": 0.95}, 30497
        )
    with pytest.raises(ValueError, match="different checkpoint"):
        verify_replay_reference(
            config,
            reference | {"source_run_id": "another-run"},
            original["model"],
            original["optimizer"],
            30497,
        )


def test_probe_changes_dispatch_without_changing_architecture():
    control = RoutingProbeConfig("scratch", "exp586-test-smoke", "cw-us-east-02a")
    candidate = dataclasses.replace(control, backend="dropless")
    left, right = (
        dataclasses.asdict(probe_model_config(control)),
        dataclasses.asdict(probe_model_config(candidate)),
    )
    assert left.pop("moe_implementation") == "fixed_pooled_wave_all_to_all"
    assert right.pop("moe_implementation") == "scatter"
    assert left == right
    assert candidate.expert_axis_size == 1
    with pytest.raises(ValueError, match="Bounded"):
        dataclasses.replace(control, steps=1001)


def test_dropless_expert_output_and_gradients_under_concentrated_routing():
    # Every real token selects the same two experts; one row is padding.
    # This exceeds a balanced per-expert allocation by several times.
    x = jnp.arange(24, dtype=jnp.float32).reshape(6, 4) / 20
    up = jnp.sin(jnp.arange(8 * 4 * 6, dtype=jnp.float32)).reshape(8, 4, 6) / 5
    down = jnp.cos(jnp.arange(8 * 3 * 4, dtype=jnp.float32)).reshape(8, 3, 4) / 5
    selected = jnp.tile(jnp.asarray([[0, 1]]), (6, 1))
    combine = jnp.tile(jnp.asarray([[0.7, 0.3]]), (6, 1))
    valid = jnp.asarray([True] * 5 + [False])

    def reference(x, up, down):
        result = jnp.zeros_like(x)
        for expert, weight in ((0, 0.7), (1, 0.3)):
            gate, activation = jnp.split(x @ up[expert], 2, axis=-1)
            result += weight * ((jax.nn.silu(gate) * activation) @ down[expert])
        return jnp.where(valid[:, None], result, 0)

    def actual(x, up, down):
        return moe_mlp(
            x,
            selected,
            combine,
            up,
            down,
            token_valid=valid,
            implementation="scatter",
            report_capacity_overflow=True,
        )

    with jax.set_mesh(compact_grug_mesh()):
        output, counts = actual(x, up, down)
        np.testing.assert_allclose(output, reference(x, up, down), atol=1e-6)
        assert int(counts.sender_dropped) == int(counts.receiver_dropped) == 0
        assert int(counts.padding_skipped) == 2
        expected_grad = jax.grad(
            lambda x, u, d: jnp.sum(reference(x, u, d) ** 2), argnums=(0, 1, 2)
        )(x, up, down)
        observed_grad = jax.grad(
            lambda x, u, d: jnp.sum(actual(x, u, d)[0] ** 2), argnums=(0, 1, 2)
        )(x, up, down)
        for observed, expected in zip(observed_grad, expected_grad, strict=True):
            np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=1e-5)
