"""Preserve every metric and exact wide integer accounting after batched copies."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from levanter.grug.grug_moe import (
    MOE_DROPPED_ASSIGNMENTS_METRIC,
    MOE_RECEIVER_DROPPED_ASSIGNMENTS_METRIC,
    MOE_SENDER_DROPPED_ASSIGNMENTS_METRIC,
    MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC,
    MOE_VALID_ASSIGNMENTS_METRIC,
)

from exp586_moe.routing_probe import routing_diagnostics


def test_batched_metric_transfers_preserve_values_and_int64_sums(monkeypatch):
    model = SimpleNamespace(num_layers=6, num_experts_per_token=8)
    batch = 8192
    valid = jnp.full((6,), batch * 8192 * 8, dtype=jnp.int32)
    zeros = jnp.zeros((6,), dtype=jnp.int32)
    metrics = {
        MOE_DROPPED_ASSIGNMENTS_METRIC: zeros,
        MOE_RECEIVER_DROPPED_ASSIGNMENTS_METRIC: zeros,
        MOE_SENDER_DROPPED_ASSIGNMENTS_METRIC: zeros,
        MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC: zeros,
        MOE_VALID_ASSIGNMENTS_METRIC: valid,
        "train/router/routing_counts_per_layer": jnp.arange(
            6 * 384, dtype=jnp.int32
        ).reshape(6, 384),
        "train/loss": jnp.asarray(0.978, dtype=jnp.float32),
        "train/gradient_norm": jnp.asarray(0.015, dtype=jnp.float32),
        "train/router/bias_min": jnp.asarray(-3.0),
        "train/router/entropy": jnp.asarray(4.1),
        "unused_qb_tensor": jnp.ones((6, 384)),
    }
    original = routing_diagnostics(metrics, batch_size=batch, model_config=model)
    copied = {}
    get = jax.device_get

    def capture(values):
        copied.update(values)
        return get(values)

    monkeypatch.setattr(jax, "device_get", capture)
    candidate = routing_diagnostics(
        metrics, batch_size=batch, model_config=model, prefetch_to_host=True
    )
    assert original == candidate
    assert candidate[MOE_VALID_ASSIGNMENTS_METRIC] == 3_221_225_472
    assert "train/gradient_norm" in candidate
    assert "unused_qb_tensor" not in copied
    monkeypatch.setattr(jax, "device_get", get)
    packed = routing_diagnostics(
        metrics,
        batch_size=batch,
        model_config=model,
        prefetch_to_host=True,
        pack_floating_scalars=True,
    )
    assert packed == original
