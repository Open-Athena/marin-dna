"""Bounded, dropless frozen hidden-state extraction without a vocabulary head."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask


class FrozenFeatures:
    """Extract last-real-token and real-token mean vectors, never full token arrays."""

    def __init__(self, model: Any, mesh: Any, *, sequence_length: int) -> None:
        self.model, self.mesh = model, mesh
        self.sequence_length = sequence_length
        self.batch_size = mesh.devices.size

        @eqx.filter_jit
        def forward(
            model: Any, ids: jax.Array, segments: jax.Array, lengths: jax.Array
        ) -> tuple[jax.Array, ...]:
            hidden, metrics = model(
                ids, AttentionMask.causal().with_segment_ids(segments)
            )
            last = jnp.take_along_axis(hidden, (lengths - 1)[:, None, None], axis=1)[
                :, 0
            ].astype(jnp.float32)
            valid = segments >= 0
            mean = (
                jnp.sum(hidden.astype(jnp.float32) * valid[..., None], axis=1)
                / lengths[:, None]
            )
            return last, mean, metrics["capacity_overflow_per_layer"]

        self._forward = forward

    def evaluate(self, sequences: list[list[int]]) -> dict[str, np.ndarray]:
        """Pad each independent sequence; discard duplicated batch-fill requests."""
        if not sequences or len(sequences) > self.batch_size:
            raise ValueError("Expected a nonempty bounded inference batch")
        if any(not 2 <= len(tokens) <= self.sequence_length for tokens in sequences):
            raise ValueError("Truncation and empty contexts are forbidden")
        count = len(sequences)
        repeated = sequences + [sequences[0]] * (self.batch_size - count)
        ids = np.zeros((self.batch_size, self.sequence_length), dtype=np.int32)
        segments = np.full_like(ids, -1)
        lengths = np.asarray([len(tokens) for tokens in repeated], dtype=np.int32)
        for index, tokens in enumerate(repeated):
            ids[index, : len(tokens)] = tokens
            segments[index, : len(tokens)] = 0
        with jax.set_mesh(self.mesh):
            placed = []
            for value in (ids, segments, lengths):
                sharding = NamedSharding(
                    self.mesh,
                    P(("replica_dcn", "data", "expert"), *([None] * (value.ndim - 1))),
                )
                placed.append(
                    jax.make_array_from_callback(
                        value.shape, sharding, lambda index, x=value: x[index]
                    )
                )
            outputs = self._forward(self.model, *placed)
            arrays = [
                np.asarray(
                    value
                    if value.is_fully_addressable
                    else multihost_utils.process_allgather(value, tiled=True)
                )
                for value in outputs
            ]
        if np.any(arrays[-1] != 0):
            raise ValueError("Frozen inference dropped expert assignments")
        if any(not np.isfinite(value).all() for value in arrays[:-1]):
            raise ValueError("Frozen inference produced nonfinite features")
        return {"last": arrays[0][:count], "mean": arrays[1][:count]}
