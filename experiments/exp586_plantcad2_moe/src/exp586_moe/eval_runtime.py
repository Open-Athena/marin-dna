"""Native model adapter for validation of base-coordinate evaluation protocols."""

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask

from experiments.grug.moe_hero_ep.model import Transformer


def _prepare_pilot_batch(
    ids: list[list[int]],
    positions: list[list[int]],
    feature_positions: list[int],
    *,
    batch_size: int,
    sequence_length: int,
) -> list[np.ndarray]:
    """Pad requests without turning padding or query metadata into real context."""
    actual = len(ids)
    real_length = len(ids[0])
    padded_ids = np.zeros((batch_size, sequence_length), dtype=np.int32)
    segments = np.full_like(padded_ids, -1)
    repeated = ids + [ids[0]] * (batch_size - actual)
    for i, tokens in enumerate(repeated):
        padded_ids[i, :real_length] = tokens
        segments[i, :real_length] = 0
    arrays = [
        padded_ids,
        segments,
        np.zeros((batch_size, max(10, max(map(len, positions)))), dtype=np.int32),
        np.asarray(
            feature_positions + [feature_positions[0]] * (batch_size - actual),
            dtype=np.int32,
        ),
    ]
    for i, points in enumerate(positions + [positions[0]] * (batch_size - actual)):
        arrays[2][i, : len(points)] = points
    return arrays


def dna_output_head(model: Transformer, base_ids: np.ndarray) -> jax.Array:
    """Replicate only the four DNA columns from the FSDP-sharded vocabulary head."""
    return model.output_proj.at[:, base_ids].get(out_sharding=P(None, None))


class NativeScorer:
    """Return real-token probabilities and hidden states without packing examples.

    The caller supplies a dropless inference model on an expert-collapsed mesh.
    Padding to a whole data-parallel batch repeats examples, then discards the
    copies; it never concatenates genomic windows or updates router biases.
    """

    def __init__(
        self, model: Transformer, mesh: jax.sharding.Mesh, *, pad_to: int = 8192
    ) -> None:
        self.model = model
        self.mesh = mesh
        self.pad_to = pad_to
        self.batch_size = mesh.devices.size

        @eqx.filter_jit
        def forward(
            model: Transformer, ids: jax.Array, segments: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            hidden, metrics = model(
                ids, AttentionMask.causal().with_segment_ids(segments)
            )
            logits = jnp.einsum(
                "bsh,hv->bsv",
                hidden[:, :-1],
                model.output_proj,
                preferred_element_type=jnp.float32,
                out_sharding=P(("replica_dcn", "data", "expert"), None, None),
            )
            selected = jnp.take_along_axis(logits, ids[:, 1:, None], axis=-1)[..., 0]
            log_probs = selected - jax.scipy.special.logsumexp(logits, axis=-1)
            return log_probs, hidden, metrics["capacity_overflow_per_layer"]

        self._forward = forward

    def evaluate(
        self, sequences: Sequence[Sequence[int]]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not sequences:
            return [], []
        output_logs, output_hidden = [], []
        with jax.set_mesh(self.mesh):
            sharding = NamedSharding(
                self.mesh, P(("replica_dcn", "data", "expert"), None)
            )

            def put(array: np.ndarray) -> jax.Array:
                return jax.make_array_from_callback(
                    array.shape, sharding, lambda index: array[index]
                )

            for start in range(0, len(sequences), self.batch_size):
                group = list(sequences[start : start + self.batch_size])
                actual = len(group)
                group += [group[0]] * (self.batch_size - actual)
                ids = np.zeros((self.batch_size, self.pad_to), dtype=np.int32)
                segments = np.full_like(ids, -1)
                for i, tokens in enumerate(group):
                    if not 2 <= len(tokens) <= self.pad_to:
                        raise ValueError(
                            "Need 2..pad_to real tokens; truncation is forbidden"
                        )
                    ids[i, : len(tokens)] = tokens
                    segments[i, : len(tokens)] = 0
                logs, hidden, dropped = self._forward(
                    self.model, put(ids), put(segments)
                )
                # Global arrays are not locally addressable in the eight-process runtime.
                logs, hidden, dropped = [
                    np.asarray(
                        x
                        if x.is_fully_addressable
                        else multihost_utils.process_allgather(x, tiled=True)
                    )
                    for x in (logs, hidden, dropped)
                ]
                if np.any(dropped != 0):
                    raise ValueError("Evaluation dropped routed expert assignments")
                for i in range(actual):
                    length = len(group[i])
                    output_logs.append(logs[i, : length - 1].astype(np.float64))
                    output_hidden.append(hidden[i, :length].astype(np.float32))
        return output_logs, output_hidden

    def token_log_probs(self, sequences: list[list[int]]) -> list[np.ndarray]:
        return self.evaluate(sequences)[0]


class PilotScorer:
    """Export selected DNA probabilities, suffix likelihoods and pooled features."""

    def __init__(
        self,
        model: Transformer,
        mesh: jax.sharding.Mesh,
        base_ids: list[int],
        *,
        sequence_length: int = 8192,
    ) -> None:
        self.model, self.mesh = model, mesh
        self.base_ids = np.asarray(base_ids, dtype=np.int32)
        self.batch_size = mesh.devices.size
        max_sequence_length = getattr(
            getattr(model, "config", None), "max_seq_len", 8192
        )
        if sequence_length < 2 or sequence_length > max_sequence_length:
            raise ValueError("Scoring length must fit the configured model context")
        self.sequence_length = sequence_length

        @eqx.filter_jit
        def forward(
            model: Transformer,
            ids: jax.Array,
            segments: jax.Array,
            positions: jax.Array,
            feature_positions: jax.Array,
            allele_frequency: bool,
        ) -> tuple[jax.Array, ...]:
            hidden, metrics = model(
                ids, AttentionMask.causal().with_segment_ids(segments)
            )
            head = dna_output_head(model, self.base_ids)
            dna_logits = jnp.einsum(
                "bsh,hv->bsv",
                hidden,
                head,
                preferred_element_type=jnp.float32,
                out_sharding=P(("replica_dcn", "data", "expert"), None, None),
            )
            dna_logs = jax.nn.log_softmax(dna_logits[:, :-1], axis=-1)
            rows = jnp.arange(ids.shape[0])[:, None]
            queries = jnp.exp(
                dna_logs.at[rows, jnp.maximum(positions - 1, 0)].get(
                    out_sharding=P(("replica_dcn", "data", "expert"), None, None)
                )
            )
            queries = jnp.where((positions > 0)[..., None], queries, 0)
            if allele_frequency:
                logits = jnp.einsum(
                    "bsh,hv->bsv",
                    hidden[:, :-1],
                    model.output_proj,
                    preferred_element_type=jnp.float32,
                    out_sharding=P(("replica_dcn", "data", "expert"), None, None),
                )
                full = jnp.take_along_axis(logits, ids[:, 1:, None], axis=-1)[
                    ..., 0
                ] - jax.scipy.special.logsumexp(logits, axis=-1)
                matches = ids[:, 1:, None] == self.base_ids
                valid = jnp.any(matches, axis=-1)
                targets = jnp.argmax(matches, axis=-1)
                acgt = jnp.take_along_axis(dna_logs, targets[..., None], axis=-1)[
                    ..., 0
                ]
                token_valid = segments >= 0
                mean = jnp.sum(
                    hidden.astype(jnp.float32) * token_valid[..., None], axis=1
                ) / jnp.maximum(jnp.sum(token_valid, axis=1, keepdims=True), 1)
                variant = jnp.take_along_axis(
                    hidden, feature_positions[:, None, None], axis=1
                )[:, 0].astype(jnp.float32)
            else:
                full = acgt = jnp.zeros((ids.shape[0], 1), dtype=jnp.float32)
                valid = jnp.ones_like(full, dtype=bool)
                mean = variant = jnp.zeros((ids.shape[0], 1), dtype=jnp.float32)
            return (
                queries,
                full,
                acgt,
                valid,
                mean,
                variant,
                metrics["capacity_overflow_per_layer"],
            )

        self._forward = forward

    def evaluate(
        self,
        ids: list[list[int]],
        positions: list[list[int]],
        feature_positions: list[int],
        *,
        allele_frequency: bool,
    ) -> dict[str, np.ndarray]:
        lengths = {len(x) for x in ids}
        if not ids or len(ids) > self.batch_size or len(lengths) != 1:
            raise ValueError("Scoring requires one batch of equal-length windows")
        real_length = next(iter(lengths))
        if real_length < 2 or real_length > self.sequence_length:
            raise ValueError("Real tokens must fit the static scoring length")
        if len(ids) != len(positions) or len(ids) != len(feature_positions):
            raise ValueError("Request metadata count differs")
        if any(
            not 1 <= len(x) <= real_length or any(p < 0 or p >= real_length for p in x)
            for x in positions
        ):
            raise ValueError("Invalid query positions")
        actual = len(ids)
        arrays = _prepare_pilot_batch(
            ids,
            positions,
            feature_positions,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
        )
        with jax.set_mesh(self.mesh):
            device = []
            for array in arrays:
                sharding = NamedSharding(
                    self.mesh,
                    P(("replica_dcn", "data", "expert"), *([None] * (array.ndim - 1))),
                )
                device.append(
                    jax.make_array_from_callback(
                        array.shape, sharding, lambda index, x=array: x[index]
                    )
                )
            outputs = self._forward(self.model, *device, allele_frequency)
            outputs = [
                np.asarray(
                    x
                    if x.is_fully_addressable
                    else multihost_utils.process_allgather(x, tiled=True)
                )
                for x in outputs
            ]
        if np.any(outputs[-1] != 0):
            raise ValueError(
                "Pilot inference dropped expert assignments: "
                f"max={int(np.max(outputs[-1]))}, sum={int(np.sum(outputs[-1]))}"
            )
        nonfinite = [
            name
            for name, value in zip(
                ["probabilities", "full_logs", "acgt_logs", "valid", "mean", "variant"],
                outputs[:-1],
                strict=True,
            )
            if not np.isfinite(value).all()
        ]
        if nonfinite:
            raise ValueError(
                f"Nonfinite pilot outputs at static={self.sequence_length}, real={real_length}: "
                + ", ".join(nonfinite)
            )
        result = {
            name: x[:actual]
            for name, x in zip(
                ["probabilities", "full_logs", "acgt_logs", "valid", "mean", "variant"],
                outputs[:-1],
                strict=True,
            )
        }
        for name in ("full_logs", "acgt_logs", "valid"):
            result[name] = result[name][:, : real_length - 1]
        return result
