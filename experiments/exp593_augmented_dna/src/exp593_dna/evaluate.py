"""Accelerator evaluation of the frozen historical, disjoint and paired FIM panels."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from exp586_moe.data import WindowBatch
from exp586_moe.gpu_smoke import device_batch
from experiments.grug.moe_hero_ep.model import apply_qb_betas
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from exp593_dna.corpus import open_corpus
from exp593_dna.fim_validation import FIMObservation, fim_metrics, prompt_and_target
from exp593_dna.inputs import PreparedInputs
from exp593_dna.pipeline import (
    Encoder,
    Example,
    RawWindow,
    Span,
    collate,
    reverse_complement,
)
from exp593_dna.train_step import COMPILER_OPTIONS
from exp593_dna.validation import loss_views

PRIMARY_LOWER_SHA256 = (
    "62334d75edfef471c248b36c01e1542885acb5c0ecf3ac88afbd164ebe56e112"
)
BATCH_SIZE = 64


def causal_example(raw: RawWindow, encoder: Encoder) -> Example:
    """Use exactly the old lowercase, character-tokenized validation protocol."""
    if raw.split != "validation":
        raise ValueError("Evaluation requires held-out rows")
    spans = (Span(raw.seq.lower(), "dna", True),)
    tokens, weights = encoder.encode(spans)
    return Example(tokens, weights, spans, 8192, {"raw_row": raw.row})


def frozen_fim(
    record: dict[str, Any], raw: RawWindow, encoder: Encoder
) -> tuple[Example, Example]:
    """Re-encode frozen spans and verify that their DNA still comes from that row."""
    if (
        raw.split != "validation"
        or record["source_row"] != raw.row
        or record["source_sequence_sha256"]
        != hashlib.sha256(raw.seq.encode()).hexdigest()
    ):
        raise ValueError("Frozen FIM source identity differs")
    members = []
    for member in record["members"]:
        spans = tuple(Span(**s) for s in member["spans"])
        tokens, weights = encoder.encode(spans)
        a = member["attributes"]
        example = Example(tokens, weights, spans, member["bucket"], a)
        dna = {s.role: s.text for s in spans if s.role != "text"}
        expected = reverse_complement(raw.seq) if a["rc"] else raw.seq
        start = a["crop_start_in_oriented_window"]
        expected = expected[start : start + a["dna_tokens"]]
        if a["case"] == "unmasked_lower":
            expected = expected.lower()
        elif a["case"] == "unmasked_upper":
            expected = expected.upper()
        elif a["case"] != "native_softmask":
            raise ValueError("Unknown frozen DNA case convention")
        if (
            dna["left"] + dna["middle"] + dna["right"] != expected
            or len(dna["left"]) != a["gap_start"]
            or len(tokens) != a["input_tokens"]
            or len(tokens) > example.bucket
        ):
            raise ValueError("Frozen FIM geometry differs from the original DNA")
        prompt_and_target(example)
        members.append(example)
    if len(members) != 2 or [e.attributes["presentation"] for e in members] != [
        "minimal",
        "formatted",
    ]:
        raise ValueError("Expected one matched minimal/formatted pair")
    if (
        prompt_and_target(members[0])[1].tolist()
        != prompt_and_target(members[1])[1].tolist()
    ):
        raise ValueError("Matched FIM targets differ")
    return members[0], members[1]


@dataclass(frozen=True)
class ValidationPanels:
    primary: tuple[RawWindow, ...]
    disjoint: tuple[RawWindow, ...]
    fim: tuple[Example, ...]
    generation_indices: tuple[int, ...]

    @classmethod
    def open(cls, inputs: PreparedInputs, encoder: Encoder) -> ValidationPanels:
        primary = tuple(
            RawWindow.from_dict(json.loads(line))
            for line in inputs.path("validation/primary_validation_512.jsonl")
            .read_text()
            .splitlines()
        )
        digest = hashlib.sha256(
            "".join(r.seq.lower() for r in primary).encode()
        ).hexdigest()
        if (
            len(primary) != 512
            or [r.row for r in primary] != [i * 329_832 // 512 for i in range(512)]
            or digest != PRIMARY_LOWER_SHA256
        ):
            raise ValueError("Historical validation membership or encoding changed")
        selected = json.loads(
            inputs.path("audit/validation_disjoint_512.json").read_text()
        )["examples"]
        disjoint = tuple(
            open_corpus(inputs, "validation", 0).get_batch([r["row"] for r in selected])
        )
        if len(disjoint) != 512 or any(
            any(getattr(raw, k) != v for k, v in record.items())
            for raw, record in zip(disjoint, selected, strict=True)
        ):
            raise ValueError("Disjoint validation membership changed")
        sources = {r.row: r for r in disjoint}
        records = [
            json.loads(line)
            for line in inputs.path("audit/fim_validation_pairs.jsonl")
            .read_text()
            .splitlines()
        ]
        if len(records) != 512 or {r["source_row"] for r in records} != sources.keys():
            raise ValueError("FIM panel does not match the frozen disjoint rows")
        fim: list[Example] = []
        generation = []
        for record in records:
            if record["generation_candidate"]:
                generation.extend((len(fim), len(fim) + 1))
            fim.extend(frozen_fim(record, sources[record["source_row"]], encoder))
        if len(generation) != 36:
            raise ValueError("Fixed generation panel membership changed")
        return cls(primary, disjoint, tuple(fim), tuple(generation))


def binary_batch(examples: Sequence[Example], mesh: jax.sharding.Mesh) -> Any:
    """Use binary target masks for raw NLL, never training's 1/target_length weights.

    Repeat a real context for unused rows, with zero target weight, rather than
    presenting an all-padding document to the attention implementation.
    """
    if not 1 <= len(examples) <= BATCH_SIZE:
        raise ValueError("Invalid validation batch size")
    padded = list(examples) + [examples[0]] * (BATCH_SIZE - len(examples))
    data = collate(padded)
    data["loss_weights"] = (data["loss_weights"] > 0).astype(np.float32)
    data["loss_weights"][len(examples) :] = 0
    return device_batch(
        WindowBatch(
            data["token_ids"],
            data["segment_ids"],
            data["loss_weights"],
            data["lengths"],
        ),
        mesh,
    )


def _gather(value: jax.Array) -> np.ndarray:
    return np.asarray(multihost_utils.process_allgather(value, tiled=True))


def make_evaluator(mp: jmp.Policy) -> tuple[Any, Any, Any]:
    """Compile causal raw NLL, FIM target readout and an actual greedy decode step."""

    @partial(eqx.filter_jit, compiler_options=COMPILER_OPTIONS)
    def causal(params: Any, pending: Any, batch: Any) -> jax.Array:
        model = mp.cast_to_compute(apply_qb_betas(params, pending))
        return model.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="none",
            logsumexp_weight=None,
        )

    @partial(eqx.filter_jit, compiler_options=COMPILER_OPTIONS)
    def fim(
        params: Any, pending: Any, batch: Any, positions: jax.Array, targets: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        model = mp.cast_to_compute(apply_qb_betas(params, pending))
        hidden, _ = model(batch.tokens, mask=batch.attn_mask)
        selected = jnp.take_along_axis(hidden, positions[:, :, None], axis=1)
        # Only the missing spans receive a vocabulary projection (at most 512
        # positions per row); never materialize [64,8192,128256] logits.
        logits = jnp.einsum(
            "bth,hv->btv",
            selected,
            model.output_proj,
            preferred_element_type=jnp.float32,
        )
        nll = (
            jax.nn.logsumexp(logits, axis=-1)
            - jnp.take_along_axis(logits, targets[:, :, None], axis=-1)[:, :, 0]
        )
        return nll, jnp.argmax(logits, axis=-1).astype(jnp.int32)

    @partial(eqx.filter_jit, compiler_options=COMPILER_OPTIONS)
    def greedy(
        params: Any, pending: Any, batch: Any, positions: jax.Array
    ) -> jax.Array:
        model = mp.cast_to_compute(apply_qb_betas(params, pending))
        hidden, _ = model(batch.tokens, mask=batch.attn_mask)
        selected = jnp.take_along_axis(hidden, positions[:, None, None], axis=1)[:, 0]
        logits = jnp.einsum(
            "bh,hv->bv", selected, model.output_proj, preferred_element_type=jnp.float32
        )
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    return causal, fim, greedy


class Evaluator:
    """Retain compiled functions and frozen membership across validation events."""

    def __init__(
        self,
        panels: ValidationPanels,
        encoder: Encoder,
        mesh: jax.sharding.Mesh,
        mp: jmp.Policy,
    ) -> None:
        self.panels, self.encoder, self.mesh = panels, encoder, mesh
        self.causal_step, self.fim_step, self.greedy_step = make_evaluator(mp)

    def _put(self, values: np.ndarray) -> jax.Array:
        spec = P(("replica_dcn", "data", "expert"), *([None] * (values.ndim - 1)))
        return jax.make_array_from_callback(
            values.shape, NamedSharding(self.mesh, spec), lambda index: values[index]
        )

    def causal_losses(self, state: Any, rows: Sequence[RawWindow]) -> np.ndarray:
        batches = []
        for start in range(0, len(rows), BATCH_SIZE):
            examples = [
                causal_example(r, self.encoder)
                for r in rows[start : start + BATCH_SIZE]
            ]
            result = self.causal_step(
                state.params, state.pending_qb_betas, binary_batch(examples, self.mesh)
            )
            batches.append(_gather(result)[: len(examples), :-1])
        return np.concatenate(batches)

    def generate(self, state: Any, examples: Sequence[Example]) -> list[np.ndarray]:
        """Only generated tokens enter subsequent prefixes; length is caller controlled."""
        if not examples:
            return []
        prompts, targets = zip(*(prompt_and_target(e) for e in examples), strict=True)
        generated: list[list[int]] = [[] for _ in examples]
        for offset in range(max(map(len, targets))):
            contexts = [
                np.concatenate((p, np.asarray(g, dtype=np.int32)))
                for p, g in zip(prompts, generated, strict=True)
            ]
            lengths = np.asarray(
                [len(c) for c in contexts]
                + [len(contexts[0])] * (BATCH_SIZE - len(contexts)),
                dtype=np.int32,
            )
            tokens = np.zeros((BATCH_SIZE, 8192), dtype=np.int32)
            segments = np.full_like(tokens, -1)
            for row, context in enumerate(
                contexts + [contexts[0]] * (BATCH_SIZE - len(contexts))
            ):
                tokens[row, : len(context)] = context
                segments[row, : len(context)] = 0
            batch = device_batch(
                WindowBatch(
                    tokens, segments, np.zeros_like(tokens, dtype=np.float32), lengths
                ),
                self.mesh,
            )
            predictions = _gather(
                self.greedy_step(
                    state.params, state.pending_qb_betas, batch, self._put(lengths - 1)
                )
            )
            for row, target in enumerate(targets):
                if offset < len(target):
                    generated[row].append(int(predictions[row]))
        return [np.asarray(g, dtype=np.int32) for g in generated]

    def evaluate(self, state: Any, *, generate: bool = True) -> dict[str, float | int]:
        started = time.monotonic()
        losses = self.causal_losses(state, self.panels.primary)
        result = loss_views([r.seq for r in self.panels.primary], losses)
        if result["eval/softmasked_targets"] != 396_290 or losses.size != 4_193_792:
            raise ValueError("Historical validation target counts changed")
        disjoint = self.causal_losses(state, self.panels.disjoint)
        result.update(
            {
                "eval/loss_disjoint": float(disjoint.mean()),
                "eval/disjoint_examples": len(self.panels.disjoint),
            }
        )
        generated = {}
        generation_start = time.monotonic()
        if generate:
            chosen = self.panels.generation_indices
            outputs = self.generate(state, [self.panels.fim[i] for i in chosen])
            generated = dict(zip(chosen, outputs, strict=True))
        result["eval/generation_seconds"] = time.monotonic() - generation_start
        observations = []
        for start in range(0, len(self.panels.fim), BATCH_SIZE):
            examples = self.panels.fim[start : start + BATCH_SIZE]
            positions = np.zeros((BATCH_SIZE, 512), dtype=np.int32)
            targets = np.zeros_like(positions)
            for row, example in enumerate(examples):
                prompt, target = prompt_and_target(example)
                positions[row, : len(target)] = np.arange(
                    len(prompt) - 1, len(prompt) + len(target) - 1
                )
                targets[row, : len(target)] = target
            nll, predictions = self.fim_step(
                state.params,
                state.pending_qb_betas,
                binary_batch(examples, self.mesh),
                self._put(positions),
                self._put(targets),
            )
            nll, predictions = _gather(nll), _gather(predictions)
            for row, example in enumerate(examples):
                length = example.scored_tokens
                observations.append(
                    FIMObservation(
                        example,
                        nll[row, :length],
                        predictions[row, :length],
                        generated.get(start + row),
                    )
                )
        result.update(fim_metrics(observations, self.encoder))
        result["eval/seconds"] = time.monotonic() - started
        if any(not np.isfinite(v) for v in result.values()):
            raise ValueError("Nonfinite validation metric")
        return result
