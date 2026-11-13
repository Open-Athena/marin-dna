"""Batched native Grug inference for the frozen, evaluation-only diagnostics."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import jax
import numpy as np

from exp593_dna.development_readout import (
    DevelopmentPanels,
    binary_ranking_metrics,
    completion_target,
    conservation_examples,
)
from exp593_dna.evaluate import (
    BATCH_SIZE,
    Evaluator,
    ValidationPanels,
    _gather,
    binary_batch,
)
from exp593_dna.pipeline import ALPHABET, Example


class DevelopmentEvaluator(Evaluator):
    """Reuse the validated full-vocabulary kernels and actual greedy decoder.

    Hugging Face prediction loops cannot express this native sharded Grug state
    or its character-forced DNA encoding. All accelerator collectives run in the
    same order on the caller's training thread, without another GPU allocation.
    The empty historical panel is intentional: its evaluator is not called.
    """

    def __init__(self, panels: DevelopmentPanels, *args: Any, **kwargs: Any) -> None:
        super().__init__(ValidationPanels((), (), (), ()), *args, **kwargs)
        self.development_panels = panels

    def readout(
        self, state: Any, examples: Sequence[Example]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Project only the final one/two target positions; keep input order."""
        outputs = []
        for start in range(0, len(examples), BATCH_SIZE):
            batch_examples = examples[start : start + BATCH_SIZE]
            target_length = max(e.scored_tokens for e in batch_examples)
            positions = np.zeros((BATCH_SIZE, target_length), dtype=np.int32)
            targets = np.zeros_like(positions)
            for row, example in enumerate(batch_examples):
                prompt, target = completion_target(example)
                positions[row, : len(target)] = np.arange(
                    len(prompt) - 1, len(prompt) + len(target) - 1
                )
                targets[row, : len(target)] = target
            nll, predictions = self.fim_step(
                state.params,
                state.pending_qb_betas,
                binary_batch(batch_examples, self.mesh),
                self._put(positions),
                self._put(targets),
            )
            nll, predictions = _gather(nll), _gather(predictions)
            for row, example in enumerate(batch_examples):
                length = example.scored_tokens
                losses = nll[row, :length]
                if not np.isfinite(losses).all():
                    raise ValueError("Nonfinite development target likelihood")
                outputs.append((losses, predictions[row, :length]))
        return outputs

    def evaluate_development(
        self, state: Any
    ) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
        """Return scalar metrics plus ordered per-site audit records, without fitting."""
        started = time.monotonic()
        metrics: dict[str, float | int] = {}
        records: list[dict[str, Any]] = []
        sites = self.development_panels.conservation
        self.progress("conservation_development", 0, len(sites))
        # View-major ordering keeps target shapes and context buckets identical.
        examples = [conservation_examples(s, self.encoder) for s in sites]
        outputs = [
            self.readout(state, [e[view] for e in examples]) for view in range(3)
        ]
        bare_scores = []
        fim_scores = []
        for index, site in enumerate(sites):
            losses = [float(view[index][0][0]) for view in outputs]
            bare, fim = -(losses[0] + losses[1]) / 2, -losses[2]
            bare_scores.append(bare)
            fim_scores.append(fim)
            records.append(
                {
                    "task": "conservation",
                    "source_row": site.source_row,
                    "dna_sha256": site.dna_sha256,
                    "label": site.label,
                    "left_nll": losses[0],
                    "right_rc_nll": losses[1],
                    "bare_score": bare,
                    "fim_score": fim,
                }
            )
        labels = np.asarray([s.label for s in sites])
        for name, values in (("bare", bare_scores), ("fim", fim_scores)):
            metrics.update(
                {
                    f"eval/development_conservation_{name}_{key}": value
                    for key, value in binary_ranking_metrics(
                        labels, np.asarray(values)
                    ).items()
                }
            )
        junctions = self.development_panels.junctions
        self.progress("junction_development", 0, len(junctions))
        teacher = self.readout(state, junctions)
        generated = []
        for start in range(0, len(junctions), BATCH_SIZE):
            generated.extend(
                self.generate(state, junctions[start : start + BATCH_SIZE])
            )
        casefold = {
            int(self.encoder.dna_lookup[ord(c)]): c.casefold() for c in ALPHABET
        }
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for example, (nll, prediction), completion in zip(
            junctions, teacher, generated, strict=True
        ):
            _, target = completion_target(example)
            record = {
                "task": example.attributes["development_task"],
                "source_row": example.attributes["raw_row"],
                "rc": example.attributes["rc"],
                "source_class": example.attributes["source_class"],
                "target_ids": target.tolist(),
                "target_nll": nll.tolist(),
                "teacher_prediction_ids": prediction.tolist(),
                "generated_ids": completion.tolist(),
                "loss": float(np.mean(nll)),
                "base_accuracy": float(np.mean(prediction == target)),
                "exact_match": int(np.array_equal(completion, target)),
                "dna_exact_match": int(
                    [casefold.get(int(t), "<non-DNA>") for t in completion]
                    == [casefold[int(t)] for t in target]
                ),
            }
            groups[record["task"]].append(record)
            records.append(record)
        for task, members in groups.items():
            prefix = f"eval/development_junction_{task}"
            metrics[prefix + "_examples"] = len(members)
            metrics[prefix + "_targets"] = 2 * len(members)
            for name in ("loss", "base_accuracy", "exact_match", "dna_exact_match"):
                metrics[prefix + "_" + name] = float(
                    np.mean([r[name] for r in members])
                )
        metrics["eval/development_seconds"] = time.monotonic() - started
        if any(not np.isfinite(v) for v in metrics.values()):
            raise ValueError("Nonfinite development metric")
        jax.block_until_ready(state.params)
        return metrics, records
