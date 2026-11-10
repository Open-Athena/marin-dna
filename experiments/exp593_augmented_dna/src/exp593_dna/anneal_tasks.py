"""Frozen real-task monitoring, with native Grug inference and no training feedback.

The 2,048-example panels are diagnostic subsets of the existing benchmark test
sets, not independent development sets. Their immutable membership and methods
are bound to each new training run. No task data enters the training sampler.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections.abc import Sequence
from functools import partial
from importlib.resources import files
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from experiments.grug.moe_hero_ep.model import apply_qb_betas
from filelock import FileLock
from rigging.filesystem.storage_path import StoragePath

from exp593_dna.development_readout import binary_ranking_metrics
from exp593_dna.evaluate import BATCH_SIZE, Evaluator, _gather, binary_batch
from exp593_dna.pipeline import Encoder, Example, Span, reverse_complement
from exp593_dna.train_step import COMPILER_OPTIONS

BASES = "acgt"
SLOTS = 16


def panel_binding() -> dict[str, Any]:
    return json.loads(
        files("exp593_dna").joinpath("assets/anneal_task_panel.json").read_text()
    )


def fetch_panels(directory: Path) -> dict[str, Any]:
    """Verify all bytes once per node under a lock; never accept partial caches."""
    directory.mkdir(parents=True, exist_ok=True)
    binding = panel_binding()
    with FileLock(str(directory / ".fetch.lock")):
        manifest_bytes = StoragePath(binding["uri"]).read_bytes()
        if (
            len(manifest_bytes) != binding["bytes"]
            or hashlib.sha256(manifest_bytes).hexdigest() != binding["sha256"]
        ):
            raise ValueError("Frozen task manifest identity changed")
        manifest = json.loads(manifest_bytes)
        result = {}
        for key, record in manifest["tasks"].items():
            path = directory / record["path"]
            payload = path.read_bytes() if path.exists() else b""
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                payload = StoragePath(record["uri"]).read_bytes()
            if (
                len(payload) != record["bytes"]
                or hashlib.sha256(payload).hexdigest() != record["sha256"]
            ):
                raise ValueError(f"Task payload identity changed: {key}")
            if not path.exists():
                path.write_bytes(payload)
            rows = [json.loads(line) for line in gzip.decompress(payload).splitlines()]
            if (
                len(rows) != 2048
                or [r["monitor_source_row"] for r in rows] != record["membership"]
            ):
                raise ValueError(f"Task membership changed: {key}")
            result[key] = {"spec": record["spec"], "rows": rows}
        if len(result) != 21:
            raise ValueError("Expected twenty benchmark tasks and maize AF")
        return result


def dna_text(sequence: str) -> str:
    """Match benchmark readers: lowercase canonical bases, map ambiguities to n."""
    return "".join(c if c in BASES else "n" for c in sequence.lower())


def task_example(
    sequence: str, positions: Sequence[int], encoder: Encoder, *, fim: bool
) -> tuple[Example, np.ndarray]:
    """Return target-token positions (not prediction positions) in the new prompt.

    Causal queries retain their source positions. FIM removes the entire queried
    contiguous span from both flanks, then teacher-forces that answer at the end.
    Only remote flank ends are trimmed to accommodate structural labels.
    """
    dna = dna_text(sequence)
    positions = np.asarray(positions, dtype=np.int32)
    if (
        not 0 < len(positions) <= SLOTS
        or positions[0] < 1
        or positions[-1] >= len(dna)
        or np.any(np.diff(positions) <= 0)
        or len(dna) not in (2048, 8192)
    ):
        raise ValueError("Invalid benchmark target geometry")
    if fim:
        if not np.all(np.diff(positions) == 1):
            raise ValueError("A FIM query requires one contiguous missing span")
        left, answer, right = (
            dna[: positions[0]],
            dna[positions[0] : positions[-1] + 1],
            dna[positions[-1] + 1 :],
        )

        def encode() -> tuple[tuple[Span, ...], np.ndarray, np.ndarray]:
            spans = (
                Span("[left] ", "text"),
                Span(left, "left"),
                Span("\n[right] ", "text"),
                Span(right, "right"),
                Span(f"\n[gap:{len(answer)}]\n[fill] ", "text"),
                Span(answer, "middle", True),
            )
            tokens, weights = encoder.encode(spans)
            return spans, tokens, weights

        spans, tokens, weights = encode()
        excess = max(0, len(tokens) - len(dna))
        left_trim = min(len(left) - 1, (excess + 1) // 2)
        right_trim = excess - left_trim
        if right_trim >= len(right):
            raise ValueError("Insufficient flanks for structural labels")
        left, right = left[left_trim:], right[: len(right) - right_trim]
        spans, tokens, weights = encode()
        query = np.arange(len(tokens) - len(answer), len(tokens), dtype=np.int32)
    else:
        spans = (Span(dna, "dna", True),)
        tokens, weights = encoder.encode(spans)
        query = positions
    if len(tokens) > len(dna):
        raise ValueError("Benchmark rendering exceeds its context")
    return Example(tokens, weights, spans, len(dna), {}), query


def combine_strands(forward: np.ndarray, reverse: np.ndarray) -> np.ndarray:
    """Align reverse target/base axes, geometric mean, then condition on ACGT."""
    aligned = reverse[..., ::-1, :][..., [3, 2, 1, 0]]
    values = 0.5 * (forward + aligned)  # inputs are normalized log probabilities
    values -= np.max(values, axis=-1, keepdims=True)
    probabilities = np.exp(values)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def true_probabilities(probabilities: np.ndarray, answers: Sequence[str]) -> np.ndarray:
    """Unknown answer bases receive zero probability, matching legacy readouts."""
    output = np.zeros(probabilities.shape[:2], dtype=np.float64)
    for row, answer in enumerate(answers):
        if len(answer) != probabilities.shape[1]:
            raise ValueError("Answer length differs from the queried positions")
        for column, base in enumerate(answer.lower()):
            if base in BASES:
                output[row, column] = probabilities[row, column, BASES.index(base)]
    return output


def sv_queries(row: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Historical five bases on either side of the deletion junction."""
    left, right = int(row["left"]), int(row["right"])
    return list(range(left - 6, left - 1)) + list(range(right, right + 5)), list(
        range(4091, 4101)
    )


def sv_scores(ref: np.ndarray, alt: np.ndarray, answers: Sequence[str]) -> np.ndarray:
    """Negative mean ALT/REF boundary log ratio; both use the ALT target base."""
    p_ref = true_probabilities(ref, answers)
    p_alt = true_probabilities(alt, answers)
    return -np.mean(np.log(np.maximum(p_alt, 1e-12) / np.maximum(p_ref, 1e-12)), axis=1)


def af_alleles(row: dict[str, Any]) -> tuple[str, str, int]:
    """The AF source has its reference at 4096, unlike motif targets at 4095."""
    sequence = dna_text(row["sequence"])
    ref, alt = row["ref"].lower(), row["alt"].lower()
    if len(sequence) != 8192 or ref not in BASES or alt not in BASES:
        raise ValueError("Unsupported AF example")
    if sequence[4096] != ref or ref == alt:
        raise ValueError("AF reference allele or center differs")
    sequence = sequence[3072:5120]
    return sequence, sequence[:1024] + alt + sequence[1025:], 1024


def make_base_step(mp: Any, encoder: Encoder) -> Any:
    """Project eight case variants only, sum case mass, then normalize four bases."""
    ids = tuple(int(encoder.dna_lookup[ord(c)]) for c in "ACGTacgt")

    @partial(eqx.filter_jit, compiler_options=COMPILER_OPTIONS)
    def step(params: Any, pending: Any, batch: Any, positions: Any) -> jax.Array:
        model = mp.cast_to_compute(apply_qb_betas(params, pending))
        hidden, _ = model(batch.tokens, mask=batch.attn_mask)
        selected = jnp.take_along_axis(hidden, positions[:, :, None], axis=1)
        head = model.output_proj[:, jnp.asarray(ids)]
        logits = jnp.einsum(
            "bth,hv->btv", selected, head, preferred_element_type=jnp.float32
        )
        folded = jnp.logaddexp(logits[..., :4], logits[..., 4:])
        return jax.nn.log_softmax(folded, axis=-1)

    return step


class TaskEvaluator:
    """Run approved fixed diagnostic panels on the existing training gang."""

    def __init__(
        self, base: Evaluator, panels: dict[str, Any], mp: Any, *, pilot: bool = False
    ) -> None:
        self.base, self.panels, self.pilot = base, panels, pilot
        self.step = make_base_step(mp, base.encoder)

    def base_logs(
        self, state: Any, queries: Sequence[tuple[Example, np.ndarray]]
    ) -> np.ndarray:
        outputs = []
        for start in range(0, len(queries), BATCH_SIZE):
            selected = queries[start : start + BATCH_SIZE]
            positions = np.zeros((BATCH_SIZE, SLOTS), dtype=np.int32)
            for row, (_, indices) in enumerate(selected):
                positions[row, : len(indices)] = indices - 1
            values = self.step(
                state.params,
                state.pending_qb_betas,
                binary_batch([q[0] for q in selected], self.base.mesh),
                self.base._put(positions),
            )
            values = _gather(values)
            outputs.extend(values[r, : len(q[1])] for r, q in enumerate(selected))
        result = np.asarray(outputs, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite task base log probabilities")
        return result

    def standard(
        self, state: Any, spec: dict[str, Any], rows: list[dict[str, Any]], *, fim: bool
    ) -> dict[str, float]:
        sequences = [dna_text(r["sequence"]) for r in rows]
        positions = spec["positions"]
        orientations = (
            (sequences, positions),
            (
                [reverse_complement(s) for s in sequences],
                sorted(8191 - p for p in positions),
            ),
        )
        outputs = [
            self.base_logs(
                state,
                [task_example(s, points, self.base.encoder, fim=fim) for s in strings],
            )
            for strings, points in orientations
        ]
        probs = combine_strands(*outputs)
        answers = ["".join(s[p] for p in positions) for s in sequences]
        if spec["kind"] == "motif":
            predictions = np.argmax(probs, axis=-1)
            correct = [
                "".join(BASES[int(p)] for p in prediction) == answer
                for prediction, answer in zip(predictions, answers, strict=True)
            ]
            return {"accuracy": float(np.mean(correct))}
        scores = true_probabilities(probs, answers).mean(axis=1)
        metrics = binary_ranking_metrics(np.asarray([r["label"] for r in rows]), scores)
        # One scalar per binary task; conservation uses AUROC, core uses AP.
        name = "auroc" if spec["kind"] == "conservation" else "average_precision"
        return {name: float(metrics[name])}

    def sv(
        self, state: Any, rows: list[dict[str, Any]], *, fim: bool
    ) -> dict[str, float]:
        allele_outputs = []
        for allele, field in enumerate(("RefSeq", "MutSeq")):
            if fim:
                # The established SV FIM readout hides one base per boundary
                # query. A discontiguous ten-base FIM would change the task.
                queries = [
                    task_example(r[field], [p], self.base.encoder, fim=True)
                    for r in rows
                    for p in sv_queries(r)[allele]
                ]
                values = self.base_logs(state, queries).reshape(len(rows), 10, 4)
            else:
                values = self.base_logs(
                    state,
                    [
                        task_example(
                            r[field],
                            sv_queries(r)[allele],
                            self.base.encoder,
                            fim=False,
                        )
                        for r in rows
                    ],
                )
            allele_outputs.append(np.exp(values))
        answers = [dna_text(r["MutSeq"])[4091:4101] for r in rows]
        scores = sv_scores(*allele_outputs, answers)
        metrics = binary_ranking_metrics(np.asarray([r["label"] for r in rows]), scores)
        return {"average_precision": float(metrics["average_precision"])}

    def af(
        self, state: Any, rows: list[dict[str, Any]], *, fim: bool
    ) -> dict[str, float]:
        from scipy.stats import spearmanr

        alleles = [af_alleles(r) for r in rows]
        scores = []
        for rc in (False, True):
            ref_sequences = [reverse_complement(a[0]) if rc else a[0] for a in alleles]
            alt_sequences = [reverse_complement(a[1]) if rc else a[1] for a in alleles]
            index = 1023 if rc else 1024
            if fim:
                log_probs = self.base_logs(
                    state,
                    [
                        task_example(s, [index], self.base.encoder, fim=True)
                        for s in ref_sequences
                    ],
                )[:, 0]
                score = [
                    p[BASES.index(a[index])] - p[BASES.index(r[index])]
                    for p, r, a in zip(
                        log_probs, ref_sequences, alt_sequences, strict=True
                    )
                ]
            else:
                # Legacy AF score includes the variant and downstream suffix,
                # using full-vocabulary lowercase-token sequence likelihood.
                losses = []
                for sequences in (ref_sequences, alt_sequences):
                    values = []
                    for start in range(0, len(sequences), BATCH_SIZE):
                        examples = [
                            task_example(s, [index], self.base.encoder, fim=False)[0]
                            for s in sequences[start : start + BATCH_SIZE]
                        ]
                        nll = _gather(
                            self.base.causal_step(
                                state.params,
                                state.pending_qb_betas,
                                binary_batch(examples, self.base.mesh),
                            )
                        )
                        values.extend(nll[: len(examples), index - 1 : -1].sum(axis=1))
                    losses.append(np.asarray(values))
                score = losses[0] - losses[1]
            scores.append(np.asarray(score))
        score = np.mean(scores, axis=0)
        correlation = float(spearmanr(score, [float(r["AF"]) for r in rows]).statistic)
        if not np.isfinite(correlation):
            raise ValueError("AF ranking is undefined")
        return {"spearman": correlation}

    def evaluate(self, state: Any) -> dict[str, float | int]:
        started = time.monotonic()
        prefix = "validation/tasks" if self.pilot else "eval/tasks"
        metrics: dict[str, float | int] = {}
        for key, panel in self.panels.items():
            spec, rows = panel["spec"], panel["rows"]
            if self.pilot:
                # Bounded implementation checks only, never production scores.
                rows = rows[:32]
                if spec["kind"] not in ("motif", "af"):
                    for label in (0, 1):
                        if not any(int(r["label"]) == label for r in rows):
                            rows.append(
                                next(
                                    r for r in panel["rows"] if int(r["label"]) == label
                                )
                            )
            for fim, view in ((False, "raw"), (True, "fim")):
                self.base.progress(f"task/{key}/{view}", 0, len(rows))
                if spec["kind"] == "sv":
                    result = self.sv(state, rows, fim=fim)
                elif spec["kind"] == "af":
                    result = self.af(state, rows, fim=fim)
                else:
                    result = self.standard(state, spec, rows, fim=fim)
                metrics.update(
                    {f"{prefix}/{key}/{view}_{k}": v for k, v in result.items()}
                )
            metrics[f"{prefix}/{key}/examples"] = len(rows)
        metrics[f"{prefix}/seconds"] = time.monotonic() - started
        if not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError("Nonfinite benchmark metric")
        return metrics
