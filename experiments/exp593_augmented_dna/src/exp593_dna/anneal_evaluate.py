"""Compact monitoring: historical repeat splits, two FIM accuracies, suffix lift."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np

from exp593_dna.development_evaluate import DevelopmentEvaluator
from exp593_dna.development_readout import DevelopmentPanels
from exp593_dna.evaluate import ValidationPanels
from exp593_dna.fim_validation import prompt_and_target
from exp593_dna.pipeline import Encoder, Example, Span
from exp593_dna.validation import loss_views


def lowercase_minimal(
    example: Example, encoder: Encoder, *, suffix: bool = True
) -> Example:
    """Use identical held-out source crops/answers and only structural labels."""
    dna = {s.role: s.text.lower() for s in example.spans if s.role != "text"}
    spans = (
        Span("[left] ", "text"),
        Span(dna["left"], "left"),
        Span("\n[right] ", "text"),
        Span(dna["right"] if suffix else "", "right"),
        Span(f"\n[gap:{len(dna['middle'])}]\n[fill] ", "text"),
        Span(dna["middle"], "middle", True),
    )
    tokens, weights = encoder.encode(spans)
    # Original minimal-member label lengths can differ from our canonical form.
    # Trim only the remote end of the left flank, identically for both arms.
    if suffix and len(tokens) > 8192:
        extra = len(tokens) - 8192
        dna["left"] = dna["left"][extra:]
        spans = tuple(
            Span(dna["left"], "left") if s.role == "left" else s for s in spans
        )
        tokens, weights = encoder.encode(spans)
    if len(tokens) > 8192:
        raise ValueError("Compact held-out FIM presentation exceeds context")
    return Example(
        tokens,
        weights,
        spans,
        8192,
        {
            **example.attributes,
            "case": "unmasked_lower",
            "metadata_fields": [],
            "input_tokens": len(tokens),
            "suffix_present": suffix,
        },
    )


def remove_suffix(example: Example, encoder: Encoder) -> Example:
    """Remove only the right DNA: preserve the target, left flank, and labels."""
    spans = tuple(Span("", "right") if s.role == "right" else s for s in example.spans)
    tokens, weights = encoder.encode(spans)
    result = replace(
        example,
        token_ids=tokens,
        loss_weights=weights,
        spans=spans,
        attributes={
            **example.attributes,
            "input_tokens": len(tokens),
            "suffix_present": False,
        },
    )
    assert np.array_equal(prompt_and_target(example)[1], prompt_and_target(result)[1])
    return result


class CompactEvaluator(DevelopmentEvaluator):
    """Keep the old primary exactly; retain only approved aggregate FIM curves."""

    def __init__(
        self, panels: ValidationPanels, encoder: Encoder, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(DevelopmentPanels((), ()), encoder, *args, **kwargs)
        self.primary = panels.primary
        self.examples = tuple(lowercase_minimal(e, encoder) for e in panels.fim[::2])
        self.short = tuple(e for e in self.examples if e.scored_tokens <= 3)[:64]
        self.suffix_cases = self.short
        if len(self.examples) != 512 or len(self.short) != 64:
            raise ValueError("Compact FIM panel membership differs")

    def evaluate(self, state: Any) -> dict[str, float | int]:
        start = time.monotonic()
        self.progress("primary", 0, len(self.primary))
        losses = self.causal_losses(state, self.primary)
        result = loss_views([r.seq for r in self.primary], losses)
        if losses.size != 4_193_792 or result["eval/softmasked_targets"] != 396_290:
            raise ValueError("Historical validation membership/targets changed")
        self.progress("compact_fim", 0, len(self.examples))
        teacher = self.readout(state, self.examples)
        result["eval/fim_base_accuracy"] = float(
            np.mean(
                [
                    np.mean(prediction == prompt_and_target(example)[1])
                    for example, (_, prediction) in zip(
                        self.examples, teacher, strict=True
                    )
                ]
            )
        )
        generated = self.generate(state, self.short)
        result["eval/fim_short_exact_match"] = float(
            np.mean(
                [
                    np.array_equal(prediction, prompt_and_target(example)[1])
                    for example, prediction in zip(self.short, generated, strict=True)
                ]
            )
        )
        paired = self.readout(state, self.suffix_cases)
        ablated = self.readout(
            state, [remove_suffix(e, self.encoder) for e in self.suffix_cases]
        )
        result["eval/fim_suffix_benefit"] = float(
            np.mean(
                [
                    np.mean(without[0]) - np.mean(with_right[0])
                    for with_right, without in zip(paired, ablated, strict=True)
                ]
            )
        )
        result.update(
            {
                "eval/fim_examples": len(self.examples),
                "eval/fim_generated_examples": len(self.short),
                "eval/fim_suffix_examples": len(self.suffix_cases),
                "eval/seconds": time.monotonic() - start,
            }
        )
        if not all(np.isfinite(v) for v in result.values()):
            raise ValueError("Nonfinite compact evaluation result")
        return result
