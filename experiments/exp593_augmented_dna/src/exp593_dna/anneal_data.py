"""Metadata-free causal DNA and strictly annotation-guided FIM presentations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from exp593_dna.annotations import Landmark, RowAnnotations, candidate_starts
from exp593_dna.pipeline import (
    AugmentationConfig,
    Encoder,
    Example,
    RawWindow,
    Span,
    rc_selected,
    reverse_complement,
    rng_for,
)
from exp593_dna.templates import phrase_catalog, pick


class NoEligibleLandmark(ValueError):
    """The selected original window cannot express the requested guided gap."""


@dataclass(frozen=True)
class Presentation:
    """A sampled task, independent of source-row identity and presentation text."""

    occurrence: int
    bucket: int
    length: int
    fim: bool
    gap: int
    mode: str
    rc: bool


class GuidedAugmenter:
    """Use equal scored-base weights with RC, lengths and structural diversity."""

    def __init__(self, encoder: Encoder, *, seed: int = 472) -> None:
        self.encoder = encoder
        self.config = AugmentationConfig(seed=seed)
        self.phrases = phrase_catalog()

    def presentation(
        self, occurrence: int, fim_probability: float, *, length: int | None = None
    ) -> Presentation:
        if not 0 <= fim_probability <= 0.30:
            raise ValueError("Unapproved FIM curriculum probability")
        c = self.config
        bucket_rng = rng_for(c.seed, occurrence // 64, "batch_bucket")
        bucket = int(bucket_rng.choice(c.length_buckets, p=c.bucket_probabilities))
        length_rng = rng_for(c.seed, occurrence, "length")
        if length is None:
            length = int(
                length_rng.integers(max(2048, int(bucket * 0.875)), bucket + 1)
            )
        else:
            bucket = next(b for b in c.length_buckets if b >= length)
        kind_rng = rng_for(c.seed, occurrence, "anneal_kind")
        fim = bool(kind_rng.random() < fim_probability)
        short = bool(kind_rng.random() < 0.75)
        gap = int(kind_rng.choice((1, 2, 3) if short else (16, 32, 64, 128, 256, 512)))
        mode = "nearby" if short and kind_rng.random() < 0.5 else "overlap"
        return Presentation(
            occurrence,
            bucket,
            length,
            fim,
            gap if fim else 0,
            mode if fim else "",
            rc_selected(occurrence, c.seed),
        )

    def _structure(self, p: Presentation) -> tuple[Span, ...]:
        """Only task structure is rendered; biological metadata never enters."""
        rng = rng_for(self.config.seed, p.occurrence, "presentation")
        minimal = rng.random() < 0.25
        compact = minimal or rng.random() < 0.5
        labels = (
            (
                pick(rng, ("left", "prefix", "L")),
                pick(rng, ("right", "suffix", "R")),
                pick(rng, ("fill", "answer", "middle")),
            )
            if minimal
            else tuple(
                pick(rng, self.phrases[k])
                for k in ("left_label", "right_label", "answer_label")
            )
        )
        instruction = (
            "" if minimal else pick(rng, self.phrases["fim_instruction"]) + "\n"
        )
        instruction_first = bool(rng.random() < 0.5)
        separator = pick(rng, (":\n", " = ", ": "))
        visible_gap = (
            rng_for(self.config.seed, p.occurrence, "gap_length_hint").random() < 0.80
        )
        hint = f"[gap:{p.gap}]\n" if visible_gap else ""
        flanks = [(labels[0], "left"), (labels[1], "right")]
        if rng.random() < 0.25:
            flanks.reverse()
        spans = [Span(instruction if instruction_first else "", "text")]
        for label, role in flanks:
            spans.extend(
                (
                    Span(f"[{label}] " if compact else label + separator, "text"),
                    Span("", role),
                    Span("\n", "text"),
                )
            )
        spans.extend(
            (
                Span(("" if instruction_first else instruction) + hint, "text"),
                Span(f"[{labels[2]}] " if compact else labels[2] + separator, "text"),
                Span("", "middle", True),
            )
        )
        return tuple(spans)

    def build(
        self, raw: RawWindow, p: Presentation, annotation: RowAnnotations | None = None
    ) -> Example:
        if raw.split != "train":
            raise ValueError("Annealing can only consume original train rows")
        spans = self._structure(p) if p.fim else (Span("", "dna", True),)
        text_tokens = sum(
            len(self.encoder.text_ids(s.text)) for s in spans if s.role == "text"
        )
        dna_length = p.length - text_tokens
        if not 2048 <= p.length <= p.bucket or dna_length < p.gap + 256:
            raise ValueError("Task text leaves insufficient DNA context")
        rng = rng_for(self.config.seed, p.occurrence, "anneal_crop")
        oriented = reverse_complement(raw.seq) if p.rc else raw.seq
        landmark: Landmark | None = None
        absolute_gap = 0
        if p.fim:
            if annotation is None or annotation.row != raw.row:
                raise ValueError(
                    "Every FIM example requires its exact train annotation"
                )
            annotation.verify(raw.seq, raw.split)
            by_class: dict[str, list[tuple[Landmark, int, tuple[int, ...]]]] = (
                defaultdict(list)
            )
            seen = set()
            for feature in annotation.landmarks:
                start, end = (
                    (8192 - feature.end, 8192 - feature.start)
                    if p.rc
                    else (feature.start, feature.end)
                )
                key = (feature.kind, start, end)
                if key in seen:
                    continue
                seen.add(key)
                starts = candidate_starts(start, end, p.gap, 8192, p.mode)
                if starts:
                    by_class[feature.kind].append((feature, start, starts))
            if not by_class:
                raise NoEligibleLandmark(
                    "No valid annotated gap in this original window"
                )
            classes = sorted(by_class)
            values = by_class[classes[int(rng.integers(len(classes)))]]
            landmark, landmark_start, starts = values[int(rng.integers(len(values)))]
            absolute_gap = starts[int(rng.integers(len(starts)))]
            lo = max(0, absolute_gap + p.gap + 128 - dna_length)
            hi = min(8192 - dna_length, absolute_gap - 128)
            if lo > hi:
                raise NoEligibleLandmark(
                    "Selected gap cannot retain both context flanks"
                )
            crop = int(rng.integers(lo, hi + 1))
        else:
            crop = int(rng.integers(8192 - dna_length + 1))
            landmark_start = 0
        dna = oriented[crop : crop + dna_length].lower()
        gap_start = absolute_gap - crop if p.fim else 0
        pieces = {
            "dna": dna,
            "left": dna[:gap_start],
            "middle": dna[gap_start : gap_start + p.gap],
            "right": dna[gap_start + p.gap :],
        }
        spans = tuple(
            s if s.role == "text" else Span(pieces[s.role], s.role, s.score)
            for s in spans
        )
        tokens, weights = self.encoder.encode(spans)
        expected_targets = p.gap if p.fim else dna_length - 1
        if len(tokens) != p.length or int(weights.sum()) != expected_targets:
            raise ValueError("Annealing input length or scored target count differs")
        attributes = {
            "occurrence": p.occurrence,
            "raw_row": raw.row,
            "assembly": raw.assembly,
            "kind": ("short_fim" if p.gap <= 3 else "long_fim")
            if p.fim
            else "bare_causal",
            "case": "unmasked_lower",
            "rc": p.rc,
            "bucket": p.bucket,
            "input_tokens": len(tokens),
            "dna_tokens": dna_length,
            "text_tokens": text_tokens,
            "scored_tokens": int(np.count_nonzero(weights)),
            "gap_length": p.gap,
            "gap_start": gap_start,
            "crop_start_in_oriented_window": crop,
            "annotation_guided": p.fim,
            "annotation_eligible": p.fim,
            "landmark_class": landmark.kind if landmark else "",
            "landmark_mode": p.mode,
            "landmark_offset": absolute_gap - landmark_start if p.fim else 0,
            "annotation_source": landmark.source if landmark else "",
            "annotation_fallback": "",
            "minimal_fim": p.fim,
            "metadata_fields": [],
            "mask_description": False,
            "right_flank_first": next(
                (s.role for s in spans if s.role in ("left", "right")), ""
            )
            == "right",
        }
        return Example(tokens, weights, spans, p.bucket, attributes)


class GuidedSource:
    """Keep the causal cursor; independently sample annotated original FIM rows."""

    def __init__(
        self,
        shuffled: Any,
        original: Any,
        annotations: Mapping[int, RowAnnotations],
        augmenter: GuidedAugmenter,
    ) -> None:
        self.shuffled, self.original = shuffled, original
        self.annotations, self.augmenter = annotations, augmenter
        self.rows = np.asarray(sorted(annotations), dtype=np.int32)
        if len(self.rows) == 0:
            raise ValueError("The guided FIM source has no annotated training rows")

    def get_examples(self, presentations: Sequence[Presentation]) -> list[Example]:
        result: list[Example | None] = [None] * len(presentations)
        causal = [i for i, p in enumerate(presentations) if not p.fim]
        if causal:
            raw = self.shuffled.get_batch([presentations[i].occurrence for i in causal])
            for i, row in zip(causal, raw, strict=True):
                result[i] = self.augmenter.build(row, presentations[i])
        pending = [i for i, p in enumerate(presentations) if p.fim]
        for attempt in range(64):
            if not pending:
                break
            rows = [
                int(
                    self.rows[
                        rng_for(
                            self.augmenter.config.seed,
                            presentations[i].occurrence,
                            f"anneal_annotated_row_{attempt}",
                        ).integers(len(self.rows))
                    ]
                )
                for i in pending
            ]
            raw = self.original.get_batch(rows)
            retry = []
            for i, row in zip(pending, raw, strict=True):
                try:
                    result[i] = self.augmenter.build(
                        row, presentations[i], self.annotations[row.row]
                    )
                except NoEligibleLandmark:
                    retry.append(i)
            pending = retry
        if pending or any(e is None for e in result):
            raise ValueError("Annotation-only FIM exhausted bounded source selection")
        return list(result)
