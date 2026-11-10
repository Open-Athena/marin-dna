"""Audit the approved curriculum on verified original train samples, without GPUs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from exp593_dna.anneal import TRIALS
from exp593_dna.anneal_data import GuidedAugmenter, GuidedSource
from exp593_dna.annotations import load_annotations
from exp593_dna.audit import verify_prep_identity
from exp593_dna.pipeline import RawWindow, load_encoder, reverse_complement


class SampleRows:
    """Audit-only finite original-row view, never substituted for production data."""

    def __init__(self, rows: list[RawWindow], *, original: bool) -> None:
        self.rows = rows
        self.by_id = {r.row: r for r in rows}
        self.original = original

    def get_batch(self, indices: list[int]) -> list[RawWindow]:
        return [
            self.by_id[i] if self.original else self.rows[i % len(self.rows)]
            for i in indices
        ]


def audit(prep: Path, annotations: Path, count: int) -> dict[str, Any]:
    verify_prep_identity(prep)
    raw = [
        RawWindow.from_dict(json.loads(line))
        for path in sorted((prep / "metadata/train").glob("*.sample.jsonl"))
        for line in path.read_text().splitlines()
    ]
    bank = load_annotations(annotations, rows={r.row for r in raw})
    augmenter = GuidedAugmenter(load_encoder())
    source = GuidedSource(
        SampleRows(raw, original=False), SampleRows(raw, original=True), bank, augmenter
    )
    stages = []
    for probability in (0.0, 0.075, 0.15, 0.30):
        counts: Counter[str] = Counter()
        target_counts: Counter[str] = Counter()
        length_counts: Counter[str] = Counter()
        inputs = pads = 0
        for start in range(0, count, 64):
            presentations = [
                augmenter.presentation(i, probability)
                for i in range(start, min(start + 64, count))
            ]
            examples = source.get_examples(presentations)
            for e in examples:
                a = e.attributes
                counts[a["kind"]] += 1
                counts["fim"] += int(a["annotation_guided"])
                counts["rc"] += int(a["rc"])
                counts["rc_fim"] += int(a["rc"] and a["annotation_guided"])
                counts["rc_causal"] += int(a["rc"] and not a["annotation_guided"])
                counts["guided"] += int(a["annotation_guided"])
                if a["annotation_guided"]:
                    counts["placement/" + a["landmark_mode"]] += 1
                    counts["landmark/" + a["landmark_class"]] += 1
                    counts["gap/" + str(a["gap_length"])] += 1
                counts["species/" + a["assembly"]] += 1
                target_counts[a["kind"]] += a["scored_tokens"]
                length_counts[str(a["bucket"])] += 1
                inputs += len(e.token_ids)
                pads += e.bucket
                assert np.isclose(e.loss_weights.sum(), 1)
                assert not a["metadata_fields"] and a["case"] == "unmasked_lower"
                assert all(
                    s.text == s.text.lower() for s in e.spans if s.role != "text"
                )
                original = source.original.by_id[a["raw_row"]]
                native = reverse_complement(original.seq) if a["rc"] else original.seq
                crop = a["crop_start_in_oriented_window"]
                roles = {s.role: s.text for s in e.spans if s.role != "text"}
                observed = (
                    roles["left"] + roles["middle"] + roles["right"]
                    if a["annotation_guided"]
                    else roles["dna"]
                )
                assert observed == native[crop : crop + a["dna_tokens"]].lower()
                if a["kind"].endswith("fim"):
                    assert a["annotation_guided"] and a["raw_row"] in bank
        stages.append(
            {
                "fim_probability": probability,
                "examples": count,
                "counts": dict(counts),
                "target_counts": dict(target_counts),
                "fim_example_and_objective_fraction": counts["fim"] / count,
                "guided_fraction_of_fim": counts["guided"] / max(counts["fim"], 1),
                "input_tokens": inputs,
                "padding_fraction": 1 - inputs / pads,
                "bucket_counts": dict(length_counts),
            }
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in stages[-1].items()
                    if k not in ("counts", "target_counts")
                }
            ),
            flush=True,
        )
    return {
        "scope": "Audit of verified original prep sample rows, not a full-corpus prevalence estimate",
        "raw_sample_rows": len(raw),
        "annotated_sample_rows": len(bank),
        "trials": {k: asdict(t) for k, t in TRIALS.items()},
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=25_000)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.examples < 1 or a.output.exists():
        raise ValueError("Require a positive audit size and a fresh output path")
    result = audit(a.prep, a.annotations, a.examples)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
