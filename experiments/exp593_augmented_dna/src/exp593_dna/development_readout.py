"""Frozen development completions and tie-aware metrics, independent of training."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from exp593_dna.development import ConservationSite
from exp593_dna.pipeline import Encoder, Example, Span, reverse_complement

DEVELOPMENT_MANIFEST_SHA256 = (
    "73904d171e8d90b1345b0688d20338f31dc9146de9e5c7ff1be3ab66d98a2c0a"
)
DEVELOPMENT_MANIFEST_URI = (
    "s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/validation/"
    "development-v1-09067653218f16aad613263fe30d51bbb2a66c796903145df6243ca244cdff48/manifest.json"
)


def completion_target(example: Example) -> tuple[np.ndarray, np.ndarray]:
    """Require one final contiguous scored span, including bare causal targets."""
    length = example.scored_tokens
    expected = np.arange(
        len(example.token_ids) - length - 1, len(example.token_ids) - 1
    )
    if length < 1 or not np.array_equal(np.flatnonzero(example.loss_weights), expected):
        raise ValueError("Development targets must be the final contiguous span")
    return example.token_ids[:-length], example.token_ids[-length:]


def conservation_examples(
    site: ConservationSite, encoder: Encoder
) -> tuple[Example, ...]:
    """Read the true base from left, right-RC, and blind two-flank contexts.

    Bare views contain only lowercase DNA and no task instruction. The forward
    target is source index 255; the reverse target is index 256, not 255.
    """
    dna, index = site.sequence.lower(), site.target_index
    rc = reverse_complement(dna)
    reverse_index = len(dna) - index - 1
    views = (
        ("left", (Span(dna[:index], "left"), Span(dna[index], "middle", True))),
        (
            "right_rc",
            (Span(rc[:reverse_index], "left"), Span(rc[reverse_index], "middle", True)),
        ),
        (
            "fim",
            (
                Span("[left] ", "text"),
                Span(dna[:index], "left"),
                Span("\n[right] ", "text"),
                Span(dna[index + 1 :], "right"),
                Span("\n[gap:1]\n[fill] ", "text"),
                Span(dna[index], "middle", True),
            ),
        ),
    )
    examples = []
    for view, spans in views:
        tokens, weights = encoder.encode(spans)
        if len(tokens) > 1024:
            raise ValueError("Conservation readout exceeds its fixed bucket")
        example = Example(
            tokens,
            weights,
            spans,
            1024,
            {
                "conditioning": view,
                "source_row": site.source_row,
                "use": "evaluation_only",
            },
        )
        completion_target(example)
        examples.append(example)
    return tuple(examples)


def junction_members(
    record: dict[str, Any], encoder: Encoder
) -> tuple[Example, Example]:
    """Re-encode both frozen prompts and verify their shared crop and orientation."""
    examples = []
    for member in record["members"]:
        spans = tuple(Span(**s) for s in member["spans"])
        tokens, weights = encoder.encode(spans)
        attributes = member["attributes"]
        example = Example(tokens, weights, spans, member["bucket"], attributes)
        parts = {s.role: s.text for s in spans if s.role != "text"}
        expected_text = "[left] \n[right] \n[gap:2]\n[fill] "
        if (
            "".join(s.text for s in spans if s.role == "text") != expected_text
            or len(tokens) != attributes["input_tokens"]
            or len(tokens) > example.bucket
            or len(completion_target(example)[1]) != 2
            or attributes["raw_row"] != record["source_row"]
            or attributes["use"] != "evaluation_only"
            or attributes["case"] != "unmasked_lower"
            or len(parts["left"]) != attributes["gap_start"]
        ):
            raise ValueError("Frozen blind junction presentation changed")
        dna = parts["left"] + parts["middle"] + parts["right"]
        if len(dna) != 8064 or dna != dna.lower():
            raise ValueError("Frozen junction crop changed")
        examples.append(example)
    if len(examples) != 2:
        raise ValueError("Expected one junction and its paired random control")
    first, second = examples
    for key in ("raw_row", "rc", "assembly", "source_class"):
        if first.attributes[key] != second.attributes[key]:
            raise ValueError("Junction/control source identity differs")
    dna = [
        "".join(
            next(s.text for s in e.spans if s.role == role)
            for role in ("left", "middle", "right")
        )
        for e in examples
    ]
    if (
        dna[0] != dna[1]
        or first.attributes["development_task"] != record["annotation"]["kind"]
        or second.attributes["development_task"] != "random_control"
    ):
        raise ValueError("Junction/control pairing differs")
    return first, second


@dataclass(frozen=True)
class DevelopmentPanels:
    conservation: tuple[ConservationSite, ...]
    junctions: tuple[Example, ...]

    @classmethod
    def open(cls, directory: Path, encoder: Encoder) -> DevelopmentPanels:
        raw_manifest = (directory / "manifest.json").read_bytes()
        if hashlib.sha256(raw_manifest).hexdigest() != DEVELOPMENT_MANIFEST_SHA256:
            raise ValueError("Frozen development manifest changed")
        manifest = json.loads(raw_manifest)
        for name, record in manifest["files"].items():
            payload = (directory / name).read_bytes()
            if (
                len(payload) != record["bytes"]
                or hashlib.sha256(payload).hexdigest() != record["sha256"]
            ):
                raise ValueError("Frozen development payload changed")
        conservation = tuple(
            ConservationSite(**json.loads(line))
            for line in (directory / "conservation_development.jsonl")
            .read_text()
            .splitlines()
        )
        junctions = tuple(
            e
            for line in (directory / "junction_development.jsonl")
            .read_text()
            .splitlines()
            for e in junction_members(json.loads(line), encoder)
        )
        junction_manifest = json.loads(
            (directory / "junction_development_manifest.json").read_text()
        )
        if (
            len(conservation) != 512
            or Counter(s.label for s in conservation) != {0: 256, 1: 256}
            or len(junctions) != 128
            or len({e.attributes["raw_row"] for e in junctions}) != 64
            or encoder.digest != junction_manifest["tokenizer_sha256"]
        ):
            raise ValueError("Frozen development membership or tokenizer changed")
        return cls(conservation, junctions)


def fetch_development(directory: Path, encoder: Encoder) -> DevelopmentPanels:
    """Cache the tiny pinned bundle once per node, then verify every input byte."""
    from filelock import FileLock
    from rigging.filesystem.storage_path import StoragePath

    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".fetch.lock")):
        raw_manifest = StoragePath(DEVELOPMENT_MANIFEST_URI).read_bytes()
        if hashlib.sha256(raw_manifest).hexdigest() != DEVELOPMENT_MANIFEST_SHA256:
            raise ValueError("Native development manifest changed")
        manifest = json.loads(raw_manifest)
        for name, record in manifest["files"].items():
            path = directory / name
            if (
                path.exists()
                and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            ):
                continue
            payload = StoragePath(record["uri"]).read_bytes()
            if (
                len(payload) != record["bytes"]
                or hashlib.sha256(payload).hexdigest() != record["sha256"]
            ):
                raise ValueError("Native development payload changed")
            path.write_bytes(payload)
        (directory / "manifest.json").write_bytes(raw_manifest)
        return DevelopmentPanels.open(directory, encoder)


def binary_ranking_metrics(
    labels: np.ndarray, scores: np.ndarray
) -> dict[str, float | int]:
    """AUROC gives ties half credit; AP groups equal-score thresholds together."""
    y, s = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if (
        y.ndim != 1
        or s.shape != y.shape
        or set(y.tolist()) != {0, 1}
        or not np.isfinite(s).all()
    ):
        raise ValueError("Ranking requires finite binary scores with both classes")
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]) + 1, len(s)]
    positives = np.cumsum(y, dtype=np.int64)[ends - 1]
    negatives = ends - positives
    p, n = int(positives[-1]), int(negatives[-1])
    recall = np.r_[0, positives / p]
    false_positive_rate = np.r_[0, negatives / n]
    return {
        "auroc": float(np.trapezoid(recall, false_positive_rate)),
        "average_precision": float(np.sum(np.diff(recall) * positives / ends)),
        "examples": len(y),
        "positives": p,
        "negatives": n,
    }
