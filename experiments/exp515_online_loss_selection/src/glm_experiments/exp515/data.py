"""Deterministic sequence plans, collation, and case-distribution audit."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from glm_experiments.data.lm_datamodule import build_soft_mask
from glm_experiments.exp515.config import (
    NUCLEOTIDE_LENGTH,
    SEQUENCE_LENGTH,
    SHUFFLE_BUFFER_SIZE,
    TRAIN_DATASET,
    TRAIN_REVISION,
    TRAIN_SPECIES_KEY,
    TRAIN_TEXT_KEY,
)

PlanSignature = tuple[tuple[int, int, int], ...]
_PLAN_VALIDATION_CACHE: dict[Path, tuple[PlanSignature, dict[str, Any]]] = {}


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_stream(seed: int) -> Any:
    from datasets import load_dataset

    stream = load_dataset(
        TRAIN_DATASET,
        revision=TRAIN_REVISION,
        split="train",
        streaming=True,
    )
    return stream.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER_SIZE)


def _eligible_sequence(sequence: str) -> bool:
    return len(sequence) == NUCLEOTIDE_LENGTH and any(
        character.isupper() for character in sequence
    )


def build_sequence_plan(destination: Path, *, rows: int, seed: int) -> dict[str, Any]:
    """Materialize a compact fixed-width plan shared by every arm."""

    if rows <= 0:
        raise ValueError("sequence-plan rows must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    sequence_path = destination / "sequences.bin"
    species_path = destination / "species.u16"
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["rows"]) < rows:
            raise ValueError("existing sequence plan is shorter than requested")
        validate_sequence_plan(destination)
        return manifest

    temporary_sequence = destination / "sequences.bin.partial"
    temporary_species = destination / "species.u16.partial"
    species_to_id: dict[str, int] = {}
    filtered_all_lowercase = 0
    filtered_wrong_length = 0
    written = 0
    with (
        temporary_sequence.open("wb") as sequence_handle,
        temporary_species.open("wb") as species_handle,
    ):
        for record in _training_stream(seed):
            sequence = str(record[TRAIN_TEXT_KEY])
            if len(sequence) != NUCLEOTIDE_LENGTH:
                filtered_wrong_length += 1
                continue
            if not _eligible_sequence(sequence):
                filtered_all_lowercase += 1
                continue
            species = str(record[TRAIN_SPECIES_KEY])
            species_id = species_to_id.setdefault(species, len(species_to_id))
            if species_id >= 2**16:
                raise ValueError("sequence plan exceeds the uint16 species capacity")
            encoded = sequence.encode("ascii")
            if len(encoded) != NUCLEOTIDE_LENGTH:
                raise ValueError("training sequence is not one-byte ASCII")
            sequence_handle.write(encoded)
            species_handle.write(struct.pack("<H", species_id))
            written += 1
            if written == rows:
                break
    if written != rows:
        raise RuntimeError(f"training stream ended after {written} of {rows} rows")
    os.replace(temporary_sequence, sequence_path)
    os.replace(temporary_species, species_path)
    manifest = {
        "dataset": TRAIN_DATASET,
        "revision": TRAIN_REVISION,
        "seed": seed,
        "rows": rows,
        "nucleotide_length": NUCLEOTIDE_LENGTH,
        "species": species_to_id,
        "filtered_all_lowercase": filtered_all_lowercase,
        "filtered_wrong_length": filtered_wrong_length,
        "sequences_sha256": sha256_file(sequence_path),
        "species_sha256": sha256_file(species_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    validate_sequence_plan(destination)
    return manifest


def _plan_signature(directory: Path) -> PlanSignature:
    paths = (
        directory / "manifest.json",
        directory / "sequences.bin",
        directory / "species.u16",
    )
    return tuple(
        (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
        for path in paths
    )


def validate_sequence_plan(directory: Path) -> dict[str, Any]:
    """Validate plan lengths and immutable hashes."""

    cache_key = directory.resolve()
    signature = _plan_signature(directory)
    cached = _PLAN_VALIDATION_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = int(manifest["rows"])
    sequence_path = directory / "sequences.bin"
    species_path = directory / "species.u16"
    if sequence_path.stat().st_size != rows * NUCLEOTIDE_LENGTH:
        raise ValueError("sequence-plan byte length does not match its manifest")
    if species_path.stat().st_size != rows * 2:
        raise ValueError("species-plan byte length does not match its manifest")
    if sha256_file(sequence_path) != manifest["sequences_sha256"]:
        raise ValueError("sequence-plan checksum mismatch")
    if sha256_file(species_path) != manifest["species_sha256"]:
        raise ValueError("species-plan checksum mismatch")
    _PLAN_VALIDATION_CACHE[cache_key] = (_plan_signature(directory), manifest)
    return manifest


class SequencePlanDataset(Dataset[dict[str, Any]]):
    """Random-access view over one contiguous part of a fixed-width plan."""

    def __init__(self, directory: Path, *, start: int, rows: int) -> None:
        self.directory = directory
        self.manifest = validate_sequence_plan(directory)
        if start < 0 or rows <= 0 or start + rows > int(self.manifest["rows"]):
            raise ValueError("sequence-plan view is outside the materialized plan")
        self.start = start
        self.rows = rows
        self._sequence_handle: Any | None = None
        self._species_handle: Any | None = None
        self._id_to_species = {
            int(identifier): species
            for species, identifier in self.manifest["species"].items()
        }

    @property
    def sha256(self) -> str:
        return str(self.manifest["sequences_sha256"])

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.rows:
            raise IndexError(index)
        if self._sequence_handle is None:
            self._sequence_handle = (self.directory / "sequences.bin").open("rb")
            self._species_handle = (self.directory / "species.u16").open("rb")
        absolute = self.start + index
        self._sequence_handle.seek(absolute * NUCLEOTIDE_LENGTH)
        sequence = self._sequence_handle.read(NUCLEOTIDE_LENGTH).decode("ascii")
        assert self._species_handle is not None
        self._species_handle.seek(absolute * 2)
        species_id = struct.unpack("<H", self._species_handle.read(2))[0]
        return {
            "sample_id": absolute,
            "sequence": sequence,
            "species": self._id_to_species[species_id],
        }


class SequenceCollator:
    """Tokenize a plan batch and align source-case flags after one BOS."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        if tokenizer.bos_token_id is None:
            raise ValueError("checkpoint tokenizer must define a BOS token")
        self.tokenizer = tokenizer

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        sequences = [str(row["sequence"]) for row in rows]
        encoded = self.tokenizer(
            sequences,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].long()
        attention_mask = encoded["attention_mask"].bool()
        expected = (len(rows), SEQUENCE_LENGTH)
        if tuple(input_ids.shape) != expected:
            raise ValueError(
                f"tokenized plan batch has shape {tuple(input_ids.shape)}, expected {expected}"
            )
        soft_masked = build_soft_mask(
            sequences,
            input_ids,
            bos_token_id=int(self.tokenizer.bos_token_id),
            require_leading_bos=True,
        )
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": attention_mask,
            "soft_masked": soft_masked,
            "sample_ids": torch.tensor(
                [int(row["sample_id"]) for row in rows], dtype=torch.long
            ),
            "species": [str(row["species"]) for row in rows],
        }


def _run_lengths(sequence: str, *, lowercase: bool) -> list[int]:
    lengths: list[int] = []
    current = 0
    for character in sequence:
        matches = character.islower() == lowercase
        if matches:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    ordered = sorted(values)
    result: dict[str, float] = {}
    for name, fraction in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        index = round((len(ordered) - 1) * fraction)
        result[name] = float(ordered[index])
    return result


def audit_case_distribution(
    output_path: Path,
    *,
    samples_per_species: int = 128,
    expected_species: int = 108,
    seed: int = 1515,
) -> dict[str, Any]:
    """Stream a fixed sample from every species and audit source-case behavior."""

    if samples_per_species <= 0:
        raise ValueError("samples_per_species must be positive")
    samples: dict[str, list[str]] = defaultdict(list)
    scanned = 0
    for record in _training_stream(seed):
        scanned += 1
        sequence = str(record[TRAIN_TEXT_KEY])
        species = str(record[TRAIN_SPECIES_KEY])
        if len(sequence) != NUCLEOTIDE_LENGTH:
            continue
        if len(samples[species]) < samples_per_species:
            samples[species].append(sequence)
        if len(samples) == expected_species and all(
            len(values) == samples_per_species for values in samples.values()
        ):
            break
    if len(samples) != expected_species:
        raise RuntimeError(
            f"case audit observed {len(samples)} species, expected {expected_species}"
        )
    incomplete = {
        species: len(values)
        for species, values in samples.items()
        if len(values) != samples_per_species
    }
    if incomplete:
        raise RuntimeError(f"case audit has incomplete species samples: {incomplete}")

    rows: list[dict[str, Any]] = []
    for species, sequences in sorted(samples.items()):
        lowercase_bases = sum(
            character.islower() for sequence in sequences for character in sequence
        )
        eligible = [
            sum(character.isupper() for character in sequence) for sequence in sequences
        ]
        lowercase_runs = [
            length
            for sequence in sequences
            for length in _run_lengths(sequence, lowercase=True)
        ]
        uppercase_runs = [
            length
            for sequence in sequences
            for length in _run_lengths(sequence, lowercase=False)
        ]
        row = {
            "species": species,
            "samples": len(sequences),
            "lowercase_base_fraction": lowercase_bases
            / (len(sequences) * NUCLEOTIDE_LENGTH),
            "all_lowercase_sequence_fraction": sum(
                not any(character.isupper() for character in sequence)
                for sequence in sequences
            )
            / len(sequences),
            "eligible_targets": _quantiles(eligible),
            "lowercase_run_length": _quantiles(lowercase_runs),
            "uppercase_run_length": _quantiles(uppercase_runs),
        }
        row["obvious_case_encoding_break"] = bool(
            row["eligible_targets"]["p50"] == 0.0
            or row["all_lowercase_sequence_fraction"] >= 0.99
        )
        rows.append(row)
    payload = {
        "dataset": TRAIN_DATASET,
        "revision": TRAIN_REVISION,
        "seed": seed,
        "samples_per_species": samples_per_species,
        "expected_species": expected_species,
        "scanned_rows": scanned,
        "fallback_required": any(row["obvious_case_encoding_break"] for row in rows),
        "species": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
