"""Pinned sequence plans and deterministic training/validation collation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from exp479_mntp.config import DATA_COMPONENTS, NUCLEOTIDE_LENGTH, SEQUENCE_LENGTH
from exp479_mntp.loss import causal_supervision
from exp479_mntp.masking import (
    corrupt_fixed_rate_mntp,
    corrupt_for_mntp,
    corrupt_single_mask,
)

Objective = Literal["mntp", "clm"]
ValidationMode = Literal["diffusion", "single"]


def plan_sha256(path: Path) -> str:
    """Hash a sequence plan without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SequencePlanDataset(Dataset[dict[str, Any]]):
    """A compact, immutable sequence plan shared by every trained arm."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with path.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if not self.rows:
            raise ValueError(f"empty sequence plan: {path}")
        for expected_id, row in enumerate(self.rows):
            if row["sample_id"] != expected_id:
                raise ValueError(
                    f"plan sample IDs must be contiguous; row {expected_id} has {row['sample_id']}"
                )
            if len(row["sequence"]) != NUCLEOTIDE_LENGTH:
                raise ValueError(
                    f"sample {expected_id} has {len(row['sequence'])} bases, expected {NUCLEOTIDE_LENGTH}"
                )
        self.sha256 = plan_sha256(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _lowercase_mask(sequences: Sequence[str]) -> torch.Tensor:
    mask = torch.zeros((len(sequences), SEQUENCE_LENGTH), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        for position, base in enumerate(sequence, start=1):
            mask[row, position] = base in "acgt"
    return mask


class SequenceCollator:
    """Tokenize raw sequences and build deterministic MNTP or CLM supervision."""

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        objective: Objective,
        canonical_token_ids: tuple[int, ...],
        mask_token_id: int | None,
        seed: int,
        validation_mode: ValidationMode = "diffusion",
        fixed_mask_probability: float | None = None,
    ) -> None:
        if objective == "mntp" and mask_token_id is None:
            raise ValueError("MNTP collation requires a mask token")
        self.tokenizer = tokenizer
        self.objective = objective
        self.canonical_token_ids = canonical_token_ids
        self.mask_token_id = mask_token_id
        self.seed = seed
        self.validation_mode = validation_mode
        self.fixed_mask_probability = fixed_mask_probability

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        sequences = [str(row["sequence"]) for row in rows]
        sample_ids = torch.tensor([int(row["sample_id"]) for row in rows], dtype=torch.long)
        encoded = self.tokenizer(
            sequences,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].long()
        attention_mask = encoded["attention_mask"].long()
        if input_ids.shape != (len(rows), SEQUENCE_LENGTH):
            raise ValueError(f"unexpected tokenized shape {tuple(input_ids.shape)}")
        if not torch.all(input_ids[:, 0] == self.tokenizer.bos_token_id):
            raise ValueError("every sequence must begin with BOS")
        lowercase = _lowercase_mask(sequences)

        batch: dict[str, Any] = {
            "attention_mask": attention_mask,
            "components": [str(row["component"]) for row in rows],
            "sample_ids": sample_ids,
        }
        if self.objective == "clm":
            labels, weights = causal_supervision(input_ids, lowercase)
            batch.update(
                input_ids=input_ids,
                labels=labels,
                loss_weights=weights,
                mask_probabilities=torch.zeros(len(rows), dtype=torch.float32),
            )
            return batch

        assert self.mask_token_id is not None
        if self.validation_mode == "diffusion" and self.fixed_mask_probability is not None:
            corrupted = corrupt_fixed_rate_mntp(
                input_ids,
                lowercase,
                sample_ids,
                mask_token_id=self.mask_token_id,
                canonical_token_ids=self.canonical_token_ids,
                seed=self.seed,
                mask_probability=self.fixed_mask_probability,
            )
        else:
            corruption_fn = (
                corrupt_for_mntp if self.validation_mode == "diffusion" else corrupt_single_mask
            )
            corrupted = corruption_fn(
                input_ids,
                lowercase,
                sample_ids,
                mask_token_id=self.mask_token_id,
                canonical_token_ids=self.canonical_token_ids,
                seed=self.seed,
            )
        batch.update(
            input_ids=corrupted.input_ids,
            labels=corrupted.labels,
            loss_weights=corrupted.loss_weights,
            mask_probabilities=corrupted.mask_probabilities,
        )
        return batch


def _stream_component(
    *,
    repo: str,
    revision: str,
    text_key: str,
    split: Literal["train", "validation"],
    seed: int,
    shuffle_buffer_size: int,
) -> Iterator[str]:
    from datasets import load_dataset

    dataset = load_dataset(repo, split=split, revision=revision, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    for row in dataset:
        sequence = str(row[text_key])
        if len(sequence) != NUCLEOTIDE_LENGTH:
            raise ValueError(
                f"{repo}@{revision} yielded {len(sequence)} bases; expected {NUCLEOTIDE_LENGTH}"
            )
        if not any(base in "ACGTacgt" for base in sequence):
            continue
        yield sequence


def build_sequence_plan(
    output_path: Path,
    *,
    samples_per_component: int,
    seed: int,
    validation: bool,
    shuffle_buffer_size: int = 10_000,
) -> str:
    """Materialize one pinned, uniform five-component sequence plan."""

    if samples_per_component <= 0:
        raise ValueError("samples_per_component must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    streams: list[Iterator[str]] = []
    for index, component in enumerate(DATA_COMPONENTS):
        streams.append(
            _stream_component(
                repo=component.validation_repo if validation else component.train_repo,
                revision=component.validation_revision if validation else component.train_revision,
                text_key=(
                    component.validation_text_key if validation else component.train_text_key
                ),
                split="validation" if validation else "train",
                seed=seed + index,
                shuffle_buffer_size=shuffle_buffer_size,
            )
        )

    with output_path.open("w", encoding="utf-8") as handle:
        sample_id = 0
        for _ in range(samples_per_component):
            for component, stream in zip(DATA_COMPONENTS, streams, strict=True):
                row = {
                    "sample_id": sample_id,
                    "component": component.name,
                    "sequence": next(stream),
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                sample_id += 1
    return plan_sha256(output_path)
