"""Detached selector-composition, throughput, memory, and budget diagnostics."""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from lightning import Callback, LightningModule, Trainer

from glm_experiments.exp515.config import (
    EFFECTIVE_BATCH_SIZE,
    GPU_COMPUTE_CAP_USD,
    GPU_PRICE_PER_HOUR_USD,
    SEQUENCE_LENGTH,
)

CountKey = tuple[str, str]


def _add_grouped(
    counts: dict[CountKey, list[int]],
    dimension: str,
    categories: torch.Tensor,
    eligible: torch.Tensor,
    selected: torch.Tensor,
    labels: dict[int, str],
) -> None:
    for value, label in labels.items():
        group = categories == value
        counts[(dimension, label)][0] += int((eligible & group).sum())
        counts[(dimension, label)][1] += int((selected & group).sum())


def _repeat_boundary_buckets(soft_masked: torch.Tensor) -> torch.Tensor:
    result = torch.full(soft_masked.shape, 5, dtype=torch.long)
    for row in range(soft_masked.shape[0]):
        values = soft_masked[row]
        boundaries = torch.nonzero(values[1:] != values[:-1]).flatten() + 1
        if not boundaries.numel():
            continue
        positions = torch.arange(values.numel())
        distances = (positions[:, None] - boundaries[None, :]).abs().min(dim=1).values
        result[row][distances == 0] = 0
        result[row][distances == 1] = 1
        result[row][(distances >= 2) & (distances <= 3)] = 2
        result[row][(distances >= 4) & (distances <= 7)] = 3
        result[row][(distances >= 8) & (distances <= 15)] = 4
    return result


def _local_gc_buckets(
    target_ids: torch.Tensor,
    *,
    gc_token_ids: tuple[int, int],
) -> torch.Tensor:
    gc = (
        ((target_ids == gc_token_ids[0]) | (target_ids == gc_token_ids[1]))
        .float()
        .unsqueeze(1)
    )
    fractions = F.avg_pool1d(
        gc,
        kernel_size=21,
        stride=1,
        padding=10,
        count_include_pad=False,
    ).squeeze(1)
    return torch.clamp((fractions * 10).floor().long(), max=9)


def _sevenmer_frequency_buckets(
    target_ids: torch.Tensor,
    *,
    nucleotide_token_ids: tuple[int, int, int, int],
) -> torch.Tensor:
    digits = torch.full_like(target_ids, -1)
    for digit, token_id in enumerate(nucleotide_token_ids):
        digits[target_ids == token_id] = digit
    result = torch.zeros_like(target_ids)
    if target_ids.shape[1] < 7:
        return result
    windows = digits.unfold(1, 7, 1)
    valid = (windows >= 0).all(dim=-1)
    powers = torch.tensor([4**value for value in range(6, -1, -1)])
    codes = (windows.clamp_min(0) * powers).sum(dim=-1)
    flattened = codes[valid]
    frequencies = torch.zeros_like(codes)
    if flattened.numel():
        unique, inverse, counts = torch.unique(
            flattened,
            return_inverse=True,
            return_counts=True,
        )
        del unique
        frequencies[valid] = counts[inverse]
    buckets = torch.ones_like(frequencies)
    buckets[(frequencies >= 2) & (frequencies <= 3)] = 2
    buckets[(frequencies >= 4) & (frequencies <= 7)] = 3
    buckets[frequencies >= 8] = 4
    buckets[~valid] = 0
    result[:, 3:-3] = buckets
    return result


def selector_composition_counts(
    diagnostic: dict[str, torch.Tensor],
    *,
    nucleotide_token_ids: dict[str, int] | None = None,
) -> dict[CountKey, list[int]]:
    """Summarize selection composition from detached current-batch tensors."""

    eligible = diagnostic["eligible_mask"].bool().cpu()
    selected = diagnostic["selected_mask"].bool().cpu()
    target_ids = diagnostic["input_ids"][:, 1:].long().cpu()
    target_soft_masked = diagnostic["soft_masked"][:, 1:].bool().cpu()
    if eligible.shape != target_ids.shape or selected.shape != target_ids.shape:
        raise ValueError("selector diagnostics have inconsistent causal alignment")
    token_ids = nucleotide_token_ids or {"A": 3, "C": 4, "G": 5, "T": 6}
    if set(token_ids) != set("ACGT") or len(set(token_ids.values())) != 4:
        raise ValueError("nucleotide token IDs must map distinct A/C/G/T tokens")
    counts: dict[CountKey, list[int]] = defaultdict(lambda: [0, 0])
    _add_grouped(
        counts,
        "target_nucleotide",
        target_ids,
        eligible,
        selected,
        {identifier: base for base, identifier in token_ids.items()},
    )
    positions = torch.arange(1, target_ids.shape[1] + 1).expand_as(target_ids)
    _add_grouped(
        counts,
        "sequence_position",
        positions,
        eligible,
        selected,
        {position: str(position) for position in range(1, target_ids.shape[1] + 1)},
    )
    _add_grouped(
        counts,
        "repeat_boundary_distance",
        _repeat_boundary_buckets(target_soft_masked),
        eligible,
        selected,
        {0: "0", 1: "1", 2: "2-3", 3: "4-7", 4: "8-15", 5: "16+_or_none"},
    )
    _add_grouped(
        counts,
        "local_gc_fraction",
        _local_gc_buckets(
            target_ids,
            gc_token_ids=(token_ids["C"], token_ids["G"]),
        ),
        eligible,
        selected,
        {index: f"{index / 10:.1f}-{(index + 1) / 10:.1f}" for index in range(10)},
    )
    _add_grouped(
        counts,
        "local_7mer_frequency",
        _sevenmer_frequency_buckets(
            target_ids,
            nucleotide_token_ids=tuple(token_ids[base] for base in "ACGT"),
        ),
        eligible,
        selected,
        {0: "edge_or_noncanonical", 1: "1", 2: "2-3", 3: "4-7", 4: "8+"},
    )
    return counts


def _append_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


class Exp515Diagnostics(Callback):
    """Write authoritative CSV diagnostics and enforce the compute stop."""

    def __init__(
        self,
        output_dir: Path,
        *,
        every_n_steps: int = 10,
        sequence_length: int = SEQUENCE_LENGTH,
        nucleotide_token_ids: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")
        self.output_dir = output_dir
        self.every_n_steps = every_n_steps
        self.sequence_length = sequence_length
        self.nucleotide_token_ids = nucleotide_token_ids
        self._last_step: int | None = None
        self._last_resource_step: int | None = None
        self._starting_step = 0
        self._counts: dict[CountKey, list[int]] = defaultdict(lambda: [0, 0])
        self._start = time.time()
        self._instance_start = float(
            os.getenv("EXP515_INSTANCE_START_UNIX", self._start)
        )
        self._prior_cost = float(os.getenv("EXP515_PRIOR_COST_USD", "0"))

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Capture the restored global step for phase-local throughput."""

        del pl_module
        self._starting_step = int(trainer.global_step)
        self._start = time.time()

    def _flush(self, step: int) -> None:
        if step % self.every_n_steps or not self._counts:
            self._counts.clear()
            return
        rows = []
        for (dimension, category), (eligible, selected) in sorted(self._counts.items()):
            rows.append(
                {
                    "optimizer_step": step,
                    "dimension": dimension,
                    "category": category,
                    "eligible_count": eligible,
                    "selected_count": selected,
                    "selection_rate": selected / eligible if eligible else "",
                }
            )
        _append_csv(
            self.output_dir / "selection_composition.csv",
            [
                "optimizer_step",
                "dimension",
                "category",
                "eligible_count",
                "selected_count",
                "selection_rate",
            ],
            rows,
        )
        self._counts.clear()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        step = int(trainer.global_step)
        if self._last_step is not None and step != self._last_step:
            self._flush(self._last_step)
        self._last_step = step
        diagnostic = getattr(pl_module, "last_selector_diagnostics", None)
        if step > 0 and step % self.every_n_steps == 0 and diagnostic is not None:
            observed = selector_composition_counts(
                diagnostic,
                nucleotide_token_ids=self.nucleotide_token_ids,
            )
            for key, values in observed.items():
                self._counts[key][0] += values[0]
                self._counts[key][1] += values[1]
        elapsed = time.time() - self._instance_start
        compute_cost = self._prior_cost + elapsed / 3600 * GPU_PRICE_PER_HOUR_USD
        if compute_cost >= GPU_COMPUTE_CAP_USD:
            raise RuntimeError(
                f"issue #515 compute stop reached ${compute_cost:.2f} of ${GPU_COMPUTE_CAP_USD:.2f}"
            )
        if step and step % self.every_n_steps == 0 and step != self._last_resource_step:
            self._last_resource_step = step
            peak_memory = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )
            wall = time.time() - self._start
            phase_steps = step - self._starting_step
            _append_csv(
                self.output_dir / "resource_usage.csv",
                [
                    "optimizer_step",
                    "wall_seconds",
                    "processed_input_tokens",
                    "input_tokens_per_second",
                    "peak_memory_bytes",
                    "estimated_compute_cost_usd",
                ],
                [
                    {
                        "optimizer_step": step,
                        "wall_seconds": wall,
                        "processed_input_tokens": phase_steps
                        * EFFECTIVE_BATCH_SIZE
                        * self.sequence_length,
                        "input_tokens_per_second": phase_steps
                        * EFFECTIVE_BATCH_SIZE
                        * self.sequence_length
                        / max(wall, 1e-9),
                        "peak_memory_bytes": peak_memory,
                        "estimated_compute_cost_usd": compute_cost,
                    }
                ],
            )

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer, pl_module
        if self._last_step is not None:
            self._flush(self._last_step)
