"""Append-only benchmark evidence and per-backend saturation accounting."""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

Backend = Literal["torch", "vllm", "jax"]


@dataclass(frozen=True)
class Trial:
    """One timed optimization attempt on an available single H100."""

    backend: Backend
    name: str
    productive_seconds: float
    sequences_per_second: float
    median_latency_seconds: float
    accepted: bool
    correctness_max_abs_error: float
    notes: str
    timestamp: str
    tokens_per_sequence: int = 8192
    batch_size: int = 1

    @property
    def input_tokens_per_second(self) -> float:
        """Return portable input throughput for fixed-length scoring workloads."""
        return self.sequences_per_second * self.tokens_per_sequence

    @classmethod
    def now(
        cls,
        *,
        backend: Backend,
        name: str,
        productive_seconds: float,
        sequences_per_second: float,
        median_latency_seconds: float,
        accepted: bool,
        correctness_max_abs_error: float,
        notes: str,
        tokens_per_sequence: int = 8192,
        batch_size: int = 1,
    ) -> "Trial":
        """Create a trial with a stable UTC timestamp."""
        return cls(
            backend=backend,
            name=name,
            productive_seconds=productive_seconds,
            sequences_per_second=sequences_per_second,
            median_latency_seconds=median_latency_seconds,
            accepted=accepted,
            correctness_max_abs_error=correctness_max_abs_error,
            notes=notes,
            timestamp=datetime.now(UTC).isoformat(),
            tokens_per_sequence=tokens_per_sequence,
            batch_size=batch_size,
        )


def append_trial(path: Path, trial: Trial) -> None:
    """Persist each attempt immediately so preemption cannot erase evidence."""
    if trial.productive_seconds < 0:
        raise ValueError("Productive time cannot be negative")
    if trial.sequences_per_second <= 0 or trial.median_latency_seconds <= 0:
        raise ValueError("Timing metrics must be positive")
    if trial.tokens_per_sequence <= 0:
        raise ValueError("Tokens per sequence must be positive")
    if trial.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if trial.correctness_max_abs_error < 0:
        raise ValueError("Correctness error cannot be negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(asdict(trial), sort_keys=True) + "\n")


def load_trials(path: Path) -> list[Trial]:
    """Read a ledger while rejecting malformed partial records."""
    if not path.exists():
        return []
    return [Trial(**json.loads(line)) for line in path.read_text().splitlines() if line]


def progress(trials: list[Trial]) -> list[tuple[Trial, float, float]]:
    """Return trials with cumulative productive time and best accepted throughput."""
    elapsed = 0.0
    best = 0.0
    points: list[tuple[Trial, float, float]] = []
    for trial in trials:
        elapsed += trial.productive_seconds
        if trial.accepted:
            best = max(best, trial.sequences_per_second)
        points.append((trial, elapsed, best))
    return points


def productive_seconds_since_gain(trials: list[Trial]) -> float:
    """Measure only available-H100 experiment time after the last accepted gain."""
    best = 0.0
    elapsed = 0.0
    for trial in trials:
        elapsed += trial.productive_seconds
        if trial.accepted and trial.sequences_per_second > best:
            best = trial.sequences_per_second
            elapsed = 0.0
    return elapsed
