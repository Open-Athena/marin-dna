from pathlib import Path

import pytest

from exp586_moe.inference.ledger import (
    Trial,
    append_trial,
    load_trials,
    productive_seconds_since_gain,
    progress,
)
from exp586_moe.inference.plot_progress import plot_backend


def trial(name: str, *, accepted: bool, rate: float, seconds: float = 60) -> Trial:
    return Trial.now(
        backend="torch",
        name=name,
        productive_seconds=seconds,
        sequences_per_second=rate,
        median_latency_seconds=1 / rate,
        accepted=accepted,
        correctness_max_abs_error=0,
        notes="test",
    )


def test_ledger_tracks_frontier_and_stagnation(tmp_path: Path) -> None:
    path = tmp_path / "torch.jsonl"
    records = [
        trial("baseline", accepted=True, rate=1),
        trial("rejected", accepted=False, rate=0.9, seconds=100),
        trial("gain", accepted=True, rate=1.2, seconds=200),
        trial("later", accepted=False, rate=1.1, seconds=300),
    ]
    for record in records:
        append_trial(path, record)

    loaded = load_trials(path)
    assert [best for _, _, best in progress(loaded)] == [1, 1, 1.2, 1.2]
    assert loaded[0].input_tokens_per_second == 8192
    assert productive_seconds_since_gain(loaded) == 300
    output = tmp_path / "torch.png"
    plot_backend("torch", path, output)
    assert output.stat().st_size > 1_000


def test_ledger_rejects_impossible_timing(tmp_path: Path) -> None:
    invalid = trial("bad", accepted=False, rate=1)
    invalid = Trial(**{**invalid.__dict__, "median_latency_seconds": 0})
    with pytest.raises(ValueError, match="positive"):
        append_trial(tmp_path / "torch.jsonl", invalid)


def test_ledger_records_batch_throughput(tmp_path: Path) -> None:
    batched = Trial.now(
        backend="torch",
        name="batched",
        productive_seconds=1,
        sequences_per_second=10,
        median_latency_seconds=0.4,
        accepted=True,
        correctness_max_abs_error=0,
        notes="test",
        batch_size=4,
    )
    append_trial(tmp_path / "torch.jsonl", batched)
    assert batched.input_tokens_per_second == 81_920
