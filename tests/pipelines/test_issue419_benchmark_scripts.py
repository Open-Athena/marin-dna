"""Focused tests for the issue #419 benchmark reporting scripts."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.issue419_validate_benchmark import (
    _flat_index_to_position,
    summarize_gpu_csv,
)


def test_flat_index_to_position_uses_half_open_window_offsets():
    shard = SimpleNamespace(
        log_probabilities=np.zeros((5, 4), dtype=np.float32),
        score_offsets=np.array([0, 2, 5]),
        emit_start=np.array([10, 20]),
        emit_end=np.array([12, 23]),
    )
    assert [_flat_index_to_position(shard, index) for index in range(5)] == [
        10,
        11,
        20,
        21,
        22,
    ]


def test_summarize_gpu_csv_reports_active_utilization_and_peaks(tmp_path):
    path = tmp_path / "gpu.csv"
    path.write_text(
        "timestamp, name, utilization.gpu [%], utilization.memory [%], "
        "memory.used [MiB], memory.total [MiB], power.draw [W], "
        "temperature.gpu, clocks.current.sm [MHz]\n"
        "2026/07/31 00:00:00, NVIDIA A10G, 0 %, 0 %, 0 MiB, 23028 MiB, "
        "10 W, 30, 0 MHz\n"
        "2026/07/31 00:00:05, NVIDIA A10G, 50 %, 10 %, 5000 MiB, 23028 MiB, "
        "100 W, 40, 1500 MHz\n"
        "2026/07/31 00:00:10, NVIDIA A10G, 100 %, 20 %, 6000 MiB, 23028 MiB, "
        "200 W, 50, 1700 MHz\n"
    )
    summary = summarize_gpu_csv(path)
    assert summary["sample_count"] == 3
    assert summary["active_sample_count"] == 2
    assert summary["active_utilization_mean_percent"] == pytest.approx(75)
    assert summary["active_utilization_p50_percent"] == pytest.approx(75)
    assert summary["active_utilization_p95_percent"] == pytest.approx(97.5)
    assert summary["peak_memory_mib"] == 6000
    assert summary["peak_power_watts"] == 200
    assert summary["peak_temperature_c"] == 50
