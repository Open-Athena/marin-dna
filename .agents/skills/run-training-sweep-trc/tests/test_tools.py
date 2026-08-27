# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

SKILL_ROOT = Path(__file__).parents[1]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


persistence = _load_module(
    "run_training_sweep_persistence",
    SKILL_ROOT / "scripts" / "persistence.py",
)
utilization = _load_module(
    "run_training_sweep_utilization",
    SKILL_ROOT / "scripts" / "utilization.py",
)


def _insert_trial_run(
    connection: sqlite3.Connection,
    trial_id: str,
    run_id: str,
    wandb_id: str,
    progress: float,
) -> None:
    connection.execute(
        "INSERT INTO trials(trial_id, env_json, status) VALUES (?, '{}', 'active')",
        (trial_id,),
    )
    connection.execute(
        """
        INSERT INTO runs(
            regional_run_id, trial_id, region, wandb_run_id,
            checkpoint_root, status, high_water_progress
        ) VALUES (?, ?, 'us-central1', ?, 'gs://checkpoints', 'running', ?)
        """,
        (run_id, trial_id, wandb_id, progress),
    )


def _insert_dispatch(
    connection: sqlite3.Connection,
    dispatch_id: str,
    run_id: str,
    attempt: int,
    tpu_slice: str,
    chips: int,
    submitted_at: str,
    ended_at: str | None,
) -> None:
    connection.execute(
        """
        INSERT INTO dispatches(
            dispatch_id, regional_run_id, attempt, iris_job_id, tpu_slice,
            chips, priority_band, command_redacted, submitted_at, ended_at,
            active, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, 'batch', 'launch', ?, ?, ?, ?)
        """,
        (
            dispatch_id,
            run_id,
            attempt,
            f"iris-{dispatch_id}",
            tpu_slice,
            chips,
            submitted_at,
            ended_at,
            int(ended_at is None),
            None if ended_at is None else "stopped",
        ),
    )


def _insert_observation(
    connection: sqlite3.Connection,
    run_id: str,
    dispatch_id: str,
    observed_at: str,
    progress: float,
    iris_running: bool,
) -> None:
    connection.execute(
        """
        INSERT INTO observations(
            regional_run_id, dispatch_id, observed_at,
            wandb_state, run_progress, iris_running
        ) VALUES (?, ?, ?, 'running', ?, ?)
        """,
        (run_id, dispatch_id, observed_at, progress, int(iris_running)),
    )


def test_target_rate_pools_trials_and_zero_progress_time(tmp_path: Path) -> None:
    database = tmp_path / "sweep.sqlite"
    persistence.init(database)
    with sqlite3.connect(database) as connection:
        _insert_trial_run(connection, "trial-1", "run-1", "wandb-1", 0.12)
        _insert_trial_run(connection, "trial-2", "run-2", "wandb-2", 0.0)
        _insert_dispatch(
            connection,
            "dispatch-1",
            "run-1",
            1,
            "v5p-8",
            8,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T01:00:00+00:00",
        )
        _insert_dispatch(
            connection,
            "dispatch-2",
            "run-2",
            1,
            "v5p-8",
            8,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T02:00:00+00:00",
        )
        _insert_observation(
            connection,
            "run-1",
            "dispatch-1",
            "2026-01-01T00:30:00+00:00",
            0.12,
            True,
        )

    placement = persistence.snapshot(database, recent_event_limit=0)["placements"]

    assert placement == [
        {
            "region": "us-central1",
            "slice": "v5p-8",
            "chips": 8,
            "dispatch_count": 2,
            "dispatches_with_progress": 1,
            "zero_progress_dispatches": 1,
            "active_dispatches": 0,
            "productive_dispatches": 0,
            "pending_dispatches": 0,
            "total_progress": 0.12,
            "total_wall_time_seconds": 10800.0,
            "target_rate": 0.04,
            "average_time_to_first_progress_seconds": 1800.0,
        }
    ]


def test_replacement_uses_only_its_own_observations(tmp_path: Path) -> None:
    database = tmp_path / "sweep.sqlite"
    persistence.init(database)
    with sqlite3.connect(database) as connection:
        _insert_trial_run(connection, "trial", "run", "wandb", 0.2)
        _insert_dispatch(
            connection,
            "old",
            "run",
            1,
            "v5p-8",
            8,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:10:00+00:00",
        )
        _insert_dispatch(
            connection,
            "new",
            "run",
            2,
            "v5p-16",
            16,
            "2026-01-01T00:10:00+00:00",
            None,
        )
        _insert_observation(
            connection,
            "run",
            "old",
            "2026-01-01T00:05:00+00:00",
            0.1,
            True,
        )
        _insert_observation(
            connection,
            "run",
            "old",
            "2026-01-01T00:11:00+00:00",
            0.2,
            False,
        )

    before = persistence.snapshot(database, recent_event_limit=0)
    assert before["fleet"]["wandb_running_chips"] == 0
    assert before["runs"][0]["active_dispatch"]["latest_observation"] is None
    assert before["conditions"][0]["kind"] == "active_dispatch_unobserved"

    with sqlite3.connect(database) as connection:
        _insert_observation(
            connection,
            "run",
            "new",
            "2026-01-01T00:12:00+00:00",
            0.3,
            True,
        )
        connection.execute(
            "UPDATE runs SET high_water_progress = 0.3 WHERE regional_run_id = 'run'"
        )

    after = persistence.snapshot(database, recent_event_limit=0)
    progress_by_slice = {
        placement["slice"]: placement["total_progress"]
        for placement in after["placements"]
    }
    assert progress_by_slice == {"v5p-8": 0.2, "v5p-16": 0.1}
    assert after["fleet"]["wandb_running_chips"] == 16


def test_snapshot_classifies_current_target_headroom(tmp_path: Path) -> None:
    database = tmp_path / "sweep.sqlite"
    persistence.init(database)
    now = datetime.now(UTC)
    old = (now.replace(microsecond=0) - timedelta(hours=2)).isoformat()
    recent = (now.replace(microsecond=0) - timedelta(minutes=30)).isoformat()
    with sqlite3.connect(database) as connection:
        _insert_trial_run(connection, "trial-1", "run-1", "wandb-1", 0.1)
        _insert_trial_run(connection, "trial-2", "run-2", "wandb-2", 0.0)
        _insert_dispatch(connection, "dispatch-1", "run-1", 1, "v5p-8", 8, old, None)
        _insert_dispatch(connection, "dispatch-2", "run-2", 1, "v5p-8", 8, old, None)
        _insert_observation(connection, "run-1", "dispatch-1", recent, 0.1, True)
        _insert_observation(connection, "run-2", "dispatch-2", recent, 0.0, True)

    result = persistence.snapshot(database, recent_event_limit=0)
    placement = result["placements"][0]

    assert placement["active_dispatches"] == 2
    assert placement["productive_dispatches"] == 1
    assert placement["pending_dispatches"] == 1
    assert result["runs"][0]["active_dispatch"]["recent_progress"] is True
    assert result["runs"][1]["active_dispatch"]["recent_progress"] is False

    shorter_window = persistence.snapshot(
        database, recent_event_limit=0, reslice_after_hours=0.25
    )["placements"][0]
    assert shorter_window["productive_dispatches"] == 0
    assert shorter_window["pending_dispatches"] == 2


def test_check_and_snapshot_reject_the_same_semantic_error(tmp_path: Path) -> None:
    database = tmp_path / "sweep.sqlite"
    persistence.init(database)
    with sqlite3.connect(database) as connection:
        _insert_trial_run(connection, "trial", "run", "wandb", 0.5)

    assert persistence.check(database) == [
        "semantic_check: run 'run' progress high-water does not match observations"
    ]
    with pytest.raises(RuntimeError, match="progress high-water"):
        persistence.snapshot(database, recent_event_limit=0)


def _utilization_response() -> dict[str, object]:
    def group(name: str, capacity: str, tasks: int, created_at: int, availability: str):
        return {
            "name": name,
            "device_type": "tpu",
            "device_variant": "v5p-8",
            "region": "us-central1",
            "availability_status": availability,
            "slice_state_counts": {"ready": 1},
            "slices": [
                {
                    "state": "ready",
                    "capacity_status": capacity,
                    "vms": [
                        {
                            "created_at": {"epoch_ms": created_at},
                            "running_task_count": tasks,
                        }
                    ],
                }
            ],
        }

    return {
        "status": {
            "groups": [
                group("one", "in_use", 1, 1_767_225_600_000, "available"),
                group("two", "available", 0, 1_767_222_000_000, "cooldown"),
            ]
        }
    }


def test_utilization_summarizes_structured_groups() -> None:
    observed_at = datetime(2026, 1, 1, 2, tzinfo=UTC)

    result = utilization.summarize(_utilization_response(), observed_at)

    assert result["fleet"]["ready_slices"] == 2
    assert result["fleet"]["in_use_percent"] == 50.0
    assert result["targets"] == [
        {
            "region": "us-central1",
            "tpu_slice": "v5p-8",
            "chips_per_slice": 8,
            "ready_slices": 2,
            "in_use_slices": 1,
            "in_use_percent": 50.0,
            "average_age_seconds": 9000,
            "slice_capacity_counts": {"available": 1, "in_use": 1},
            "autoscaler_status": "cooldown",
            "autoscaler_reasons": [],
        }
    ]


def test_utilization_rejects_inconsistent_slice_counts() -> None:
    response = copy.deepcopy(_utilization_response())
    response["status"]["groups"][0]["slice_state_counts"] = {"ready": 2}

    with pytest.raises(utilization.SnapshotError, match="slice counts disagree"):
        utilization.summarize(response, datetime(2026, 1, 1, 2, tzinfo=UTC))
