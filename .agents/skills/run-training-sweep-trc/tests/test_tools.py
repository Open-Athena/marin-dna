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
    database: Path,
    trial_id: str,
    run_id: str,
    wandb_id: str,
    progress: float,
) -> None:
    del progress
    persistence.add_trial(database, trial_id, {})
    persistence.add_regional_run(
        database,
        run_id,
        trial_id,
        "us-central1",
        wandb_id,
        "gs://checkpoints",
    )


def _insert_dispatch(
    database: Path,
    dispatch_id: str,
    run_id: str,
    attempt: int,
    tpu_slice: str,
    chips: int,
    submitted_at: str,
    ended_at: str | None,
) -> None:
    actual_attempt = persistence.prepare_dispatch(
        database,
        run_id,
        dispatch_id,
        f"iris-{dispatch_id}",
        tpu_slice,
        chips,
        "batch",
        "launch",
        submitted_at,
    )
    assert actual_attempt == attempt
    persistence.confirm_dispatch(database, dispatch_id, submitted_at)
    if ended_at is not None:
        persistence.end_dispatch(database, dispatch_id, "stopped", ended_at=ended_at)


def _insert_observation(
    database: Path,
    run_id: str,
    dispatch_id: str,
    observed_at: str,
    progress: float,
    iris_running: bool,
) -> None:
    del run_id
    persistence.record_observation(
        database, dispatch_id, observed_at, "running", progress, iris_running
    )


def test_target_rate_pools_trials_and_zero_progress_time(tmp_path: Path) -> None:
    database = tmp_path / "sweep.sqlite"
    persistence.init(database)
    _insert_trial_run(database, "trial-1", "run-1", "wandb-1", 0.12)
    _insert_trial_run(database, "trial-2", "run-2", "wandb-2", 0.0)
    _insert_dispatch(
        database,
        "dispatch-1",
        "run-1",
        1,
        "v5p-8",
        8,
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T01:00:00+00:00",
    )
    _insert_dispatch(
        database,
        "dispatch-2",
        "run-2",
        1,
        "v5p-8",
        8,
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T02:00:00+00:00",
    )
    _insert_observation(
        database,
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
    _insert_trial_run(database, "trial", "run", "wandb", 0.2)
    _insert_dispatch(
        database,
        "old",
        "run",
        1,
        "v5p-8",
        8,
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:10:00+00:00",
    )
    _insert_dispatch(
        database,
        "new",
        "run",
        2,
        "v5p-16",
        16,
        "2026-01-01T00:10:00+00:00",
        None,
    )
    _insert_observation(
        database,
        "run",
        "old",
        "2026-01-01T00:05:00+00:00",
        0.1,
        True,
    )
    _insert_observation(
        database,
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

    _insert_observation(
        database,
        "run",
        "new",
        "2026-01-01T00:12:00+00:00",
        0.3,
        True,
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
    _insert_trial_run(database, "trial-1", "run-1", "wandb-1", 0.1)
    _insert_trial_run(database, "trial-2", "run-2", "wandb-2", 0.0)
    _insert_dispatch(database, "dispatch-1", "run-1", 1, "v5p-8", 8, old, None)
    _insert_dispatch(database, "dispatch-2", "run-2", 1, "v5p-8", 8, old, None)
    _insert_observation(database, "run-1", "dispatch-1", recent, 0.1, True)
    _insert_observation(database, "run-2", "dispatch-2", recent, 0.0, True)

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
    _insert_trial_run(database, "trial", "run", "wandb", 0.5)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE runs SET high_water_progress = 0.5 WHERE regional_run_id = 'run'"
        )

    assert persistence.check(database) == [
        "semantic_check: run 'run' progress high-water does not match observations"
    ]
    with pytest.raises(RuntimeError, match="progress high-water"):
        persistence.snapshot(database, recent_event_limit=0)


def test_persistence_normalizes_input_times_to_utc(tmp_path: Path) -> None:
    database = tmp_path / "sweep.sqlite"
    persistence.init(database)
    persistence.add_trial(database, "trial", {})
    persistence.add_regional_run(
        database,
        "run",
        "trial",
        "us-central1",
        "wandb-run",
        "gs://checkpoints/run",
    )
    assert (
        persistence.prepare_dispatch(
            database,
            "run",
            "dispatch",
            "iris-dispatch",
            "v5p-8",
            8,
            "batch",
            "launch --redacted",
            "2026-01-01T00:58:00+01:00",
        )
        == 1
    )
    assert (
        persistence.confirm_dispatch(database, "dispatch", "2025-12-31T23:59:00Z")
        == "2025-12-31T23:59:00+00:00"
    )
    persistence.record_observation(
        database,
        "dispatch",
        "2026-01-01T01:00:00+01:00",
        "running",
        0.5,
        True,
    )
    persistence.record_observation(
        database,
        "dispatch",
        "2025-12-31T19:30:00-05:00",
        "finished",
        1.0,
        False,
    )
    assert (
        persistence.end_dispatch(
            database,
            "dispatch",
            "completed",
            ended_at="2025-12-31T19:31:00-05:00",
        )
        == "2026-01-01T00:31:00+00:00"
    )
    assert (
        persistence.complete_trial(
            database,
            "trial",
            "run",
            "2026-01-01T01:32:00+01:00",
        )
        == "2026-01-01T00:32:00+00:00"
    )

    with sqlite3.connect(database) as connection:
        dispatch_times = connection.execute(
            "SELECT intent_at, submitted_at, ended_at FROM dispatches"
        ).fetchone()
        observations = connection.execute(
            "SELECT observed_at, run_progress FROM observations ORDER BY observed_at"
        ).fetchall()
        checkpoint_verified_at = connection.execute(
            "SELECT checkpoint_verified_at FROM runs"
        ).fetchone()[0]
    assert dispatch_times == (
        "2025-12-31T23:58:00+00:00",
        "2025-12-31T23:59:00+00:00",
        "2026-01-01T00:31:00+00:00",
    )
    assert observations == [
        ("2026-01-01T00:00:00+00:00", 0.5),
        ("2026-01-01T00:30:00+00:00", 1.0),
    ]
    assert checkpoint_verified_at == "2026-01-01T00:32:00+00:00"
    assert persistence.check(database) == []


def test_record_observation_rejects_nonfinite_progress(tmp_path: Path) -> None:
    database = tmp_path / "sweep.sqlite"
    persistence.init(database)
    persistence.add_trial(database, "trial", {})
    persistence.add_regional_run(
        database,
        "run",
        "trial",
        "us-central1",
        "wandb-run",
        "gs://checkpoints/run",
    )
    persistence.prepare_dispatch(
        database,
        "run",
        "dispatch",
        "iris-dispatch",
        "v5p-8",
        8,
        "batch",
        "launch --redacted",
        "2026-01-01T00:00:00+00:00",
    )
    persistence.confirm_dispatch(database, "dispatch", "2026-01-01T00:01:00Z")

    for progress in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(RuntimeError, match="finite and nonnegative"):
            persistence.record_observation(
                database,
                "dispatch",
                "2026-01-01T00:02:00Z",
                "running",
                progress,
                True,
            )

    with sqlite3.connect(database) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
        )
        assert (
            connection.execute("SELECT high_water_progress FROM runs").fetchone()[0]
            == 0.0
        )


def test_dispatch_intent_blocks_duplicate_after_backup_recovery(tmp_path: Path) -> None:
    database = tmp_path / "sweep.sqlite"
    recovered = tmp_path / "recovered.sqlite"
    persistence.init(database)
    persistence.add_trial(database, "trial", {})
    persistence.add_regional_run(
        database,
        "run",
        "trial",
        "us-central1",
        "wandb-run",
        "gs://checkpoints/run",
    )

    attempt = persistence.prepare_dispatch(
        database,
        "run",
        "dispatch-1",
        "iris-exact-1",
        "v5p-8",
        8,
        "batch",
        "launch --redacted",
        "2026-01-01T00:00:00+00:00",
    )
    assert attempt == 1
    backup = persistence.backup(database, recovered)
    assert backup["path"] == str(recovered)
    assert len(backup["sha256"]) == 64

    snapshot = persistence.snapshot(recovered, recent_event_limit=0)
    assert snapshot["fleet"]["unreconciled_dispatch_intents"] == 1
    assert snapshot["fleet"]["active_dispatches"] == 0
    assert snapshot["runs"][0]["active_dispatch"]["submission_state"] == "intent"
    assert snapshot["conditions"] == [
        {
            "kind": "dispatch_intent_unreconciled",
            "trial_id": "trial",
            "run_id": "run",
        }
    ]

    with pytest.raises(RuntimeError, match="already has active dispatch"):
        persistence.prepare_dispatch(
            recovered,
            "run",
            "dispatch-duplicate",
            "iris-duplicate",
            "v5p-16",
            16,
            "batch",
            "launch --redacted",
            "2026-01-01T00:01:00+00:00",
        )

    persistence.confirm_dispatch(recovered, "dispatch-1", "2026-01-01T00:02:00+00:00")
    persistence.end_dispatch(
        recovered,
        "dispatch-1",
        "stopped",
        ended_at="2026-01-01T00:03:00+00:00",
    )
    assert (
        persistence.prepare_dispatch(
            recovered,
            "run",
            "dispatch-2",
            "iris-exact-2",
            "v5p-16",
            16,
            "batch",
            "launch --redacted",
            "2026-01-01T00:04:00+00:00",
        )
        == 2
    )


def test_completed_trial_cannot_reopen_or_change_winner(tmp_path: Path) -> None:
    database = tmp_path / "sweep.sqlite"
    persistence.init(database)
    persistence.add_trial(database, "trial", {})
    for run_id, region in (("run-1", "us-central1"), ("run-2", "us-east1")):
        persistence.add_regional_run(
            database,
            run_id,
            "trial",
            region,
            f"wandb-{run_id}",
            f"gs://checkpoints/{run_id}",
        )
        dispatch_id = f"dispatch-{run_id}"
        persistence.prepare_dispatch(
            database,
            run_id,
            dispatch_id,
            f"iris-{run_id}",
            "v5p-8",
            8,
            "batch",
            "launch --redacted",
            "2026-01-01T00:00:00+00:00",
        )
        persistence.confirm_dispatch(database, dispatch_id, "2026-01-01T00:01:00+00:00")
        persistence.record_observation(
            database,
            dispatch_id,
            "2026-01-01T00:02:00+00:00",
            "finished",
            1.0,
            False,
        )
        persistence.end_dispatch(
            database,
            dispatch_id,
            "completed",
            ended_at="2026-01-01T00:03:00+00:00",
        )

    verified_at = "2026-01-01T00:04:00+00:00"
    assert (
        persistence.complete_trial(database, "trial", "run-1", verified_at)
        == verified_at
    )
    assert (
        persistence.complete_trial(
            database,
            "trial",
            "run-1",
            "2026-01-01T00:05:00+00:00",
        )
        == verified_at
    )
    with pytest.raises(RuntimeError, match="already completed with winner 'run-1'"):
        persistence.complete_trial(database, "trial", "run-2")
    with pytest.raises(RuntimeError, match="is completed"):
        persistence.add_regional_run(
            database,
            "run-late",
            "trial",
            "us-west1",
            "wandb-late",
            "gs://checkpoints/late",
        )
    with pytest.raises(RuntimeError, match="is completed"):
        persistence.prepare_dispatch(
            database,
            "run-2",
            "dispatch-late",
            "iris-late",
            "v5p-8",
            8,
            "batch",
            "launch --redacted",
        )
    with pytest.raises(RuntimeError, match="is completed"):
        persistence.record_observation(
            database,
            "dispatch-run-2",
            "2026-01-01T00:06:00+00:00",
            "finished",
            1.1,
            False,
        )
    with sqlite3.connect(database) as connection:
        runs = connection.execute(
            """
            SELECT regional_run_id, status, high_water_progress, is_winner
            FROM runs ORDER BY regional_run_id
            """
        ).fetchall()
    assert runs == [
        ("run-1", "completed", 1.0, 1),
        ("run-2", "race_lost", 1.0, 0),
    ]
    assert persistence.check(database) == []


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
