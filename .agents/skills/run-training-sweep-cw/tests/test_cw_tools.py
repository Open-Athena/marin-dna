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
    "run_training_sweep_cw_persistence",
    SKILL_ROOT / "scripts" / "persistence.py",
)
utilization = _load_module(
    "run_training_sweep_cw_utilization",
    SKILL_ROOT / "scripts" / "utilization.py",
)


def _insert_trial(
    connection: sqlite3.Connection,
    trial_id: str,
    progress: float,
    *,
    status: str = "active",
    checkpoint_verified_at: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO trials(
            trial_id, env_json, wandb_run_id, checkpoint_root, status,
            high_water_progress, checkpoint_verified_at
        ) VALUES (?, '{}', ?, ?, ?, ?, ?)
        """,
        (
            trial_id,
            f"wandb-{trial_id}",
            f"s3://checkpoints/{trial_id}",
            status,
            progress,
            checkpoint_verified_at,
        ),
    )


def _insert_dispatch(
    connection: sqlite3.Connection,
    dispatch_id: str,
    trial_id: str,
    attempt: int,
    cluster: str,
    gpu_variant: str,
    nodes: int,
    gpus: int,
    submitted_at: str,
    ended_at: str | None,
) -> None:
    connection.execute(
        """
        INSERT INTO dispatches(
            dispatch_id, trial_id, attempt, iris_job_id, cluster, gpu_variant,
            nodes, gpus, priority_band, command_redacted, submitted_at,
            ended_at, active, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'batch', 'launch', ?, ?, ?, ?)
        """,
        (
            dispatch_id,
            trial_id,
            attempt,
            f"iris-{dispatch_id}",
            cluster,
            gpu_variant,
            nodes,
            gpus,
            submitted_at,
            ended_at,
            int(ended_at is None),
            None if ended_at is None else "stopped",
        ),
    )


def _insert_observation(
    connection: sqlite3.Connection,
    trial_id: str,
    dispatch_id: str,
    observed_at: str,
    progress: float,
    iris_running: bool,
    *,
    wandb_state: str = "running",
) -> None:
    connection.execute(
        """
        INSERT INTO observations(
            trial_id, dispatch_id, observed_at, wandb_state,
            run_progress, iris_running
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (trial_id, dispatch_id, observed_at, wandb_state, progress, int(iris_running)),
    )


def test_target_rate_pools_trials_and_zero_progress_time(tmp_path: Path) -> None:
    database = tmp_path / "sweep.sqlite"
    persistence.init(database)
    with sqlite3.connect(database) as connection:
        _insert_trial(connection, "trial-1", 0.12)
        _insert_trial(connection, "trial-2", 0.0)
        _insert_dispatch(
            connection,
            "dispatch-1",
            "trial-1",
            1,
            "cw-rno2a",
            "H100",
            2,
            16,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T01:00:00+00:00",
        )
        _insert_dispatch(
            connection,
            "dispatch-2",
            "trial-2",
            1,
            "cw-rno2a",
            "H100",
            2,
            16,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T02:00:00+00:00",
        )
        _insert_observation(
            connection,
            "trial-1",
            "dispatch-1",
            "2026-01-01T00:30:00+00:00",
            0.12,
            True,
        )

    assert persistence.snapshot(database, recent_event_limit=0)["placements"] == [
        {
            "cluster": "cw-rno2a",
            "gpu_variant": "H100",
            "nodes": 2,
            "gpus": 16,
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


def test_reslice_attributes_progress_to_exact_dispatch(tmp_path: Path) -> None:
    database = tmp_path / "sweep.sqlite"
    persistence.init(database)
    with sqlite3.connect(database) as connection:
        _insert_trial(connection, "trial", 0.2)
        _insert_dispatch(
            connection,
            "old",
            "trial",
            1,
            "cw-rno2a",
            "H100",
            2,
            16,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:10:00+00:00",
        )
        _insert_dispatch(
            connection,
            "new",
            "trial",
            2,
            "cw-us-east-08a",
            "GB200",
            2,
            8,
            "2026-01-01T00:10:00+00:00",
            None,
        )
        _insert_observation(
            connection, "trial", "old", "2026-01-01T00:05:00+00:00", 0.1, True
        )
        _insert_observation(
            connection, "trial", "old", "2026-01-01T00:11:00+00:00", 0.2, False
        )

    before = persistence.snapshot(database, recent_event_limit=0)
    assert before["fleet"]["wandb_running_gpus"] == 0
    assert before["trials"][0]["active_dispatch"]["latest_observation"] is None
    assert before["conditions"][0]["kind"] == "active_dispatch_unobserved"

    with sqlite3.connect(database) as connection:
        _insert_observation(
            connection, "trial", "new", "2026-01-01T00:12:00+00:00", 0.3, True
        )
        connection.execute(
            "UPDATE trials SET high_water_progress = 0.3 WHERE trial_id = 'trial'"
        )

    after = persistence.snapshot(database, recent_event_limit=0)
    progress_by_gpu = {
        placement["gpu_variant"]: placement["total_progress"]
        for placement in after["placements"]
    }
    assert progress_by_gpu == {"H100": 0.2, "GB200": 0.1}
    assert after["fleet"]["wandb_running_gpus"] == 8


def test_snapshot_classifies_target_headroom(tmp_path: Path) -> None:
    database = tmp_path / "sweep.sqlite"
    persistence.init(database)
    now = datetime.now(UTC).replace(microsecond=0)
    old = (now - timedelta(hours=2)).isoformat()
    recent = (now - timedelta(minutes=30)).isoformat()
    with sqlite3.connect(database) as connection:
        _insert_trial(connection, "trial-1", 0.1)
        _insert_trial(connection, "trial-2", 0.0)
        _insert_dispatch(
            connection, "dispatch-1", "trial-1", 1, "cw-rno2a", "H100", 2, 16, old, None
        )
        _insert_dispatch(
            connection, "dispatch-2", "trial-2", 1, "cw-rno2a", "H100", 2, 16, old, None
        )
        _insert_observation(connection, "trial-1", "dispatch-1", recent, 0.1, True)
        _insert_observation(connection, "trial-2", "dispatch-2", recent, 0.0, True)

    result = persistence.snapshot(database, recent_event_limit=0)
    placement = result["placements"][0]
    assert placement["active_dispatches"] == 2
    assert placement["productive_dispatches"] == 1
    assert placement["pending_dispatches"] == 1
    assert result["trials"][0]["active_dispatch"]["recent_progress"] is True
    assert result["trials"][1]["active_dispatch"]["recent_progress"] is False

    shorter = persistence.snapshot(
        database, recent_event_limit=0, reslice_after_hours=0.25
    )["placements"][0]
    assert shorter["productive_dispatches"] == 0
    assert shorter["pending_dispatches"] == 2


def test_completion_and_snapshot_reject_semantic_errors(tmp_path: Path) -> None:
    database = tmp_path / "sweep.sqlite"
    persistence.init(database)
    with sqlite3.connect(database) as connection:
        _insert_trial(connection, "trial", 0.5)

    assert persistence.check(database) == [
        "semantic_check: trial 'trial' progress high-water does not match observations"
    ]
    with pytest.raises(RuntimeError, match="progress high-water"):
        persistence.snapshot(database, recent_event_limit=0)


def test_schema_rejects_non_batch_dispatch(tmp_path: Path) -> None:
    database = tmp_path / "sweep.sqlite"
    persistence.init(database)
    with sqlite3.connect(database) as connection:
        _insert_trial(connection, "trial", 0.0)
        with pytest.raises(sqlite3.IntegrityError, match="priority_band"):
            connection.execute(
                """
                INSERT INTO dispatches(
                    dispatch_id, trial_id, attempt, iris_job_id, cluster,
                    gpu_variant, nodes, gpus, priority_band, command_redacted,
                    submitted_at, active
                ) VALUES (
                    'dispatch', 'trial', 1, 'iris-dispatch', 'cw-rno2a',
                    'H100', 1, 8, 'interactive', 'launch',
                    '2026-01-01T00:00:00+00:00', 1
                )
                """
            )


def _availability(
    gpu: str,
    free: int,
    total: int,
    observed_ms: int,
    held: list[tuple[str, int]],
) -> dict[str, object]:
    return {
        "version": "2",
        "observation_epoch_ms": str(observed_ms),
        "amounts": {gpu: str(free)},
        "total_amounts": {gpu: str(total)},
        "held_by_band": [
            {"band": band, "amounts": {gpu: str(amount)}} for band, amount in held
        ],
    }


def _utilization_response(observed_at: datetime) -> dict[str, object]:
    observed_ms = int((observed_at - timedelta(seconds=20)).timestamp() * 1000)

    def peer(
        peer_id: str,
        gpu: str,
        free: int,
        total: int,
        held: list[tuple[str, int]],
    ) -> dict[str, object]:
        return {
            "peer_id": peer_id,
            "reachable": True,
            "backends": [
                {
                    "backend_id": "default",
                    "pending_task_count": 0,
                    "running_task_count": 4,
                    "availability": _availability(gpu, free, total, observed_ms, held),
                }
            ],
        }

    return {
        "peers": [
            peer(
                "cw-rno2a",
                "h100",
                0,
                512,
                [
                    ("PRIORITY_BAND_INTERACTIVE", 112),
                    ("PRIORITY_BAND_BATCH", 400),
                ],
            ),
            peer(
                "cw-us-east-02a",
                "h100",
                112,
                256,
                [
                    ("PRIORITY_BAND_PRODUCTION", 16),
                    ("PRIORITY_BAND_BATCH", 128),
                ],
            ),
            peer(
                "cw-us-east-08a",
                "gb200",
                476,
                804,
                [
                    ("PRIORITY_BAND_PRODUCTION", 320),
                    ("PRIORITY_BAND_INTERACTIVE", 4),
                    ("PRIORITY_BAND_BATCH", 4),
                ],
            ),
        ]
    }


def test_utilization_summarizes_all_production_peers() -> None:
    observed_at = datetime(2026, 8, 16, 13, tzinfo=UTC)
    result = utilization.summarize(_utilization_response(observed_at), observed_at)

    assert result["fleet_by_gpu"] == {
        "GB200": {
            "free_gpus": 476,
            "held_gpus": 328,
            "total_gpus": 804,
            "held_by_band": {
                "PRIORITY_BAND_BATCH": 4,
                "PRIORITY_BAND_INTERACTIVE": 4,
                "PRIORITY_BAND_PRODUCTION": 320,
            },
        },
        "H100": {
            "free_gpus": 112,
            "held_gpus": 656,
            "total_gpus": 768,
            "held_by_band": {
                "PRIORITY_BAND_BATCH": 528,
                "PRIORITY_BAND_INTERACTIVE": 112,
                "PRIORITY_BAND_PRODUCTION": 16,
            },
        },
    }
    assert [target["cluster"] for target in result["targets"]] == [
        "cw-us-east-08a",
        "cw-rno2a",
        "cw-us-east-02a",
    ]
    assert all(target["observation_age_seconds"] == 20 for target in result["targets"])


def test_utilization_rejects_stale_and_inconsistent_capacity() -> None:
    observed_at = datetime(2026, 8, 16, 13, tzinfo=UTC)
    stale = _utilization_response(observed_at)
    stale["peers"][0]["backends"][0]["availability"]["observation_epoch_ms"] = str(
        int((observed_at - timedelta(minutes=2)).timestamp() * 1000)
    )
    with pytest.raises(utilization.SnapshotError, match="stale"):
        utilization.summarize(stale, observed_at)

    inconsistent = _utilization_response(observed_at)
    inconsistent["peers"][1]["backends"][0]["availability"]["amounts"]["h100"] = "111"
    with pytest.raises(utilization.SnapshotError, match="accounting disagrees"):
        utilization.summarize(inconsistent, observed_at)


def test_utilization_requires_reachable_selected_peers() -> None:
    observed_at = datetime(2026, 8, 16, 13, tzinfo=UTC)
    response = copy.deepcopy(_utilization_response(observed_at))
    response["peers"][2]["reachable"] = False
    with pytest.raises(utilization.SnapshotError, match="unreachable"):
        utilization.summarize(response, observed_at)

    result = utilization.summarize(
        response,
        observed_at,
        peers=("cw-rno2a", "cw-us-east-02a"),
    )
    assert set(result["fleet_by_gpu"]) == {"H100"}
