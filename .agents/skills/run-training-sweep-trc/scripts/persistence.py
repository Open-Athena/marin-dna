"""Initialize, inspect, and check training-sweep persistence."""

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 2

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trials (
    trial_id TEXT PRIMARY KEY,
    env_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'planned'
);

CREATE TABLE IF NOT EXISTS runs (
    regional_run_id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL REFERENCES trials(trial_id),
    region TEXT NOT NULL,
    wandb_run_id TEXT NOT NULL,
    wandb_url TEXT,
    checkpoint_root TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'planned',
    high_water_progress REAL NOT NULL DEFAULT 0 CHECK (high_water_progress >= 0),
    checkpoint_verified_at TEXT,
    is_winner INTEGER NOT NULL DEFAULT 0 CHECK (is_winner IN (0, 1)),
    UNIQUE (trial_id, region),
    UNIQUE (wandb_run_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_winner_per_trial
ON runs(trial_id) WHERE is_winner = 1;

CREATE TABLE IF NOT EXISTS dispatches (
    dispatch_id TEXT PRIMARY KEY,
    regional_run_id TEXT NOT NULL REFERENCES runs(regional_run_id),
    attempt INTEGER NOT NULL CHECK (attempt > 0),
    iris_job_id TEXT NOT NULL UNIQUE,
    tpu_slice TEXT NOT NULL,
    chips INTEGER NOT NULL CHECK (chips > 0),
    priority_band TEXT NOT NULL,
    command_redacted TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    ended_at TEXT,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    outcome TEXT,
    stop_reason TEXT,
    UNIQUE (regional_run_id, attempt)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_dispatch_per_regional_run
ON dispatches(regional_run_id) WHERE active = 1;

CREATE INDEX IF NOT EXISTS dispatch_target_history
ON dispatches(regional_run_id, tpu_slice, chips, submitted_at);

CREATE TABLE IF NOT EXISTS observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    regional_run_id TEXT NOT NULL REFERENCES runs(regional_run_id),
    dispatch_id TEXT NOT NULL REFERENCES dispatches(dispatch_id),
    observed_at TEXT NOT NULL,
    wandb_state TEXT,
    run_progress REAL CHECK (run_progress IS NULL OR run_progress >= 0),
    iris_running INTEGER CHECK (iris_running IS NULL OR iris_running IN (0, 1)),
    UNIQUE (regional_run_id, observed_at)
);

CREATE INDEX IF NOT EXISTS observations_by_dispatch_time
ON observations(dispatch_id, observed_at);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    trial_id TEXT REFERENCES trials(trial_id),
    regional_run_id TEXT REFERENCES runs(regional_run_id),
    dispatch_id TEXT REFERENCES dispatches(dispatch_id),
    detail TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS events_by_time
ON events(recorded_at);

"""

REQUIRED_COLUMNS = {
    "meta": {"key", "value"},
    "trials": {"trial_id", "env_json", "status"},
    "runs": {
        "regional_run_id",
        "trial_id",
        "region",
        "wandb_run_id",
        "checkpoint_root",
        "status",
        "high_water_progress",
        "checkpoint_verified_at",
        "is_winner",
    },
    "dispatches": {
        "dispatch_id",
        "regional_run_id",
        "attempt",
        "iris_job_id",
        "tpu_slice",
        "chips",
        "priority_band",
        "command_redacted",
        "submitted_at",
        "ended_at",
        "active",
        "outcome",
        "stop_reason",
    },
    "observations": {
        "observation_id",
        "regional_run_id",
        "dispatch_id",
        "observed_at",
        "wandb_state",
        "run_progress",
        "iris_running",
    },
    "events": {
        "event_id",
        "recorded_at",
        "kind",
        "trial_id",
        "regional_run_id",
        "dispatch_id",
        "detail",
    },
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_time(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RuntimeError(
            f"{field} is not an ISO-8601 timestamp: {value!r}"
        ) from error
    if parsed.tzinfo is None:
        raise RuntimeError(f"{field} has no timezone: {value!r}")
    return parsed.astimezone(UTC)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _user_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """
        )
    }


def _schema_errors(connection: sqlite3.Connection) -> list[str]:
    errors: list[str] = []
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version != SCHEMA_VERSION:
        errors.append(f"schema version {version}; expected {SCHEMA_VERSION}")

    for table, required in REQUIRED_COLUMNS.items():
        actual = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
        missing = required - actual
        if missing:
            errors.append(f"{table}: missing columns {sorted(missing)}")
    return errors


def init(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if _user_tables(connection) and version != SCHEMA_VERSION:
            raise RuntimeError(
                f"{path} has schema version {version}; expected {SCHEMA_VERSION}"
            )
        connection.executescript(SCHEMA)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
            ("created_at", _now()),
        )
        errors = _schema_errors(connection)
        if errors:
            raise RuntimeError("; ".join(errors))


def record_event(
    path: Path,
    kind: str,
    detail: str,
    trial_id: str | None,
    regional_run_id: str | None,
    dispatch_id: str | None,
) -> None:
    with _connect(path) as connection:
        connection.execute(
            """
            INSERT INTO events(
                recorded_at, kind, trial_id, regional_run_id, dispatch_id, detail
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (_now(), kind, trial_id, regional_run_id, dispatch_id, detail),
        )


def _structural_errors(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"database does not exist: {path}"]
    with _connect(path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            errors.append(f"integrity_check: {integrity}")
        for row in connection.execute("PRAGMA foreign_key_check"):
            errors.append(f"foreign_key_check: {row}")
        errors.extend(_schema_errors(connection))
        if errors:
            return errors
        duplicate_active = connection.execute(
            """
            SELECT regional_run_id, COUNT(*)
            FROM dispatches
            WHERE active = 1
            GROUP BY regional_run_id
            HAVING COUNT(*) > 1
            """
        ).fetchall()
        for row in duplicate_active:
            errors.append(f"multiple active dispatches: {row}")
        inconsistent_dispatches = connection.execute(
            """
            SELECT dispatch_id, active, ended_at
            FROM dispatches
            WHERE (active = 1 AND ended_at IS NOT NULL)
               OR (active = 0 AND ended_at IS NULL)
            """
        ).fetchall()
        for row in inconsistent_dispatches:
            errors.append(f"inconsistent dispatch end state: {row}")
        mismatched_observations = connection.execute(
            """
            SELECT observations.observation_id
            FROM observations
            JOIN dispatches USING (dispatch_id)
            WHERE observations.regional_run_id != dispatches.regional_run_id
            """
        ).fetchall()
        for row in mismatched_observations:
            errors.append(f"observation dispatch/run mismatch: {row[0]}")
    return errors


def _build_snapshot(
    path: Path, recent_event_limit: int, reslice_after_hours: float
) -> dict[str, object]:
    if recent_event_limit < 0:
        raise RuntimeError("recent event limit must be nonnegative")
    if reslice_after_hours <= 0:
        raise RuntimeError("reslice-after hours must be positive")

    snapshot_at = datetime.now(UTC)
    reslice_after_seconds = reslice_after_hours * 3600
    with _connect(path) as connection:
        connection.row_factory = sqlite3.Row
        trials = list(connection.execute("SELECT * FROM trials ORDER BY trial_id"))
        runs = list(connection.execute("SELECT * FROM runs ORDER BY regional_run_id"))
        dispatches = list(
            connection.execute(
                "SELECT * FROM dispatches ORDER BY submitted_at, dispatch_id"
            )
        )
        observations = list(
            connection.execute(
                """
                SELECT *
                FROM observations
                ORDER BY regional_run_id, observed_at, observation_id
                """
            )
        )
        event_counts = {
            row["kind"]: row["count"]
            for row in connection.execute(
                "SELECT kind, COUNT(*) AS count FROM events GROUP BY kind ORDER BY kind"
            )
        }
        event_total = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        recent_events = list(
            connection.execute(
                """
                SELECT event_id, recorded_at, kind, trial_id, regional_run_id,
                       dispatch_id, detail
                FROM events
                ORDER BY event_id DESC
                LIMIT ?
                """,
                (recent_event_limit,),
            )
        )

    run_by_id = {row["regional_run_id"]: row for row in runs}
    dispatch_by_id = {row["dispatch_id"]: row for row in dispatches}
    runs_by_trial: dict[str, list[sqlite3.Row]] = defaultdict(list)
    dispatches_by_run: dict[str, list[sqlite3.Row]] = defaultdict(list)
    observations_by_run: dict[str, list[tuple[sqlite3.Row, datetime]]] = defaultdict(
        list
    )

    for run in runs:
        runs_by_trial[run["trial_id"]].append(run)
    for dispatch in dispatches:
        dispatches_by_run[dispatch["regional_run_id"]].append(dispatch)
    for observation in observations:
        observed_at = _parse_time(
            observation["observed_at"],
            f"observation {observation['observation_id']} observed_at",
        )
        if observed_at > snapshot_at:
            raise RuntimeError(
                f"observation {observation['observation_id']} is in the future"
            )
        if observation["dispatch_id"] is not None:
            dispatch = dispatch_by_id[observation["dispatch_id"]]
            submitted_at = _parse_time(
                dispatch["submitted_at"],
                f"dispatch {dispatch['dispatch_id']} submitted_at",
            )
            if observed_at < submitted_at:
                raise RuntimeError(
                    f"observation {observation['observation_id']} predates its dispatch"
                )
        observations_by_run[observation["regional_run_id"]].append(
            (observation, observed_at)
        )
    for run_observations in observations_by_run.values():
        run_observations.sort(key=lambda item: (item[1], item[0]["observation_id"]))

    progress_by_dispatch: dict[str, float] = defaultdict(float)
    first_progress_at_by_dispatch: dict[str, datetime] = {}
    last_progress_at_by_dispatch: dict[str, datetime] = {}
    for run_observations in observations_by_run.values():
        progress_high_water = 0.0
        for observation, observed_at in run_observations:
            progress = observation["run_progress"]
            if progress is None or progress <= progress_high_water:
                continue
            progress_delta = progress - progress_high_water
            progress_high_water = progress
            dispatch_id = observation["dispatch_id"]
            if dispatch_id is None:
                continue
            progress_by_dispatch[dispatch_id] += progress_delta
            first_progress_at_by_dispatch.setdefault(dispatch_id, observed_at)
            last_progress_at_by_dispatch[dispatch_id] = observed_at

    completed_trial_ids = {
        trial["trial_id"]
        for trial in trials
        if trial["status"].casefold() == "completed"
    }
    for trial_id in completed_trial_ids:
        trial_runs = runs_by_trial[trial_id]
        winners = [run for run in trial_runs if run["is_winner"]]
        if len(winners) != 1:
            raise RuntimeError(
                f"completed trial {trial_id!r} has {len(winners)} winning runs"
            )
        winner = winners[0]
        if winner["high_water_progress"] < 1:
            raise RuntimeError(
                f"completed trial {trial_id!r} has winner progress below 1"
            )
        if winner["checkpoint_verified_at"] is None:
            raise RuntimeError(
                f"completed trial {trial_id!r} has no verified checkpoint"
            )
        if any(
            dispatch["active"]
            for run in trial_runs
            for dispatch in dispatches_by_run[run["regional_run_id"]]
        ):
            raise RuntimeError(f"completed trial {trial_id!r} has an active dispatch")

    run_snapshots: list[dict[str, object]] = []
    current_run_by_id: dict[str, dict[str, object]] = {}
    conditions: list[dict[str, object]] = []
    latest_observation_times: list[datetime] = []

    for run in runs:
        run_id = run["regional_run_id"]
        run_observations = observations_by_run[run_id]
        progress_values = [
            observation["run_progress"]
            for observation, _ in run_observations
            if observation["run_progress"] is not None
        ]
        observed_progress_high_water = max(progress_values, default=0.0)
        if abs(observed_progress_high_water - run["high_water_progress"]) > 1e-12:
            raise RuntimeError(
                f"run {run_id!r} progress high-water does not match observations"
            )

        progress_high_water = 0.0
        last_progress_at = None
        for observation, observed_at in run_observations:
            progress = observation["run_progress"]
            if progress is not None and progress > progress_high_water:
                progress_high_water = progress
                last_progress_at = observed_at
        if last_progress_at is None and dispatches_by_run[run_id]:
            last_progress_at = min(
                _parse_time(
                    dispatch["submitted_at"],
                    f"dispatch {dispatch['dispatch_id']} submitted_at",
                )
                for dispatch in dispatches_by_run[run_id]
            )

        latest_observation = run_observations[-1] if run_observations else None
        latest_row = latest_observation[0] if latest_observation else None
        latest_at = latest_observation[1] if latest_observation else None
        if latest_at is not None:
            latest_observation_times.append(latest_at)

        prior_progress_high_water = max(
            (
                observation["run_progress"]
                for observation, _ in run_observations[:-1]
                if observation["run_progress"] is not None
            ),
            default=0.0,
        )
        active_dispatches = [
            dispatch for dispatch in dispatches_by_run[run_id] if dispatch["active"]
        ]
        if len(active_dispatches) > 1:
            raise RuntimeError(f"run {run_id!r} has multiple active dispatches")
        active_dispatch = active_dispatches[0] if active_dispatches else None

        stall_seconds = None
        if last_progress_at is not None:
            stall_seconds = (snapshot_at - last_progress_at).total_seconds()
            if stall_seconds < 0:
                raise RuntimeError(f"run {run_id!r} stall clock is in the future")

        active_dispatch_snapshot = None
        active_latest_row = None
        active_latest_at = None
        if active_dispatch is not None:
            submitted_at = _parse_time(
                active_dispatch["submitted_at"],
                f"dispatch {active_dispatch['dispatch_id']} submitted_at",
            )
            if submitted_at > snapshot_at:
                raise RuntimeError(
                    f"dispatch {active_dispatch['dispatch_id']!r} is in the future"
                )
            active_observations = [
                (observation, observed_at)
                for observation, observed_at in run_observations
                if observation["dispatch_id"] == active_dispatch["dispatch_id"]
                and observed_at >= submitted_at
            ]
            if active_observations:
                active_latest_row, active_latest_at = active_observations[-1]
            active_last_progress_at = last_progress_at_by_dispatch.get(
                active_dispatch["dispatch_id"]
            )
            dispatch_stall_since = active_last_progress_at or submitted_at
            dispatch_stall_seconds = (
                snapshot_at - dispatch_stall_since
            ).total_seconds()
            active_dispatch_snapshot = {
                "dispatch_id": active_dispatch["dispatch_id"],
                "attempt": active_dispatch["attempt"],
                "iris_job_id": active_dispatch["iris_job_id"],
                "slice": active_dispatch["tpu_slice"],
                "chips": active_dispatch["chips"],
                "submitted_at": active_dispatch["submitted_at"],
                "age_seconds": round((snapshot_at - submitted_at).total_seconds(), 3),
                "last_progress_at": (
                    None
                    if active_last_progress_at is None
                    else active_last_progress_at.isoformat()
                ),
                "stall_since": dispatch_stall_since.isoformat(),
                "stall_seconds": round(dispatch_stall_seconds, 3),
                "recent_progress": (
                    active_last_progress_at is not None
                    and dispatch_stall_seconds <= reslice_after_seconds
                ),
                "latest_observation": (
                    None
                    if active_latest_row is None or active_latest_at is None
                    else {
                        "observed_at": active_latest_row["observed_at"],
                        "age_seconds": round(
                            (snapshot_at - active_latest_at).total_seconds(), 3
                        ),
                        "wandb_state": active_latest_row["wandb_state"],
                        "run_progress": active_latest_row["run_progress"],
                        "iris_running": (
                            None
                            if active_latest_row["iris_running"] is None
                            else bool(active_latest_row["iris_running"])
                        ),
                    }
                ),
            }

        latest_observation_snapshot = None
        if latest_row is not None and latest_at is not None:
            latest_observation_snapshot = {
                "dispatch_id": latest_row["dispatch_id"],
                "observed_at": latest_row["observed_at"],
                "age_seconds": round((snapshot_at - latest_at).total_seconds(), 3),
                "wandb_state": latest_row["wandb_state"],
                "run_progress": latest_row["run_progress"],
                "iris_running": (
                    None
                    if latest_row["iris_running"] is None
                    else bool(latest_row["iris_running"])
                ),
            }

        run_snapshot = {
            "run_id": run_id,
            "trial_id": run["trial_id"],
            "region": run["region"],
            "status": run["status"],
            "wandb_run_id": run["wandb_run_id"],
            "checkpoint_root": run["checkpoint_root"],
            "is_winner": bool(run["is_winner"]),
            "run_progress_high_water": run["high_water_progress"],
            "progress_delta_since_previous_observation": round(
                max(0.0, observed_progress_high_water - prior_progress_high_water), 12
            ),
            "stall_since": None
            if last_progress_at is None
            else last_progress_at.isoformat(),
            "stall_seconds": None if stall_seconds is None else round(stall_seconds, 3),
            "checkpoint_verified_at": run["checkpoint_verified_at"],
            "dispatch_count": len(dispatches_by_run[run_id]),
            "active_dispatch": active_dispatch_snapshot,
            "latest_observation": latest_observation_snapshot,
        }
        current_run_by_id[run_id] = run_snapshot
        if run["trial_id"] not in completed_trial_ids:
            run_snapshots.append(run_snapshot)

        run_conditions: list[str] = []
        state_row = active_latest_row if active_dispatch is not None else latest_row
        latest_state = (
            state_row["wandb_state"].casefold()
            if state_row is not None and state_row["wandb_state"] is not None
            else None
        )
        if active_dispatch is not None and active_latest_row is None:
            run_conditions.append("active_dispatch_unobserved")
        if latest_state == "failed":
            run_conditions.append("wandb_failed")
        if latest_state == "finished" and run["high_water_progress"] < 1:
            run_conditions.append("wandb_finished_incomplete")
        if (
            active_dispatch is not None
            and active_latest_row is not None
            and active_latest_row["iris_running"] == 0
        ):
            run_conditions.append("iris_not_running")
        if run["high_water_progress"] >= 1 and run["checkpoint_verified_at"] is None:
            run_conditions.append("checkpoint_unverified")
        if (
            active_dispatch is None
            and run["trial_id"] not in completed_trial_ids
            and run["status"].casefold() in {"planned", "active", "running", "failed"}
        ):
            run_conditions.append("no_active_dispatch")
        if run["trial_id"] not in completed_trial_ids:
            conditions.extend(
                {
                    "kind": kind,
                    "trial_id": run["trial_id"],
                    "run_id": run_id,
                }
                for kind in run_conditions
            )

    trial_snapshots: list[dict[str, object]] = []
    for trial in trials:
        trial_id = trial["trial_id"]
        if trial_id in completed_trial_ids:
            continue
        trial_runs = runs_by_trial[trial_id]
        trial_snapshots.append(
            {
                "trial_id": trial_id,
                "status": trial["status"],
                "run_count": len(trial_runs),
                "active_dispatch_count": sum(
                    1
                    for run in trial_runs
                    for dispatch in dispatches_by_run[run["regional_run_id"]]
                    if dispatch["active"]
                ),
                "run_progress_high_water": max(
                    (run["high_water_progress"] for run in trial_runs), default=0.0
                ),
                "winning_run_id": next(
                    (run["regional_run_id"] for run in trial_runs if run["is_winner"]),
                    None,
                ),
            }
        )

    placement_groups: dict[tuple[str, str, int], dict[str, object]] = {}
    for dispatch in dispatches:
        run = run_by_id[dispatch["regional_run_id"]]
        run_observations = observations_by_run[run["regional_run_id"]]
        submitted_at = _parse_time(
            dispatch["submitted_at"], f"dispatch {dispatch['dispatch_id']} submitted_at"
        )
        dispatch_observations = [
            (observation, observed_at)
            for observation, observed_at in run_observations
            if observation["dispatch_id"] == dispatch["dispatch_id"]
            and observed_at >= submitted_at
        ]
        if dispatch["active"]:
            measured_at = snapshot_at
        else:
            measured_at = _parse_time(
                dispatch["ended_at"], f"dispatch {dispatch['dispatch_id']} ended_at"
            )
            if dispatch_observations:
                measured_at = max(
                    measured_at,
                    max(observed_at for _, observed_at in dispatch_observations),
                )
        if measured_at < submitted_at:
            raise RuntimeError(
                f"dispatch {dispatch['dispatch_id']!r} ends before submission"
            )

        progress_delta = progress_by_dispatch[dispatch["dispatch_id"]]
        wall_seconds = (measured_at - submitted_at).total_seconds()
        first_progress_at = first_progress_at_by_dispatch.get(dispatch["dispatch_id"])
        first_progress_seconds = (
            None
            if first_progress_at is None
            else (first_progress_at - submitted_at).total_seconds()
        )

        key = (
            run["region"],
            dispatch["tpu_slice"],
            dispatch["chips"],
        )
        group = placement_groups.setdefault(
            key,
            {
                "region": run["region"],
                "slice": dispatch["tpu_slice"],
                "chips": dispatch["chips"],
                "dispatch_count": 0,
                "dispatches_with_progress": 0,
                "zero_progress_dispatches": 0,
                "active_dispatches": 0,
                "productive_dispatches": 0,
                "pending_dispatches": 0,
                "total_progress": 0.0,
                "total_wall_time_seconds": 0.0,
                "first_progress_seconds": [],
            },
        )
        group["dispatch_count"] += 1
        group["total_progress"] += progress_delta
        group["total_wall_time_seconds"] += wall_seconds
        if progress_delta > 0:
            group["dispatches_with_progress"] += 1
        else:
            group["zero_progress_dispatches"] += 1
        if first_progress_seconds is not None:
            group["first_progress_seconds"].append(first_progress_seconds)
        if dispatch["active"]:
            group["active_dispatches"] += 1
            last_progress_at = last_progress_at_by_dispatch.get(dispatch["dispatch_id"])
            if (
                last_progress_at is not None
                and (snapshot_at - last_progress_at).total_seconds()
                <= reslice_after_seconds
            ):
                group["productive_dispatches"] += 1
            else:
                group["pending_dispatches"] += 1

    placement_snapshots: list[dict[str, object]] = []
    for group in placement_groups.values():
        wall_seconds = group.pop("total_wall_time_seconds")
        progress = group.pop("total_progress")
        first_progress_values = group.pop("first_progress_seconds")
        group["total_progress"] = round(progress, 12)
        group["total_wall_time_seconds"] = round(wall_seconds, 3)
        group["target_rate"] = (
            None if wall_seconds <= 0 else round(progress * 3600 / wall_seconds, 12)
        )
        group["average_time_to_first_progress_seconds"] = (
            None
            if not first_progress_values
            else round(sum(first_progress_values) / len(first_progress_values), 3)
        )
        placement_snapshots.append(group)
    placement_snapshots.sort(
        key=lambda group: (
            -1.0 if group["target_rate"] is None else -group["target_rate"],
            group["region"],
            group["slice"],
            group["chips"],
        )
    )

    trial_status_counts = Counter(trial["status"] for trial in trials)
    run_status_counts = Counter(run["status"] for run in runs)
    active_dispatches = [dispatch for dispatch in dispatches if dispatch["active"]]
    training_chips = 0
    training_runs = 0
    for dispatch in active_dispatches:
        current_run = current_run_by_id[dispatch["regional_run_id"]]
        active = current_run["active_dispatch"]
        latest = None if active is None else active["latest_observation"]
        if (
            latest is not None
            and latest["wandb_state"] is not None
            and latest["wandb_state"].casefold() == "running"
        ):
            training_runs += 1
            training_chips += dispatch["chips"]

    recent_event_snapshots = [
        {
            "event_id": event["event_id"],
            "recorded_at": event["recorded_at"],
            "kind": event["kind"],
            "trial_id": event["trial_id"],
            "run_id": event["regional_run_id"],
            "dispatch_id": event["dispatch_id"],
            "detail": event["detail"],
        }
        for event in recent_events
    ]

    return {
        "snapshot_at_utc": snapshot_at.isoformat(),
        "reslice_after_seconds": reslice_after_seconds,
        "coverage": {
            "trials_total": len(trials),
            "unfinished_trials_included": len(trial_snapshots),
            "completed_trials_omitted": len(completed_trial_ids),
            "runs_total": len(runs),
            "unfinished_runs_included": len(run_snapshots),
            "dispatches_total": len(dispatches),
            "observations_total": len(observations),
            "events_total": event_total,
            "recent_events_included": len(recent_event_snapshots),
            "recent_events_omitted": event_total - len(recent_event_snapshots),
            "latest_observation_at_utc": (
                None
                if not latest_observation_times
                else max(latest_observation_times).isoformat()
            ),
            "unfinished_runs_without_observations": sum(
                1
                for run_snapshot in run_snapshots
                if run_snapshot["latest_observation"] is None
            ),
        },
        "fleet": {
            "trial_status_counts": dict(sorted(trial_status_counts.items())),
            "run_status_counts": dict(sorted(run_status_counts.items())),
            "active_dispatches": len(active_dispatches),
            "submitted_chips": sum(dispatch["chips"] for dispatch in active_dispatches),
            "wandb_running_runs": training_runs,
            "wandb_running_chips": training_chips,
        },
        "trials": trial_snapshots,
        "runs": run_snapshots,
        "conditions": conditions,
        "placements": placement_snapshots,
        "event_counts": event_counts,
        "recent_events": recent_event_snapshots,
    }


def check(path: Path) -> list[str]:
    errors = _structural_errors(path)
    if errors:
        return errors
    try:
        _build_snapshot(path, recent_event_limit=0, reslice_after_hours=1)
    except (RuntimeError, sqlite3.Error) as error:
        errors.append(f"semantic_check: {error}")
    return errors


def snapshot(
    path: Path, recent_event_limit: int, reslice_after_hours: float = 1
) -> dict[str, object]:
    errors = _structural_errors(path)
    if errors:
        raise RuntimeError("; ".join(errors))
    return _build_snapshot(path, recent_event_limit, reslice_after_hours)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("database", type=Path)

    event_parser = subparsers.add_parser("event")
    event_parser.add_argument("database", type=Path)
    event_parser.add_argument("--kind", required=True)
    event_parser.add_argument("--detail", required=True)
    event_parser.add_argument("--trial-id")
    event_parser.add_argument("--regional-run-id")
    event_parser.add_argument("--dispatch-id")

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("database", type=Path)

    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("database", type=Path)
    snapshot_parser.add_argument("--recent-events", type=int, default=20)
    snapshot_parser.add_argument("--reslice-after-hours", type=float, default=1)

    args = parser.parse_args()
    if args.command == "init":
        init(args.database)
        print(f"OK: initialized {args.database}")
        return 0
    if args.command == "event":
        record_event(
            args.database,
            args.kind,
            args.detail,
            args.trial_id,
            args.regional_run_id,
            args.dispatch_id,
        )
        print("OK: event recorded")
        return 0
    if args.command == "check":
        errors = check(args.database)
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        print(f"OK: {args.database}")
        return 0
    if args.command == "snapshot":
        try:
            result = snapshot(
                args.database, args.recent_events, args.reslice_after_hours
            )
        except (RuntimeError, sqlite3.Error) as error:
            print(f"ERROR: {error}")
            return 1
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
