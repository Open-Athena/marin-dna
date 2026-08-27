"""Initialize, inspect, and check CoreWeave training-sweep persistence."""

import argparse
import hashlib
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
    wandb_run_id TEXT NOT NULL UNIQUE,
    wandb_url TEXT,
    checkpoint_root TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'planned',
    high_water_progress REAL NOT NULL DEFAULT 0 CHECK (high_water_progress >= 0),
    checkpoint_verified_at TEXT
);

CREATE TABLE IF NOT EXISTS dispatches (
    dispatch_id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL REFERENCES trials(trial_id),
    attempt INTEGER NOT NULL CHECK (attempt > 0),
    iris_job_id TEXT NOT NULL UNIQUE,
    cluster TEXT NOT NULL,
    gpu_variant TEXT NOT NULL,
    nodes INTEGER NOT NULL CHECK (nodes > 0),
    gpus INTEGER NOT NULL CHECK (gpus > 0),
    priority_band TEXT NOT NULL CHECK (priority_band = 'batch'),
    command_redacted TEXT NOT NULL,
    intent_at TEXT NOT NULL,
    submitted_at TEXT,
    submission_state TEXT NOT NULL DEFAULT 'intent'
        CHECK (submission_state IN ('intent', 'submitted', 'ended')),
    ended_at TEXT,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    outcome TEXT,
    stop_reason TEXT,
    UNIQUE (trial_id, attempt),
    CHECK (
        (submission_state = 'intent' AND active = 1
            AND submitted_at IS NULL AND ended_at IS NULL)
        OR (submission_state = 'submitted' AND active = 1
            AND submitted_at IS NOT NULL AND ended_at IS NULL)
        OR (submission_state = 'ended' AND active = 0 AND ended_at IS NOT NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_dispatch_per_trial
ON dispatches(trial_id) WHERE active = 1;

CREATE INDEX IF NOT EXISTS dispatch_target_history
ON dispatches(cluster, gpu_variant, nodes, gpus, intent_at);

CREATE TABLE IF NOT EXISTS observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id TEXT NOT NULL REFERENCES trials(trial_id),
    dispatch_id TEXT NOT NULL REFERENCES dispatches(dispatch_id),
    observed_at TEXT NOT NULL,
    wandb_state TEXT,
    run_progress REAL CHECK (run_progress IS NULL OR run_progress >= 0),
    iris_running INTEGER CHECK (iris_running IS NULL OR iris_running IN (0, 1)),
    UNIQUE (trial_id, observed_at)
);

CREATE INDEX IF NOT EXISTS observations_by_dispatch_time
ON observations(dispatch_id, observed_at);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    trial_id TEXT REFERENCES trials(trial_id),
    dispatch_id TEXT REFERENCES dispatches(dispatch_id),
    detail TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS events_by_time ON events(recorded_at);
"""

REQUIRED_COLUMNS = {
    "meta": {"key", "value"},
    "trials": {
        "trial_id",
        "env_json",
        "wandb_run_id",
        "checkpoint_root",
        "status",
        "high_water_progress",
        "checkpoint_verified_at",
    },
    "dispatches": {
        "dispatch_id",
        "trial_id",
        "attempt",
        "iris_job_id",
        "cluster",
        "gpu_variant",
        "nodes",
        "gpus",
        "priority_band",
        "command_redacted",
        "intent_at",
        "submitted_at",
        "submission_state",
        "ended_at",
        "active",
        "outcome",
        "stop_reason",
    },
    "observations": {
        "observation_id",
        "trial_id",
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
            SELECT name FROM sqlite_master
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
    dispatch_id: str | None,
) -> None:
    with _connect(path) as connection:
        _record_event(connection, kind, detail, trial_id, dispatch_id)


def _record_event(
    connection: sqlite3.Connection,
    kind: str,
    detail: str,
    trial_id: str | None,
    dispatch_id: str | None,
) -> None:
    connection.execute(
        """
        INSERT INTO events(recorded_at, kind, trial_id, dispatch_id, detail)
        VALUES (?, ?, ?, ?, ?)
        """,
        (_now(), kind, trial_id, dispatch_id, detail),
    )


def add_trial(
    path: Path,
    trial_id: str,
    env: dict[str, object],
    wandb_run_id: str,
    checkpoint_root: str,
    wandb_url: str | None = None,
) -> None:
    with _connect(path) as connection:
        connection.execute(
            """
            INSERT INTO trials(
                trial_id, env_json, wandb_run_id, wandb_url, checkpoint_root
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                trial_id,
                json.dumps(env, sort_keys=True),
                wandb_run_id,
                wandb_url,
                checkpoint_root,
            ),
        )
        _record_event(connection, "trial_added", "trial registered", trial_id, None)


def prepare_dispatch(
    path: Path,
    trial_id: str,
    dispatch_id: str,
    iris_job_id: str,
    cluster: str,
    gpu_variant: str,
    nodes: int,
    gpus: int,
    command_redacted: str,
    intent_at: str | None = None,
) -> int:
    recorded_at = intent_at or _now()
    _parse_time(recorded_at, "intent_at")
    with _connect(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        trial = connection.execute(
            "SELECT status FROM trials WHERE trial_id = ?", (trial_id,)
        ).fetchone()
        if trial is None:
            raise RuntimeError(f"unknown trial: {trial_id}")
        if trial[0].casefold() == "completed":
            raise RuntimeError(f"trial {trial_id!r} is completed")
        active = connection.execute(
            """
            SELECT dispatch_id, submission_state FROM dispatches
            WHERE trial_id = ? AND active = 1
            """,
            (trial_id,),
        ).fetchone()
        if active is not None:
            raise RuntimeError(
                f"trial {trial_id!r} already has active dispatch {active[0]!r} "
                f"in state {active[1]!r}"
            )
        attempt = connection.execute(
            "SELECT COALESCE(MAX(attempt), 0) + 1 FROM dispatches WHERE trial_id = ?",
            (trial_id,),
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO dispatches(
                dispatch_id, trial_id, attempt, iris_job_id, cluster, gpu_variant,
                nodes, gpus, priority_band, command_redacted, intent_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'batch', ?, ?)
            """,
            (
                dispatch_id,
                trial_id,
                attempt,
                iris_job_id,
                cluster,
                gpu_variant,
                nodes,
                gpus,
                command_redacted,
                recorded_at,
            ),
        )
        _record_event(
            connection,
            "dispatch_intent",
            f"iris_job_id={iris_job_id} target={cluster}/{gpu_variant}/{nodes}/{gpus}",
            trial_id,
            dispatch_id,
        )
    return attempt


def confirm_dispatch(
    path: Path, dispatch_id: str, submitted_at: str | None = None
) -> str:
    confirmed_at = submitted_at or _now()
    _parse_time(confirmed_at, "submitted_at")
    with _connect(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT trial_id, intent_at, submitted_at, submission_state
            FROM dispatches WHERE dispatch_id = ?
            """,
            (dispatch_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"unknown dispatch: {dispatch_id}")
        trial_id, intent_at, existing_submitted_at, state = row
        if state == "submitted":
            return existing_submitted_at
        if state != "intent":
            raise RuntimeError(f"dispatch {dispatch_id!r} is already ended")
        if _parse_time(confirmed_at, "submitted_at") < _parse_time(
            intent_at, "intent_at"
        ):
            raise RuntimeError("submitted_at predates intent_at")
        connection.execute(
            """
            UPDATE dispatches
            SET submitted_at = ?, submission_state = 'submitted'
            WHERE dispatch_id = ?
            """,
            (confirmed_at, dispatch_id),
        )
        _record_event(
            connection,
            "dispatch_submitted",
            "exact Iris job accepted",
            trial_id,
            dispatch_id,
        )
    return confirmed_at


def end_dispatch(
    path: Path,
    dispatch_id: str,
    outcome: str,
    stop_reason: str | None = None,
    ended_at: str | None = None,
) -> str:
    terminal_at = ended_at or _now()
    _parse_time(terminal_at, "ended_at")
    with _connect(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT trial_id, intent_at, submitted_at, submission_state, outcome
            FROM dispatches WHERE dispatch_id = ?
            """,
            (dispatch_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"unknown dispatch: {dispatch_id}")
        trial_id, intent_at, submitted_at, state, existing_outcome = row
        if state == "ended":
            if existing_outcome != outcome:
                raise RuntimeError(
                    f"dispatch {dispatch_id!r} already ended as {existing_outcome!r}"
                )
            return terminal_at
        lower_bound = submitted_at or intent_at
        if _parse_time(terminal_at, "ended_at") < _parse_time(
            lower_bound, "dispatch start"
        ):
            raise RuntimeError("ended_at predates dispatch intent or submission")
        connection.execute(
            """
            UPDATE dispatches
            SET submission_state = 'ended', ended_at = ?, active = 0,
                outcome = ?, stop_reason = ?
            WHERE dispatch_id = ?
            """,
            (terminal_at, outcome, stop_reason, dispatch_id),
        )
        event_kind = "dispatch_terminal" if submitted_at else "dispatch_not_submitted"
        _record_event(
            connection,
            event_kind,
            f"outcome={outcome}; reason={stop_reason or '-'}",
            trial_id,
            dispatch_id,
        )
    return terminal_at


def record_observation(
    path: Path,
    dispatch_id: str,
    observed_at: str,
    wandb_state: str | None,
    run_progress: float | None,
    iris_running: bool | None,
) -> None:
    observed_time = _parse_time(observed_at, "observed_at")
    if run_progress is not None and run_progress < 0:
        raise RuntimeError("run_progress must be nonnegative")
    with _connect(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT trial_id, submitted_at FROM dispatches WHERE dispatch_id = ?
            """,
            (dispatch_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"unknown dispatch: {dispatch_id}")
        trial_id, submitted_at = row
        if submitted_at is None:
            raise RuntimeError(
                f"dispatch {dispatch_id!r} has not been confirmed as submitted"
            )
        if observed_time < _parse_time(submitted_at, "submitted_at"):
            raise RuntimeError("observed_at predates submitted_at")
        previous = connection.execute(
            "SELECT high_water_progress FROM trials WHERE trial_id = ?", (trial_id,)
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO observations(
                trial_id, dispatch_id, observed_at, wandb_state,
                run_progress, iris_running
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                trial_id,
                dispatch_id,
                observed_at,
                wandb_state,
                run_progress,
                None if iris_running is None else int(iris_running),
            ),
        )
        if run_progress is not None and run_progress > previous:
            connection.execute(
                """
                UPDATE trials SET high_water_progress = ?, status = 'active'
                WHERE trial_id = ? AND status != 'completed'
                """,
                (run_progress, trial_id),
            )
            _record_event(
                connection,
                "progress",
                f"run_progress={run_progress}",
                trial_id,
                dispatch_id,
            )


def complete_trial(
    path: Path, trial_id: str, checkpoint_verified_at: str | None = None
) -> str:
    verified_at = checkpoint_verified_at or _now()
    _parse_time(verified_at, "checkpoint_verified_at")
    with _connect(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT status, high_water_progress, checkpoint_verified_at
            FROM trials WHERE trial_id = ?
            """,
            (trial_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"unknown trial: {trial_id}")
        if row[0].casefold() == "completed":
            if row[2] is None:
                raise RuntimeError(
                    f"completed trial {trial_id!r} has no verified checkpoint time"
                )
            return row[2]
        if row[1] < 1:
            raise RuntimeError(f"trial {trial_id!r} has progress below 1")
        if (
            connection.execute(
                "SELECT 1 FROM dispatches WHERE trial_id = ? AND active = 1",
                (trial_id,),
            ).fetchone()
            is not None
        ):
            raise RuntimeError(f"trial {trial_id!r} still has an active dispatch")
        connection.execute(
            """
            UPDATE trials
            SET status = 'completed', checkpoint_verified_at = ?
            WHERE trial_id = ?
            """,
            (verified_at, trial_id),
        )
        _record_event(
            connection,
            "trial_completed",
            "progress complete and checkpoint verified",
            trial_id,
            None,
        )
    return verified_at


def backup(path: Path, destination: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"database does not exist: {path}")
    if destination.exists():
        raise RuntimeError(f"backup destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as source, sqlite3.connect(destination) as target:
        source.backup(target)
    errors = check(destination)
    if errors:
        raise RuntimeError("backup validation failed: " + "; ".join(errors))
    data = destination.read_bytes()
    return {
        "path": str(destination),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _structural_errors(path: Path) -> list[str]:
    if not path.is_file():
        return [f"database does not exist: {path}"]
    errors: list[str] = []
    with _connect(path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            errors.append(f"integrity_check: {integrity}")
        errors.extend(
            f"foreign_key_check: {row}"
            for row in connection.execute("PRAGMA foreign_key_check")
        )
        errors.extend(_schema_errors(connection))
        if errors:
            return errors
        duplicate_active = connection.execute(
            """
            SELECT trial_id, COUNT(*) FROM dispatches
            WHERE active = 1 GROUP BY trial_id HAVING COUNT(*) > 1
            """
        ).fetchall()
        errors.extend(f"multiple active dispatches: {row}" for row in duplicate_active)
        inconsistent_dispatches = connection.execute(
            """
            SELECT dispatch_id, submission_state, active, submitted_at, ended_at
            FROM dispatches
            WHERE (submission_state = 'intent'
                    AND (active != 1 OR submitted_at IS NOT NULL OR ended_at IS NOT NULL))
               OR (submission_state = 'submitted'
                    AND (active != 1 OR submitted_at IS NULL OR ended_at IS NOT NULL))
               OR (submission_state = 'ended'
                    AND (active != 0 OR ended_at IS NULL))
            """
        ).fetchall()
        errors.extend(
            f"inconsistent dispatch end state: {row}" for row in inconsistent_dispatches
        )
        mismatches = connection.execute(
            """
            SELECT observations.observation_id FROM observations
            JOIN dispatches USING (dispatch_id)
            WHERE observations.trial_id != dispatches.trial_id
            """
        ).fetchall()
        errors.extend(
            f"observation dispatch/trial mismatch: {row[0]}" for row in mismatches
        )
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
        dispatches = list(
            connection.execute(
                "SELECT * FROM dispatches ORDER BY intent_at, dispatch_id"
            )
        )
        observations = list(
            connection.execute(
                "SELECT * FROM observations ORDER BY trial_id, observed_at, observation_id"
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
                SELECT event_id, recorded_at, kind, trial_id, dispatch_id, detail
                FROM events ORDER BY event_id DESC LIMIT ?
                """,
                (recent_event_limit,),
            )
        )

    trial_by_id = {row["trial_id"]: row for row in trials}
    dispatch_by_id = {row["dispatch_id"]: row for row in dispatches}
    dispatches_by_trial: dict[str, list[sqlite3.Row]] = defaultdict(list)
    observations_by_trial: dict[str, list[tuple[sqlite3.Row, datetime]]] = defaultdict(
        list
    )
    observations_by_dispatch: dict[str, list[tuple[sqlite3.Row, datetime]]] = (
        defaultdict(list)
    )

    for dispatch in dispatches:
        dispatches_by_trial[dispatch["trial_id"]].append(dispatch)
    for observation in observations:
        observed_at = _parse_time(
            observation["observed_at"],
            f"observation {observation['observation_id']} observed_at",
        )
        if observed_at > snapshot_at:
            raise RuntimeError(
                f"observation {observation['observation_id']} is in the future"
            )
        dispatch = dispatch_by_id[observation["dispatch_id"]]
        if dispatch["submitted_at"] is None:
            raise RuntimeError(
                f"observation {observation['observation_id']} belongs to an "
                "unsubmitted dispatch intent"
            )
        submitted_at = _parse_time(
            dispatch["submitted_at"], f"dispatch {dispatch['dispatch_id']} submitted_at"
        )
        if observed_at < submitted_at:
            raise RuntimeError(
                f"observation {observation['observation_id']} predates its dispatch"
            )
        item = (observation, observed_at)
        observations_by_trial[observation["trial_id"]].append(item)
        observations_by_dispatch[observation["dispatch_id"]].append(item)

    progress_by_dispatch: dict[str, float] = defaultdict(float)
    first_progress_at: dict[str, datetime] = {}
    last_progress_at: dict[str, datetime] = {}
    last_trial_progress_at: dict[str, datetime] = {}
    for trial_id, trial_observations in observations_by_trial.items():
        high_water = 0.0
        for observation, observed_at in trial_observations:
            progress = observation["run_progress"]
            if progress is None or progress <= high_water:
                continue
            progress_by_dispatch[observation["dispatch_id"]] += progress - high_water
            high_water = progress
            dispatch_id = observation["dispatch_id"]
            first_progress_at.setdefault(dispatch_id, observed_at)
            last_progress_at[dispatch_id] = observed_at
            last_trial_progress_at[trial_id] = observed_at

    completed_ids = {
        trial["trial_id"]
        for trial in trials
        if trial["status"].casefold() == "completed"
    }
    for trial_id in completed_ids:
        trial = trial_by_id[trial_id]
        if trial["high_water_progress"] < 1:
            raise RuntimeError(f"completed trial {trial_id!r} has progress below 1")
        if trial["checkpoint_verified_at"] is None:
            raise RuntimeError(
                f"completed trial {trial_id!r} has no verified checkpoint"
            )
        if any(dispatch["active"] for dispatch in dispatches_by_trial[trial_id]):
            raise RuntimeError(f"completed trial {trial_id!r} has an active dispatch")

    trial_snapshots: list[dict[str, object]] = []
    current_trial_by_id: dict[str, dict[str, object]] = {}
    conditions: list[dict[str, object]] = []
    latest_observation_times: list[datetime] = []

    for trial in trials:
        trial_id = trial["trial_id"]
        trial_observations = observations_by_trial[trial_id]
        progress_values = [
            row["run_progress"]
            for row, _ in trial_observations
            if row["run_progress"] is not None
        ]
        observed_high_water = max(progress_values, default=0.0)
        if abs(observed_high_water - trial["high_water_progress"]) > 1e-12:
            raise RuntimeError(
                f"trial {trial_id!r} progress high-water does not match observations"
            )

        latest = trial_observations[-1] if trial_observations else None
        latest_row = latest[0] if latest else None
        latest_at = latest[1] if latest else None
        if latest_at is not None:
            latest_observation_times.append(latest_at)
        prior_high_water = max(
            (
                row["run_progress"]
                for row, _ in trial_observations[:-1]
                if row["run_progress"] is not None
            ),
            default=0.0,
        )

        active = [row for row in dispatches_by_trial[trial_id] if row["active"]]
        if len(active) > 1:
            raise RuntimeError(f"trial {trial_id!r} has multiple active dispatches")
        active_dispatch = active[0] if active else None
        active_snapshot = None
        active_latest_row = None
        active_latest_at = None
        if active_dispatch is not None:
            dispatch_id = active_dispatch["dispatch_id"]
            intent_at = _parse_time(
                active_dispatch["intent_at"], f"dispatch {dispatch_id} intent_at"
            )
            if intent_at > snapshot_at:
                raise RuntimeError(f"dispatch {dispatch_id!r} intent is in the future")
            submitted_at = None
            progress_at = None
            stall_since = None
            stall_seconds = None
            if active_dispatch["submission_state"] == "submitted":
                submitted_at = _parse_time(
                    active_dispatch["submitted_at"],
                    f"dispatch {dispatch_id} submitted_at",
                )
                if submitted_at > snapshot_at:
                    raise RuntimeError(f"dispatch {dispatch_id!r} is in the future")
                active_observations = observations_by_dispatch[dispatch_id]
                if active_observations:
                    active_latest_row, active_latest_at = active_observations[-1]
                progress_at = last_progress_at.get(dispatch_id)
                stall_since = progress_at or submitted_at
                stall_seconds = (snapshot_at - stall_since).total_seconds()
            active_snapshot = {
                "dispatch_id": dispatch_id,
                "attempt": active_dispatch["attempt"],
                "iris_job_id": active_dispatch["iris_job_id"],
                "submission_state": active_dispatch["submission_state"],
                "cluster": active_dispatch["cluster"],
                "gpu_variant": active_dispatch["gpu_variant"],
                "nodes": active_dispatch["nodes"],
                "gpus": active_dispatch["gpus"],
                "intent_at": active_dispatch["intent_at"],
                "submitted_at": active_dispatch["submitted_at"],
                "age_seconds": round(
                    (snapshot_at - (submitted_at or intent_at)).total_seconds(), 3
                ),
                "last_progress_at": None
                if progress_at is None
                else progress_at.isoformat(),
                "stall_since": None if stall_since is None else stall_since.isoformat(),
                "stall_seconds": None
                if stall_seconds is None
                else round(stall_seconds, 3),
                "recent_progress": progress_at is not None
                and stall_seconds is not None
                and stall_seconds <= reslice_after_seconds,
                "latest_observation": None
                if active_latest_row is None or active_latest_at is None
                else {
                    "observed_at": active_latest_row["observed_at"],
                    "age_seconds": round(
                        (snapshot_at - active_latest_at).total_seconds(), 3
                    ),
                    "wandb_state": active_latest_row["wandb_state"],
                    "run_progress": active_latest_row["run_progress"],
                    "iris_running": None
                    if active_latest_row["iris_running"] is None
                    else bool(active_latest_row["iris_running"]),
                },
            }

        latest_snapshot = None
        if latest_row is not None and latest_at is not None:
            latest_snapshot = {
                "dispatch_id": latest_row["dispatch_id"],
                "observed_at": latest_row["observed_at"],
                "age_seconds": round((snapshot_at - latest_at).total_seconds(), 3),
                "wandb_state": latest_row["wandb_state"],
                "run_progress": latest_row["run_progress"],
                "iris_running": None
                if latest_row["iris_running"] is None
                else bool(latest_row["iris_running"]),
            }

        trial_progress_at = last_trial_progress_at.get(trial_id)
        submitted_dispatches = [
            row
            for row in dispatches_by_trial[trial_id]
            if row["submitted_at"] is not None
        ]
        if trial_progress_at is None and submitted_dispatches:
            trial_progress_at = min(
                _parse_time(
                    row["submitted_at"], f"dispatch {row['dispatch_id']} submitted_at"
                )
                for row in submitted_dispatches
            )
        snapshot = {
            "trial_id": trial_id,
            "status": trial["status"],
            "wandb_run_id": trial["wandb_run_id"],
            "wandb_url": trial["wandb_url"],
            "checkpoint_root": trial["checkpoint_root"],
            "high_water_progress": trial["high_water_progress"],
            "progress_delta_since_latest_prior_observation": max(
                0.0, trial["high_water_progress"] - prior_high_water
            ),
            "last_progress_at": None
            if trial_progress_at is None
            else trial_progress_at.isoformat(),
            "stall_seconds": None
            if trial_progress_at is None
            else round((snapshot_at - trial_progress_at).total_seconds(), 3),
            "checkpoint_verified_at": trial["checkpoint_verified_at"],
            "dispatch_count": len(dispatches_by_trial[trial_id]),
            "active_dispatch": active_snapshot,
            "latest_observation": latest_snapshot,
        }
        current_trial_by_id[trial_id] = snapshot

        if trial_id not in completed_ids:
            kinds: list[str] = []
            if active_dispatch is None:
                kinds.append("no_active_dispatch")
            elif active_dispatch["submission_state"] == "intent":
                kinds.append("dispatch_intent_unreconciled")
            elif active_latest_row is None:
                kinds.append("active_dispatch_unobserved")
            else:
                if active_latest_row["wandb_state"] == "failed":
                    kinds.append("wandb_failed")
                if (
                    active_latest_row["wandb_state"] == "finished"
                    and trial["high_water_progress"] < 1
                ):
                    kinds.append("wandb_finished_before_completion")
                if active_latest_row["iris_running"] == 0:
                    kinds.append("iris_not_running")
            for kind in kinds:
                conditions.append({"kind": kind, "trial_id": trial_id})
            trial_snapshots.append(snapshot)

    placement_groups: dict[tuple[str, str, int, int], dict[str, object]] = {}
    for dispatch in dispatches:
        if dispatch["submitted_at"] is None:
            continue
        submitted_at = _parse_time(
            dispatch["submitted_at"], f"dispatch {dispatch['dispatch_id']} submitted_at"
        )
        dispatch_observations = observations_by_dispatch[dispatch["dispatch_id"]]
        if dispatch["active"]:
            effective_end = snapshot_at
        else:
            ended_at = _parse_time(
                dispatch["ended_at"], f"dispatch {dispatch['dispatch_id']} ended_at"
            )
            effective_end = max(
                [ended_at, *(observed_at for _, observed_at in dispatch_observations)]
            )
        if effective_end < submitted_at:
            raise RuntimeError(
                f"dispatch {dispatch['dispatch_id']!r} ends before submission"
            )

        key = (
            dispatch["cluster"],
            dispatch["gpu_variant"],
            dispatch["nodes"],
            dispatch["gpus"],
        )
        group = placement_groups.setdefault(
            key,
            {
                "cluster": dispatch["cluster"],
                "gpu_variant": dispatch["gpu_variant"],
                "nodes": dispatch["nodes"],
                "gpus": dispatch["gpus"],
                "dispatch_count": 0,
                "dispatches_with_progress": 0,
                "zero_progress_dispatches": 0,
                "active_dispatches": 0,
                "productive_dispatches": 0,
                "pending_dispatches": 0,
                "total_progress": 0.0,
                "total_wall_time_seconds": 0.0,
                "time_to_first_progress_seconds": [],
            },
        )
        delta = progress_by_dispatch[dispatch["dispatch_id"]]
        group["dispatch_count"] += 1
        group["total_progress"] += delta
        group["total_wall_time_seconds"] += (
            effective_end - submitted_at
        ).total_seconds()
        if delta > 0:
            group["dispatches_with_progress"] += 1
            group["time_to_first_progress_seconds"].append(
                (
                    first_progress_at[dispatch["dispatch_id"]] - submitted_at
                ).total_seconds()
            )
        else:
            group["zero_progress_dispatches"] += 1
        if dispatch["active"]:
            group["active_dispatches"] += 1
            progress_at = last_progress_at.get(dispatch["dispatch_id"])
            if (
                progress_at is not None
                and (snapshot_at - progress_at).total_seconds() <= reslice_after_seconds
            ):
                group["productive_dispatches"] += 1
            else:
                group["pending_dispatches"] += 1

    placements: list[dict[str, object]] = []
    for group in placement_groups.values():
        first_times = group.pop("time_to_first_progress_seconds")
        wall_hours = group["total_wall_time_seconds"] / 3600
        group["total_progress"] = round(group["total_progress"], 12)
        group["total_wall_time_seconds"] = round(group["total_wall_time_seconds"], 3)
        group["target_rate"] = (
            None if wall_hours == 0 else round(group["total_progress"] / wall_hours, 12)
        )
        group["average_time_to_first_progress_seconds"] = (
            None if not first_times else round(sum(first_times) / len(first_times), 3)
        )
        placements.append(group)
    placements.sort(
        key=lambda row: (
            -1.0 if row["target_rate"] is None else -row["target_rate"],
            row["cluster"],
            row["gpu_variant"],
            row["nodes"],
        )
    )

    active_intents = [
        row
        for row in dispatches
        if row["active"] and row["submission_state"] == "intent"
    ]
    active_dispatches = [
        row
        for row in dispatches
        if row["active"] and row["submission_state"] == "submitted"
    ]
    training_dispatches = 0
    training_gpus = 0
    for dispatch in active_dispatches:
        active = current_trial_by_id[dispatch["trial_id"]]["active_dispatch"]
        latest = None if active is None else active["latest_observation"]
        if latest is not None and latest["wandb_state"] == "running":
            training_dispatches += 1
            training_gpus += dispatch["gpus"]

    recent_event_snapshots = [
        {
            "event_id": row["event_id"],
            "recorded_at": row["recorded_at"],
            "kind": row["kind"],
            "trial_id": row["trial_id"],
            "dispatch_id": row["dispatch_id"],
            "detail": row["detail"],
        }
        for row in reversed(recent_events)
    ]
    trial_status_counts = Counter(row["status"] for row in trials)

    return {
        "snapshot_at_utc": snapshot_at.isoformat(),
        "reslice_after_seconds": reslice_after_seconds,
        "coverage": {
            "trials_total": len(trials),
            "unfinished_trials_included": len(trial_snapshots),
            "completed_trials_omitted": len(completed_ids),
            "dispatches_total": len(dispatches),
            "observations_total": len(observations),
            "events_total": event_total,
            "recent_events_included": len(recent_event_snapshots),
            "recent_events_omitted": event_total - len(recent_event_snapshots),
            "latest_observation_at_utc": None
            if not latest_observation_times
            else max(latest_observation_times).isoformat(),
            "unfinished_trials_without_observations": sum(
                1 for row in trial_snapshots if row["latest_observation"] is None
            ),
        },
        "fleet": {
            "trial_status_counts": dict(sorted(trial_status_counts.items())),
            "unreconciled_dispatch_intents": len(active_intents),
            "active_dispatches": len(active_dispatches),
            "submitted_gpus": sum(row["gpus"] for row in active_dispatches),
            "wandb_running_dispatches": training_dispatches,
            "wandb_running_gpus": training_gpus,
        },
        "trials": trial_snapshots,
        "conditions": sorted(
            conditions, key=lambda row: (row["trial_id"], row["kind"])
        ),
        "placements": placements,
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


def _optional_bool(value: str) -> bool | None:
    normalized = value.casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    if normalized in {"unknown", "none", "null"}:
        return None
    raise argparse.ArgumentTypeError("expected true, false, or unknown")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("database", type=Path)

    trial_parser = subparsers.add_parser("trial-add")
    trial_parser.add_argument("database", type=Path)
    trial_parser.add_argument("--trial-id", required=True)
    trial_parser.add_argument("--env-json", required=True)
    trial_parser.add_argument("--wandb-run-id", required=True)
    trial_parser.add_argument("--wandb-url")
    trial_parser.add_argument("--checkpoint-root", required=True)

    intent_parser = subparsers.add_parser("dispatch-intent")
    intent_parser.add_argument("database", type=Path)
    intent_parser.add_argument("--trial-id", required=True)
    intent_parser.add_argument("--dispatch-id", required=True)
    intent_parser.add_argument("--iris-job-id", required=True)
    intent_parser.add_argument("--cluster", required=True)
    intent_parser.add_argument("--gpu-variant", required=True)
    intent_parser.add_argument("--nodes", type=int, required=True)
    intent_parser.add_argument("--gpus", type=int, required=True)
    intent_parser.add_argument("--command-redacted", required=True)
    intent_parser.add_argument("--intent-at")

    confirm_parser = subparsers.add_parser("dispatch-confirm")
    confirm_parser.add_argument("database", type=Path)
    confirm_parser.add_argument("--dispatch-id", required=True)
    confirm_parser.add_argument("--submitted-at")

    end_parser = subparsers.add_parser("dispatch-end")
    end_parser.add_argument("database", type=Path)
    end_parser.add_argument("--dispatch-id", required=True)
    end_parser.add_argument("--outcome", required=True)
    end_parser.add_argument("--stop-reason")
    end_parser.add_argument("--ended-at")

    observation_parser = subparsers.add_parser("observe")
    observation_parser.add_argument("database", type=Path)
    observation_parser.add_argument("--dispatch-id", required=True)
    observation_parser.add_argument("--observed-at", required=True)
    observation_parser.add_argument("--wandb-state")
    observation_parser.add_argument("--run-progress", type=float)
    observation_parser.add_argument("--iris-running", type=_optional_bool)

    complete_parser = subparsers.add_parser("trial-complete")
    complete_parser.add_argument("database", type=Path)
    complete_parser.add_argument("--trial-id", required=True)
    complete_parser.add_argument("--checkpoint-verified-at")

    backup_parser = subparsers.add_parser("backup")
    backup_parser.add_argument("database", type=Path)
    backup_parser.add_argument("destination", type=Path)

    event_parser = subparsers.add_parser("event")
    event_parser.add_argument("database", type=Path)
    event_parser.add_argument("--kind", required=True)
    event_parser.add_argument("--detail", required=True)
    event_parser.add_argument("--trial-id")
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
    if args.command == "trial-add":
        env = json.loads(args.env_json)
        if not isinstance(env, dict):
            raise RuntimeError("--env-json must decode to an object")
        add_trial(
            args.database,
            args.trial_id,
            env,
            args.wandb_run_id,
            args.checkpoint_root,
            args.wandb_url,
        )
        print("OK: trial added")
        return 0
    if args.command == "dispatch-intent":
        attempt = prepare_dispatch(
            args.database,
            args.trial_id,
            args.dispatch_id,
            args.iris_job_id,
            args.cluster,
            args.gpu_variant,
            args.nodes,
            args.gpus,
            args.command_redacted,
            args.intent_at,
        )
        print(json.dumps({"attempt": attempt, "dispatch_id": args.dispatch_id}))
        return 0
    if args.command == "dispatch-confirm":
        submitted_at = confirm_dispatch(
            args.database, args.dispatch_id, args.submitted_at
        )
        print(
            json.dumps({"dispatch_id": args.dispatch_id, "submitted_at": submitted_at})
        )
        return 0
    if args.command == "dispatch-end":
        ended_at = end_dispatch(
            args.database,
            args.dispatch_id,
            args.outcome,
            args.stop_reason,
            args.ended_at,
        )
        print(json.dumps({"dispatch_id": args.dispatch_id, "ended_at": ended_at}))
        return 0
    if args.command == "observe":
        record_observation(
            args.database,
            args.dispatch_id,
            args.observed_at,
            args.wandb_state,
            args.run_progress,
            args.iris_running,
        )
        print("OK: observation recorded")
        return 0
    if args.command == "trial-complete":
        verified_at = complete_trial(
            args.database, args.trial_id, args.checkpoint_verified_at
        )
        print(
            json.dumps(
                {"trial_id": args.trial_id, "checkpoint_verified_at": verified_at}
            )
        )
        return 0
    if args.command == "backup":
        print(json.dumps(backup(args.database, args.destination), sort_keys=True))
        return 0
    if args.command == "event":
        record_event(
            args.database,
            args.kind,
            args.detail,
            args.trial_id,
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
