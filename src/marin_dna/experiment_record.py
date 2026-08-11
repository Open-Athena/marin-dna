# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment-level provenance records — one ``.experiment.json`` per ``dna-exp<N>``.

Marin's executor writes a per-step ``.artifact.json`` (config + fingerprint + launch
provenance) at every step output path, but no on-disk record ties those outputs to the
*experiment*: the ``dna-exp<N>`` identity, its tracking issue, the arms it swept, the
pinned inputs each arm consumed, and the wandb runs that trained it. Those live only in
wandb and the issue thread. This module owns that record: ``launch.py`` builds an
:class:`ExperimentRecord` and writes it once the runs succeed, so an experiment is
catalogable and reproducible from storage alone (same motivation as marin#7967, which
did this for marin's ablation sweep cells).

The record is ``.artifact.json``-style (name / config / deps / provenance) but uses a
distinct filename so marin's ``read_record`` never mistakes it for a step record, and
adds the experiment-grain fields marin has no notion of (tracking issue, arms, wandb
runs).

Storage-agnostic via fsspec: local paths and any ``<scheme>://`` URL work, provided the
scheme's fsspec backend is importable where ``launch.py`` runs (the marin experiment
env ships ``gcsfs``, so ``gs://`` works on the iris coordinator).
"""

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

import fsspec

RECORD_FILENAME = ".experiment.json"
SCHEMA_VERSION = 1

# Experiment numbers equal their tracking-issue numbers by convention (see AGENTS.md:
# "<N> = the experiment number from the issue"), so both are validated together.
_EXPERIMENT_NAME = re.compile(r"dna-exp([1-9][0-9]*)")
_ISSUE_URL = re.compile(r"https://github\.com/[^/]+/[^/]+/issues/([1-9][0-9]*)")
_FULL_SHA = re.compile(r"[0-9a-f]{40}")

DepKind = Literal["hf-dataset", "hf-model", "git"]
_DEP_KINDS: tuple[str, ...] = ("hf-dataset", "hf-model", "git")

type JSONValue = (
    None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
)


@dataclass(frozen=True)
class ExperimentDep:
    """One pinned input: a HF dataset/model or a git repo, at an immutable revision.

    ``revision`` must be a full 40-hex sha — branch names, tags, and abbreviated shas
    are not pins.
    """

    kind: DepKind
    id: str
    revision: str

    def __post_init__(self) -> None:
        assert self.kind in _DEP_KINDS, (
            f"unknown dep kind {self.kind!r}; expected one of {_DEP_KINDS}"
        )
        assert self.id, "dep id must be non-empty"
        assert _FULL_SHA.fullmatch(self.revision), (
            f"dep {self.id!r} revision {self.revision!r} is not a full 40-hex sha"
        )


@dataclass(frozen=True)
class ExperimentRun:
    """One training run (arm) of the experiment.

    ``run_id`` is the wandb run name and must carry the ``dna-exp<N>`` tag (validated
    at the record level); ``output_path`` is the checkpoint root the run wrote.
    """

    run_id: str
    output_path: str
    arm: str | None = None
    wandb_url: str | None = None

    def __post_init__(self) -> None:
        assert self.run_id, "run_id must be non-empty"
        assert self.output_path, f"run {self.run_id!r} output_path must be non-empty"


@dataclass(frozen=True)
class ExperimentRecord:
    """The durable description of one ``dna-exp<N>``: identity, pins, runs, provenance.

    ``config`` holds the experiment-level knobs shared across arms (model geometry,
    optimizer, batch/step horizon — whatever the launch pins as constants); per-arm
    variation belongs in ``runs`` and per-arm deps. ``provenance`` is free-form string
    metadata about the launch itself (experiment branch commit, marin pin, launcher).
    """

    name: str
    issue: str
    created_at: str
    config: dict[str, JSONValue]
    deps: list[ExperimentDep]
    runs: list[ExperimentRun]
    provenance: dict[str, str]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        assert self.schema_version == SCHEMA_VERSION, (
            f"unsupported schema_version {self.schema_version!r}"
        )
        name_match = _EXPERIMENT_NAME.fullmatch(self.name)
        assert name_match, f"name {self.name!r} does not match 'dna-exp<N>'"
        issue_match = _ISSUE_URL.fullmatch(self.issue)
        assert issue_match, f"issue {self.issue!r} is not a GitHub issue URL"
        assert issue_match.group(1) == name_match.group(1), (
            f"{self.name} must track issue #{name_match.group(1)}, got {self.issue!r}"
        )
        created = datetime.fromisoformat(self.created_at)
        assert created.tzinfo is not None, (
            f"created_at {self.created_at!r} must be timezone-aware"
        )
        assert self.runs, "an experiment record must list at least one run"
        run_ids = [run.run_id for run in self.runs]
        assert len(set(run_ids)) == len(run_ids), f"duplicate run_ids: {run_ids}"
        output_paths = [run.output_path for run in self.runs]
        assert len(set(output_paths)) == len(output_paths), (
            f"duplicate output_paths: {output_paths}"
        )
        for run in self.runs:
            # (?![0-9]) so 'dna-exp41' never matches inside 'dna-exp417-...'.
            assert re.search(rf"{re.escape(self.name)}(?![0-9])", run.run_id), (
                f"run_id {run.run_id!r} does not carry {self.name!r} — wandb runs must"
                " filter by experiment (see AGENTS.md: wandb run names)"
            )
        for key, value in self.provenance.items():
            assert isinstance(value, str), (
                f"provenance[{key!r}] must be a string, got {type(value).__name__}"
            )
        # Fail at construction, near the bug, not at write time on the coordinator.
        json.dumps(self.config)

    @property
    def experiment_number(self) -> int:
        match = _EXPERIMENT_NAME.fullmatch(self.name)
        assert match is not None
        return int(match.group(1))

    def to_json_dict(self) -> dict[str, JSONValue]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict[str, JSONValue]) -> "ExperimentRecord":
        """Rebuild a record from its JSON form, re-running every validation."""
        data = dict(payload)
        deps_raw = data.pop("deps")
        runs_raw = data.pop("runs")
        assert isinstance(deps_raw, list) and isinstance(runs_raw, list), (
            "malformed record: 'deps' and 'runs' must be lists"
        )
        deps: list[ExperimentDep] = []
        for dep in deps_raw:
            assert isinstance(dep, dict), "malformed record: each dep must be an object"
            deps.append(ExperimentDep(**cast(dict[str, Any], dep)))
        runs: list[ExperimentRun] = []
        for run in runs_raw:
            assert isinstance(run, dict), "malformed record: each run must be an object"
            runs.append(ExperimentRun(**cast(dict[str, Any], run)))
        return cls(deps=deps, runs=runs, **cast(dict[str, Any], data))


def utc_now_iso() -> str:
    """Timezone-aware ISO-8601 timestamp for :attr:`ExperimentRecord.created_at`."""
    return datetime.now(UTC).isoformat()


def _record_url(prefix: str) -> str:
    assert prefix, "record prefix must be non-empty"
    return prefix.rstrip("/") + "/" + RECORD_FILENAME


def write_experiment_record(record: ExperimentRecord, prefix: str) -> str:
    """Write ``record`` to ``<prefix>/.experiment.json`` and return that URL.

    ``prefix`` is the experiment's record directory — a local path or any fsspec URL
    (``gs://``, ``s3://``, ``memory://``). Parent directories are created if the
    filesystem has them.
    """
    url = _record_url(prefix)
    fs, path = fsspec.core.url_to_fs(url)
    fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(path, "w") as f:
        json.dump(record.to_json_dict(), f, indent=2)
        f.write("\n")
    return url


def read_experiment_record(prefix: str) -> ExperimentRecord:
    """Read back the record at ``<prefix>/.experiment.json``, re-validating it.

    Raises ``FileNotFoundError`` if absent: a missing record where one is expected is
    a failure worth surfacing, not a probe result.
    """
    url = _record_url(prefix)
    with fsspec.open(url, "r") as f:
        payload = json.load(f)
    assert isinstance(payload, dict), f"malformed record at {url}: not a JSON object"
    return ExperimentRecord.from_json_dict(payload)
