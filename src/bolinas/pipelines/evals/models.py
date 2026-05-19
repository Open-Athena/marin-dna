"""Load and validate ``dashboard/models.yaml``.

The YAML file is the single source of truth for every method that appears on
any leaderboard — its display name, family (which drives parquet path +
filtering in ``leaderboard.py``), description, training metadata, links to
wandb / checkpoints / issues, and the set of datasets it was evaluated on.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import yaml

Family = Literal["bolinas", "conservation", "alphagenome", "gpn_star"]
ALL_FAMILIES: tuple[Family, ...] = (
    "bolinas",
    "conservation",
    "alphagenome",
    "gpn_star",
)
ALL_DATASETS: tuple[str, ...] = ("mendelian_traits", "complex_traits")


@dataclass(frozen=True)
class TrainingMeta:
    data: str | None = None
    params: float | None = None
    window_size: int | None = None
    objective: str | None = None


@dataclass(frozen=True)
class CheckpointMeta:
    gcs: str | None = None
    hf: str | None = None


@dataclass(frozen=True)
class Model:
    id: str
    display: str
    family: Family
    description: str
    datasets: tuple[str, ...]
    training: TrainingMeta | None = None
    checkpoint: CheckpointMeta | None = None
    experiment: int | None = None
    issue: str | None = None
    source_code: str | None = None
    wandb: str | None = None
    paper: str | None = None
    msa: str | None = None


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"could not find pyproject.toml above {start}")


MODELS_YAML: Path = (
    _find_repo_root(Path(__file__).resolve()) / "dashboard" / "models.yaml"
)


def _coerce(raw: dict) -> Model:
    family = raw["family"]
    assert family in ALL_FAMILIES, (
        f"unknown family {family!r} for id={raw.get('id')!r}; expected one of {ALL_FAMILIES}"
    )
    datasets = tuple(raw["datasets"])
    for d in datasets:
        assert d in ALL_DATASETS, (
            f"unknown dataset {d!r} on id={raw['id']!r}; expected subset of {ALL_DATASETS}"
        )
    training = TrainingMeta(**raw["training"]) if "training" in raw else None
    checkpoint = CheckpointMeta(**raw["checkpoint"]) if "checkpoint" in raw else None
    if family == "bolinas":
        assert checkpoint is not None, (
            f"family=bolinas requires `checkpoint:` block (id={raw['id']!r})"
        )
        assert checkpoint.gcs or checkpoint.hf, (
            f"family=bolinas requires checkpoint.gcs or checkpoint.hf (id={raw['id']!r})"
        )
    return Model(
        id=raw["id"],
        display=raw["display"],
        family=family,
        description=raw["description"],
        datasets=datasets,
        training=training,
        checkpoint=checkpoint,
        experiment=raw.get("experiment"),
        issue=raw.get("issue"),
        source_code=raw.get("source_code"),
        wandb=raw.get("wandb"),
        paper=raw.get("paper"),
        msa=raw.get("msa"),
    )


@cache
def load_models(path: Path = MODELS_YAML) -> tuple[Model, ...]:
    """Load and validate ``dashboard/models.yaml``. Cached per (process, path)."""
    raw_list = yaml.safe_load(path.read_text())
    assert isinstance(raw_list, list), (
        f"models.yaml must be a top-level list, got {type(raw_list).__name__}"
    )
    methods = tuple(_coerce(r) for r in raw_list)
    seen: set[str] = set()
    for m in methods:
        assert m.id not in seen, f"duplicate method id {m.id!r}"
        seen.add(m.id)
    return methods


def models_for_dataset(dataset: str) -> tuple[Model, ...]:
    """Methods registered for ``dataset``, in registry order."""
    assert dataset in ALL_DATASETS, f"unknown dataset {dataset!r}"
    return tuple(m for m in load_models() if dataset in m.datasets)
