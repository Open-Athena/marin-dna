"""Tests for ``marin_dna_evals.models`` — loader + validator for
``dashboard/models.yaml``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from marin_dna_evals.models import (
    ALL_DATASETS,
    ALL_FAMILIES,
    MODELS_YAML,
    Model,
    load_models,
    models_for_dataset,
)


def test_methods_yaml_exists_at_dashboard_path():
    assert MODELS_YAML.exists(), (
        f"models.yaml missing at {MODELS_YAML}; the loader anchors to the "
        f"repo root and expects dashboard/models.yaml there"
    )
    assert MODELS_YAML.parent.name == "dashboard"


def test_load_methods_returns_dataclasses():
    methods = load_models()
    assert len(methods) > 0
    for m in methods:
        assert isinstance(m, Model)
        assert m.family in ALL_FAMILIES
        for d in m.datasets:
            assert d in ALL_DATASETS


def test_load_methods_ids_unique():
    methods = load_models()
    ids = [m.id for m in methods]
    assert len(ids) == len(set(ids)), (
        f"duplicate id(s) in models.yaml: {[i for i in ids if ids.count(i) > 1]}"
    )


def test_models_for_dataset_filters():
    mendelian = models_for_dataset("mendelian_traits")
    complex_ = models_for_dataset("complex_traits")
    for m in mendelian:
        assert "mendelian_traits" in m.datasets
    for m in complex_:
        assert "complex_traits" in m.datasets


def test_models_for_dataset_unknown_raises():
    with pytest.raises(AssertionError):
        models_for_dataset("not_a_dataset")


def test_every_family_has_at_least_one_method():
    methods = load_models()
    families_present = {m.family for m in methods}
    assert families_present == set(ALL_FAMILIES), (
        f"missing families: {set(ALL_FAMILIES) - families_present}; "
        f"unexpected: {families_present - set(ALL_FAMILIES)}"
    )


def test_bolinas_methods_have_checkpoint(tmp_path: Path):
    methods = load_models()
    for m in methods:
        if m.family == "marin_dna":
            assert m.checkpoint is not None, m.id
            assert m.checkpoint.gcs or m.checkpoint.hf, m.id


def test_invalid_family_rejected(tmp_path: Path):
    bad = tmp_path / "models.yaml"
    bad.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "x",
                    "display": "x",
                    "family": "not_a_family",
                    "description": "test",
                    "datasets": ["mendelian_traits"],
                }
            ]
        )
    )
    with pytest.raises(AssertionError, match="unknown family"):
        load_models.__wrapped__(bad)  # type: ignore[attr-defined]


def test_invalid_dataset_rejected(tmp_path: Path):
    bad = tmp_path / "models.yaml"
    bad.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "x",
                    "display": "x",
                    "family": "conservation",
                    "description": "test",
                    "datasets": ["bogus_dataset"],
                }
            ]
        )
    )
    with pytest.raises(AssertionError, match="unknown dataset"):
        load_models.__wrapped__(bad)  # type: ignore[attr-defined]


def test_duplicate_id_rejected(tmp_path: Path):
    bad = tmp_path / "models.yaml"
    bad.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "dup",
                    "display": "a",
                    "family": "conservation",
                    "description": "test",
                    "datasets": ["mendelian_traits"],
                },
                {
                    "id": "dup",
                    "display": "b",
                    "family": "conservation",
                    "description": "test",
                    "datasets": ["mendelian_traits"],
                },
            ]
        )
    )
    with pytest.raises(AssertionError, match="duplicate method id"):
        load_models.__wrapped__(bad)  # type: ignore[attr-defined]


def test_bolinas_requires_checkpoint(tmp_path: Path):
    bad = tmp_path / "models.yaml"
    bad.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "exp999",
                    "display": "exp999",
                    "family": "marin_dna",
                    "description": "test",
                    "datasets": ["mendelian_traits"],
                }
            ]
        )
    )
    with pytest.raises(AssertionError, match="requires `checkpoint:` block"):
        load_models.__wrapped__(bad)  # type: ignore[attr-defined]
