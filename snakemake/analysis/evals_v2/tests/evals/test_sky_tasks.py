"""Contracts for evals_v2 Sky task bootstraps."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parents[2]


def test_locked_sky_tasks_pin_project_uv_version() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    required_version = project["tool"]["uv"]["required-version"]
    assert required_version.startswith("==")
    uv_version = required_version.removeprefix("==")

    checked: list[str] = []
    for task_path in sorted((PROJECT_ROOT / "sky").glob("*.yaml")):
        with task_path.open(encoding="utf-8") as handle:
            task = yaml.safe_load(handle)
        setup = task.get("setup", "")
        if "uv sync" not in setup or "--locked" not in setup:
            continue

        checked.append(task_path.name)
        pin = f"uv self update {uv_version}"
        guarded_pin = (
            f'''if [[ "$(uv --version | awk '{{print $2}}')" != "{uv_version}" ]]; then\n'''
            f"    {pin}\n"
            "fi"
        )
        assert guarded_pin in setup, f"{task_path.name} does not guard {pin!r}"
        assert setup.index(pin) < setup.index("uv sync"), (
            f"{task_path.name} pins uv after the locked sync"
        )

    assert checked == ["analysis_489.yaml", "probe.yaml", "run.yaml"]
