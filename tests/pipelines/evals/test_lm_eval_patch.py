# Copyright The MarinDNA Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lm_eval.tasks.TaskManager monkeypatch.

The patch lives in ``marin_dna.pipelines.evals.lm_eval.__init__`` and runs as a side
effect of importing the package — see the module docstring there for why.
"""

import importlib.resources

import pytest

pytest.importorskip("lm_eval", reason="install with `uv sync --extra marin` to run")


@pytest.fixture
def patched_task_manager():
    """Re-import the package fresh in case sentinels need re-firing."""
    import marin_dna.pipelines.evals.lm_eval  # noqa: F401  triggers _install_task_manager_patch
    from lm_eval.tasks import TaskManager

    return TaskManager


def test_patch_marker_set(patched_task_manager):
    assert getattr(patched_task_manager, "_marin_dna_patched", False) is True


def test_patch_is_idempotent(patched_task_manager):
    """Re-running the installer must not double-wrap __init__."""
    from marin_dna.pipelines.evals.lm_eval import _install_task_manager_patch

    init_before = patched_task_manager.__init__
    _install_task_manager_patch()
    _install_task_manager_patch()
    assert patched_task_manager.__init__ is init_before


def test_custom_tasks_discoverable(patched_task_manager):
    """TaskManager() with no args should still see marin-dna's task YAMLs."""
    mgr = patched_task_manager()
    assert "mendelian_traits_255" in mgr.all_tasks


def test_caller_include_path_preserved_as_string(tmp_path, patched_task_manager):
    """A caller-supplied string ``include_path`` should be merged, not clobbered.

    Use a real empty directory so the underlying ``TaskManager`` doesn't choke
    on it, then read back the stored ``self.include_path`` to see what the
    patched ``__init__`` forwarded.
    """
    marin_dna_path = str(importlib.resources.files("marin_dna.pipelines.evals.lm_eval"))
    caller_path = str(tmp_path)
    mgr = patched_task_manager(include_path=caller_path)
    assert mgr.include_path == [caller_path, marin_dna_path]


def test_caller_include_path_preserved_as_list(tmp_path, patched_task_manager):
    marin_dna_path = str(importlib.resources.files("marin_dna.pipelines.evals.lm_eval"))
    p1 = str(tmp_path / "a")
    p2 = str(tmp_path / "b")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    mgr = patched_task_manager(include_path=[p1, p2])
    assert mgr.include_path == [p1, p2, marin_dna_path]


def test_levanter_rename_patch_handles_plain_task():
    """`LmEvalHarnessConfig.to_task_dict()` must work for our plain-Task
    subclass — upstream `_rename_tasks_for_eval_harness` only handles
    `ConfigurableTask` + dict and otherwise raises ``ValueError: Unknown task type``.
    Our package's `_install_levanter_rename_patch()` adds a Task-passthrough.
    """
    import marin_dna.pipelines.evals.lm_eval  # noqa: F401  triggers both patches
    from levanter.eval_harness import LmEvalHarnessConfig
    from levanter.eval_harness import TaskConfig as LevanterTaskConfig

    spec = [
        LevanterTaskConfig(
            task="mendelian_traits_255",
            num_fewshot=0,
            task_alias="mendelian_traits_255",
        )
    ]
    cfg = LmEvalHarnessConfig(task_spec=spec)
    task_dict = cfg.to_task_dict()
    assert "mendelian_traits_255" in task_dict
    task = task_dict["mendelian_traits_255"]
    assert type(task).__name__ == "DnaVepLlrEvalTask"


def test_levanter_rename_patch_marker_set():
    import marin_dna.pipelines.evals.lm_eval  # noqa: F401
    from levanter.eval_harness import LmEvalHarnessConfig

    assert getattr(LmEvalHarnessConfig, "_marin_dna_rename_patched", False) is True


def test_levanter_rename_patch_is_idempotent():
    import marin_dna.pipelines.evals.lm_eval  # noqa: F401
    from marin_dna.pipelines.evals.lm_eval import _install_levanter_rename_patch
    from levanter.eval_harness import LmEvalHarnessConfig

    rename_before = LmEvalHarnessConfig._rename_tasks_for_eval_harness
    _install_levanter_rename_patch()
    _install_levanter_rename_patch()
    assert LmEvalHarnessConfig._rename_tasks_for_eval_harness is rename_before
