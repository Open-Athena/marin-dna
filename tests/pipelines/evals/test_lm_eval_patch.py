# Copyright The MarinDNA Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lm_eval.tasks.TaskManager monkeypatch.

The patch lives in ``marin_dna.pipelines.evals.lm_eval.__init__`` and runs as a side
effect of importing the package — see the module docstring there for why.
"""

import importlib.resources

import pytest

pytest.importorskip(
    "lm_eval",
    reason="requires the marin runtime (not a core dep); see the marin-experiment skill",
)


@pytest.fixture
def patched_task_manager():
    """Re-import the package fresh in case sentinels need re-firing."""
    from lm_eval.tasks import TaskManager

    import marin_dna.pipelines.evals.lm_eval  # noqa: F401  triggers _install_task_manager_patch

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
    from levanter.eval_harness import LmEvalHarnessConfig
    from levanter.eval_harness import TaskConfig as LevanterTaskConfig

    import marin_dna.pipelines.evals.lm_eval  # noqa: F401  triggers both patches

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
    from levanter.eval_harness import LmEvalHarnessConfig

    import marin_dna.pipelines.evals.lm_eval  # noqa: F401

    assert getattr(LmEvalHarnessConfig, "_marin_dna_rename_patched", False) is True


# --- [BOS] fix (#257 root cause; stopgap #263; durable fix #264) -------------


@pytest.mark.parametrize(
    "ids, bos, expected",
    [
        ([[10, 11], [12]], 2, [[2, 10, 11], [2, 12]]),  # prepend
        ([[2, 10], [2, 12]], 2, [[2, 10], [2, 12]]),  # idempotent (already BOS)
        ([[2, 10], [12]], 2, [[2, 10], [2, 12]]),  # mixed: only the BOS-less one
        ([[10], [11]], None, [[10], [11]]),  # no-BOS tokenizer → no-op
        ([[]], 2, [[2]]),  # empty completion still gets BOS
    ],
)
def test_prepend_bos(ids, bos, expected):
    from marin_dna.pipelines.evals.lm_eval import _prepend_bos

    assert _prepend_bos(ids, bos) == expected


def test_bos_patch_marker_set():
    from levanter import eval_harness

    import marin_dna.pipelines.evals.lm_eval  # noqa: F401

    assert getattr(eval_harness, "_marin_dna_bos_patched", False) is True


def test_bos_patch_is_idempotent():
    """Re-running the installer must not double-wrap ``_encode_batch``."""
    from levanter import eval_harness

    import marin_dna.pipelines.evals.lm_eval as pkg

    before = eval_harness._encode_batch
    pkg._install_bos_fix()
    pkg._install_bos_fix()
    assert eval_harness._encode_batch is before


def test_bos_patch_prepends_through_encode_batch():
    """The patched ``_encode_batch`` prepends the tokenizer's BOS id; a tokenizer
    without a BOS is left untouched (datasets stay tokenizer-agnostic)."""
    from levanter import eval_harness

    import marin_dna.pipelines.evals.lm_eval  # noqa: F401  triggers the patch

    class _FakeTok:
        def __init__(self, bos):
            self.bos_token_id = bos

        def encode_batch(self, texts):  # MarinTokenizer-style: list[list[int]]
            return [[10, 11, 12] for _ in texts]

    assert eval_harness._encode_batch(_FakeTok(bos=2), ["A", "C"]) == [
        [2, 10, 11, 12],
        [2, 10, 11, 12],
    ]
    assert eval_harness._encode_batch(_FakeTok(bos=None), ["A"]) == [[10, 11, 12]]


def test_levanter_rename_patch_is_idempotent():
    from levanter.eval_harness import LmEvalHarnessConfig

    import marin_dna.pipelines.evals.lm_eval  # noqa: F401
    from marin_dna.pipelines.evals.lm_eval import _install_levanter_rename_patch

    rename_before = LmEvalHarnessConfig._rename_tasks_for_eval_harness
    _install_levanter_rename_patch()
    _install_levanter_rename_patch()
    assert LmEvalHarnessConfig._rename_tasks_for_eval_harness is rename_before
