"""Custom lm-eval-harness tasks for DNA experiments.

Importing this module installs two idempotent monkeypatches:

1. ``lm_eval.tasks.TaskManager.__init__`` — appends this package's directory
   to ``include_path`` so the YAML task definitions here are discoverable.
   Marin's ``LmEvalHarnessConfig`` no longer exposes ``include_path``, so
   without the patch our tasks are unreachable from a marin-launched run.

2. ``levanter.eval_harness.LmEvalHarnessConfig._rename_tasks_for_eval_harness``
   — passes plain ``lm_eval.api.task.Task`` subclasses through untouched.
   Upstream only handles ``dict`` and ``ConfigurableTask`` and raises
   ``ValueError`` otherwise. ``DnaVepLlrEvalTask`` extends ``Task`` directly
   because ``ConfigurableTask.__init__`` eagerly walks fewshot docs and hangs
   on large datasets.
"""

import logging
import pathlib
from functools import wraps

_logger = logging.getLogger(__name__)


def _install_task_manager_patch() -> None:
    try:
        from lm_eval.tasks import TaskManager
    except ImportError:
        # lm-eval not installed (default `uv sync`).
        return

    if getattr(TaskManager, "_marin_dna_patched", False):
        return

    # `__file__.parent` rather than `importlib.resources.files()` — the latter
    # can return a MultiplexedPath under editable installs / namespace
    # packages whose `str()` doesn't resolve to a filesystem path.
    marin_dna_path = str(pathlib.Path(__file__).parent)
    yaml_files = sorted(pathlib.Path(marin_dna_path).glob("*.yaml"))
    _logger.info(
        "marin_dna.pipelines.evals.lm_eval: patching TaskManager.__init__ to include "
        "%s (found %d task YAMLs: %s)",
        marin_dna_path,
        len(yaml_files),
        [f.name for f in yaml_files],
    )
    original_init = TaskManager.__init__

    @wraps(original_init)
    def patched_init(self, *args, include_path=None, **kwargs):
        if include_path is None:
            merged = [marin_dna_path]
        elif isinstance(include_path, str):
            merged = [include_path, marin_dna_path]
        else:
            merged = list(include_path) + [marin_dna_path]
        return original_init(self, *args, include_path=merged, **kwargs)

    TaskManager.__init__ = patched_init
    TaskManager._marin_dna_patched = True


def _install_levanter_rename_patch() -> None:
    try:
        from levanter.eval_harness import LmEvalHarnessConfig
        from lm_eval.api.task import Task
    except ImportError:
        return

    if getattr(LmEvalHarnessConfig, "_marin_dna_rename_patched", False):
        return

    original_rename = LmEvalHarnessConfig._rename_tasks_for_eval_harness

    @wraps(original_rename)
    def patched_rename(self, this_task, lm_eval_task_name, our_name):
        from lm_eval.api.task import ConfigurableTask

        if (
            not isinstance(this_task, dict)
            and isinstance(this_task, Task)
            and not isinstance(this_task, ConfigurableTask)
        ):
            return this_task
        return original_rename(self, this_task, lm_eval_task_name, our_name)

    LmEvalHarnessConfig._rename_tasks_for_eval_harness = patched_rename
    LmEvalHarnessConfig._marin_dna_rename_patched = True


_install_task_manager_patch()
_install_levanter_rename_patch()
