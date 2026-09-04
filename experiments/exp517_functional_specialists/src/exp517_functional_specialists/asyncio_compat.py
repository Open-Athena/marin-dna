"""Avoid CPython 3.12's cross-thread ``asyncio.all_tasks`` weak-set race."""

from __future__ import annotations

import asyncio
import itertools
import sys
import weakref
from typing import Any

_PATCH_MARKER = "_exp517_atomic_weakset_snapshot"


def _snapshot_weakset(
    registry: weakref.WeakSet[asyncio.Task[Any]],
) -> list[asyncio.Task[Any]]:
    """Resolve an atomic copy of a CPython weak-set registry."""
    references = registry.data.copy()
    tasks: list[asyncio.Task[Any]] = []
    for reference in references:
        task = reference()
        if task is not None:
            tasks.append(task)
    return tasks


def _all_tasks_from_atomic_snapshot(
    loop: asyncio.AbstractEventLoop | None = None,
) -> set[asyncio.Task[Any]]:
    """Implement CPython 3.12 ``all_tasks`` without weak-set iteration."""
    if loop is None:
        loop = asyncio.get_running_loop()
    eager_tasks = list(asyncio.tasks._eager_tasks.copy())
    scheduled_tasks = _snapshot_weakset(asyncio.tasks._scheduled_tasks)
    return {
        task
        for task in itertools.chain(scheduled_tasks, eager_tasks)
        if asyncio.futures._get_loop(task) is loop and not task.done()
    }


setattr(_all_tasks_from_atomic_snapshot, _PATCH_MARKER, True)


def install_all_tasks_snapshot() -> bool:
    """Install the scoped CPython 3.12 workaround once."""
    if sys.version_info[:2] != (3, 12):
        return False
    current = asyncio.tasks.all_tasks
    if getattr(current, _PATCH_MARKER, False):
        return True
    if not isinstance(asyncio.tasks._scheduled_tasks, weakref.WeakSet):
        return False
    asyncio.tasks.all_tasks = _all_tasks_from_atomic_snapshot
    asyncio.all_tasks = _all_tasks_from_atomic_snapshot
    return True
