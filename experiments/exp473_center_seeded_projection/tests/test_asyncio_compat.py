from __future__ import annotations

import asyncio
import weakref

from exp473_center_seeded_projection.asyncio_compat import (
    _PATCH_MARKER,
    _all_tasks_from_atomic_snapshot,
    _snapshot_weakset,
    install_all_tasks_snapshot,
)


class _FailingIteratorWeakSet(weakref.WeakSet):
    def __iter__(self):
        raise RuntimeError("Set changed size during iteration")


class _WeakReferenceTarget:
    pass


def test_snapshot_does_not_iterate_mutable_weakset() -> None:
    target = _WeakReferenceTarget()
    registry = _FailingIteratorWeakSet([target])
    assert _snapshot_weakset(registry) == [target]


def test_python_312_guard_is_installed_and_idempotent() -> None:
    assert install_all_tasks_snapshot() is True
    installed = asyncio.tasks.all_tasks
    assert installed is asyncio.all_tasks
    assert getattr(installed, _PATCH_MARKER) is True
    assert install_all_tasks_snapshot() is True
    assert asyncio.tasks.all_tasks is installed


def test_atomic_all_tasks_snapshot_contains_current_task() -> None:
    async def current_task_is_visible() -> None:
        current = asyncio.current_task()
        assert current is not None
        assert current in _all_tasks_from_atomic_snapshot()

    asyncio.run(current_task_is_visible())
