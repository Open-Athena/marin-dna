"""Reproduce failed-writer closure against the verified, pinned Zephyr source.

Only the writer class and its queue iterator are loaded; no cloud access or
Marin/JAX imports are performed. The blocked closer is a daemon, so this
reproducer exits without leaving a process behind.
"""

import argparse
import ast
import logging
import queue
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path, help="writers.py extracted from the pinned Zephyr wheel")
SOURCE = parser.parse_args().source
tree = ast.parse(SOURCE.read_text())
nodes = [
    node
    for node in tree.body
    if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    and node.name in {"ThreadedBatchWriter", "_queue_iterable"}
]
namespace = {
    "queue": queue,
    "threading": threading,
    "Callable": Callable,
    "Iterable": Iterable,
    "Any": Any,
    "logger": logging.getLogger("review"),
    "_SENTINEL": object(),
}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
gate = threading.Event()


def fail_after_queue_fills(_records: Iterable) -> None:
    if not gate.wait(1):
        raise RuntimeError("reproduction setup timeout")
    raise RuntimeError("simulated write failure")


writer = namespace["ThreadedBatchWriter"](fail_after_queue_fills, maxsize=1)
writer.submit([1])
gate.set()
writer._thread.join(timeout=1)
assert not writer._thread.is_alive()
assert str(writer._error) == "simulated write failure"
closer = threading.Thread(target=writer.close, daemon=True)
closer.start()
closer.join(timeout=0.1)
assert closer.is_alive(), "Expected pinned close() to block on its full queue"
print(
    {
        "writer_failed": True,
        "queued_batches": writer._queue.qsize(),
        "close_blocked_in_sentinel_put": closer.is_alive(),
    }
)
