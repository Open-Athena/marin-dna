"""Frozen constants and deterministic helpers for the issue-435 reference panel."""

from __future__ import annotations

import hashlib

PANEL_RUN_ID = "dna-exp435-repeat-reference-panel-r1"
WINDOW_BP = 255
FOCAL_INDEX = 127
UNIFORM_REPEAT_CONTEXTS = 32_768
UNIFORM_CONTROL_CONTEXTS = 32_768
CATEGORY_CONTEXTS = 128
MIN_CATEGORY_RECORDS = 256
MIN_CATEGORY_RAW_BP = 100_000
GLOBAL_SUBFAMILIES = 64
SUBFAMILIES_PER_FAMILY = 2
EXPECTED_CLASS_COUNT = 18
EXPECTED_FAMILY_COUNT = 49
EXPECTED_SUBFAMILY_COUNT = 125

assert WINDOW_BP == 2 * FOCAL_INDEX + 1
assert UNIFORM_REPEAT_CONTEXTS == UNIFORM_CONTROL_CONTEXTS


def stable_seed(namespace: str) -> int:
    digest = hashlib.sha256(namespace.encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def selection_hash(namespace: str, chrom: str, pos0: int) -> str:
    return hashlib.sha256(f"{namespace}|{chrom}|{pos0}".encode()).hexdigest()
