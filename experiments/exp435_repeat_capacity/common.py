"""Pinned constants and small helpers for experiment 435."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ISSUE = 435
SEED = 288
RUN_ID = "dna-exp435-repeat-inventory-r1"
RMSK_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz"
PRIMARY_CHROMS = tuple([str(index) for index in range(1, 23)] + ["X", "Y"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def assert_commit(value: str) -> None:
    assert len(value) == 40
    assert all(character in "0123456789abcdef" for character in value)
