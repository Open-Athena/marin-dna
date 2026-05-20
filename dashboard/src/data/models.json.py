"""Observable Framework data loader: ``dashboard/models.yaml`` → JSON.

Emits a JSON list of methods to stdout. Each entry preserves the schema
from models.yaml, dropping ``None`` fields for compactness."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Any

from marin_dna.pipelines.evals.models import Model, load_models


def _strip_nones(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_nones(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_nones(v) for v in obj]
    return obj


def _to_record(m: Model) -> dict[str, Any]:
    rec = asdict(m)
    # asdict turns the tuple back into a list, which is what we want for JSON.
    rec["datasets"] = list(m.datasets)
    return _strip_nones(rec)


def main() -> None:
    records = [_to_record(m) for m in load_models()]
    json.dump(records, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
