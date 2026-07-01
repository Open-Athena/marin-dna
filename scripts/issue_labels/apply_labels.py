#!/usr/bin/env python3
"""Apply the retroactive issue-label mapping in mapping.tsv.

Sets each issue's **Type/Area** labels to exactly what its row implies, and
removes retired labels (`enhancement`/`question`) and dropped `epic`. **Meta
labels (`agent-generated`/`marin`) and any other non-taxonomy label are left
untouched** — provenance is set at issue-creation time, so this tool never adds
or strips it (the meta column in mapping.tsv is documentation only). Idempotent.

Usage:
    python scripts/issue_labels/apply_labels.py --dry-run   # preview diffs
    python scripts/issue_labels/apply_labels.py             # apply via gh
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MAPPING = Path(__file__).with_name("mapping.tsv")

TYPES = {"research-question", "experiment", "eda", "infrastructure", "bug"}
AREAS = {
    "evals",
    "data",
    "modeling",
    "baselines",
    "hyperparameter-optimization",
    "interpretation",
}
META = {"agent-generated", "marin"}

# Labels this tool actively manages (adds/removes to match the mapping).
MANAGED = TYPES | AREAS
# Retired labels to strip wherever present; `epic` is dropped from issues too
# (kept as a label for future engineering use). META is deliberately absent —
# it is never added or removed.
REMOVABLE = MANAGED | {"enhancement", "question", "epic"}


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command, surfacing its stderr before propagating a failure."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr or "")
        raise


def parse_mapping(path: Path) -> dict[int, set[str]]:
    """number -> desired label set (Type + Area + meta, for display/validation)."""
    desired: dict[int, set[str]] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        assert len(cols) >= 4, (
            f"line {lineno}: expected >=4 tab-separated cols, got {len(cols)}: {raw!r}"
        )
        number = int(cols[0].strip())
        typ = cols[1].strip()
        areas = [a.strip() for a in cols[2].split(",") if a.strip()]
        meta = [m.strip() for m in cols[3].split(",") if m.strip()]
        labels: set[str] = set()
        if typ:
            assert typ in TYPES, f"line {lineno}: unknown type {typ!r}"
            labels.add(typ)
        for a in areas:
            assert a in AREAS, f"line {lineno}: unknown area {a!r}"
            labels.add(a)
        for m in meta:
            assert m in META, f"line {lineno}: unknown meta {m!r}"
            labels.add(m)
        assert number not in desired, f"line {lineno}: duplicate issue #{number}"
        desired[number] = labels
    return desired


def current_labels(number: int) -> set[str]:
    out = _run(["gh", "issue", "view", str(number), "--json", "labels"]).stdout
    return {lab["name"] for lab in json.loads(out)["labels"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run", action="store_true", help="print planned changes, don't apply"
    )
    ap.add_argument(
        "--only", type=int, nargs="*", help="restrict to these issue numbers"
    )
    args = ap.parse_args()

    desired = parse_mapping(MAPPING)
    if args.only is not None:
        desired = {n: v for n, v in desired.items() if n in args.only}
    print(
        f"{len(desired)} issues in mapping{' (filtered)' if args.only is not None else ''}\n"
    )

    changed = 0
    for number in sorted(desired):
        want = desired[number]
        have = current_labels(number)
        # Only add/remove Type+Area (and strip retired/epic); META and any other
        # label the issue already carries are left exactly as-is.
        add = (want & MANAGED) - have
        remove = (have - want) & REMOVABLE
        if not add and not remove:
            continue
        changed += 1
        print(f"#{number}: {sorted(have)} -> {sorted((have - remove) | add)}")
        print(f"    +{sorted(add)}  -{sorted(remove)}")
        if args.dry_run:
            continue
        cmd = ["gh", "issue", "edit", str(number)]
        for lab in sorted(add):
            cmd += ["--add-label", lab]
        for lab in sorted(remove):
            cmd += ["--remove-label", lab]
        _run(cmd)

    print(f"\n{'Would change' if args.dry_run else 'Changed'} {changed} issues.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
