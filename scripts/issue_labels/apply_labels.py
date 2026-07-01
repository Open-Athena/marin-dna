#!/usr/bin/env python3
"""Apply the retroactive issue-label mapping in mapping.tsv.

Sets each issue to EXACTLY the label set implied by its row (type + areas +
meta), adding what's missing and removing anything else (including retired
labels like `enhancement`/`question` and `epic` where dropped). Idempotent.

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
ALLOWED = TYPES | AREAS | META


def parse_mapping(path: Path) -> dict[int, set[str]]:
    """number -> desired exact label set."""
    desired: dict[int, set[str]] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
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
    out = subprocess.run(
        ["gh", "issue", "view", str(number), "--json", "labels"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
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
    if args.only:
        desired = {n: v for n, v in desired.items() if n in args.only}
    print(f"{len(desired)} issues in mapping{' (filtered)' if args.only else ''}\n")

    changed = 0
    for number in sorted(desired):
        want = desired[number]
        have = current_labels(number)
        # only reconcile taxonomy-relevant labels; leave any unknown label
        # untouched EXCEPT the ones we explicitly retire.
        retire = {"enhancement", "question"}
        add = want - have
        remove = (have - want) & (ALLOWED | retire | {"epic"})
        if not add and not remove:
            continue
        changed += 1
        print(f"#{number}: {sorted(have)} -> {sorted(want)}")
        print(f"    +{sorted(add)}  -{sorted(remove)}")
        if args.dry_run:
            continue
        cmd = ["gh", "issue", "edit", str(number)]
        for lab in sorted(add):
            cmd += ["--add-label", lab]
        for lab in sorted(remove):
            cmd += ["--remove-label", lab]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    print(f"\n{'Would change' if args.dry_run else 'Changed'} {changed} issues.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
