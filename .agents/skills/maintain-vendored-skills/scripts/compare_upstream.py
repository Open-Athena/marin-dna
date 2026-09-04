#!/usr/bin/env python3
"""Report exact differences between pinned Marin skills and local vendors."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

IGNORED_DIRECTORY_NAMES = {".pytest_cache", "__pycache__"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--upstream-commit")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Vendor manifest; defaults to references/manifest.json",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def file_identity(mode: str, data: bytes) -> str:
    kind = "symlink" if mode == "120000" else "file"
    return (
        f"mode={mode} type={kind} size={len(data)} "
        f"sha256={hashlib.sha256(data).hexdigest()}"
    )


def _entries(root: Path) -> dict[Path, tuple[str, bytes]]:
    entries: dict[Path, tuple[str, bytes]] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if IGNORED_DIRECTORY_NAMES.intersection(relative.parts):
            continue
        if path.suffix == ".pyc":
            continue
        if path.is_symlink():
            entries[relative] = ("120000", os.fsencode(os.readlink(path)))
        elif path.is_file():
            executable = bool(path.lstat().st_mode & stat.S_IXUSR)
            entries[relative] = (
                "100755" if executable else "100644",
                path.read_bytes(),
            )
    return entries


def directory_diff(
    upstream: Path,
    local: Path,
    upstream_label: str,
    local_label: str,
) -> str:
    upstream_files = _entries(upstream)
    local_files = _entries(local)
    chunks: list[str] = []
    for relative in sorted(upstream_files.keys() | local_files.keys()):
        upstream_entry = upstream_files.get(relative)
        local_entry = local_files.get(relative)
        upstream_mode, upstream_bytes = upstream_entry or ("missing", b"")
        local_mode, local_bytes = local_entry or ("missing", b"")
        if upstream_entry == local_entry:
            continue

        upstream_name = (
            f"{upstream_label}/.agents/skills/{upstream.name}/{relative}"
            if upstream_entry is not None
            else "/dev/null"
        )
        local_name = (
            f"{local_label}/.agents/skills/{local.name}/{relative}"
            if local_entry is not None
            else "/dev/null"
        )
        if (
            (upstream_entry is None) != (local_entry is None)
            and not upstream_bytes
            and not local_bytes
        ):
            upstream_identity = (
                file_identity(upstream_mode, upstream_bytes)
                if upstream_entry
                else "missing"
            )
            local_identity = (
                file_identity(local_mode, local_bytes) if local_entry else "missing"
            )
            chunks.extend(
                [
                    f"--- {upstream_name}\n",
                    f"+++ {local_name}\n",
                    "@@ empty file @@\n",
                    f"-{upstream_identity}\n",
                    f"+{local_identity}\n",
                ]
            )
            continue
        if upstream_mode != local_mode:
            upstream_identity = (
                file_identity(upstream_mode, upstream_bytes)
                if upstream_entry
                else "missing"
            )
            local_identity = (
                file_identity(local_mode, local_bytes) if local_entry else "missing"
            )
            chunks.extend(
                [
                    f"--- {upstream_name}\n",
                    f"+++ {local_name}\n",
                    "@@ metadata @@\n",
                    f"-{upstream_identity}\n",
                    f"+{local_identity}\n",
                ]
            )
            if upstream_bytes == local_bytes or "120000" in {
                upstream_mode,
                local_mode,
            }:
                continue
        try:
            upstream_lines = upstream_bytes.decode("utf-8").splitlines(keepends=True)
            local_lines = local_bytes.decode("utf-8").splitlines(keepends=True)
        except UnicodeDecodeError:
            upstream_identity = (
                file_identity(upstream_mode, upstream_bytes)
                if upstream_entry
                else "missing"
            )
            local_identity = (
                file_identity(local_mode, local_bytes) if local_entry else "missing"
            )
            chunks.extend(
                [
                    f"--- {upstream_name}\n",
                    f"+++ {local_name}\n",
                    "@@ binary @@\n",
                    f"-{upstream_identity}\n",
                    f"+{local_identity}\n",
                ]
            )
            continue

        chunks.extend(
            difflib.unified_diff(
                upstream_lines,
                local_lines,
                fromfile=upstream_name,
                tofile=local_name,
            )
        )
    return "".join(chunks)


def upstream_provenance(
    root: Path, expected_commit: str, declared_commit: str | None
) -> tuple[str, list[str]]:
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or "git rev-parse failed"
            return "Git checkout could not be verified", [message]
        actual_commit = result.stdout.strip()
        errors = []
        if actual_commit != expected_commit:
            errors.append(
                f"upstream Git HEAD is {actual_commit}, expected {expected_commit}"
            )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                ".agents/skills",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode != 0:
            message = status.stderr.strip() or "git status failed"
            errors.append(
                f"upstream skill-tree cleanliness could not be verified: {message}"
            )
        elif status.stdout.strip():
            dirty_lines = [
                line
                for line in status.stdout.splitlines()
                if line.strip()
                and not IGNORED_DIRECTORY_NAMES.intersection(Path(line[3:]).parts)
                and Path(line[3:]).suffix != ".pyc"
            ]
            if not dirty_lines:
                state = "verified clean Git HEAD" if not errors else "Git HEAD"
                return f"{state} `{actual_commit}`", errors
            dirty_paths = ", ".join(line[3:] for line in dirty_lines)
            errors.append(f"upstream skill tree is dirty: {dirty_paths}")
        state = "verified clean Git HEAD" if not errors else "Git HEAD"
        return f"{state} `{actual_commit}`", errors

    if declared_commit is None:
        return "unverified non-Git tree", [
            "--upstream-commit is required for a non-Git upstream tree"
        ]
    errors = (
        []
        if declared_commit == expected_commit
        else [
            f"declared upstream commit is {declared_commit}, expected {expected_commit}"
        ]
    )
    provenance = f"caller-declared non-Git tree at `{declared_commit}`"
    provenance += " (not independently verified)"
    return provenance, errors


def main() -> int:
    args = parse_args()
    skill_root = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest or skill_root / "references/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    local_root = args.repo_root.resolve() / ".agents/skills"
    upstream_repo = manifest["upstream_repo"]
    expected_commit = manifest["commit"]
    upstream_checkout = args.upstream_root.resolve()
    upstream_root = upstream_checkout / ".agents/skills"
    provenance, errors = upstream_provenance(
        upstream_checkout, expected_commit, args.upstream_commit
    )

    lines = [
        f"# {upstream_repo} vendored-skill deltas",
        "",
        f"Expected upstream: `{upstream_repo}@{expected_commit}`",
        "",
        f"Input provenance: {provenance}",
        "",
        "## Exact vendors",
        "",
    ]

    for name in manifest["unchanged"]:
        upstream_skill = upstream_root / name
        local_skill = local_root / name
        missing = [
            str(path) for path in (upstream_skill, local_skill) if not path.is_dir()
        ]
        if missing:
            lines.append(f"- `{name}`: missing directory")
            errors.extend(f"missing skill directory: {path}" for path in missing)
            continue
        diff = directory_diff(
            upstream_skill,
            local_skill,
            upstream_repo,
            "Open-Athena/marin-dna",
        )
        state = "content/type/mode-identical" if not diff else "unexpected local diff"
        lines.append(f"- `{name}`: {state}")
        if diff:
            errors.append(f"{name} is classified unchanged but has a diff")
            lines.extend(
                [
                    "",
                    f"### Unexpected diff: `{name}`",
                    "",
                    "```diff",
                    diff.rstrip(),
                    "```",
                ]
            )

    lines.extend(["", "## Adapted vendors", ""])
    for item in manifest["adapted"]:
        name = item["name"]
        upstream_skill = upstream_root / name
        local_skill = local_root / name
        lines.append(f"### `{name}`")
        lines.append("")
        lines.extend(f"- {reason}" for reason in item["deviations"])
        lines.append("")
        missing = [
            str(path) for path in (upstream_skill, local_skill) if not path.is_dir()
        ]
        if missing:
            lines.extend(["Missing skill directory.", ""])
            errors.extend(f"missing skill directory: {path}" for path in missing)
            continue
        diff = directory_diff(
            upstream_skill,
            local_skill,
            upstream_repo,
            "Open-Athena/marin-dna",
        )
        if diff:
            lines.extend(["```diff", diff.rstrip(), "```", ""])
        else:
            lines.extend(["No upstream-to-local diff.", ""])
            errors.append(f"{name} is classified adapted but has no diff")

    lines.extend(["## MarinDNA-only skills", ""])
    for name in manifest["local"]:
        local_skill = local_root / name
        state = "" if local_skill.is_dir() else " (missing directory)"
        lines.append(f"- `{name}`{state}")
        if state:
            errors.append(f"missing local skill directory: {local_skill}")
    lines.append("")
    report = "\n".join(lines)

    if args.output is None:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(args.output)

    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
