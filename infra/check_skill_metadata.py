#!/usr/bin/env python3
"""Validate every skill under ``.agents/skills/``.

The pre-commit hook runs this on every commit and the Quality workflow runs it
on every pull request and push to ``main``. Every run validates every skill
because skills link to each other and to the repository layout, so a rename
anywhere can break a link elsewhere.

Per skill directory the check enforces:

- the directory holds ``SKILL.md`` whose frontmatter is a YAML mapping with
  identifier-like keys, closed by a ``---`` line;
- ``name`` is a non-empty string equal to the directory name, and no two
  skills share a name;
- ``description`` is a non-empty single-line string;
- ``schedule_cron`` and ``schedule_tz`` appear together, as a five-field cron
  expression and an IANA time zone;
- ``allowed-tools`` is a string;
- every repository path in inline code, fenced code, or a relative Markdown
  link — and every well-known root file cited by bare name — resolves from the
  skill directory or the repository root, with exact case and without escaping
  the repository;
- known drift traps: paths and skill names that moved or were retired.

Skills a vendor manifest lists as ``unchanged`` are upstream content that
``maintain-vendored-skills`` verifies byte-for-byte, so their path references
and drift traps are not checked; their frontmatter still is. ``adapted``
vendored skills are checked in full: when a local path one of them links to
moves, update the link and record the deviation in the manifest in the same
change.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zoneinfo
from pathlib import Path

import yaml

SKILLS_DIR = Path(".agents") / "skills"
VENDOR_MANIFEST_GLOB = "maintain-vendored-skills/references/*manifest.json"

# Top-level directories a skill may reference.
REPOSITORY_PATH_PREFIXES = (
    ".agents",
    ".claude",
    ".github",
    "dashboard",
    "docs",
    "infra",
    "snakemake",
    "src",
    "tests",
)
_PREFIX_ALTERNATION = "|".join(re.escape(prefix) for prefix in REPOSITORY_PATH_PREFIXES)
# A repository path token inside code: not glued to a preceding path or word
# (so ``s3://bucket/docs/x`` and ``a/docs/x`` do not match) and running to the
# next whitespace or quote.
PATH_TOKEN_PATTERN = re.compile(rf"(?<![\w./-])((?:{_PREFIX_ALTERNATION})/[^\s`'\"|]*)")
# Well-known files referenced by bare name; validated like paths so renaming
# one fails lint in every skill that cites it. SKILL.md resolves to the citing
# skill's own file, the rest to the repository root.
ROOT_FILE_NAMES = (
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "SKILL.md",
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
    ".pre-commit-config.yaml",
    ".python-version",
)
_ROOT_FILE_ALTERNATION = "|".join(re.escape(name) for name in ROOT_FILE_NAMES)
ROOT_FILE_PATTERN = re.compile(rf"(?<![\w./-])({_ROOT_FILE_ALTERNATION})(?![\w-])")
INLINE_CODE_PATTERN = re.compile(r"`([^`\n]+)`")
FENCED_BLOCK_PATTERN = re.compile(
    r"^ {0,3}```[^\n]*\n(.*?)^ {0,3}```[ \t]*$", re.MULTILINE | re.DOTALL
)
MARKDOWN_LINK_PATTERN = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
LINE_SUFFIX_PATTERN = re.compile(r":\d+(?:-\d+)?$")
TRAILING_REFERENCE_PUNCTUATION = ".,;:)"
REFERENCE_PLACEHOLDER_TOKENS = ("<", ">", "*", "{", "}", "...", "YYYY", "XXXX", "$")
URL_SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

FRONTMATTER_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
CRON_FIELD_COUNT = 5
_CRON_MONTHS = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)
_CRON_DAYS = ("sun", "mon", "tue", "wed", "thu", "fri", "sat")
# (low, high, names) per field: minute, hour, day of month, month, day of week.
CRON_FIELD_SPECS: tuple[tuple[int, int, tuple[str, ...]], ...] = (
    (0, 59, ()),
    (0, 23, ()),
    (1, 31, ()),
    (1, 12, _CRON_MONTHS),
    (0, 7, _CRON_DAYS),
)


def _cron_value_valid(value: str, low: int, high: int, names: tuple[str, ...]) -> bool:
    if value.lower() in names:
        return True
    return value.isdigit() and low <= int(value) <= high


def _cron_field_valid(field: str, low: int, high: int, names: tuple[str, ...]) -> bool:
    for token in field.split(","):
        value, slash, step = token.partition("/")
        if slash and not (step.isdigit() and int(step) > 0):
            return False
        if value == "*":
            continue
        bounds = value.split("-")
        if len(bounds) > 2:
            return False
        if not all(_cron_value_valid(bound, low, high, names) for bound in bounds):
            return False
    return True


# Drift traps for mistakes this repository has already made once. Each entry
# is (pattern, message); a match anywhere in the skill text is an error. When
# a skill is renamed or retired, add its old name here so bare mentions fail.
DRIFT_TRAPS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"src/marin_dna/pipelines/"),
        (
            "src/marin_dna/pipelines/ no longer exists; pipeline packages live under "
            "snakemake/<pipeline>/src/ since pipelines became independently locked "
            "projects"
        ),
    ),
    (
        re.compile(r"(?<![\w./-])(?:experiments|plots)/"),
        (
            "top-level experiments/ and plots/ were removed from main (#333); "
            "one-off code and figures live on permanent branches"
        ),
    ),
    (
        re.compile(r"\bagent-research\b"),
        "the agent-research skill was replaced by run-research (#455)",
    ),
    (
        re.compile(r"\bgh-upload-asset\b"),
        (
            "the gh-upload-asset skill was retired (#466); commit artifacts under "
            ".agents/artifacts/<topic>/ on the permanent branch instead"
        ),
    ),
    (
        re.compile(r"\bmaintain-research-question\b"),
        (
            "the maintain-research-question skill was replaced by "
            "maintain-knowledge-base (#476)"
        ),
    ),
)

Error = tuple[Path, str]


def split_frontmatter(text: str) -> tuple[str, str] | None:
    """Return ``(frontmatter, body)`` or ``None`` when the delimiters are missing."""
    text = text.removeprefix("﻿")
    if not text.startswith("---\n"):
        return None
    lines = text.split("\n")
    for index in range(1, len(lines)):
        if lines[index].rstrip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    return None


def is_placeholder_reference(reference: str) -> bool:
    return any(token in reference for token in REFERENCE_PLACEHOLDER_TOKENS)


def normalize_reference(raw: str) -> str:
    """Strip fragments, ``:line`` suffixes, and trailing punctuation."""
    reference = raw.split("#", 1)[0]
    reference = LINE_SUFFIX_PATTERN.sub("", reference)
    return reference.rstrip(TRAILING_REFERENCE_PUNCTUATION)


def iter_path_references(text: str) -> list[str]:
    """Repository paths in code spans and fenced blocks, plus relative link targets."""
    references: list[str] = []
    fenced_blocks = FENCED_BLOCK_PATTERN.findall(text)
    prose = FENCED_BLOCK_PATTERN.sub("", text)
    inline_code = INLINE_CODE_PATTERN.findall(prose)
    prose = INLINE_CODE_PATTERN.sub("", prose)
    for code in [*fenced_blocks, *inline_code]:
        references.extend(match.group(1) for match in PATH_TOKEN_PATTERN.finditer(code))
        references.extend(match.group(1) for match in ROOT_FILE_PATTERN.finditer(code))
    for match in MARKDOWN_LINK_PATTERN.finditer(prose):
        target = match.group(1)
        if target.startswith("#") or URL_SCHEME_PATTERN.match(target):
            continue
        references.append(target)
    return references


def locate_reference(skill_dir: Path, root: Path, reference: str) -> str | None:
    """Return an error message when ``reference`` does not resolve inside ``root``.

    A leading ``/`` is repository-root relative, as GitHub renders it. Paths are
    compared component by component so a reference passes only with the exact
    case recorded on disk, which keeps macOS and Linux CI in agreement.
    """
    if reference.startswith("/"):
        candidates = [root / reference.lstrip("/")]
    else:
        candidates = [skill_dir / reference, root / reference]
    problems: list[str] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError, RuntimeError):
            problems.append("missing local reference")
            continue
        if not resolved.is_relative_to(root):
            problems.append("reference escapes the repository")
            continue
        current = root
        exact = True
        for part in resolved.relative_to(root).parts:
            if part not in os.listdir(current):
                exact = False
                break
            current = current / part
        if exact:
            return None
        problems.append("reference case differs from the file system")
    # Report the most specific problem: an escape or case mismatch beats "missing".
    for message in (
        "reference escapes the repository",
        "reference case differs from the file system",
    ):
        if message in problems:
            return f"{message}: {reference}"
    return f"missing local reference: {reference}"


def vendored_unchanged_skills(root: Path) -> tuple[set[str], list[Error]]:
    """Skill names every vendor manifest classifies as ``unchanged``."""
    names: set[str] = set()
    errors: list[Error] = []
    for manifest_path in sorted((root / SKILLS_DIR).glob(VENDOR_MANIFEST_GLOB)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["unchanged"]:
            if not isinstance(entry, str):
                errors.append(
                    (manifest_path, f"unchanged entries must be names, got {entry!r}")
                )
                continue
            if not (root / SKILLS_DIR / entry / "SKILL.md").is_file():
                errors.append(
                    (manifest_path, f"unchanged skill {entry!r} has no SKILL.md")
                )
                continue
            names.add(entry)
    return names, errors


def check_schedule(metadata: dict[str, object]) -> list[str]:
    errors: list[str] = []
    has_cron = "schedule_cron" in metadata
    has_tz = "schedule_tz" in metadata
    if has_cron != has_tz:
        errors.append("schedule_cron and schedule_tz must be specified together")
    if has_cron:
        cron = metadata["schedule_cron"]
        if not isinstance(cron, str):
            errors.append(f"schedule_cron must be a string, got {type(cron).__name__}")
        else:
            fields = cron.split()
            valid = len(fields) == CRON_FIELD_COUNT and all(
                _cron_field_valid(field, low, high, names)
                for field, (low, high, names) in zip(fields, CRON_FIELD_SPECS)
            )
            if not valid:
                errors.append(
                    f"schedule_cron must be a {CRON_FIELD_COUNT}-field cron expression, "
                    f"got {cron!r}"
                )
    if has_tz:
        tz = metadata["schedule_tz"]
        if not isinstance(tz, str):
            errors.append(f"schedule_tz must be a string, got {type(tz).__name__}")
        else:
            try:
                zoneinfo.ZoneInfo(tz)
            except (zoneinfo.ZoneInfoNotFoundError, ValueError):
                errors.append(f"schedule_tz must be an IANA time zone, got {tz!r}")
    return errors


def check_metadata(skill_file: Path, frontmatter: str) -> tuple[list[str], str | None]:
    """Validate the frontmatter mapping; return ``(errors, name)``."""
    try:
        metadata = yaml.safe_load(frontmatter)
    except yaml.YAMLError as error:
        return [f"invalid YAML frontmatter: {error}"], None
    if not isinstance(metadata, dict):
        return [
            f"frontmatter must be a YAML mapping, got {type(metadata).__name__}"
        ], None
    bad_keys = [
        key
        for key in metadata
        if not isinstance(key, str) or not FRONTMATTER_KEY_PATTERN.match(key)
    ]
    if bad_keys:
        return [
            f"frontmatter key {bad_keys[0]!r} is not an identifier; is the closing --- missing?"
        ], None

    errors: list[str] = []
    name = metadata.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append(f"frontmatter must include a non-empty string name, got {name!r}")
        name = None
    elif name != skill_file.parent.name:
        errors.append(
            f"name {name!r} must match directory name {skill_file.parent.name!r}"
        )

    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append(
            f"frontmatter must include a non-empty string description, got {description!r}"
        )
    elif "\n" in description.strip():
        errors.append("description must be a single-line string")

    errors.extend(check_schedule(metadata))

    allowed_tools = metadata.get("allowed-tools")
    if allowed_tools is not None and not isinstance(allowed_tools, str):
        errors.append(
            f"allowed-tools must be a string, got {type(allowed_tools).__name__}"
        )
    return errors, name


def check_references(skill_file: Path, root: Path, text: str) -> list[str]:
    errors: list[str] = []
    for pattern, message in DRIFT_TRAPS:
        if pattern.search(text):
            errors.append(message)
    seen: set[str] = set()
    for raw_reference in iter_path_references(text):
        if is_placeholder_reference(raw_reference):
            continue
        reference = normalize_reference(raw_reference)
        if not reference or reference in seen:
            continue
        seen.add(reference)
        problem = locate_reference(skill_file.parent, root, reference)
        if problem is not None:
            errors.append(problem)
    return errors


def check_skills(root: Path) -> list[Error]:
    """Validate every skill under ``root``; return ``(path, message)`` errors."""
    root = root.resolve()
    skills_dir = root / SKILLS_DIR
    skill_dirs = sorted(
        path
        for path in skills_dir.iterdir()
        if path.is_dir() and not path.name.startswith((".", "__"))
    )
    if not skill_dirs:
        return [(skills_dir, "no skill directories found")]
    unchanged_vendors, errors = vendored_unchanged_skills(root)

    names: dict[str, list[Path]] = {}
    for skill_dir in skill_dirs:
        skill_file = skill_dir / "SKILL.md"
        if skill_file.name not in os.listdir(skill_dir):
            errors.append((skill_dir, "skill directory has no SKILL.md"))
            continue
        text = skill_file.read_text(encoding="utf-8")
        parts = split_frontmatter(text)
        if parts is None:
            errors.append((skill_file, "missing frontmatter delimiters"))
            continue
        frontmatter, _body = parts
        metadata_errors, name = check_metadata(skill_file, frontmatter)
        errors.extend((skill_file, error) for error in metadata_errors)
        if name is not None:
            names.setdefault(name, []).append(skill_file)
        if skill_dir.name in unchanged_vendors:
            continue
        errors.extend(
            (skill_file, error) for error in check_references(skill_file, root, text)
        )

    for name, paths in names.items():
        if len(paths) <= 1:
            continue
        joined = ", ".join(str(path.relative_to(root)) for path in paths)
        errors.extend(
            (path, f"duplicate skill name {name!r}: {joined}") for path in paths
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate every skill under .agents/skills/."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (default: the parent of infra/)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    errors = check_skills(root)
    if not errors:
        return 0
    print(f"Skill metadata check failed ({len(errors)} error(s)):")
    for path, error in errors:
        print(f"  - {path.relative_to(root)}: {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
