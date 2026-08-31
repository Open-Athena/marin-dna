#!/usr/bin/env python3
"""Validate every skill under ``.agents/skills/``.

The pre-commit hook runs this whenever anything under ``.agents/skills/``
changes, and the Quality workflow runs it on every push and pull request.
One skill change re-validates all skills because skills link to each other
and to the repository layout, so a rename anywhere can break a link elsewhere.

Per ``SKILL.md`` the check enforces:

- the frontmatter is a YAML mapping;
- ``name`` is a non-empty string equal to the skill directory name, and no two
  skills share a name;
- ``description`` is a non-empty single-line string;
- ``schedule_cron`` and ``schedule_tz`` appear together, as a five-field cron
  expression and an IANA time zone;
- ``allowed-tools`` is a string;
- every backticked repository path and every relative Markdown link resolves
  from the skill directory or the repository root;
- known drift traps: paths and skill names that moved or were retired.

Skills a vendor manifest lists as ``unchanged`` are upstream content that
``maintain-vendored-skills`` verifies byte-for-byte, so their local path
references and drift traps are not checked; their frontmatter still is.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zoneinfo
from pathlib import Path

import yaml

SKILLS_DIR = Path(".agents") / "skills"
VENDOR_MANIFEST_GLOB = "maintain-vendored-skills/references/*manifest.json"

# Top-level directories a skill may reference. The last four were removed from
# ``main`` in #333, so a reference to them is stale by construction.
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
    "experiments",
    "lib",
    "plots",
    "scripts",
)
_PREFIX_ALTERNATION = "|".join(re.escape(prefix) for prefix in REPOSITORY_PATH_PREFIXES)
BACKTICK_PATH_PATTERN = re.compile(rf"`((?:{_PREFIX_ALTERNATION})/[^`\s,;:)]*)`")
MARKDOWN_LINK_PATTERN = re.compile(r"\]\(([^)\s]+)\)")
REFERENCE_PLACEHOLDER_TOKENS = ("<", ">", "*", "{", "}", "...", "YYYY", "XXXX", "$")
URL_SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

# Drift traps for mistakes this repository has already made once. Each entry
# is (pattern, message); a match anywhere in the skill text is an error.
DRIFT_TRAPS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"src/marin_dna/pipelines/"),
        "src/marin_dna/pipelines/ no longer exists; pipeline packages live under "
        "snakemake/<pipeline>/src/ since pipelines became independently locked "
        "projects",
    ),
    (
        re.compile(r"\bagent-research\b"),
        "the agent-research skill was replaced by run-research (#455)",
    ),
    (
        re.compile(r"\bgh-upload-asset\b"),
        "the gh-upload-asset skill was retired (#466); commit artifacts under "
        ".agents/artifacts/<topic>/ on the permanent branch instead",
    ),
    (
        re.compile(r"\bmaintain-research-question\b"),
        "the maintain-research-question skill was replaced by "
        "maintain-knowledge-base (#476)",
    ),
)

CRON_FIELD_COUNT = 5

Error = tuple[Path, str]


def split_frontmatter(text: str) -> tuple[str, str] | None:
    """Return ``(frontmatter, body)`` or ``None`` when the delimiters are missing."""
    if not text.startswith("---\n"):
        return None
    lines = text.split("\n")
    for index in range(1, len(lines)):
        if lines[index].rstrip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    return None


def is_placeholder_reference(reference: str) -> bool:
    return any(token in reference for token in REFERENCE_PLACEHOLDER_TOKENS)


def reference_exists(skill_dir: Path, root: Path, reference: str) -> bool:
    return (skill_dir / reference).exists() or (root / reference).exists()


def iter_path_references(text: str) -> list[str]:
    """Backticked repository paths and relative Markdown link targets in ``text``."""
    references: list[str] = []
    for match in BACKTICK_PATH_PATTERN.finditer(text):
        references.append(match.group(1))
    for match in MARKDOWN_LINK_PATTERN.finditer(text):
        target = match.group(1)
        if target.startswith("#") or URL_SCHEME_PATTERN.match(target):
            continue
        references.append(target)
    return references


def vendored_unchanged_skills(root: Path) -> set[str]:
    """Skill names every vendor manifest classifies as ``unchanged``."""
    names: set[str] = set()
    for manifest_path in sorted((root / SKILLS_DIR).glob(VENDOR_MANIFEST_GLOB)):
        manifest = json.loads(manifest_path.read_text())
        unchanged = manifest.get("unchanged", [])
        assert isinstance(unchanged, list), f"{manifest_path}: unchanged must be a list"
        names.update(unchanged)
    return names


def check_metadata(skill_file: Path, frontmatter: str) -> tuple[list[str], str | None]:
    """Validate the frontmatter mapping; return ``(errors, name)``."""
    errors: list[str] = []
    try:
        metadata = yaml.safe_load(frontmatter)
    except yaml.YAMLError as error:
        return [f"invalid YAML frontmatter: {error}"], None
    if not isinstance(metadata, dict):
        return [
            f"frontmatter must be a YAML mapping, got {type(metadata).__name__}"
        ], None

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
            "frontmatter must include a non-empty string description, "
            f"got {description!r}"
        )
    elif "\n" in description.strip():
        errors.append("description must be a single-line string")

    has_cron = "schedule_cron" in metadata
    has_tz = "schedule_tz" in metadata
    if has_cron != has_tz:
        errors.append("schedule_cron and schedule_tz must be specified together")
    if has_cron:
        cron = metadata["schedule_cron"]
        if not isinstance(cron, str):
            errors.append(f"schedule_cron must be a string, got {type(cron).__name__}")
        elif len(cron.split()) != CRON_FIELD_COUNT:
            errors.append(
                f"schedule_cron must be a {CRON_FIELD_COUNT}-field cron expression, "
                f"got {cron!r}"
            )
    if has_tz:
        tz = metadata["schedule_tz"]
        if not isinstance(tz, str):
            errors.append(f"schedule_tz must be a string, got {type(tz).__name__}")
        else:
            known_zones = zoneinfo.available_timezones()
            if known_zones and tz not in known_zones:
                errors.append(f"schedule_tz must be an IANA time zone, got {tz!r}")

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
        reference = raw_reference.split("#", 1)[0]
        if not reference or reference in seen or is_placeholder_reference(reference):
            continue
        seen.add(reference)
        if not reference_exists(skill_file.parent, root, reference):
            errors.append(f"missing local reference: {reference}")
    return errors


def check_skills(root: Path, changed: list[Path] | None = None) -> list[Error]:
    """Validate every skill under ``root``.

    ``changed`` narrows the report to those SKILL.md paths when any of them has
    an error; otherwise every error is reported so a broken untouched skill
    cannot hide behind a clean one.
    """
    root = root.resolve()
    skill_files = sorted((root / SKILLS_DIR).glob("*/SKILL.md"))
    assert skill_files, f"no skills found under {root / SKILLS_DIR}"
    unchanged_vendors = vendored_unchanged_skills(root)

    errors: list[Error] = []
    names: dict[str, list[Path]] = {}
    for skill_file in skill_files:
        text = skill_file.read_text()
        parts = split_frontmatter(text)
        if parts is None:
            errors.append((skill_file, "missing frontmatter delimiters"))
            continue
        frontmatter, _body = parts
        metadata_errors, name = check_metadata(skill_file, frontmatter)
        errors.extend((skill_file, error) for error in metadata_errors)
        if name is not None:
            names.setdefault(name, []).append(skill_file)
        if skill_file.parent.name in unchanged_vendors:
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

    if changed:
        changed_set = {path.resolve() for path in changed}
        relevant = [(path, error) for path, error in errors if path in changed_set]
        if relevant:
            return relevant
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (default: the parent of infra/)",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="changed SKILL.md files; every skill is validated regardless",
    )
    args = parser.parse_args(argv)
    errors = check_skills(args.root, args.files or None)
    if not errors:
        return 0
    root = args.root.resolve()
    print(f"Skill metadata check failed ({len(errors)} error(s)):")
    for path, error in errors:
        print(f"  - {path.relative_to(root)}: {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
