from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = REPO_ROOT / "infra/check_skill_metadata.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_skill_metadata", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = load_script()

VALID_FRONTMATTER = "---\nname: {name}\ndescription: Do the thing.\n---\n"


def write_skill(
    root: Path, name: str, body: str = "", frontmatter: str | None = None
) -> Path:
    skill_dir = root / ".agents/skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    header = (
        frontmatter if frontmatter is not None else VALID_FRONTMATTER.format(name=name)
    )
    skill_file.write_text(header + f"\n# {name}\n\n{body}", encoding="utf-8")
    return skill_file


def write_file(root: Path, relative: str, text: str = "x\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def messages(errors: list[tuple[Path, str]]) -> list[str]:
    return [message for _path, message in errors]


def test_valid_skill_has_no_errors(tmp_path: Path) -> None:
    write_skill(
        tmp_path,
        "demo",
        "Uses `docs/guide.md:12`, [ref](references/a.md#top), and\n"
        '[root](/docs/guide.md "Guide").\n\n'
        "```bash\nuv run .agents/skills/demo/scripts/run.py --in docs/guide.md\n```\n",
    )
    write_file(tmp_path, "docs/guide.md")
    write_file(tmp_path, ".agents/skills/demo/references/a.md")
    write_file(tmp_path, ".agents/skills/demo/scripts/run.py")

    assert checker.check_skills(tmp_path) == []


def test_missing_frontmatter_delimiters(tmp_path: Path) -> None:
    write_skill(tmp_path, "demo", frontmatter="")
    write_skill(tmp_path, "open", frontmatter="---\nname: open\ndescription: x\n")

    assert messages(checker.check_skills(tmp_path)) == [
        "missing frontmatter delimiters",
        "missing frontmatter delimiters",
    ]


def test_body_swallowed_by_a_later_rule_is_rejected(tmp_path: Path) -> None:
    text = (
        "---\nname: demo\ndescription: x\n\n# demo\n\n"
        "Use when: the task needs it.\n\n---\n\nMore prose\n"
    )
    (tmp_path / ".agents/skills/demo").mkdir(parents=True)
    (tmp_path / ".agents/skills/demo/SKILL.md").write_text(text, encoding="utf-8")

    assert messages(checker.check_skills(tmp_path)) == [
        "frontmatter key 'Use when' is not an identifier; is the closing --- missing?"
    ]


def test_byte_order_mark_is_tolerated(tmp_path: Path) -> None:
    write_skill(
        tmp_path, "demo", frontmatter="﻿" + VALID_FRONTMATTER.format(name="demo")
    )

    assert checker.check_skills(tmp_path) == []


def test_frontmatter_must_be_mapping(tmp_path: Path) -> None:
    write_skill(tmp_path, "demo", frontmatter="---\n- not\n- a mapping\n---\n")

    assert messages(checker.check_skills(tmp_path)) == [
        "frontmatter must be a YAML mapping, got list"
    ]


def test_invalid_yaml_is_reported(tmp_path: Path) -> None:
    write_skill(tmp_path, "demo", frontmatter="---\nname: [unclosed\n---\n")

    (message,) = messages(checker.check_skills(tmp_path))
    assert message.startswith("invalid YAML frontmatter:")


def test_name_must_match_directory(tmp_path: Path) -> None:
    write_skill(tmp_path, "demo", frontmatter="---\nname: other\ndescription: x\n---\n")

    assert messages(checker.check_skills(tmp_path)) == [
        "name 'other' must match directory name 'demo'"
    ]


def test_name_and_description_must_be_non_empty_strings(tmp_path: Path) -> None:
    write_skill(tmp_path, "demo", frontmatter="---\nname: ''\ndescription: 3\n---\n")

    assert messages(checker.check_skills(tmp_path)) == [
        "frontmatter must include a non-empty string name, got ''",
        "frontmatter must include a non-empty string description, got 3",
    ]


def test_description_must_be_single_line(tmp_path: Path) -> None:
    frontmatter = "---\nname: demo\ndescription: |\n  first line\n  second line\n---\n"
    write_skill(tmp_path, "demo", frontmatter=frontmatter)

    assert messages(checker.check_skills(tmp_path)) == [
        "description must be a single-line string"
    ]


def test_schedule_fields_must_appear_together(tmp_path: Path) -> None:
    write_skill(
        tmp_path,
        "cron-only",
        frontmatter='---\nname: cron-only\ndescription: x\nschedule_cron: "0 10 * * *"\n---\n',
    )
    write_skill(
        tmp_path,
        "tz-only",
        frontmatter="---\nname: tz-only\ndescription: x\nschedule_tz: America/New_York\n---\n",
    )

    assert messages(checker.check_skills(tmp_path)) == [
        "schedule_cron and schedule_tz must be specified together",
        "schedule_cron and schedule_tz must be specified together",
    ]


def test_schedule_fields_are_validated(tmp_path: Path) -> None:
    def scheduled(name: str, cron: str, tz: str) -> None:
        write_skill(
            tmp_path,
            name,
            frontmatter=(
                f"---\nname: {name}\ndescription: x\n"
                f"schedule_cron: {cron}\nschedule_tz: {tz}\n---\n"
            ),
        )

    scheduled("six-fields", '"0 10 2 * * *"', "America/New_York")
    scheduled("letters", '"a b c d e"', "America/New_York")
    scheduled("macro", '"@daily"', "America/New_York")
    scheduled("bad-zone", '"0 10 * * *"', "Mars/Olympus_Mons")
    scheduled("typed", "5", "[a]")
    scheduled("range-dom", '"0 0 32 * *"', "America/New_York")
    scheduled("range-dow", '"0 0 * * 8"', "America/New_York")
    scheduled("range-minute", '"99 0 * * *"', "America/New_York")
    scheduled("range-month", '"0 0 1 13 *"', "America/New_York")
    scheduled("zero-step", '"*/0 0 * * *"', "America/New_York")
    scheduled("good", '"*/15 0-6,22 1,15 jan-jun mon-fri"', "America/New_York")

    assert messages(checker.check_skills(tmp_path)) == [
        "schedule_tz must be an IANA time zone, got 'Mars/Olympus_Mons'",
        "schedule_cron must be a 5-field cron expression, got 'a b c d e'",
        "schedule_cron must be a 5-field cron expression, got '@daily'",
        "schedule_cron must be a 5-field cron expression, got '0 0 32 * *'",
        "schedule_cron must be a 5-field cron expression, got '0 0 * * 8'",
        "schedule_cron must be a 5-field cron expression, got '99 0 * * *'",
        "schedule_cron must be a 5-field cron expression, got '0 0 1 13 *'",
        "schedule_cron must be a 5-field cron expression, got '0 10 2 * * *'",
        "schedule_cron must be a string, got int",
        "schedule_tz must be a string, got list",
        "schedule_cron must be a 5-field cron expression, got '*/0 0 * * *'",
    ]


def test_allowed_tools_must_be_string(tmp_path: Path) -> None:
    write_skill(
        tmp_path,
        "demo",
        frontmatter="---\nname: demo\ndescription: x\nallowed-tools:\n  - Bash\n---\n",
    )
    write_skill(
        tmp_path,
        "ok",
        frontmatter="---\nname: ok\ndescription: x\nallowed-tools: Bash(git status:*)\n---\n",
    )

    assert messages(checker.check_skills(tmp_path)) == [
        "allowed-tools must be a string, got list"
    ]


def test_duplicate_names_across_directories(tmp_path: Path) -> None:
    write_skill(tmp_path, "one", frontmatter="---\nname: shared\ndescription: x\n---\n")
    write_skill(tmp_path, "two", frontmatter="---\nname: shared\ndescription: x\n---\n")

    found = messages(checker.check_skills(tmp_path))
    assert found.count("name 'shared' must match directory name 'one'") == 1
    assert found.count("name 'shared' must match directory name 'two'") == 1
    duplicate = (
        "duplicate skill name 'shared': "
        ".agents/skills/one/SKILL.md, .agents/skills/two/SKILL.md"
    )
    assert found.count(duplicate) == 2


def test_skill_directory_without_skill_md(tmp_path: Path) -> None:
    write_skill(tmp_path, "demo")
    (tmp_path / ".agents/skills/stray").mkdir()
    (tmp_path / ".agents/skills/stray/skill.md").write_text(
        "lowercase\n", encoding="utf-8"
    )
    (tmp_path / ".agents/skills/__pycache__").mkdir()

    assert [(p.name, m) for p, m in checker.check_skills(tmp_path)] == [
        ("stray", "skill directory has no SKILL.md"),
    ]


def test_missing_references_are_reported_once_each(tmp_path: Path) -> None:
    body = (
        "Read `docs/missing.md` twice: `docs/missing.md:7`.\n"
        "Link [gone](references/gone.md) and [root](tests/nope.py).\n"
        "Anchored `.agents/skills/demo/SKILL.md#section` resolves.\n"
        "Root files: follow `AGENTS.md`, then `pyproject.toml`.\n"
        "```bash\npython3 .agents/skills/demo/scripts/absent.py --flag\n```\n"
    )
    write_skill(tmp_path, "demo", body)
    write_file(tmp_path, "AGENTS.md")

    assert messages(checker.check_skills(tmp_path)) == [
        "missing local reference: .agents/skills/demo/scripts/absent.py",
        "missing local reference: docs/missing.md",
        "missing local reference: pyproject.toml",
        "missing local reference: references/gone.md",
        "missing local reference: tests/nope.py",
    ]


def test_placeholders_urls_anchors_and_code_span_links_are_skipped(
    tmp_path: Path,
) -> None:
    body = (
        "Placeholders: `docs/<topic>/x.md`, `.agents/logbooks/*.md`, "
        "`docs/YYYY-MM-DD.md`, `docs/{a}.md`, `src/...`, `tests/$NAME.py`.\n"
        "Links: [site](https://example.com/docs/x), [mail](mailto:a@b.c), "
        "[anchor](#here), and syntax `[text](url)` in a code span.\n"
        "Glued paths: `s3://bucket/docs/x.md` and `a/docs/y.md`.\n"
        "Real: `docs/present.md`.\n"
    )
    write_skill(tmp_path, "demo", body)
    write_file(tmp_path, "docs/present.md")

    assert checker.check_skills(tmp_path) == []


def test_root_file_references_are_validated(tmp_path: Path) -> None:
    write_skill(tmp_path, "demo", "Follow `AGENTS.md`; see `SKILL.md` and `README.md`.")
    write_file(tmp_path, "AGENTS.md")

    assert messages(checker.check_skills(tmp_path)) == [
        "missing local reference: README.md"
    ]


def test_references_must_stay_inside_the_repository(tmp_path: Path) -> None:
    traversal = os.path.relpath("/etc/hosts", tmp_path / ".agents/skills/demo")
    write_skill(tmp_path, "demo", f"See [a](/etc/hosts) and [b]({traversal}).")

    assert messages(checker.check_skills(tmp_path)) == [
        "missing local reference: /etc/hosts",
        f"reference escapes the repository: {traversal}",
    ]


def test_reference_case_must_match_the_file_system(tmp_path: Path) -> None:
    write_skill(tmp_path, "demo", "See `docs/README.md` and `docs/readme.md`.")
    write_file(tmp_path, "docs/readme.md")

    (message,) = messages(checker.check_skills(tmp_path))
    # Case-insensitive file systems resolve the wrong-case path; others do not.
    assert message in {
        "reference case differs from the file system: docs/README.md",
        "missing local reference: docs/README.md",
    }


def test_drift_traps_fire_on_prose(tmp_path: Path) -> None:
    body = (
        "Old code sat in src/marin_dna/pipelines/evals and plots/output.\n"
        "Use agent-research, then gh-upload-asset the figure, "
        "then maintain-research-question.\n"
        "Fine: `docs/research/experiments/` and the Experiments section.\n"
    )
    write_skill(tmp_path, "demo", body)
    write_file(tmp_path, "docs/research/experiments/.keep")

    found = messages(checker.check_skills(tmp_path))
    assert len(found) == 5
    assert found[0].startswith("src/marin_dna/pipelines/ no longer exists")
    assert found[1].startswith("top-level experiments/ and plots/ were removed")
    assert "agent-research" in found[2]
    assert "gh-upload-asset" in found[3]
    assert "maintain-research-question" in found[4]


def write_manifest(root: Path, unchanged: list[object]) -> Path:
    write_skill(root, "maintain-vendored-skills")
    manifest = root / ".agents/skills/maintain-vendored-skills/references/manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"unchanged": unchanged, "adapted": [], "local": []}),
        encoding="utf-8",
    )
    return manifest


def test_vendored_unchanged_skills_skip_reference_checks_only(tmp_path: Path) -> None:
    write_skill(tmp_path, "vendored", "Upstream path `docs/reports/index.md`.")
    write_skill(tmp_path, "local", "Local path `docs/reports/index.md`.")
    write_skill(
        tmp_path,
        "vendored-broken",
        "Path `docs/nope.md`.",
        frontmatter="---\nname: wrong\ndescription: x\n---\n",
    )
    write_manifest(tmp_path, ["vendored", "vendored-broken"])

    errors = checker.check_skills(tmp_path)
    assert [(path.parent.name, message) for path, message in errors] == [
        ("local", "missing local reference: docs/reports/index.md"),
        ("vendored-broken", "name 'wrong' must match directory name 'vendored-broken'"),
    ]


def test_manifest_unchanged_entries_are_validated(tmp_path: Path) -> None:
    write_skill(tmp_path, "vendored")
    manifest = write_manifest(tmp_path, ["vendored", {"name": "vendored"}, "ghost"])

    assert checker.check_skills(tmp_path) == [
        (manifest, "unchanged entries must be names, got {'name': 'vendored'}"),
        (manifest, "unchanged skill 'ghost' has no SKILL.md"),
    ]


def test_cli_reports_errors_relative_to_root(tmp_path: Path) -> None:
    write_skill(tmp_path, "demo", "See `docs/nope.md`.")

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stdout.splitlines() == [
        "Skill metadata check failed (1 error(s)):",
        "  - .agents/skills/demo/SKILL.md: missing local reference: docs/nope.md",
    ]

    write_file(tmp_path, "docs/nope.md")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout == ""


def test_root_without_skills_is_an_error(tmp_path: Path) -> None:
    (tmp_path / ".agents/skills").mkdir(parents=True)

    assert messages(checker.check_skills(tmp_path)) == ["no skill directories found"]


def test_repository_skills_pass() -> None:
    assert checker.check_skills(REPO_ROOT) == []
