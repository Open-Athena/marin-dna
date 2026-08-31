from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

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
    skill_file.write_text(header + f"\n# {name}\n\n{body}")
    return skill_file


def messages(errors: list[tuple[Path, str]]) -> list[str]:
    return [message for _path, message in errors]


def test_valid_skill_has_no_errors(tmp_path: Path) -> None:
    write_skill(tmp_path, "demo", "Uses `docs/guide.md` and [ref](references/a.md).")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/guide.md").write_text("guide\n")
    (tmp_path / ".agents/skills/demo/references").mkdir()
    (tmp_path / ".agents/skills/demo/references/a.md").write_text("a\n")

    assert checker.check_skills(tmp_path) == []


def test_missing_frontmatter_delimiters(tmp_path: Path) -> None:
    write_skill(tmp_path, "demo", frontmatter="")
    write_skill(tmp_path, "open", frontmatter="---\nname: open\ndescription: x\n")

    assert messages(checker.check_skills(tmp_path)) == [
        "missing frontmatter delimiters",
        "missing frontmatter delimiters",
    ]


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
    write_skill(
        tmp_path,
        "bad",
        frontmatter=(
            "---\nname: bad\ndescription: x\n"
            'schedule_cron: "0 10 2 * * *"\nschedule_tz: Mars/Olympus_Mons\n---\n'
        ),
    )
    write_skill(
        tmp_path,
        "typed",
        frontmatter="---\nname: typed\ndescription: x\nschedule_cron: 5\nschedule_tz: [a]\n---\n",
    )
    write_skill(
        tmp_path,
        "good",
        frontmatter=(
            "---\nname: good\ndescription: x\n"
            'schedule_cron: "0 10 * * *"\nschedule_tz: America/New_York\n---\n'
        ),
    )

    assert messages(checker.check_skills(tmp_path)) == [
        "schedule_cron must be a 5-field cron expression, got '0 10 2 * * *'",
        "schedule_tz must be an IANA time zone, got 'Mars/Olympus_Mons'",
        "schedule_cron must be a string, got int",
        "schedule_tz must be a string, got list",
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
    duplicate = "duplicate skill name 'shared': .agents/skills/one/SKILL.md, .agents/skills/two/SKILL.md"
    assert found.count(duplicate) == 2


def test_missing_references_are_reported_once_each(tmp_path: Path) -> None:
    body = (
        "Read `docs/missing.md` twice: `docs/missing.md`.\n"
        "Link [gone](references/gone.md) and [root](tests/nope.py).\n"
        "Anchored `.agents/skills/demo/SKILL.md#section` resolves.\n"
    )
    write_skill(tmp_path, "demo", body)

    assert messages(checker.check_skills(tmp_path)) == [
        "missing local reference: docs/missing.md",
        "missing local reference: references/gone.md",
        "missing local reference: tests/nope.py",
    ]


def test_placeholders_urls_and_anchors_are_skipped(tmp_path: Path) -> None:
    body = (
        "Placeholders: `docs/<topic>/x.md`, `.agents/logbooks/*.md`, "
        "`scripts/YYYY-MM-DD.py`, `docs/{a,b}.md`, `src/...`.\n"
        "Links: [site](https://example.com/docs/x), [mail](mailto:a@b.c), [anchor](#here).\n"
        "Retired top-level dirs are stale: `plots/output/fig.svg`.\n"
    )
    write_skill(tmp_path, "demo", body)

    assert messages(checker.check_skills(tmp_path)) == [
        "missing local reference: plots/output/fig.svg",
    ]


def test_drift_traps_fire_on_prose(tmp_path: Path) -> None:
    body = (
        "Old code sat in src/marin_dna/pipelines/evals.\n"
        "Use agent-research, then gh-upload-asset the figure, "
        "then maintain-research-question.\n"
    )
    write_skill(tmp_path, "demo", body)

    found = messages(checker.check_skills(tmp_path))
    assert len(found) == 4
    assert found[0].startswith("src/marin_dna/pipelines/ no longer exists")
    assert "agent-research" in found[1]
    assert "gh-upload-asset" in found[2]
    assert "maintain-research-question" in found[3]


def test_vendored_unchanged_skills_skip_reference_checks_only(tmp_path: Path) -> None:
    write_skill(tmp_path, "vendored", "Upstream path `docs/reports/index.md`.")
    write_skill(tmp_path, "local", "Local path `docs/reports/index.md`.")
    write_skill(
        tmp_path,
        "vendored-broken",
        "Path `docs/nope.md`.",
        frontmatter="---\nname: wrong\ndescription: x\n---\n",
    )
    manifest_dir = tmp_path / ".agents/skills/maintain-vendored-skills/references"
    manifest_dir.mkdir(parents=True)
    write_skill(tmp_path, "maintain-vendored-skills")
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {"unchanged": ["vendored", "vendored-broken"], "adapted": [], "local": []}
        )
    )

    errors = checker.check_skills(tmp_path)
    assert [(path.parent.name, message) for path, message in errors] == [
        ("local", "missing local reference: docs/reports/index.md"),
        ("vendored-broken", "name 'wrong' must match directory name 'vendored-broken'"),
    ]


def test_changed_files_narrow_the_report_only_when_they_have_errors(
    tmp_path: Path,
) -> None:
    broken = write_skill(tmp_path, "broken", "See `docs/nope.md`.")
    clean = write_skill(tmp_path, "clean")
    also_broken = write_skill(tmp_path, "also-broken", "See `docs/nope2.md`.")

    assert [p.parent.name for p, _ in checker.check_skills(tmp_path, [broken])] == [
        "broken"
    ]
    assert [p.parent.name for p, _ in checker.check_skills(tmp_path, [clean])] == [
        "also-broken",
        "broken",
    ]
    assert [
        p.parent.name for p, _ in checker.check_skills(tmp_path, [clean, also_broken])
    ] == ["also-broken"]


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

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/nope.md").write_text("now present\n")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout == ""


def test_root_without_skills_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="no skills found"):
        checker.check_skills(tmp_path)


def test_repository_skills_pass() -> None:
    assert checker.check_skills(REPO_ROOT) == []
