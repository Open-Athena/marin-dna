"""The scrub workflows stay in sync with the scrub skills they schedule."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[1]
WORKFLOW_PATHS = sorted(REPO_ROOT.glob(".github/workflows/scrub-*.yml"))

# UTC offsets for America/New_York: EDT and EST.
NEW_YORK_UTC_OFFSETS = (4, 5)

# GitHub App installation tokens live one hour; the job must finish inside it.
APP_TOKEN_LIFETIME_MINUTES = 60


def load_workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def skill_frontmatter(name: str) -> dict:
    text = (REPO_ROOT / ".agents/skills" / name / "SKILL.md").read_text(
        encoding="utf-8"
    )
    return yaml.safe_load(text.split("---")[1])


def test_both_scrub_workflows_exist() -> None:
    assert [path.stem for path in WORKFLOW_PATHS] == [
        "scrub-docs-code-parity",
        "scrub-reflection-self-improvement",
    ]


def test_each_scrub_workflow_matches_its_skill() -> None:
    for path in WORKFLOW_PATHS:
        skill = path.stem
        assert (REPO_ROOT / ".agents/skills" / skill / "SKILL.md").is_file()

        workflow = load_workflow(path)
        job = workflow["jobs"]["scrub"]
        assert job["timeout-minutes"] < APP_TOKEN_LIFETIME_MINUTES

        prompt = job["steps"][-1]["with"]["prompt"]
        assert f".agents/skills/{skill}/SKILL.md" in prompt

        # PyYAML reads the bare `on:` trigger key as boolean True.
        triggers = workflow.get("on") or workflow[True]
        workflow_cron = triggers["schedule"][0]["cron"].split()
        metadata = skill_frontmatter(skill)
        skill_cron = metadata["schedule_cron"].split()
        assert metadata["schedule_tz"] == "America/New_York"
        expected_utc_hours = {
            (int(skill_cron[1]) + offset) % 24 for offset in NEW_YORK_UTC_OFFSETS
        }
        assert int(workflow_cron[1]) in expected_utc_hours
        assert workflow_cron[2:] == skill_cron[2:] == ["*", "*", "*"]
