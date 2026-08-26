from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[1]
    / ".agents/skills/draft-weekly-research-update/scripts/draft_weekly_research_update.py"
)


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "draft_weekly_research_update", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


weekly = load_script()


def run_git(repo: Path, *args: str, commit_date: str | None = None) -> str:
    env = os.environ.copy()
    if commit_date is not None:
        env["GIT_AUTHOR_DATE"] = commit_date
        env["GIT_COMMITTER_DATE"] = commit_date
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.stdout


def commit_file(repo: Path, path: str, content: str, commit_date: str) -> None:
    destination = repo / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content)
    run_git(repo, "add", path)
    run_git(repo, "commit", "-m", f"Update {path}", commit_date=commit_date)


def test_extract_page_reads_multiline_tldr_callout() -> None:
    page = weekly.extract_page(
        "docs/research/experiments/123-example.md",
        """# Example experiment

> [!NOTE]
> **TL;DR:** The first sentence is retained.
> The second sentence is retained too.

## Findings
Ignored.
""",
    )

    assert page == weekly.ExperimentPage(
        path="docs/research/experiments/123-example.md",
        issue_number=123,
        title="Example experiment",
        tldr="The first sentence is retained.\nThe second sentence is retained too.",
    )


def test_extract_page_requires_title_and_tldr() -> None:
    assert weekly.extract_page("missing-title.md", "> **TL;DR:** Result.\n") is None
    assert weekly.extract_page("missing-tldr.md", "# Result\n") is None
    assert (
        weekly.extract_page(
            "docs/research/experiments/unnumbered.md",
            "# Result\n\n> **TL;DR:** Result.\n",
        )
        is None
    )


def test_weekly_boundaries_requires_monday() -> None:
    with pytest.raises(ValueError, match="must be a Monday"):
        weekly.weekly_boundaries(date(2026, 8, 18))


def test_new_experiment_paths_uses_weekly_main_snapshots(tmp_path: Path) -> None:
    run_git(tmp_path, "init", "--initial-branch=main")
    run_git(tmp_path, "config", "user.name", "Test User")
    run_git(tmp_path, "config", "user.email", "test@example.com")

    commit_file(
        tmp_path,
        "docs/research/experiments/100-existing.md",
        "# Existing\n\n> **TL;DR:** Old.\n",
        "2026-08-16T23:00:00+00:00",
    )
    commit_file(
        tmp_path,
        "docs/research/experiments/201-at-start.md",
        "# At start\n\n> **TL;DR:** At start.\n",
        "2026-08-17T00:00:00+00:00",
    )
    commit_file(
        tmp_path,
        "docs/research/experiments/200-new.md",
        "# New\n\n> **TL;DR:** New.\n",
        "2026-08-18T12:00:00+00:00",
    )
    commit_file(
        tmp_path,
        "docs/research/experiments/100-existing.md",
        "# Existing\n\n> **TL;DR:** Edited.\n",
        "2026-08-19T12:00:00+00:00",
    )
    commit_file(
        tmp_path,
        "docs/research/experiments/figures/200/plot.svg",
        "<svg />\n",
        "2026-08-20T12:00:00+00:00",
    )
    commit_file(
        tmp_path,
        "docs/research/experiments/299-at-end.md",
        "# At end\n\n> **TL;DR:** At end.\n",
        "2026-08-24T00:00:00+00:00",
    )
    commit_file(
        tmp_path,
        "docs/research/experiments/300-late.md",
        "# Late\n\n> **TL;DR:** Late.\n",
        "2026-08-24T01:00:00+00:00",
    )

    start, end = weekly.weekly_boundaries(date(2026, 8, 17))
    end_commit, paths = weekly.new_experiment_paths(tmp_path, "HEAD", start, end)

    assert paths == [
        "docs/research/experiments/200-new.md",
        "docs/research/experiments/201-at-start.md",
    ]
    assert (
        weekly.read_page_at_commit(tmp_path, end_commit, paths[0])
        == "# New\n\n> **TL;DR:** New.\n"
    )


def test_new_experiment_paths_excludes_restored_page(tmp_path: Path) -> None:
    run_git(tmp_path, "init", "--initial-branch=main")
    run_git(tmp_path, "config", "user.name", "Test User")
    run_git(tmp_path, "config", "user.email", "test@example.com")

    path = "docs/research/experiments/100-restored.md"
    content = "# Restored\n\n> **TL;DR:** Already announced.\n"
    commit_file(tmp_path, path, content, "2026-08-10T12:00:00+00:00")
    run_git(tmp_path, "rm", path)
    run_git(
        tmp_path,
        "commit",
        "-m",
        "Remove experiment page",
        commit_date="2026-08-16T12:00:00+00:00",
    )
    commit_file(tmp_path, path, content, "2026-08-18T12:00:00+00:00")

    start, end = weekly.weekly_boundaries(date(2026, 8, 17))
    _, paths = weekly.new_experiment_paths(tmp_path, "HEAD", start, end)

    assert paths == []


def test_new_experiment_paths_excludes_rename_with_detection_disabled(
    tmp_path: Path,
) -> None:
    run_git(tmp_path, "init", "--initial-branch=main")
    run_git(tmp_path, "config", "user.name", "Test User")
    run_git(tmp_path, "config", "user.email", "test@example.com")
    run_git(tmp_path, "config", "diff.renames", "false")

    old_path = "docs/research/experiments/100-old-name.md"
    new_path = "docs/research/experiments/100-new-name.md"
    commit_file(
        tmp_path,
        old_path,
        "# Renamed\n\n> **TL;DR:** Already announced.\n",
        "2026-08-16T12:00:00+00:00",
    )
    run_git(tmp_path, "mv", old_path, new_path)
    run_git(
        tmp_path,
        "commit",
        "-m",
        "Rename experiment page",
        commit_date="2026-08-18T12:00:00+00:00",
    )

    start, end = weekly.weekly_boundaries(date(2026, 8, 17))
    _, paths = weekly.new_experiment_paths(tmp_path, "HEAD", start, end)

    assert paths == []


def test_format_draft_no_content_and_links_canonical_page() -> None:
    pages = [
        weekly.ExperimentPage(
            path="docs/research/experiments/20-second.md",
            issue_number=20,
            title="Second result",
            tldr="A result.",
        )
    ]

    assert weekly.format_draft(
        date(2026, 8, 17),
        pages,
        "https://github.com/Open-Athena/marin-dna",
        "main",
    ) == (
        "# MarinDNA research updates — week of August 17, 2026\n\n"
        "> **Weekly blurb:** _Add a short human-written introduction before "
        "publishing._\n\n"
        "**[Second result (#20)](https://github.com/Open-Athena/marin-dna/blob/main/"
        "docs/research/experiments/20-second.md)**\n\n"
        "A result.\n"
    )


def test_write_temporary_draft_creates_markdown_file(tmp_path: Path) -> None:
    draft = "# Weekly draft\n"

    output = weekly.write_temporary_draft(date(2026, 8, 17), draft, directory=tmp_path)

    assert output.parent == tmp_path
    assert output.name.startswith("marindna-research-updates-2026-08-17-")
    assert output.suffix == ".md"
    assert output.read_text() == draft
    assert (
        weekly.format_draft(
            date(2026, 8, 17), [], "https://github.com/Open-Athena/marin-dna", "main"
        )
        == ""
    )


def test_experiment_sort_key_uses_numeric_prefix() -> None:
    paths = [
        "docs/research/experiments/100-later.md",
        "docs/research/experiments/20-earlier.md",
        "docs/research/experiments/unnumbered.md",
    ]

    assert sorted(paths, key=weekly.experiment_sort_key) == [
        "docs/research/experiments/20-earlier.md",
        "docs/research/experiments/100-later.md",
        "docs/research/experiments/unnumbered.md",
    ]
