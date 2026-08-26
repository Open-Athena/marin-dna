"""Build a weekly draft from newly added research experiment pages."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from zoneinfo import ZoneInfo

EXPERIMENTS_DIR = PurePosixPath("docs/research/experiments")
TITLE_RE = re.compile(r"^#\s+(.+?)\s*$")
TLDR_RE = re.compile(r"^>\s*\*\*TL;DR:\*\*\s*(.*)$")
EXPERIMENT_NUMBER_RE = re.compile(r"^(\d+)-")
REPORT_TIMEZONE = ZoneInfo("UTC")
BLURB_PLACEHOLDER = (
    "> **Blurb (replace with contributor name):** _Add a short human-written note "
    "about this experiment before publishing._"
)


@dataclass(frozen=True)
class ExperimentPage:
    """The public fields extracted from one canonical experiment page."""

    path: str
    issue_number: int
    title: str
    tldr: str


def run_git(repo_root: Path, *args: str) -> str:
    """Run Git and return decoded standard output."""

    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def weekly_boundaries(week_start: date) -> tuple[datetime, datetime]:
    """Return the inclusive start and exclusive end of a UTC week."""

    if week_start.weekday() != 0:
        raise ValueError(f"Week start must be a Monday: {week_start.isoformat()}")
    start = datetime.combine(week_start, time.min, tzinfo=REPORT_TIMEZONE)
    return start, start + timedelta(days=7)


def commit_before(repo_root: Path, ref: str, boundary: datetime) -> str:
    """Resolve the last first-parent commit before a time boundary."""

    latest_included = boundary - timedelta(seconds=1)
    commit = run_git(
        repo_root,
        "rev-list",
        "-1",
        "--first-parent",
        f"--before={latest_included.isoformat()}",
        ref,
    ).strip()
    if not commit:
        raise RuntimeError(f"No commit on {ref} exists before {boundary.isoformat()}")
    return commit


def new_experiment_paths(
    repo_root: Path,
    ref: str,
    start: datetime,
    end: datetime,
) -> tuple[str, list[str]]:
    """Return the end snapshot and experiment pages added during the window."""

    start_commit = commit_before(repo_root, ref, start)
    end_commit = commit_before(repo_root, ref, end)
    if start_commit == end_commit:
        return end_commit, []

    changed = run_git(
        repo_root,
        "diff",
        "--name-status",
        "--find-renames",
        "--diff-filter=A",
        start_commit,
        end_commit,
        "--",
        str(EXPERIMENTS_DIR),
    )

    paths: list[str] = []
    for line in changed.splitlines():
        status, separator, raw_path = line.partition("\t")
        if status != "A" or not separator:
            continue
        path = PurePosixPath(raw_path)
        if (
            path.parent == EXPERIMENTS_DIR
            and path.suffix == ".md"
            and not path_was_previously_added(repo_root, start_commit, raw_path)
        ):
            paths.append(raw_path)

    return end_commit, sorted(paths, key=experiment_sort_key)


def path_was_previously_added(repo_root: Path, commit: str, path: str) -> bool:
    """Return whether a path was added anywhere in the snapshot's ancestry."""

    additions = run_git(
        repo_root,
        "log",
        "--first-parent",
        "--diff-filter=A",
        "--format=%H",
        commit,
        "--",
        path,
    )
    return bool(additions.strip())


def experiment_sort_key(path: str) -> tuple[int, int | str]:
    """Sort numbered experiment pages numerically, then any fallback paths."""

    filename = PurePosixPath(path).name
    match = EXPERIMENT_NUMBER_RE.match(filename)
    if match:
        return 0, int(match.group(1))
    return 1, path


def extract_page(path: str, markdown: str) -> ExperimentPage | None:
    """Extract the H1 title and canonical TL;DR callout text."""

    issue_match = EXPERIMENT_NUMBER_RE.match(PurePosixPath(path).name)
    if issue_match is None:
        return None

    lines = markdown.splitlines()
    title = next(
        (match.group(1) for line in lines if (match := TITLE_RE.match(line))), None
    )
    if title is None:
        return None

    for index, line in enumerate(lines):
        match = TLDR_RE.match(line)
        if match is None:
            continue

        tldr_lines = [match.group(1)]
        for continuation in lines[index + 1 :]:
            if not continuation.startswith(">"):
                break
            text = continuation[1:]
            text = text.removeprefix(" ")
            tldr_lines.append(text)

        while tldr_lines and not tldr_lines[-1]:
            tldr_lines.pop()
        tldr = "\n".join(tldr_lines).strip()
        if tldr:
            return ExperimentPage(
                path=path,
                issue_number=int(issue_match.group(1)),
                title=title,
                tldr=tldr,
            )
        return None

    return None


def read_page_at_commit(repo_root: Path, commit: str, path: str) -> str:
    """Read one page from a Git snapshot."""

    return run_git(repo_root, "show", f"{commit}:{path}")


def canonical_page_url(repository_url: str, branch: str, path: str) -> str:
    """Build the moving canonical GitHub URL for an experiment page."""

    base = repository_url.rstrip("/")
    return f"{base}/blob/{quote(branch, safe='')}/{quote(path, safe='/')}"


def format_draft(
    week_start: date,
    pages: list[ExperimentPage],
    repository_url: str,
    branch: str,
) -> str:
    """Format the Discord-ready Markdown draft."""

    if not pages:
        return ""

    date_label = f"{week_start.strftime('%B')} {week_start.day}, {week_start.year}"
    blocks = [f"# MarinDNA research updates — week of {date_label}"]
    for page in pages:
        url = canonical_page_url(repository_url, branch, page.path)
        blocks.append(
            f"**[{page.title} (#{page.issue_number})]({url})**\n\n"
            f"{BLURB_PLACEHOLDER}\n\n{page.tldr}"
        )
    return "\n\n".join(blocks) + "\n"


def write_temporary_draft(
    week_start: date,
    draft: str,
    directory: Path | None = None,
) -> Path:
    """Write a draft to a private temporary Markdown file."""

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"marindna-research-updates-{week_start.isoformat()}-",
        suffix=".md",
        dir=directory,
        delete=False,
    ) as output:
        output.write(draft)
        return Path(output.name)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--week-start",
        required=True,
        type=date.fromisoformat,
        help="Monday starting the report window, in YYYY-MM-DD form.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ref", default="origin/main")
    parser.add_argument("--branch", default="main")
    parser.add_argument(
        "--repository-url", default="https://github.com/Open-Athena/marin-dna"
    )
    return parser.parse_args()


def main() -> int:
    """Write the draft to a temporary file and print its path."""

    args = parse_args()
    start, end = weekly_boundaries(args.week_start)
    end_commit, paths = new_experiment_paths(args.repo_root, args.ref, start, end)

    pages: list[ExperimentPage] = []
    for path in paths:
        page = extract_page(path, read_page_at_commit(args.repo_root, end_commit, path))
        if page is None:
            print(
                f"WARNING: omitted {path}: missing issue-number filename, H1 title, "
                "or TL;DR callout",
                file=sys.stderr,
            )
            continue
        pages.append(page)

    draft = format_draft(
        week_start=args.week_start,
        pages=pages,
        repository_url=args.repository_url,
        branch=args.branch,
    )
    if draft:
        print(write_temporary_draft(args.week_start, draft))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
