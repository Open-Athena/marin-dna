#!/usr/bin/env python3
"""Convert legacy research-question issue bodies into canonical Markdown documents.

This script reads issues through the GitHub CLI. It writes repository files only;
archiving comments, body notices, labels, and issue state remains a post-merge step.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class LegacyIssue:
    number: int
    title: str
    body: str
    state: str
    updated_at: str


def fetch_issue(repo: str, number: int) -> LegacyIssue:
    result = subprocess.run(
        [
            "gh",
            "issue",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "number,title,body,state,updatedAt",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return LegacyIssue(
        number=payload["number"],
        title=payload["title"],
        body=payload["body"] or "",
        state=payload["state"],
        updated_at=payload["updatedAt"],
    )


def extract_h2(body: str, heading: str) -> str:
    pattern = re.compile(
        rf"^## {re.escape(heading)}\s*$\n(.*?)(?=^## |\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(body)
    return match.group(1).strip() if match else ""


def extract_details(body: str, summary: str) -> str:
    pattern = re.compile(
        rf"<details>\s*<summary>{re.escape(summary)}</summary>\s*(.*?)\s*</details>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(body)
    return match.group(1).strip() if match else ""


def remove_details(value: str) -> str:
    return re.sub(
        r"<details>.*?</details>", "", value, flags=re.DOTALL | re.IGNORECASE
    ).strip()


def link_issue_references(value: str, repo: str) -> str:
    return re.sub(
        r"(?<![\w/\[])#(\d+)\b",
        lambda match: (
            f"[#{match.group(1)}](https://github.com/{repo}/issues/{match.group(1)})"
        ),
        value,
    )


def infer_confidence(body: str) -> str:
    levels: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        if "confidence" not in sentence.lower():
            continue
        for level in re.findall(
            r"\b(high|moderate|medium|low)\b", sentence, flags=re.IGNORECASE
        ):
            levels.add("medium" if level.lower() == "moderate" else level.lower())
    return levels.pop() if len(levels) == 1 else "unknown"


def confidence_and_limitations(tldr: str, current_answer: str, confidence: str) -> str:
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", f"{tldr}\n\n{current_answer}")
        if paragraph.strip()
    ]
    selected = [
        paragraph
        for paragraph in paragraphs
        if re.search(
            r"\b(confiden|limitation|confound|caveat|unknown|no evidence|not enough evidence|untested)\w*\b",
            paragraph,
            flags=re.IGNORECASE,
        )
    ]
    if selected:
        return "\n\n".join(dict.fromkeys(selected))
    return (
        f"Overall confidence is {confidence}. The predecessor issue did not maintain a separate "
        "limitations section; limitations remain embedded in the current answer and evidence ledger."
    )


def operational_consequence(current_answer: str) -> str:
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", current_answer)
        if paragraph.strip()
    ]
    if paragraphs:
        return paragraphs[0]
    return "No operational change was recorded in the predecessor issue."


def contradictory_evidence(current_answer: str, related_work: str) -> str:
    del current_answer, related_work
    return (
        "The predecessor issue did not maintain a separate contradictory-evidence section. "
        "Its caveats and negative results are preserved in Current answer and Supporting evidence."
    )


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:72].rstrip("-") or "question"


def render_document(
    issue: LegacyIssue, repo: str, migration_date: str
) -> tuple[str, str]:
    question_id = f"RQ-{issue.number:04d}"
    filename = f"{question_id.lower()}-{slugify(issue.title)}.md"
    tldr = extract_h2(issue.body, "TL;DR")
    question = extract_h2(issue.body, "Question") or issue.title
    current_answer = remove_details(extract_h2(issue.body, "Current answer"))
    related_work = extract_details(issue.body, "Related work")
    related_experiments = extract_details(issue.body, "Related experiments")
    open_questions = extract_h2(issue.body, "Open questions")
    confidence = infer_confidence(tldr or current_answer)
    issue_url = f"https://github.com/{repo}/issues/{issue.number}"
    status = "active" if issue.state.upper() == "OPEN" else "closed"

    current_text = "\n\n".join(part for part in (tldr, current_answer) if part).strip()
    if not current_text:
        current_text = "The predecessor issue did not record a current answer."
    support_text = (
        related_work
        or "No supporting evidence was listed in the predecessor issue body."
    )
    experiment_text = (
        related_experiments or "- None listed in the predecessor issue body."
    )
    if not re.search(r"^\s*-\s+", experiment_text, flags=re.MULTILINE):
        experiment_text = f"- {experiment_text}"
    open_text = (
        open_questions
        or "- No open questions were listed in the predecessor issue body."
    )
    if not re.search(r"^\s*(?:-|###)\s+", open_text, flags=re.MULTILINE):
        open_text = f"- {open_text}"

    document = f"""# {issue.title}

## Metadata

| Field | Value |
|---|---|
| Question ID | `{question_id}` |
| Status | `{status}` |
| Overall confidence | `{confidence}` |
| Evidence considered through | `{issue.updated_at[:10]}` |
| Predecessor issues | [#{issue.number}]({issue_url}) |

## Question and scope

{question}

## Current answer

{current_text}

## Confidence and limitations

{confidence_and_limitations(tldr, current_answer, confidence)}

## Operational consequence

{operational_consequence(current_answer)}

## Supporting evidence

{support_text}

## Contradictory evidence

{contradictory_evidence(current_answer, related_work)}

## Related experiments

{experiment_text}

## Open questions

{open_text}

## History

- {migration_date} — Migrated from the predecessor research-question issue [#{issue.number}]({issue_url}). The issue remains the historical source for its original body and comments.
"""
    document = document.replace(
        "This research-question issue does not authorize",
        "This question document does not authorize",
    )
    document = link_issue_references(document, repo)
    document = "\n".join(line.rstrip() for line in document.splitlines()) + "\n"
    return filename, document


def render_index(documents: list[tuple[str, str]]) -> str:
    rows: list[tuple[str, str, str, str, str, str]] = []
    for filename, text in documents:
        title = re.search(r"^# (.+)$", text, flags=re.MULTILINE).group(1)
        values = {
            key: value.strip("`")
            for key, value in re.findall(
                r"^\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*$", text, flags=re.MULTILINE
            )
            if key.strip() not in {"Field", "---"}
        }
        rows.append(
            (
                values["Question ID"],
                filename,
                title,
                values["Status"],
                values["Overall confidence"],
                values["Evidence considered through"],
            )
        )
    lines = [
        "# Research questions",
        "",
        "This index lists every canonical MarinDNA research-question document, including superseded and closed questions. See the [schema and workflow](README.md).",
        "",
        "| ID | Question | Status | Confidence | Evidence considered through |",
        "|---|---|---|---|---|",
    ]
    for question_id, filename, title, status, confidence, evidence_date in sorted(rows):
        lines.append(
            f"| `{question_id}` | [{title}]({filename}) | `{status}` | `{confidence}` | {evidence_date} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Open-Athena/marin-dna")
    parser.add_argument("--issue", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/research/questions"))
    parser.add_argument(
        "--migration-date", default=datetime.now(tz=UTC).date().isoformat()
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    documents = [
        render_document(fetch_issue(args.repo, number), args.repo, args.migration_date)
        for number in args.issue
    ]
    if not args.write:
        for filename, _ in documents:
            print(args.output / filename)
        print("Dry run only; pass --write to create documents and replace index.md.")
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    for filename, text in documents:
        (args.output / filename).write_text(text, encoding="utf-8")
    (args.output / "index.md").write_text(render_index(documents), encoding="utf-8")
    print(f"Wrote {len(documents)} documents and {args.output / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
