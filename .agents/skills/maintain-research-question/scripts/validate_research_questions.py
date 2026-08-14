#!/usr/bin/env python3
"""Validate MarinDNA research-question documents."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import unquote

QUESTION_DIRECTORY = Path("docs/research/questions")
QUESTION_FILE_PATTERN = re.compile(r"rq-\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*\.md")
QUESTION_ID_PATTERN = re.compile(r"RQ-\d{4}")
ALLOWED_STATUSES = {"active", "superseded", "closed"}
ALLOWED_CONFIDENCE = {"low", "medium", "high", "unknown"}
REQUIRED_METADATA = (
    "Question ID",
    "Status",
    "Overall confidence",
    "Evidence considered through",
    "Predecessor issues",
)
REQUIRED_SECTIONS = (
    "Metadata",
    "Question and scope",
    "Current answer",
    "Confidence and limitations",
    "Operational consequence",
    "Supporting evidence",
    "Contradictory evidence",
    "Related experiments",
    "Open questions",
    "History",
)
PLACEHOLDER_PATTERN = re.compile(
    r"<[^>\n]*\s+[^>\n]*>|\b(?:TODO|TBD|YYYY-MM-DD)\b", re.IGNORECASE
)
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
EXPERIMENT_ISSUE_LINK_PATTERN = re.compile(
    r"\[[^\]\n]+\]\(https://github\.com/Open-Athena/marin-dna/issues/\d+\)"
)
EXPERIMENT_ISSUE_TARGET_PATTERN = re.compile(
    r"https://github\.com/Open-Athena/marin-dna/issues/\d+"
)
INDEX_ROW_PATTERN = re.compile(
    r"^\|\s*`(?P<question_id>RQ-\d{4})`\s*\|\s*"
    r"\[(?P<title>[^\]]+)\]\((?:\./)?"
    r"(?P<filename>rq-\d{4}-[a-z0-9-]+\.md)\)\s*\|\s*"
    r"`(?P<status>[^`]+)`\s*\|\s*"
    r"`(?P<confidence>[^`]+)`\s*\|\s*"
    r"(?P<evidence_date>\d{4}-\d{2}-\d{2})\s*\|\s*$"
)


@dataclass(frozen=True)
class QuestionDocument:
    path: Path
    title: str
    metadata: dict[str, str]
    sections: dict[str, str]


def _strip_comments(value: str) -> str:
    return re.sub(r"<!--.*?-->", "", value, flags=re.DOTALL).strip()


def _heading_blocks(text: str, level: int) -> list[tuple[str, str]]:
    marker = "#" * level
    pattern = re.compile(rf"^{marker} (.+?)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    return [
        (
            match.group(1).strip(),
            text[
                match.end() : matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            ].strip(),
        )
        for index, match in enumerate(matches)
    ]


def _parse_metadata(block: str) -> tuple[dict[str, str], list[str]]:
    metadata: dict[str, str] = {}
    errors: list[str] = []
    for line in block.splitlines():
        match = re.match(r"^\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*$", line)
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip().strip("`")
        if key.lower() == "field" or set(key) <= {"-", ":"}:
            continue
        if key in metadata:
            errors.append(f"metadata field {key!r} appears more than once")
        metadata[key] = value
    return metadata, errors


def parse_question_document(path: Path) -> tuple[QuestionDocument, list[str]]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    title_matches = re.findall(r"^# ([^#].+?)\s*$", text, flags=re.MULTILINE)
    if len(title_matches) != 1:
        errors.append(f"expected one level-one title, found {len(title_matches)}")
        title = ""
    else:
        title = title_matches[0].strip()

    blocks = _heading_blocks(text, 2)
    section_names = [name for name, _ in blocks]
    if section_names != list(REQUIRED_SECTIONS):
        errors.append(
            "level-two sections must appear exactly in this order: "
            + ", ".join(REQUIRED_SECTIONS)
        )
    if len(section_names) != len(set(section_names)):
        errors.append("level-two section names must be unique")

    sections = dict(blocks)
    metadata, metadata_errors = _parse_metadata(sections.get("Metadata", ""))
    errors.extend(metadata_errors)
    for field in REQUIRED_METADATA:
        value = metadata.get(field, "")
        if not value:
            errors.append(f"missing metadata field {field!r}")
        elif PLACEHOLDER_PATTERN.search(value):
            errors.append(f"metadata field {field!r} contains a placeholder")

    question_id = metadata.get("Question ID", "")
    if question_id and not QUESTION_ID_PATTERN.fullmatch(question_id):
        errors.append("Question ID must match RQ-NNNN")
    if question_id and not path.name.startswith(question_id.lower() + "-"):
        errors.append(
            "filename must start with the lowercase Question ID followed by a hyphen"
        )

    status = metadata.get("Status", "").lower()
    if status and status not in ALLOWED_STATUSES:
        errors.append(f"Status must be one of {sorted(ALLOWED_STATUSES)}")
    confidence = metadata.get("Overall confidence", "").lower()
    if confidence and confidence not in ALLOWED_CONFIDENCE:
        errors.append(f"Overall confidence must be one of {sorted(ALLOWED_CONFIDENCE)}")
    evidence_date = metadata.get("Evidence considered through", "")
    if evidence_date:
        try:
            date.fromisoformat(evidence_date)
        except ValueError:
            errors.append("Evidence considered through must be an ISO date")

    for section in REQUIRED_SECTIONS[1:]:
        value = _strip_comments(sections.get(section, ""))
        if not value:
            errors.append(f"section {section!r} must not be empty")
        elif PLACEHOLDER_PATTERN.search(value):
            errors.append(f"section {section!r} contains a placeholder")

    related_experiments = sections.get("Related experiments", "")
    bullets = [
        line for line in related_experiments.splitlines() if re.match(r"^\s*-\s+", line)
    ]
    if not bullets:
        errors.append("Related experiments must contain a Markdown bullet")
    for bullet in bullets:
        if re.match(r"^\s*-\s+None(?:\.|\s|$)", bullet, flags=re.IGNORECASE):
            continue
        targets = MARKDOWN_LINK_PATTERN.findall(bullet)
        if not any(
            EXPERIMENT_ISSUE_TARGET_PATTERN.fullmatch(target) for target in targets
        ):
            errors.append(
                "each Related experiments bullet must contain an exact Markdown link "
                "to an Open-Athena/marin-dna issue"
            )
            continue
        contribution = EXPERIMENT_ISSUE_LINK_PATTERN.sub("", bullet)
        contribution = re.sub(r"^\s*-\s*", "", contribution).strip(" \t—–-:.;")
        if not contribution:
            errors.append(
                "each Related experiments bullet must state the experiment's contribution"
            )

    return QuestionDocument(
        path=path, title=title, metadata=metadata, sections=sections
    ), errors


def _validate_internal_links(path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    for raw_target in MARKDOWN_LINK_PATTERN.findall(text):
        target = raw_target.strip().strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        relative_target = unquote(target.split("#", 1)[0].split("?", 1)[0])
        resolved = (path.parent / relative_target).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            errors.append(f"internal link escapes the repository: {target}")
            continue
        if not resolved.exists():
            errors.append(f"broken internal link: {target}")
    return errors


def _validate_index(
    question_directory: Path, documents: list[QuestionDocument]
) -> list[str]:
    index_path = question_directory / "index.md"
    if not index_path.exists():
        return ["docs/research/questions/index.md is missing"]
    text = index_path.read_text(encoding="utf-8")
    linked_names = re.findall(
        r"\]\((?:\./)?(rq-\d{4}-[a-z0-9-]+\.md)(?:#[^)]+)?\)", text
    )
    rows = [
        match.groupdict()
        for line in text.splitlines()
        if (match := INDEX_ROW_PATTERN.fullmatch(line))
    ]
    rows_by_name: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_name.setdefault(row["filename"], []).append(row)
    errors: list[str] = []
    expected_names = {document.path.name for document in documents}
    linked_set = set(linked_names)
    for name in sorted(expected_names - linked_set):
        errors.append(f"index is missing {name}")
    for name in sorted(linked_set - expected_names):
        errors.append(f"index links non-question document {name}")
    for name in sorted(linked_set):
        count = linked_names.count(name)
        if count != 1:
            errors.append(f"index must link {name} exactly once, found {count}")
    for document in documents:
        name = document.path.name
        matching_rows = rows_by_name.get(name, [])
        if len(matching_rows) != 1:
            errors.append(
                f"index entry for {name} must be one canonical metadata table row"
            )
            continue
        row = matching_rows[0]
        expected = {
            "question_id": document.metadata.get("Question ID", ""),
            "title": document.title,
            "status": document.metadata.get("Status", ""),
            "confidence": document.metadata.get("Overall confidence", ""),
            "evidence_date": document.metadata.get("Evidence considered through", ""),
        }
        for field, expected_value in expected.items():
            if row[field] != expected_value:
                errors.append(
                    f"index {field.replace('_', ' ')} for {name} is {row[field]!r}; "
                    f"document has {expected_value!r}"
                )
    return errors


def validate_repository(root: Path) -> list[str]:
    question_directory = root / QUESTION_DIRECTORY
    errors: list[str] = []
    if not question_directory.is_dir():
        return [f"{QUESTION_DIRECTORY} is missing"]

    paths = sorted(
        path for path in question_directory.glob("rq-*.md") if path.is_file()
    )
    documents: list[QuestionDocument] = []
    question_ids: dict[str, Path] = {}
    for path in paths:
        relative = path.relative_to(root)
        if not QUESTION_FILE_PATTERN.fullmatch(path.name):
            errors.append(f"{relative}: filename must match rq-NNNN-short-slug.md")
        document, document_errors = parse_question_document(path)
        errors.extend(f"{relative}: {error}" for error in document_errors)
        documents.append(document)
        question_id = document.metadata.get("Question ID", "")
        if question_id in question_ids:
            errors.append(
                f"{relative}: duplicate Question ID {question_id!r}; first used by "
                f"{question_ids[question_id].relative_to(root)}"
            )
        elif question_id:
            question_ids[question_id] = path

    errors.extend(_validate_index(question_directory, documents))
    for path in sorted(question_directory.glob("*.md")):
        if path.name == "_template.md":
            continue
        errors.extend(
            f"{path.relative_to(root)}: {error}"
            for error in _validate_internal_links(path, root)
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.root.resolve()
    errors = validate_repository(root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Validated research synthesis in {root / QUESTION_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
