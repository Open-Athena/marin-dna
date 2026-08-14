from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2]
    / ".agents/skills/maintain-research-question/scripts/validate_research_questions.py"
)
SPEC = importlib.util.spec_from_file_location("validate_research_questions", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


def question_text(question_id: str = "RQ-0001") -> str:
    return f"""# Does the test question work?

## Metadata

| Field | Value |
|---|---|
| Question ID | `{question_id}` |
| Status | `active` |
| Overall confidence | `medium` |
| Evidence considered through | `2026-08-13` |
| Predecessor issues | None |

## Question and scope

Test the maintained schema without reading external evidence.

## Current answer

The schema is testable.

## Confidence and limitations

Confidence is medium because this is a fixture.

## Operational consequence

Run the validator in CI.

## Supporting evidence

- [Local schema](README.md) defines the checks.

## Contradictory evidence

- None found through 2026-08-13.

## Related experiments

- [#123](https://github.com/Open-Athena/marin-dna/issues/123) — Exercises the schema.

## Open questions

- Whether another field will be needed.

## History

- 2026-08-14 — Created for a validator test.
"""


def index_text() -> str:
    return """# Research questions

| ID | Question | Status | Confidence | Evidence considered through |
|---|---|---|---|---|
| `RQ-0001` | [Does the test question work?](rq-0001-test-question.md) | `active` | `medium` | 2026-08-13 |
"""


def write_repository(root: Path) -> Path:
    directory = root / "docs/research/questions"
    directory.mkdir(parents=True)
    path = directory / "rq-0001-test-question.md"
    path.write_text(question_text(), encoding="utf-8")
    (directory / "index.md").write_text(
        index_text(),
        encoding="utf-8",
    )
    (directory / "README.md").write_text("# Schema\n", encoding="utf-8")
    return path


def test_valid_repository(tmp_path: Path) -> None:
    write_repository(tmp_path)

    assert VALIDATOR.validate_repository(tmp_path) == []


def test_duplicate_question_id_is_rejected(tmp_path: Path) -> None:
    first = write_repository(tmp_path)
    second = first.parent / "rq-0001-second-question.md"
    second.write_text(question_text(), encoding="utf-8")

    errors = VALIDATOR.validate_repository(tmp_path)

    assert any("duplicate Question ID" in error for error in errors)
    assert any(
        "index is missing rq-0001-second-question.md" in error for error in errors
    )


def test_broken_internal_link_is_rejected(tmp_path: Path) -> None:
    path = write_repository(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "README.md",
            "missing-evidence.md",
        ),
        encoding="utf-8",
    )

    errors = VALIDATOR.validate_repository(tmp_path)

    assert any("broken internal link: missing-evidence.md" in error for error in errors)


def test_bare_index_link_is_rejected(tmp_path: Path) -> None:
    path = write_repository(tmp_path)
    (path.parent / "index.md").write_text(
        "# Research questions\n\n"
        "[Does the test question work?](rq-0001-test-question.md)\n",
        encoding="utf-8",
    )

    errors = VALIDATOR.validate_repository(tmp_path)

    assert any("must be one canonical metadata table row" in error for error in errors)


def test_stale_index_metadata_is_rejected(tmp_path: Path) -> None:
    path = write_repository(tmp_path)
    index_path = path.parent / "index.md"
    index_path.write_text(
        index_path.read_text(encoding="utf-8").replace("`medium`", "`low`"),
        encoding="utf-8",
    )

    errors = VALIDATOR.validate_repository(tmp_path)

    assert any(
        "index confidence" in error and "document has 'medium'" in error
        for error in errors
    )


def test_malformed_experiment_issue_link_is_rejected(tmp_path: Path) -> None:
    path = write_repository(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "https://github.com/Open-Athena/marin-dna/issues/123",
            "https://github.com/Open-Athena/marin-dna/pull/123",
        ),
        encoding="utf-8",
    )

    errors = VALIDATOR.validate_repository(tmp_path)

    assert any("exact Markdown link" in error for error in errors)


def test_experiment_contribution_is_required(tmp_path: Path) -> None:
    path = write_repository(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(" — Exercises the schema.", ""),
        encoding="utf-8",
    )

    errors = VALIDATOR.validate_repository(tmp_path)

    assert any("must state the experiment's contribution" in error for error in errors)
