from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2]
    / ".agents/skills/maintain-research-question/scripts/migrate_legacy_research_questions.py"
)
SPEC = importlib.util.spec_from_file_location(
    "migrate_legacy_research_questions", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MIGRATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MIGRATION
SPEC.loader.exec_module(MIGRATION)


def legacy_issue() -> object:
    return MIGRATION.LegacyIssue(
        number=12,
        title="Does migration preserve evidence?",
        body="""## TL;DR

Keep the current method. Confidence is moderate.

## Question

Does the method work with `<dna>` tags?

## Current answer

Keep the method while #34 supplies the only direct evidence.

This research-question issue does not authorize a paid run.

<details>
<summary>Related work</summary>

- Source A supports the method but has one limitation.

</details>

<details>
<summary>Related experiments</summary>

- #34 is the direct experiment.

</details>

## Open questions

- Does the result replicate?
""",
        state="CLOSED",
        updated_at="2026-08-13T12:00:00Z",
    )


def test_render_document_preserves_links_and_minimal_metadata() -> None:
    filename, text = MIGRATION.render_document(
        legacy_issue(),
        "Open-Athena/marin-dna",
        "2026-08-14",
    )

    assert filename == "rq-0012-does-migration-preserve-evidence.md"
    assert "| Status | `closed` |" in text
    assert "| Overall confidence | `medium` |" in text
    assert "Owners" not in text
    assert "Last human review" not in text
    assert "[#12](https://github.com/Open-Athena/marin-dna/issues/12)" in text
    assert "[[#12]" not in text
    assert "[#34](https://github.com/Open-Athena/marin-dna/issues/34)" in text
    assert "`<dna>`" in text
    assert "This question document does not authorize a paid run." in text
    assert "This research-question issue does not authorize" not in text
    assert all(line == line.rstrip() for line in text.splitlines())


def test_render_index_uses_evidence_cutoff() -> None:
    document = MIGRATION.render_document(
        legacy_issue(),
        "Open-Athena/marin-dna",
        "2026-08-14",
    )

    index = MIGRATION.render_index([document])

    assert (
        "| ID | Question | Status | Confidence | Evidence considered through |" in index
    )
    assert "rq-0012-does-migration-preserve-evidence.md" in index
    assert "2026-08-13" in index


def test_mixed_confidence_is_unknown() -> None:
    assert (
        MIGRATION.infer_confidence("Confidence is high for A and low for B.")
        == "unknown"
    )
