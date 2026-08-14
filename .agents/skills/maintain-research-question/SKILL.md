---
name: maintain-research-question
description: Maintain MarinDNA research-question Markdown documents as reviewed synthesis. Use when a human declares a research question, evidence or confidence may change its answer, experiments must be related to a question, a question is superseded or closed, or a legacy research-question issue must be migrated.
---

# Maintain Research Question

Maintain canonical research synthesis under `docs/research/questions/`. A document on `main` is the accepted answer. Experiment issues, logbooks, research branches, source ledgers, and open pull requests may contain newer evidence.

Do not create or maintain active research-question issues. Use GitHub issues for bounded experiments.

## Compose Existing Skills

- Read `docs/research/questions/README.md`, `index.md`, and `_template.md` for the schema and lifecycle.
- Use `background-research` for prior work, contradictions, negative search results, and source ledgers.
- Use `run-research` for bounded experiments and `task-logbook` for append-only records and issue updates.
- Use `update-docs` when accepted findings change durable guidance.
- Use `communicate-on-github` for experiment links, migration comments, and pull-request communication.
- Apply `writing-style`, `reference-docs.md`, `pull-requests.md`, and `ai-writing-donts.md`.
- Follow `AGENTS.md`, including protected evaluation splits and approval boundaries.

Do not copy those skills' literature, experiment, logbook, documentation, GitHub, or prose rules here.

## Promote Material Evidence

Open or revise a synthesis pull request when evidence changes the answer, a material claim, confidence, an important limitation, the operational consequence, scope, relationships, status, or successor. Keep routine progress, dense results, failures, and one-off analysis in experiment issues, logbooks, permanent branches, W&B, or another artifact store.

Humans approve new question scope and decide when a synthesis pull request may merge. Follow the repository's normal pull-request process and the explicit approval boundary in `AGENTS.md`.

## Create A Question

1. Confirm the human-approved question and exclusions.
2. Search active, superseded, and closed documents for overlapping scope.
3. Allocate the next unused `RQ-NNNN` identity. Never reuse or renumber an ID.
4. Copy `_template.md` to `rq-NNNN-short-slug.md` and fill every field and section.
5. Add the document to `index.md`.
6. Open a pull request and wait for human review before merge.

If the scope overlaps an existing question, show the overlap and ask whether to revise, supersede, or create a distinct document.

## Revise The Synthesis

1. Read the accepted document, open synthesis pull requests, linked experiments and comments, logbooks, permanent branches, source ledgers, and cited literature.
2. Record the evidence cutoff. Do not imply that `main` includes later evidence.
3. Separate supporting evidence, contradictory evidence, MarinDNA interpretation, and untested hypotheses.
4. Update the answer, confidence and limitations, operational consequence, open questions, metadata, and history together.
5. Preserve commit-pinned evidence and predecessor issue links. Link detailed results instead of copying them.
6. Update `index.md` when its displayed fields change.
7. Run the validator and open or update the pull request.

The branch and pull request hold the proposed answer until merge.

## Relate Experiments

Relationships are many-to-many links between documents and experiment issues.

1. Verify that each target is an issue with the `experiment` Kind label.
2. Keep the document's `Related experiments` list exhaustive and state each issue's contribution.
3. After the document merges, link the experiment issue to the canonical document on `main`.
4. Propose removals through a pull request, then update the issue after merge.

Never use GitHub sub-issue metadata for question-to-experiment relationships.

## Supersede Or Close

Set `Status` to `superseded` when a successor absorbs the useful scope, and link both documents in `History`. Set it to `closed` when the question is answered, abandoned, or no longer actionable. Retain the final synthesis, evidence, caveats, and stable ID.

## Validate

Run:

```bash
python3 .agents/skills/maintain-research-question/scripts/validate_research_questions.py --root .
```

## Migrate Legacy Issues

Migration is the only research-question issue mutation this skill performs.

1. Inventory open and closed issues carrying the historical `research-question` label.
2. Generate drafts with `scripts/migrate_legacy_research_questions.py`, supplying the explicit `--evidence-reviewed-through` cutoff. The helper refuses to overwrite existing documents by default and rebuilds the index from all question files.
3. Compare every draft with the complete predecessor body. Preserve conclusions, evidence, experiment links, open questions, and the predecessor URL. Replace the generated operational-consequence `TODO` with a concrete decision.
4. Open a migration pull request and wait for review and merge.
5. After merge, prepend this notice without changing the previous body below it:

   ```markdown
   > [!IMPORTANT]
   > This research question is archived. The synthesis is [RQ-NNNN: <title>](<canonical-main-url>). See <migration-pr-url> for the migration review. The original issue body and comments remain below as historical evidence.
   ```

6. Post one final comment beginning with `🤖` that links the document and migration pull request.
7. Close formerly open legacy issues. Add the notice to already closed issues without reopening them. Retain the historical label if useful for discovery.

Do not edit legacy issues before the document merges. Do not create a replacement tracker issue or rewrite predecessor comments.
