---
name: maintain-research-question
description: Maintain MarinDNA's active research-question Markdown documents and root README index. Use when a human declares a question, new evidence may change an answer, an experiment must be related to a question, a question may leave the active set, or priorities change.
---

# Maintain Research Questions

Keep active research synthesis in `docs/research/questions/`. Treat a document on `main` as the accepted answer. Use experiment issues, logbooks, research branches, and source ledgers for the detailed research record.

## Respect Human Decisions

- Obtain explicit human approval before creating a question or changing its scope.
- Ask before removing a question from the active set.
- Change `Current priorities` only when a human directs the change. Agents do not assign, infer, promote, demote, or rank priorities.
- Treat importance to MarinDNA's current thinking as the inclusion criterion. A question may remain active while its experiments are paused.

## Write The Document

Use a short topic filename such as `evolutionary-timescale.md`. Do not add IDs, status, confidence, dates, predecessor metadata, or history sections.

Keep this structure:

```markdown
# <Question title>

## TL;DR

<Short current answer.>

## Question

<Precise question and scope.>

## Current answer

<Current synthesis, uncertainty, and caveats.>

<details>
<summary>Related work</summary>

<Curated external work that materially informs the current answer.>

</details>

<details>
<summary>Related experiments</summary>

<Exhaustive linked experiments and what each contributes.>

</details>

## Possible directions

<Curated promising ways to reduce the important uncertainty.>
```

Keep `Related work` curated. Include counterevidence when it materially affects the answer. Keep `Related experiments` exhaustive: include experiments created for the question and experiments later found to materially inform it. Use explicit clickable MarinDNA issue links.

Treat `Possible directions` as suggestions, not commitments or a chronological backlog.

## Create Or Revise A Question

1. Confirm the human-approved question and exclusions.
2. Search the active documents and Git history for overlapping scope.
3. Read the accepted document, open synthesis pull requests, linked experiments and comments, relevant logbooks and branches, source ledgers, and material external work.
4. Preserve the existing format and human-authored content unless the evidence requires a change.
5. Update the TL;DR, current answer, related work, related experiments, and possible directions together when needed.
6. Update the root README and open a pull request.

Agents may independently open a synthesis pull request when evidence materially changes the answer, an important caveat, or the most promising directions. Keep routine progress, dense results, and failures in experiment issues and logbooks.

## Maintain The Root Index

Use the root `README.md` as the sole active-question index. List every active question once under either `Current priorities` or `Other active questions`. Keep each entry to one linked line and keep both lists alphabetical.

## Relate Experiments

Treat question-to-experiment relationships as many-to-many.

1. Verify that each target is an issue with the `experiment` Kind label.
2. Add every materially informative experiment to the document's collapsed `Related experiments` section.
3. After the document change merges, add a link to the canonical document on `main` to the experiment issue.
4. Remove a relationship through the document's normal pull-request workflow, then update the issue after merge.

## Remove A Question

Ask the human before removal. Remove the document and its root README entry through a pull request. Use Git history as the record of past synthesis; do not keep inactive question documents solely as an archive.

## Compose Existing Skills

- Use `background-research` for prior work, contradictions, negative searches, and source ledgers.
- Use `run-research` and `task-logbook` for bounded research records.
- Use `update-docs` for durable guidance.
- Use `communicate-on-github` for experiment links, migration comments, and pull-request communication.
- Apply `writing-style` to prose and GitHub communication.
