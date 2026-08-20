---
name: maintain-knowledge-base
description: Maintain MarinDNA's research knowledge base in docs/research/questions and docs/research/experiments, plus the root README question index. Use when a human declares a question, an experiment has an accepted interpretation, new evidence may change an answer or experiment interpretation, an experiment must be related to a question, a question may leave the active set, or priorities change.
---

# Maintain The Research Knowledge Base

Treat research documents on `main` as MarinDNA's current accepted knowledge:

- `docs/research/questions/` synthesizes the current answer across internal experiments and external work.
- `docs/research/experiments/` records the accepted interpretation of a stable empirical investigation.

Keep chronology, progress, commands, superseded paths, and dense results in experiment issues, comments, logbooks, research branches, W&B, and source ledgers.
Rewrite knowledge-base pages when understanding changes; Git history preserves earlier accepted versions.

## Respect Human Decisions

- Obtain explicit human approval before creating a question or changing its scope.
- Ask before removing a question from the active set.
- Change `Current priorities` only when a human directs the change.
  Agents do not assign, infer, promote, demote, or rank priorities.
- Treat importance to MarinDNA's current thinking as the inclusion criterion.
  A question may remain active while its experiments are paused.

## Write The Document

Use a short topic filename such as `evolutionary-timescale.md`.
Do not add IDs, status, confidence, dates, predecessor metadata, or history sections.

Keep this structure:

```markdown
# <Question title>

> [!NOTE]
> **TL;DR:** <Concise current answer on exactly one source line.>

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

<details>
<summary>Possible directions</summary>

<Curated promising ways to reduce the important uncertainty.>

</details>
```

Write the TL;DR as exactly one source line after the callout marker.
Keep the answer concise enough to scan as a summary; do not move evidence or extended caveats into the callout.

Keep `Related work` curated.
Include counterevidence when it materially affects the answer.
Keep `Related experiments` exhaustive over accepted experiment pages that materially inform the question and ongoing experiment issues created for it.
Use explicit clickable MarinDNA links.

Keep `Related work`, `Related experiments`, and `Possible directions` collapsed.
Treat `Possible directions` as suggestions, not commitments or a chronological backlog.

## Write An Experiment Page

Create `docs/research/experiments/<issue>-<short-slug>.md` when a completed experiment produces valid scientific evidence, including a well-designed null or negative result.
Use the originating experiment issue number as the stable identifier.
An experiment may be a training comparison, dataset analysis, evaluation, probing study, robustness study, or another bounded empirical investigation.

Keep one page while the empirical object, primary question, and interpretation boundary remain stable.
Update it through later pull requests when new evaluations add evidence to the same investigation.
Create a new experiment when the intervention, estimand, or primary claim changes.
Do not create one page per training run, checkpoint, metric, or evaluation episode.

Use this structure:

```markdown
# <Experiment title>

> [!NOTE]
> **TL;DR:** <Current accepted interpretation on exactly one source line.>

![<Accessible description of the lead figure>](figures/<issue>/<figure>.svg)

_<One-line caption stating what the figure shows and its inferential boundary.>_

## Findings

<Accepted claims and their scope.>

## Evidence

<The design, controls, datasets, metrics, and results needed to evaluate the claims.>

## Limitations

<Confounders, uncertainty, missing controls, and unsupported claims.>

## Research record

<Canonical experiment issue.>
```

Write the page as current knowledge, not as a chronology or an exhaustive report.
Do not add the original question, status, progress, iteration history, decision logs, operational failures, or next-action sections.
Include enough quantitative evidence and setup to review the findings.
Treat inferential scope as part of the findings and limitations.

Include one reviewed lead figure immediately after the TL;DR.
Use the decisive result plot for a simple experiment and a multi-panel graphical abstract when the conclusion depends on several forms of evidence.
A graphical abstract should communicate the setup, decisive evidence, finding, and main boundary.
Include additional figures when they materially support the accepted findings.
Commit every referenced figure as SVG under `docs/research/experiments/figures/<issue>/` on `main`, and use relative links from the experiment page.
Keep unused, superseded, exploratory, and dense diagnostic figures in the issue or experiment branch.
Add a PNG only for a downstream sharing surface that needs one.
Check that each figure agrees with the prose, labels units and uncertainty, uses accessible colors, has useful alt text, and remains legible in rendered Markdown.

Distinguish direct measurements, synthesized comparisons across prior work, and attributed expert judgment.
Human judgment may support the accepted interpretation when the page names the researcher and basis.
Do not present an attributed assessment as a controlled measurement.

Under `Research record`, link only the canonical experiment issue by default.
The issue owns the complete provenance chain to code, data, runs, artifacts, and discussion.
List multiple source issues only when independent records jointly support the page and no single coordinating issue owns the provenance.

Use these disposition rules:

- Assess validity per claim.
  A design may support measurements or secondary findings while failing to support its intended primary inference.
- Promote each valid positive, null, or negative claim.
  For a partially informative design, state unsupported intended inferences under limitations.
- Use no promotion only when no scientifically valid claim remains.
  State why and obtain human approval for that disposition before closure.
- Promote a reusable lesson from an invalid design to the nearest methodology, experiment, or question document without promoting the invalid result.
- If accepted evidence is later invalidated, replace the existing page with a concise warning that states why no inference is valid, update every affected question, and preserve the page path for existing links.

## Create Or Revise A Question

1. Confirm the human-approved question and exclusions.
2. Search the active documents and Git history for overlapping scope.
3. Read the accepted document, accepted experiment pages, open synthesis pull requests, linked experiment issues and comments, relevant logbooks and branches, source ledgers, and material external work.
4. Preserve the existing format and human-authored content unless the evidence requires a change.
5. Update the one-line TL;DR, current answer, related work, related experiments, and possible directions together when needed.
6. Update the root README and open a pull request.

Agents may independently open a synthesis pull request when evidence materially changes the answer, an important caveat, or the most promising directions.
Keep routine progress, dense results, and failures in experiment issues and logbooks.

## Maintain The Root Index

Use the root `README.md` as the sole active-question index.
List every active question once under either `Current priorities` or `Other active questions`.
Keep each entry to one linked line and keep both lists alphabetical.

## Relate Experiments

Treat question-to-experiment relationships as many-to-many.

1. Verify that each target is an issue with the `experiment` Kind label.
2. Link the accepted experiment page when one exists.
   Link the issue only while an informative experiment is ongoing or awaiting interpretation review.
3. Keep each entry self-contained with one or two sentences stating what the experiment contributes and its decisive limitation.
   Do not replace the synthesis with a bare list of links.
4. Keep `Related experiments` exhaustive over experiments that materially inform the current answer.
   Omit routine invalid attempts; include an invalidated design only when its methodological lesson affects the answer or future design.
5. After the knowledge-base change merges, add the canonical experiment-page and question-page links to the experiment issue.
6. Remove or revise a relationship through the normal pull-request workflow, then update the issue after merge.

## Remove A Question

Ask the human before removal.
Remove the document and its root README entry through a pull request.
Use Git history as the record of past synthesis; do not keep inactive question documents solely as an archive.

## Compose Existing Skills

- Use `background-research` for prior work, contradictions, negative searches, and source ledgers.
- Use `run-research` and `task-logbook` for bounded research records.
- Use `update-docs` for durable guidance.
- Use `communicate-on-github` for experiment links, migration comments, and pull-request communication.
- Apply `writing-style` to prose and GitHub communication.
