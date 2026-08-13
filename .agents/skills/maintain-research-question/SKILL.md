---
name: maintain-research-question
description: Maintain MarinDNA research-question issues as human-declared, living syntheses across related experiments. Use when a human explicitly asks to create a research question, when an existing question needs its answer or evidence updated, or when experiment links must be synchronized in both directions.
---

# Maintain Research Question

Use a `research-question` issue for a durable question that outlives any single experiment. The issue body is the current synthesis. Comments preserve material changes over time. Related experiment issues provide the bounded research record.

## Read First

- `.agents/skills/communicate-on-github/SKILL.md`
- `.agents/skills/writing-style/SKILL.md`
- `.agents/skills/writing-style/issues.md`
- `.agents/skills/writing-style/ai-writing-donts.md`
- `.agents/skills/file-issue/SKILL.md`

## Enforce The Human Gate

- Create a `research-question` issue only when a human explicitly declares the question or approves its exact scope.
- If an agent identifies a candidate question, draft or propose it and wait for approval before filing it.
- Do not broaden, narrow, split, supersede, or close a declared question without explicit human approval.
- Agents may create bounded `experiment` issues within research the human has already authorized. Use `file-issue` for those issues.

## Use The Canonical Body

Keep the body in this order:

```markdown
## TL;DR

<One plain-text paragraph with the current answer, confidence, and main gap. Do not put links here.>

## Question

<The human-declared question and enough context for a reader who knows genomics and ML but not this thread.>

## Current answer

<Living synthesis of the answer, supporting and contradictory evidence, confidence, and limitations.>

<details>
<summary>Related work</summary>

<Curated external literature and internal reports, datasets, methods, or neighboring questions. For each item, record the setup, finding, methodological implication, hypotheses or observables it suggests, and the remaining gap.>

</details>

<details>
<summary>Related experiments</summary>

<Exhaustive list of every experiment issue that informs this question. Use plain #N references and summarize each experiment's current contribution.>

</details>

## Open questions

<Living list of unresolved points and useful next experiments.>
```

Keep `Related work` curated. Keep `Related experiments` exhaustive.

## Create A Question

1. Confirm that the human has declared or approved the exact question and scope.
2. Search open and closed research-question issues for duplicates or overlapping scope.
3. Draft the canonical body. Leave unknown sections explicit instead of inventing an answer.
4. Add the `research-question` and `agent-generated` labels plus the smallest applicable topic-label set.
5. File the issue in `Open-Athena/marin-dna` with a unique temporary body file.
6. Add the new issue to every already-related experiment body's `Links` -> `Research questions` entry.
7. Re-fetch every changed body and verify the published Markdown and bidirectional links.

If the requested question substantially overlaps an existing one, show the overlap and ask whether to update the existing issue or create a distinct question.

## Maintain The Synthesis

Update the question when a linked experiment materially changes the answer, confidence, limitations, or next experiment. Do not update it for routine run progress.

For a material change:

1. Read the current question body and the relevant experiment bodies and updates.
2. Post an append-only comment beginning with `🤖` that states the new evidence, the synthesis change, confidence, and next implication.
3. Update the living body so a cold reader does not need the comment history to understand the current state.
4. Keep the TL;DR link-free and consistent with `Current answer`.
5. Re-fetch the issue and verify Markdown rendering.

Historical details stay in experiment comments and logbooks. Do not paste raw logs, dense tables, or a chronological transcript into the question body.

## Keep Experiment Links Bidirectional

The relationship between research questions and experiments is many-to-many. Use ordinary issue references, never GitHub sub-issue metadata.

When adding a relationship:

1. Verify that each referenced issue exists and has the expected `research-question` or `experiment` Kind label. Do not infer an issue from a bare number that resolves to a pull request.
2. Add the experiment as a plain `#N` reference under the question's `Related experiments` section.
3. Add the research question as a plain `#N` reference under the experiment's `Links` -> `Research questions` entry. If the experiment predates that structure, add the smallest compatible `Links` section without rewriting its history.
4. Preserve all other question and experiment links.
5. Re-fetch both bodies and verify that each side references the other.

Treat removing a relationship as a scope change. Explain the proposed removal and wait for explicit human approval before changing either body.

## Reference Evidence

- Use commit-pinned GitHub permalinks for code.
- Link primary sources and stable artifacts.
- Distinguish direct evidence, interpretation, and untested hypotheses.
- Put large logs, tables, or code dumps in the appropriate artifact or a collapsed `<details>` block, following `communicate-on-github`.
- Never credit the agent in the issue body.

## Finish

Propose closure when the question is answered, abandoned, or superseded. Update the final TL;DR and current answer, post the conclusion in a `🤖` comment, and wait for explicit human approval before closing the issue.
