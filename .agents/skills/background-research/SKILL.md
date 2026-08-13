---
name: background-research
description: "Forage prior work before or during MarinDNA research threads: search internal artifacts and external literature or code, then produce a cited brief with negative results and ranked experiment hypotheses."
---

# Background Research

Use this when a research, design, or experiment thread needs a prior-work pass before choosing hypotheses, drafting a design, or launching runs.

## Set The Effort

State the effort level at the top of the output. Stop when new sources no longer change the ranked hypotheses. Go longer when the user asks or the decision cost warrants it.

- `low` (3–7 minutes): read the current issue or logbook, obvious local references, and a few external sources.
- `medium` (10–15 minutes): default. Search internal MarinDNA sources plus targeted external literature or code, include a contradiction pass, and produce a source ledger with ranked experiments.
- `high` (30–60 minutes): use for expensive runs, architecture changes, data or evaluation decisions, and public claims. Record query strings and rejected-source notes.

Effort changes breadth and provenance depth. Every claim still needs evidence and a clear distinction between observation and speculation.

## Search In This Order

Search internal and external sources in parallel when useful, but do not skip the internal pass.

1. The current issue, pull request, research logbook, or task record.
2. Related GitHub issues, pull requests, experiment links, and comments.
3. Repository READMEs, `docs/`, pipeline documentation, model cards, and dataset cards.
4. Existing logbooks or research files created by the coordinating workflow.
5. Relevant code in the current tree, commit history, permanent experiment branches, and tags.
6. W&B reports, runs, and artifacts linked from those sources.
7. External papers, official documentation, codebases, arXiv, OpenReview, Semantic Scholar, and cited references.

For `medium` and `high` effort, include an adversarial query intended to find contradictory evidence or a failure mode.

## Handle Sources

- Keep raw sources as ground truth. Treat the brief as a derived artifact that may be wrong.
- Prefer primary sources: papers, official docs, code, issues, W&B runs, and reports.
- Record a source version or date when it affects interpretation.
- Grade evidence by directness to the MarinDNA regime: model scale, hardware, data, objective, optimizer, context length, evaluation harness, and implementation constraints.
- Record contradictions, negative results, and meaningful searches that found nothing.
- Use commit-pinned links for claims tied to a fixed code revision.

## Write The Brief

Write the brief in the coordinating issue, logbook, or research record selected by the parent workflow. If an experiment issue exists, also provide a short issue-ready `Prior work` block.

```md
## Background Research Brief

- Effort:
- Stop rule:
- Date:

### Question

### Current MarinDNA Context

### Internal Prior Work

### External Prior Art

### Negative Or Failed Leads

### Evidence Map

#### Claim: <short claim>
- Support:
  - <source>: <one-line evidence>
- Contradictions:
  - <source>: <one-line caveat or failed result>
- Directness to MarinDNA:
- Confidence:
- Action:

### Recommended Next Experiments

#### 1. <hypothesis>
- Minimum experiment:
- Baseline or control:
- Expected signal:
- Falsifier:
- Cost or risk:
- Sources:

### Hypothesis Queue Update
- Add:
- Revise:
- Falsify or stop:
- Promote:

### Source Ledger
| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|

### Handoff
- Suggested issue `Prior work` block:
- Suggested logbook entry:
- Open questions:
- Stop reason:
```

Use tables only for compact metadata. Use block-style entries for claims, caveats, and hypotheses that require prose.

## Rank Experiments

Make each recommendation actionable without rereading the full source set. Include:

- A falsifiable hypothesis.
- The smallest experiment that could change the decision.
- A baseline or control.
- A primary metric and expected direction.
- A falsifier.
- Cost and risk.
- Source links.
- Confidence such as `exploratory`, `replicated`, or `stable` only when the evidence supports it.

## Skip Low-Value Work

- Do not paste long paper summaries when claim-level evidence is enough.
- Do not use transient conversation as the durable record.
- Do not search upstream-only paths by assumption. Discover the current repository structure first.
