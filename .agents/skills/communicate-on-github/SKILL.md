---
name: communicate-on-github
description: Draft, post, and maintain readable MarinDNA GitHub issues, pull requests, and comments. Use for any GitHub communication, including issue classification, experiment summaries, PR descriptions, progress comments, code or artifact links, visuals, closing completed work, and maintaining current issue or PR bodies.
---

# Communicate On GitHub

Keep GitHub useful to a reader who was not present for the work. Treat bodies as the current entry point and comments as the chronological record.

## Compose The Workflow

- Read `writing-style` plus its issue or pull-request guidance before drafting non-trivial prose.
- Use `file-issue` for the mechanics and base body structures of bugs, tasks, and experiments. This skill supplies MarinDNA's repository-specific policy when the upstream skill differs.
- Use `task-logbook` and `run-research` for multi-session research records.
- When `background-research` is part of a GitHub research thread, search the current issue, PRs, permanent experiment branches, logbooks, MarinDNA code and pipeline docs, W&B, and external literature.
- Commit plots and other small artifacts under `.agents/artifacts/<topic>/` on the permanent task or research branch before linking them from GitHub.
- Follow the approval boundaries in `AGENTS.md`.

## Maintain A Living Body

- Keep an issue or PR body current when its scope, result, decision, status, or important links change.
- Keep detailed chronology in append-only comments. Do not make readers reconstruct the current state from the thread.
- Start long-running experiment issues with a short, link-free `TL;DR`.
- Start a non-trivial PR with a current summary paragraph. Do not require a separate summary heading.
- Keep citations and artifact links below the opening summary.

## Write Comments

- Begin every agent-authored issue or PR comment with `🤖`.
- Add a new comment for a material update. Edit a comment only to fix formatting or a factual error.
- State what changed, the result, confidence or caveat, and the next action.
- Update the body as well when the comment changes the current issue summary.

## Use Visuals And Collapsed Detail

- Add a concise Mermaid diagram, plot, or table only when it materially speeds up understanding of a pipeline, dependency, experiment design, comparison, or result.
- State the takeaway in text; do not make the visual the only source of the claim.
- Avoid decorative or redundant visuals.
- Wrap logs longer than 40 lines, large tables, and code dumps in `<details><summary>…</summary>…</details>`. Keep the main result outside the collapsed block.

## Link Durable Evidence

- Reference code and artifacts with immutable repository URLs, never bare paths or moving branch links.
  Use `blob/<sha>/path#Lx-Ly` for code and raw commit-pinned repository URLs for rendered images.
- Prefer committing one-off code and artifacts on the permanent task or research branch before citing them.
- Prefer stable primary sources and artifacts.
- After publishing non-trivial Markdown, re-fetch it and check lists, indentation, code blocks, links, and details blocks. Correct rendering errors.

## Classify Issues

Give every issue exactly one Kind label, zero or more Topic labels, and applicable metadata labels. Keep the label set minimal.

- Kinds: `bug` for broken behavior; `task` for a concrete improvement, feature, refactor, build, or documentation change; `experiment` for one bounded unit of research with a hypothesis or goal.
- Topics: `infrastructure`, `evals`, `data`, `modeling`, `hyperparameter-optimization`, `baselines`, `interpretation`, `communication`, and `documentation`.
- Metadata: `agent-generated`, `marin`, priority labels, and `epic`. Use `epic` only for engineering decomposition.
- Add `agent-generated` whenever an agent creates an issue or PR.

Treat `infrastructure` as a topic. A cluster-tooling improvement is `task` + `infrastructure`; broken cluster tooling is `bug` + `infrastructure`. Building a training dataset or evaluation harness is `task` + `infrastructure` plus `data` or `evals`.

Treat bounded exploratory analysis as `experiment`. Use `Hypothesis or Goal`; do not add an `eda` kind or mode field. Record a fixed hypothesis, design, primary metric, and stop criteria before execution when preregistration matters, then record deviations. Preregistration is a practice, not a label.

## Close Completed Work

- An agent may close a `bug`, `task`, or `experiment` issue after its completion criteria are met and the body and final comment record the outcome.
- Keep a research experiment with valid scientific claims open until its interpretation pull request merges.
  No-promotion closure requires explicit human approval.
- Do not close or merge a pull request without explicit user approval.

## Preserve Repository Conventions

- Use a stable lowercase branch name: `<agent>/issue-<number>-<summary>` when an issue exists, otherwise `<agent>/<summary>`.
- Put issue-closing keywords in the PR body, not the title.
- Use GitHub sub-issue metadata only to decompose one engineering work item.
- For Hugging Face uploads under `marin-dna/*`, draft the README for human review before uploading. Include a commit-pinned producing pipeline or training-script link, a short provenance description, and `biology, genomics, dna` tags.
