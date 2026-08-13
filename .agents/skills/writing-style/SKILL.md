---
name: writing-style
description: Marin house writing style. Use when drafting or revising Marin-authored prose, including commit messages and GitHub PR, issue, or comment text.
---

# Marin House Style

Start here for any non-trivial Marin-authored text. Then read the medium-specific file that matches the deliverable.

## Read The Right File

- Read [blog-posts.md](blog-posts.md) for public explainers and blog posts.
- Read [reports.md](reports.md) for technical reports aimed at peer researchers outside Marin.
- Read [tutorials.md](tutorials.md) for learning-oriented documentation that introduces Marin.
- Read [reference-docs.md](reference-docs.md) for precise usage docs aimed at readers who already know Marin.
- Read [issues.md](issues.md) for standard OSS issues and experiment issues.
- Read [pull-requests.md](pull-requests.md) for commit messages and PR titles and bodies.
- Read [discord.md](discord.md) for Discord summaries and tactical replies.
- Read [ai-writing-donts.md](ai-writing-donts.md) for the final prose-only review pass that strips generic AI-writing patterns.
- Apply this file first, then apply the medium-specific file. If a piece spans multiple media, keep the stricter rule.

## Hold The Marin Positioning

- Write from the stance of a rigorous, open-science lab building frontier-level foundation models.
- Treat process as part of the work. Experiments, decisions, and mistakes all belong in the record.
- Let the work speak. Do not substitute tone for evidence.

## Keep The Core Vibe

- Be sober, not flashy.
- Project quiet confidence.
- Keep an open door, not a megaphone.
- Stay practical and hands-on.
- Aim to be helpful and respectful.
- Assume a baseline familiarity with ML systems.
- Do not dilute discussions to accommodate every level of experience. Different media serve different audiences.
- Write like a technical peer, not an academic paper or product blog.
- Use technical language where it helps, but keep the tone natural and direct.

## Preserve Human Voice

Researchers need not suppress their own voice. Some personality is appropriate, especially in retrospectives, blog posts, and narrative writeups, while the prose remains concrete, honest, and technically disciplined.

Apply stricter discipline to agent-written text. When reviewing human prose, preserve flourishes that do not conflict with Marin's core values.

## Enforce Hard Rules

- Remove hype, marketing copy, and launch-tweet energy.
- Use emoji sparingly and only for communication.
- Remove grand claims that outrun the evidence.
- Avoid AGI rhetoric and speculation.
- Avoid adjectives that try to do the work of numbers.
- Rewrite sentences that sound like product announcements.

## Follow The Writing Principles

- Show results instead of claiming them. Prefer concrete numbers, examples, and observed behavior.
- Prefer the simplest framing that remains correct.
- State uncertainty plainly with language such as `we think`, `preliminary results suggest`, or `this seems to break down when...` when warranted.
- Treat the reader as a capable collaborator. Do not condescend or explain basics without a reason.
- Include what worked, what failed, what surprised you, and what you would try next when those facts matter.
- Cite relevant experiments, reports, issues, pull requests, papers, and other source artifacts.

## Structure Longer Prose

- Include an easy-to-scan document-level TL;DR near the top of longer work.
- Add section-level takeaway lines when they help readers navigate.
- Do not force these patterns into short-form writing.

## Set The Audience

- Assume the reader works in ML or LLMs unless the medium says otherwise.
- Do not assume deep specialization by default.
- Introduce non-standard terms before using them heavily.
- Add detail when it helps the reader reproduce, evaluate, or act.

## Use Sentence-Level Defaults

- Prefer short, direct sentences.
- Do not hard-wrap Markdown prose to a fixed width. Keep each paragraph or list item on one source line and let the renderer wrap it.
- Lead with the result or takeaway.
- Follow with method or explanation.
- End with caveats, limits, or open questions when needed.
- Prefer phrases like `we found`, `this suggests`, `in practice`, and `one limitation is`.
- Avoid phrases like `clearly`, `obviously`, `groundbreaking`, and `state-of-the-art` unless you define and defend them.

## Review For AI-Writing Tells

Do one editing pass that looks only for generic, over-smoothed, LLM-sounding prose. Then apply [ai-writing-donts.md](ai-writing-donts.md) as the detailed checklist.

## Run A Quick Self-Check

- Did you include concrete evidence?
- Did you remove hype language?
- Would this sound normal spoken aloud to a colleague?
- Did you overstate certainty or scope?
- Did you remove generic AI-writing templates and filler?

Write clearly, honestly, and with enough evidence for the work to stand on its own.
