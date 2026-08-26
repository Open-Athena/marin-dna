---
name: draft-weekly-research-update
description: Create a reviewable weekly MarinDNA public-research draft from experiment interpretation pages first added to main. Use for the scheduled Monday research update or an explicit request for that draft; do not use for issue activity, infrastructure, bugs, or edits to existing experiment pages.
---

# Draft Weekly Research Update

Create a deterministic public draft from newly deposited experiment pages.
Do not summarize or reinterpret the research.

## Selection Contract

- Select the preceding Monday-through-Sunday UTC window.
- Fetch `origin/main` before selecting files.
- Include every Markdown file first added under `docs/research/experiments/` between the two weekly boundary snapshots on `origin/main`.
- Do not include modified existing experiment pages, issues, pull requests, infrastructure work, bugs, or research-question pages.
- Attribute a page to the week when it first appears on `main`, regardless of when the experiment ran or its issue closed.

## Draft Contract

- Sort pages by the leading experiment number in the filename.
- Link each experiment title to its canonical page on `main`.
- Copy the text from the page's `> **TL;DR:**` callout without rewriting it.
- Do not add an overview, synthesis, author credit, interpretation, supporting-artifact link, or future-work section.
- Keep the voice neutral by preserving the canonical TL;DR text.
- Produce a reviewable draft only.

## Generate The Draft

From the repository root, fetch `main` and run the deterministic extractor with the reporting week's Monday date:

```bash
git fetch origin main
python3 .agents/skills/draft-weekly-research-update/scripts/draft_weekly_research_update.py \
  --week-start YYYY-MM-DD
```

Return the script's standard output unchanged as the publishable draft.
If standard output is empty, report internally that no publishable draft was produced.
Relay any standard-error warnings as an internal note after the draft.
A missing title or TL;DR omits that page and emits a warning; do not synthesize replacement text.
