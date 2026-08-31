---
name: scrub-docs-code-parity
description: Run the docs/code-parity scrub only from its scheduler or an explicit request for that scrub.
schedule_cron: "0 0 * * *"
schedule_tz: America/New_York
---

# scrub-docs-code-parity

Use this skill on scheduled scrub turns for docs/code parity in `Open-Athena/marin-dna`.

## Focus

- Prioritize high-confidence drift in `README.md`, `docs/`, workflow READMEs under `snakemake/`, `AGENTS.md`, and `.agents/skills/`.
- Confirm command examples match current tooling conventions (`uv run --locked`, `uv run --locked snakemake -n`, pre-commit, and the skill validator in `infra/check_skill_metadata.py`).
- Confirm that skill scripts under `.agents/skills/*/scripts/` still behave as their `SKILL.md` describes.
- Apply concrete corrections when drift is real; avoid status-only updates.

## Decision Heuristics

- Prefer updating docs when current behavior is clear and intentional.
- If implementation is clearly wrong relative to documented intent, update code and docs together.
- Keep scope small and land one useful parity improvement per run when possible.
- Do not edit a vendored skill (one listed under `unchanged` or `adapted` in a vendor manifest).
  Record the needed correction with `file-issue` for the next vendor refresh instead.
- If no material drift is found, choose a no-op outcome and keep cadence near daily.

## Output

- Keep the final scrub response concise and action-focused.
- Treat local-only edits as incomplete work.
  If you modify files, publish the result (commit, push, and open or update a draft pull request following the delivery steps in `AGENTS.md`) before finishing this scrub run.
- If publish is blocked (auth, permissions, CI infra, etc.), report the blocker and when the next attempt should happen instead of ending silently.
- If no material drift is found, explicitly report inspected scope and why no change was needed.
- End the run with a short status: the PR link, or the no-op rationale, plus any blocker and its retry time.
