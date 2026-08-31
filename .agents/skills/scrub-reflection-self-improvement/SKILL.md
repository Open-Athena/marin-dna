---
name: scrub-reflection-self-improvement
description: Run the repository self-improvement scrub only from its scheduler or an explicit request for that scrub.
schedule_cron: "0 10 * * *"
schedule_tz: America/New_York
---

# scrub-reflection-self-improvement

Use this skill on scheduled scrub turns to identify and land high-leverage improvements in `Open-Athena/marin-dna`.

## Focus

- Look for improvements from recent issues, PR feedback, and recurring operational friction.
- Prefer one concrete implementation per run when feasible.
- If implementation is blocked, produce a concrete plan and capture follow-up work in GitHub.

## Candidate Signals

- Repeated confusion in docs, workflow READMEs, or contributor workflows.
- Recurring failures or avoidable manual steps in experiments, pipelines, and repository tooling.
- Capability gaps that reduce the value of agent-assisted contributions.

## Triage Checklist

Run a lightweight, repeatable scan before choosing work:

1. Review recent open issues and open PRs for repeated friction clusters.
2. Explicitly check for already-open scrub-generated issues/PRs touching the same area; prefer advancing or deferring to that existing artifact instead of creating a parallel one.
3. Review the latest commits on `main` for changes that imply follow-on docs/workflow updates.
4. Search `AGENTS.md`, `.agents/skills/`, `docs/`, and workflow READMEs for stale workflow guidance related to those clusters.
5. De-duplicate against existing issues/PRs before creating new artifacts.

When possible, prefer improvements that remove recurring operator time (for example, turning ad-hoc scrub judgment into explicit repeatable guidance).

## Decision Heuristics

- Pick the highest-leverage change with the lowest coordination overhead.
- Treat open scrub-generated issues/PRs as first-class prior art during triage; if one already covers the candidate improvement, avoid opening a second artifact unless the new scope is clearly distinct.
- De-duplicate against existing issues/PRs before opening new work.
- When an improvement changes recurring workflow guidance, codify it in durable repo instructions: `AGENTS.md` for cross-cutting agent behavior, or `.agents/skills/` for repeatable task workflows.
  Follow the skill conventions in `AGENTS.md`.
- Do not edit a vendored skill (one listed under `unchanged` or `adapted` in a vendor manifest).
  Record the needed correction with `file-issue` for the next vendor refresh instead.
- If no justified improvement exists now, choose a no-op outcome.
- Prefer direct implementation over opening new issues when the change is fully in-repo and low-risk.

## Output

- Keep rationale explicit: observed gap, change made (or plan), and expected impact.
- Prefer durable artifacts over transient notes: land guidance updates in `AGENTS.md`, skills, or workflow READMEs when that is the primary improvement.
- Treat local-only edits as incomplete work.
  If you modify files, publish the result (commit, push, and open or update a draft pull request following the delivery steps in `AGENTS.md`) before finishing this scrub run.
- If publish is blocked (auth, permissions, CI infra, etc.), report the blocker and when the next attempt should happen instead of ending silently.
- If you choose no-op, include explicit inspected signals and why no justified improvement exists now.
- End the run with a short status: the PR link or plan, or the no-op rationale, plus any blocker and its retry time.

For no-op outcomes, include at minimum:

- which issue/PR/commit windows were inspected,
- why each candidate was not suitable for this run,
- and why deferring to the next scheduled run is preferable to opening a low-signal artifact now.
