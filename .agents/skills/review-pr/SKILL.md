---
name: review-pr
description: Multi-agent correctness and AGENTS.md-compliance review of a pull request. Run only when explicitly requested or from CI; `--comment` posts findings as inline PR comments.
allowed-tools: Bash(gh issue view:*), Bash(gh search:*), Bash(gh issue list:*), Bash(gh pr comment:*), Bash(gh pr diff:*), Bash(gh pr view:*), Bash(gh pr list:*), mcp__github_inline_comment__create_inline_comment
---

Provide a code review for the given pull request: `/review-pr [--comment] <PR>`.

Adapted from marin-community/marin's `.agents/skills/review-pr/SKILL.md`.
The review pipeline and the high-signal policy are unchanged; the repo-specific parts are derived from `AGENTS.md` and the local skills named below.
When `AGENTS.md` or those skills change, re-sync this file (see `maintain-vendored-skills` for the vendor comparison workflow).

**Agent assumptions (applies to all agents and subagents):**
- All tools are functional. Do not test tools or make exploratory calls.
- Only call a tool if it is required to complete the task.

Follow these steps precisely:

1. Launch a haiku agent to check if any of the following are true:
   - The PR is closed
   - The PR is a draft
   - The PR does not need code review (e.g. a dependabot bump, a trivial obviously-correct change)
   - Claude has already commented on this PR (check `gh pr view <PR> --comments`) AND a re-review was not explicitly requested. When a maintainer explicitly requests a re-review, always proceed even if a prior review exists.

   If any condition is true, stop.
   Note: still review agent-authored PRs (`codex/*` and `claude/*` branches, the `agent-generated` label).

2. Launch a haiku agent to return file paths (not contents) for all relevant guidance files:
   - The root `AGENTS.md` (`CLAUDE.md` is a symlink to it)
   - Any `AGENTS.md` or `CLAUDE.md` in directories (and parent directories) containing files modified by the PR
   - The owning project's `README.md` for every Snakemake project that contains a modified file (`AGENTS.md` requires the README to be updated when user-visible behaviour changes)
   - Area skills that govern the modified files: `.agents/skills/plot-research-results/SKILL.md` when the PR changes figures or plotting code, and `.agents/skills/develop-snakemake-pipelines/SKILL.md` when it changes files under a Snakemake project
   - `.agents/skills/writing-style/SKILL.md`, `.agents/skills/writing-style/pull-requests.md`, and `.agents/skills/communicate-on-github/SKILL.md` (for step 3)

3. Launch an opus agent to view the PR and return a summary of the changes.
   The same agent checks the PR title and description against `writing-style/pull-requests.md` and `communicate-on-github` and returns any problems it finds, for example:

   - a title over 72 characters, a non-imperative title, or a conventional-commit prefix;
   - code or artifacts referenced by bare path or moving branch link instead of an immutable `blob/<sha>/path#Lx-Ly` URL;
   - logs over ~40 lines, large tables, or code dumps not wrapped in `<details><summary>…</summary>…</details>`;
   - a body that buries what-the-change-does under inventory or boilerplate instead of leading with it.

   Flag only concrete violations of those two files.
   A terse, plain body for a small change is correct — do not flag brevity or the absence of markdown.

4. Launch 4 agents in parallel to independently review the changes.
   Each returns a list of issues; each issue includes a description and the reason it was flagged (e.g. "AGENTS.md adherence", "bug").

   Agents 1 + 2: AGENTS.md compliance opus agents.
   Audit changes for compliance.
   When evaluating a file, only consider guidance files that share its path or are parents, plus the area skills from step 2.
   The `AGENTS.md` rules concrete enough to quote, and therefore in scope:

   - **Coordinates.** 0-based, half-open genomic coordinates internally; conversion to/from 1-based or closed formats (GTF, VCF, SAM, `pyfaidx.get_seq()`, samtools-style region strings) at the tool boundary, with any deviation stated.
   - **Project boundaries.** Maintained Python lives in the owning project's `src/` package; the root `src/marin_dna/` package stays limited to lightweight genomic primitives and must not depend on training frameworks, Snakemake, W&B, plotting libraries, or pipeline-specific clients.
   - **Tests.** Non-trivial maintained behaviour in any changed project's owning package with no test; data pipelines missing contract assertions (schemas, row counts, value ranges, sequence lengths, coordinate bounds, build or strand consistency).
   - **Type annotations** on parameters and return values in every changed project's `src/` package, using current built-in generic and union syntax.
   - **Workflow README** not updated when the PR changes that workflow's user-visible behaviour, configuration, outputs, or operating procedure.
   - **Markdown prose** hard-wrapped at a fixed column or carrying multiple sentences per source line.
   - **Figures.** Judge changed plotting code against `plot-research-results`; flag only concrete violations of that skill.
   - **Lifecycle.** Experiment-only or one-off content woven into `main`-bound framework code instead of staying on its permanent branch.

   Agents 3 + 4: opus bug agents (parallel).
   Scan for obvious bugs, security issues, and incorrect logic within the changed code.
   Focus only on the diff without reading extra context.
   Flag only significant bugs you can validate from the diff alone; ignore nitpicks and likely false positives.
   Bioinformatics bugs that deserve extra suspicion: off-by-one at interval boundaries, strand handling, reference-build or chromosome-name mismatches (`chr1` vs `1`), silent NaN propagation, and parallel arrays whose lengths are never checked.

   **CRITICAL: We only want HIGH SIGNAL issues.** Flag issues where:
   - The code will fail to compile or parse (syntax errors, type errors, missing imports, unresolved references)
   - The code will definitely produce wrong results regardless of inputs (clear logic errors)
   - Clear, unambiguous AGENTS.md violations where you can quote the exact rule being broken

   Do NOT flag:
   - Code style or quality concerns
   - Potential issues that depend on specific inputs or state
   - Subjective suggestions or improvements

   If you are not certain an issue is real, do not flag it.
   False positives erode trust.

   Tell each subagent the PR title and description for author-intent context.

   **marin-dna-specific:** duplicating unstable logic across projects is intentional until a genuinely shared abstraction has emerged (`AGENTS.md`, Project Boundaries) — do not flag copy/paste or DRY concerns if behaviour is correct.
   Liberal `assert`s for data contracts are required house style, not a smell.
   Experiment branches and one-off scripts are held to correctness, not structure.

5. For each issue from step 4 — from all four agents, compliance and bug alike — launch a parallel subagent to validate it.
   Give the subagent the PR title, description, and issue description.
   It must confirm with high confidence that the issue is real — e.g. for "variable is not defined", verify that in the code; for an AGENTS.md issue, verify the rule is scoped to this file and actually violated.
   Use opus subagents throughout.

6. Filter out any issues not validated in step 5.
   The remainder is the high-signal review list.

7. Output a summary of the review findings to the terminal:
   - If issues were found, list each issue with a brief description.
   - If no issues were found, state: "No issues found. Checked for bugs and AGENTS.md compliance."
   - Separately, report any PR-description problems from step 3.

   If `--comment` argument was NOT provided, stop here.
   Do not post any GitHub comments.

   If `--comment` IS provided and step 3 found PR-description problems, post **one** top-level comment with `gh pr comment` (prefixed `🤖`, not inline) naming the specific problems and the concrete fix.
   This is independent of the code review — post it whether or not code issues were found, but skip it when the description is fine.

   If `--comment` IS provided and NO code issues were found, post the no-issues summary comment (format below) using `gh pr comment` and stop.

   If `--comment` IS provided and code issues were found, continue to step 8.

8. Draft the list of comments you plan to leave.
   For your own review only — do not post it anywhere.

9. Post inline comments for each issue using `mcp__github_inline_comment__create_inline_comment` with `confirmed: true`.
   For each comment:
   - Begin the body with `🤖` (`communicate-on-github`: every agent-authored issue or PR comment begins with it)
   - Provide a brief description of the issue
   - For small, self-contained fixes, include a committable suggestion block
   - For larger fixes (6+ lines, structural changes, or changes spanning multiple locations), describe the issue and suggested fix without a suggestion block
   - Never post a committable suggestion UNLESS committing the suggestion fixes the issue entirely. If follow-up steps are required, do not leave a committable suggestion.

   **IMPORTANT: Only post ONE comment per unique issue. Do not post duplicate comments.**

Use this list when evaluating issues in steps 4 and 5 (these are false positives, do NOT flag):

- Pre-existing issues
- Something that appears to be a bug but is actually correct
- Pedantic nitpicks that a senior engineer would not flag
- Issues that a linter will catch — ruff, mypy, snakefmt run via pre-commit in the Quality workflow (do not run them to verify)
- General code quality concerns (e.g. general security issues) unless explicitly required in AGENTS.md
- Issues mentioned in AGENTS.md but explicitly silenced in the code (e.g. via a lint ignore comment)

Notes:

- Use the gh CLI to interact with GitHub (fetch pull requests, create comments).
  Do not use web fetch.
- Create a todo list before starting.
- You must cite and link each issue in inline comments (e.g. when referring to AGENTS.md, include a permalink to it, ideally with line numbers).
- If no issues are found and `--comment` is provided, post a comment with exactly this format:

---

## 🤖 Code review

No issues found. Checked for bugs and AGENTS.md compliance.

---

- When linking to code in inline comments, follow this format precisely, otherwise the Markdown preview won't render: https://github.com/Open-Athena/marin-dna/blob/<full-40-char-sha>/AGENTS.md#L10-L15
  - Requires the full git sha. Commands like `https://github.com/owner/repo/blob/$(git rev-parse HEAD)/foo/bar` will not work, since your comment is rendered directly as Markdown.
  - Repo name must be `Open-Athena/marin-dna`.
  - `#` after the file name; line range format is `L[start]-L[end]`.
  - Provide at least 1 line of context before and after, centred on the line you are commenting about (commenting on lines 5-6 → link `L4-L7`).
