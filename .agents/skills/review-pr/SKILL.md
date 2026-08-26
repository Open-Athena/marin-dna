---
name: review-pr
description: Multi-agent correctness and CLAUDE.md-compliance review of a pull request. Run only when explicitly requested or from CI; `--comment` posts findings as inline PR comments.
allowed-tools: Bash(gh issue view:*), Bash(gh search:*), Bash(gh issue list:*), Bash(gh pr comment:*), Bash(gh pr diff:*), Bash(gh pr view:*), Bash(gh pr list:*), mcp__github_inline_comment__create_inline_comment
---

Provide a code review for the given pull request: `/review-pr [--comment] <PR>`.

Adapted from marin-community/marin's `.agents/skills/review-pr/SKILL.md`. The review pipeline and the high-signal policy are unchanged; the repo-specific parts (what counts as a quotable CLAUDE.md rule, comment conventions, permalink format) are marin-dna's.

**Agent assumptions (applies to all agents and subagents):**
- All tools are functional. Do not test tools or make exploratory calls.
- Only call a tool if it is required to complete the task.

Follow these steps precisely:

1. Launch a haiku agent to check if any of the following are true:
   - The PR is closed
   - The PR is a draft
   - The PR does not need code review (e.g. a dependabot bump, a trivial obviously-correct change)
   - Claude has already commented on this PR (check `gh pr view <PR> --comments`) AND a re-review was not explicitly requested. When a maintainer explicitly requests a re-review, always proceed even if a prior review exists.

   If any condition is true, stop. Note: still review agent-authored PRs (`codex/*` and `claude/*` branches, the `agent-generated` label).

2. Launch a haiku agent to return file paths (not contents) for all relevant guidance files:
   - The root `CLAUDE.md` and `AGENTS.md`
   - `snakemake/<pipeline>/README.md` for every pipeline directory containing files modified by the PR (CLAUDE.md requires the README to be updated in the same PR as a behaviour change)
   - Any `CLAUDE.md` or `AGENTS.md` in directories (and parent directories) containing files modified by the PR

3. Launch an opus agent to view the PR and return a summary of the changes. The same agent checks the PR title and description against the **GitHub Communication** section of `CLAUDE.md` and returns any problems it finds:

   - an issue-closing keyword (`fixes #N`, `closes #N`, `resolves #N`) in the *title* — it belongs in the body;
   - code referenced by bare path or branch link instead of a commit-pinned permalink (`blob/<sha>/path#Lx-Ly`);
   - logs over ~40 lines, large tables, or code dumps not wrapped in `<details><summary>…</summary>…</details>`;
   - experimental results (tables, leaderboards, per-genome stats) placed in a README instead of the tracking issue.

   A terse, plain body for a small change is correct — do not flag brevity or the absence of markdown.

4. Launch 4 agents in parallel to independently review the changes. Each returns a list of issues; each issue includes a description and the reason it was flagged (e.g. "CLAUDE.md adherence", "bug").

   Agents 1 + 2: CLAUDE.md compliance opus agents. Audit changes for compliance. When evaluating a file, only consider guidance files that share its path or are parents. The `CLAUDE.md` rules concrete enough to quote, and therefore in scope:

   - **Coordinates.** 0-based, half-open everywhere inside our code; conversion to/from 1-based closed formats (GTF, VCF, SAM, `pyfaidx.get_seq()`, samtools region strings) happens only at the tool boundary.
   - **Logic lives in a package, not in rule files.** `run:` blocks in `Snakefile`/`.smk` files are thin glue calling into a library. No new `.py` script files under `snakemake/` *except* inside a workflow's own `src/` package (e.g. `snakemake/analysis/evals_v2/src/marin_dna_evals/`) — AGENTS.md: "Put maintained Python in the owning project's `src/` package".
   - **Tests.** A new non-trivial function in `src/marin_dna/` with no test.
   - **Type annotations** on every parameter and return value in `src/marin_dna/`, Python 3.11+ syntax (`list[str]`, `X | None`).
   - **Pipeline README** not updated when the PR changes that pipeline's behaviour.
   - **Plots.** Figure-level seaborn functions; capless SE error bars; independent twin y-axes for non-level-comparable metrics.
   - **Scope.** Unrelated code in other pipelines or library modules removed or rewritten.

   Agents 3 + 4: opus bug agents (parallel). Scan for obvious bugs, security issues, and incorrect logic within the changed code. Focus only on the diff without reading extra context. Flag only significant bugs you can validate from the diff alone; ignore nitpicks and likely false positives. Bioinformatics bugs that deserve extra suspicion: off-by-one at interval boundaries, strand handling, reference-build or chromosome-name mismatches (`chr1` vs `1`), silent NaN propagation, and parallel arrays whose lengths are never checked.

   **CRITICAL: We only want HIGH SIGNAL issues.** Flag issues where:
   - The code will fail to compile or parse (syntax errors, type errors, missing imports, unresolved references)
   - The code will definitely produce wrong results regardless of inputs (clear logic errors)
   - Clear, unambiguous CLAUDE.md violations where you can quote the exact rule being broken

   Do NOT flag:
   - Code style or quality concerns
   - Potential issues that depend on specific inputs or state
   - Subjective suggestions or improvements

   If you are not certain an issue is real, do not flag it. False positives erode trust.

   Tell each subagent the PR title and description for author-intent context.

   **marin-dna-specific:** duplication between pipeline modules inside `src/marin_dna/` is intentional (CLAUDE.md: "Duplication beats premature abstraction *within* the library"). Do not flag copy/paste or DRY concerns if behaviour is correct. Liberal `assert`s for invariants are house style, not a smell. Experiment directories and one-off `scripts/` are deliberately not library-quality; hold them to correctness, not structure.

5. For each issue from agents 3 and 4, launch a parallel subagent to validate it. Give the subagent the PR title, description, and issue description. It must confirm with high confidence that the issue is real — e.g. for "variable is not defined", verify that in the code; for a CLAUDE.md issue, verify the rule is scoped to this file and actually violated. Use opus subagents throughout — for both bugs/logic and CLAUDE.md violations.

6. Filter out any issues not validated in step 5. The remainder is the high-signal review list.

7. Output a summary of the review findings to the terminal:
   - If issues were found, list each issue with a brief description.
   - If no issues were found, state: "No issues found. Checked for bugs and CLAUDE.md compliance."
   - Separately, report any PR-description problems from step 3.

   If `--comment` argument was NOT provided, stop here. Do not post any GitHub comments.

   If `--comment` IS provided and step 3 found PR-description problems, post **one** top-level comment with `gh pr comment` (prefixed `🤖`, not inline) naming the specific problems and the concrete fix (e.g. "move `fixes #131` from the title into the body"). This is independent of the code review — post it whether or not code issues were found, but skip it when the description is fine.

   If `--comment` IS provided and NO code issues were found, post the no-issues summary comment (format below) using `gh pr comment` and stop.

   If `--comment` IS provided and code issues were found, continue to step 8.

8. Draft the list of comments you plan to leave. For your own review only — do not post it anywhere.

9. Post inline comments for each issue using `mcp__github_inline_comment__create_inline_comment` with `confirmed: true`. For each comment:
   - Begin the body with `🤖` (CLAUDE.md: agent comments on PRs and issues must begin with it)
   - Provide a brief description of the issue
   - For small, self-contained fixes, include a committable suggestion block
   - For larger fixes (6+ lines, structural changes, or changes spanning multiple locations), describe the issue and suggested fix without a suggestion block
   - Never post a committable suggestion UNLESS committing the suggestion fixes the issue entirely. If follow-up steps are required, do not leave a committable suggestion.

   **IMPORTANT: Only post ONE comment per unique issue. Do not post duplicate comments.**

Use this list when evaluating issues in steps 4 and 5 (these are false positives, do NOT flag):

- Pre-existing issues
- Something that appears to be a bug but is actually correct
- Pedantic nitpicks that a senior engineer would not flag
- Issues that a linter will catch — ruff, mypy, snakefmt run in the Quality workflow (do not run them to verify)
- General code quality concerns (e.g. general security issues) unless explicitly required in CLAUDE.md
- Issues mentioned in CLAUDE.md but explicitly silenced in the code (e.g. via a lint ignore comment)

Notes:

- Use the gh CLI to interact with GitHub (fetch pull requests, create comments). Do not use web fetch.
- Create a todo list before starting.
- You must cite and link each issue in inline comments (e.g. when referring to CLAUDE.md, include a permalink to it, ideally with line numbers).
- If no issues are found and `--comment` is provided, post a comment with exactly this format:

---

## 🤖 Code review

No issues found. Checked for bugs and CLAUDE.md compliance.

---

- When linking to code in inline comments, follow this format precisely, otherwise the Markdown preview won't render: https://github.com/Open-Athena/marin-dna/blob/<full-40-char-sha>/CLAUDE.md#L10-L15
  - Requires the full git sha. Commands like `https://github.com/owner/repo/blob/$(git rev-parse HEAD)/foo/bar` will not work, since your comment is rendered directly as Markdown.
  - Repo name must be `Open-Athena/marin-dna`.
  - `#` after the file name; line range format is `L[start]-L[end]`.
  - Provide at least 1 line of context before and after, centred on the line you are commenting about (commenting on lines 5-6 → link `L4-L7`).
