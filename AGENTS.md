# Project Guidelines

## Project Overview

**MarinDNA** is a framework for developing genomic language models (gLMs).

## Domain Conventions

- **Coordinate system.** The codebase consistently uses 0-based, half-open intervals for all genomic coordinates. Assume this everywhere; call out any deviation explicitly. Conversions to/from 1-based closed formats (GTF, VCF, SAM) happen at the tool boundary, not inside our code.
- **Canonical public human genome.** Prefer the authenticated S3 mirror at `s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/` whenever the caller has access and runs in the same region (`us-east-2`); it offers lower and more predictable latency for byte-range queries. For credential-free, public, or non-AWS consumers, use `marin-dna/human-genome` pinned to revision `11b9433582981bb929af333bc6422f10a8fd71b4`: uncompressed FASTA for `pyfaidx` + `fsspec` HTTP queries, or BGZF plus its indexes for HTSlib/samtools and full downloads. Never migrate already-materialized inputs merely to standardize their source. Both mirrors contain the Ensembl 115 GRCh38 soft-masked primary assembly with Ensembl sequence names (`1`...`MT`). MarinDNA code and `pyfaidx` slicing use 0-based, half-open intervals; `pyfaidx.get_seq()` and samtools-style region strings use 1-based, closed intervals and require conversion at the tool boundary. Other `hg38` variants are not interchangeable.

- **VEP chromosome split.** For labeled variant-effect prediction (VEP) data, use odd-numbered autosomes (`1, 3, …, 21`) and chromosome X for training, validation, development, model selection, probing, and hyperparameter tuning. Reserve even-numbered autosomes (`2, 4, …, 22`) and chromosome Y for final test evaluation; accessing their variant labels, effect measurements, predictions, or aggregate metrics requires explicit user permission and should be a rare final-evaluation event. This holdout applies to labeled variants and VEP-derived supervision, not to genomic sequence itself: unlabeled/reference-sequence pretraining may use every chromosome, as may functional-genomics data describing reference sequences unless the dataset defines a stricter split.

## Research Code Values

This is research code. Prioritize **reproducibility** and **correctness** over architectural elegance.

- **Put Python logic in the owning project package so pytest can reach it.** Reusable genomic primitives belong in the root `src/marin_dna/`; pipeline-specific functions belong in that pipeline's local `src/` package. Inline Python in Snakemake `run:` blocks should be thin glue calling tested functions. Maintained pipeline CLIs are package entry points, not loose files under a `scripts/` directory.
- **Duplication beats premature abstraction *within* the library.** The "testable home" rule governs *entry* into `src/marin_dna/` — move logic in freely, even if similar code already exists elsewhere. A separate, weaker rule governs *deduplication*: only merge two similar functions into one shared helper when the shape has stabilized and they're genuinely doing the same thing. Until then, two near-copies in two pipeline modules is better than a premature abstraction coupling unrelated experiments.
- **Modularity is a means, not a goal.** Don't refactor for reuse that may never come. Straight-line code that reads top-to-bottom is often preferable to layered abstractions.
- **Test aggressively.** Every non-trivial function in `src/marin_dna/` should have tests — that's the whole reason logic lives there. For pipelines, add sanity checks on outputs (row counts, value ranges, coordinate invariants) rather than trusting that "it ran".
- **Assert defensively, everywhere.** Use `assert` liberally for invariants that *should* hold: coordinate bounds, dataframe shapes, no NaNs where none are expected, set membership, monotonicity, matching lengths between parallel arrays. A loud failure near the bug is worth far more than a silently corrupted result feeding into training.
- **Fail fast on silent-corruption risks.** Bioinformatics is full of off-by-one errors, strand mix-ups, and reference-build mismatches. When a result could be quietly wrong, prefer a check that crashes over a comment saying "this should be correct".
- **No premature generalizations.** If asked to implement a specific backend, dataset, or model variant, stick to that. Don't generalize to related use-cases on your own — offer the option, but only expand the scope when explicitly told to.
- **Stay in scope.** Don't remove or rewrite unrelated code in other pipelines or library modules while working on a task. Unrelated experiments may depend on exact current behavior.

## Code Structure

The codebase has five main components:

1. **Python core** (`src/marin_dna/`) - lightweight reusable genomic primitives shared by independent projects. Core must not depend on Torch, Transformers, Snakemake, WandB, plotting libraries, or pipeline-specific upload clients.

2. **Pipelines** (`snakemake/`) - Data processing workflows implemented in Snakemake
   - Read the pipeline's README before working on it — each `snakemake/<pipeline>/` has its own. If you change pipeline behaviour, update the README in the same PR so the next human or agent can onboard from it.
   - Always dry-run first (`-n` / `--dry-run`) before any real invocation.
   - Stop before reruns of steps the changes you made for this task did not intentionally touch. If the dry-run shows Snakemake planning to rerun an upstream or unrelated step — retriggered by a timestamp change, an unrelated code edit, `--rerun-triggers` defaults, etc. — stop and ask before running. Default assumption: such reruns are unintended and potentially expensive (training, genome downloads, large bedtools jobs).
   - Enter the pipeline project, sync its committed lockfile with `uv sync --locked --group dev`, and invoke as `uv run --locked snakemake …`.
   - Put pipeline-wide defaults (`cores`, `use-conda`, `default-storage-provider`, etc.) in the pipeline's `workflow/profiles/default/config.yaml`, not on the CLI. Snakemake auto-loads that profile, so every invocation picks them up.

3. **Experiments** - Marin-launched training/eval scripts. Each experiment is a **self-contained directory on its own branch** (its own `pyproject.toml` with marin in *base* deps + a `launch.py`), **not merged to `main`** (see "What gets merged to `main`" above) — cite it from its tracking issue via commit-pinned permalinks. Full setup, launch flow, and hard-won lessons live in the **`marin-experiment` skill** (`.agents/skills/marin-experiment/`).
   - **wandb run names.** Set run names that include `dna-exp<N>` (`<N>` = the experiment number from the issue) so runs filter by experiment.

4. **Plots** (`plots/`) - Self-contained Python scripts that turn pipeline metric parquets into figures. One file per recipe; outputs to gitignored `plots/output/<recipe>/`.
   - Load parquets from S3 with `polars.read_parquet("s3://…")` — native object_store support, no fsspec/s3fs needed.
   - Emit both `figure.svg` and `figure.png` to the output dir. SVG is the artifact to upload to GitHub (PR/issue/gist embeds); PNG is the local-iteration format — agents can `Read` it back to visually sanity-check (the `Read` tool renders raster but not SVG) and PNGs also render inline in agent conversations.
   - **Prefer seaborn's figure-level functions** (`relplot`/`catplot`/`displot`) over `sns.lineplot`/`scatterplot` drawn onto hand-built `plt.subplots` axes — seaborn bolted onto manual matplotlib axes reads as neither (missing labels, inconsistent fonts, off title placement). The figure-level call owns faceting, axis labels, facet titles, legend placement, and font scaling; set labels/titles via `g.set_axis_labels`/`g.set_titles` and the suptitle via `g.figure.suptitle`. Drop to manual `plt` axes only when a figure-level call genuinely can't express the layout (e.g. a custom per-facet SE band — add it by iterating `g.axes_dict`). For a **numeric hue**, seaborn auto-sorts it: use a continuous palette + `hue_norm` (e.g. `palette="viridis", hue_norm=(0, 0.65)`), not a dict palette (which forces categorical treatment), and don't pass `hue_order`.
   - **Error bars for standard error get no caps.** Draw ±1 SE (or any dispersion indicator) capless — `matplotlib.errorbar(..., capsize=0)`, which is the default. Caps read as a *bounded interval* with defined endpoints (a CI or range) and misrepresent an SE bar, so reserve them for genuine CIs/ranges. Keep the SE meaning explicit in the label (e.g. `"error bars = ±1 SE (bootstrap)"`).
   - **Two non-level-comparable metrics go on independent twin y-axes.** When two series aren't level-comparable by construction (different scale / matching / aggregation — e.g. two benchmarks, or matched-pair vs unmatched AUPRC), give each its own `ax.twinx()` autoscaled to its own range, color-code each axis to its line, and footnote that the axes are independent. A shared y-axis implies a level comparison we don't intend — the reader should compare shapes/trends, never the vertical gap between the lines.

5. **One-offs** - There is no top-level `scripts/` directory on `main`. Reusable code belongs to core or the owning pipeline project. One-off analysis and experiment code is committed on its permanent branch and cited with a commit-pinned permalink. `scratch/` contains only disposable local artifacts.

### What gets merged to `main`

`main` is the **reusable-core framework**: the library kernel, training-data construction, our-model evaluation (`evals_v2`), and `dashboard`/`docs`. Open a PR to `main` only for that core.

- **Assume non-core work never merges.** Experiments, one-off analyses, competitor baselines, and dead-ends stay on their own branches — the aim isn't to prune them out of `main` later, it's to never entangle them in the first place. Keep them self-contained on their permanent branches, not woven into a merge-bound project. Lifecycle complement to **Stay in scope** and **No premature generalizations** above.
- **A branch is a permanent reference — merging isn't.** Commit and push freely: a commit-pinned permalink to an unmerged branch is all you need to cite a result or reproduce an experiment from its tracking issue. Nothing has to land on `main` to stay reachable.

## Development Practices

- **Package management**: Use `uv` for Python dependencies
- **Bioinformatics tools**: Use Conda for external CLI tools (bedtools, twoBitToFa, etc.)
- **Testing**: Run `uv run --locked pytest` in every changed Python project before committing
- **Code quality**: Pre-commit hooks enforce ruff formatting and linting
- **Documentation**: Before merging a PR, make sure all the relevant READMEs are updated. READMEs describe how to run or use a thing, not what was found — experimental results (tables, leaderboards, key findings, per-genome stats, etc.) belong in the GitHub issue tracking that work, not in any README. Results drift; READMEs shouldn't.
- **Markdown source formatting**: Do not hard-wrap prose to a fixed column width. Keep each paragraph or list item on one source line and let editors/renderers wrap it visually; use manual line breaks only when Markdown structure requires them.
- **Where to run.** For quick work (small data, smoke tests, dev iteration), run locally on the current node — but first check system load (`uptime` / `cat /proc/loadavg`); multiple agent sessions share this small instance. Be careful about parallelizing local subprocesses: it has crashed the instance more than once (requiring reboot). Cap parallel jobs conservatively (rule of thumb: `nproc/2` or less). For heavy work (training, large-scale evals, anything GPU-bound), launch on SkyPilot. Always confirm with the user before launching SkyPilot resources — they're not free.
- **Babysit new jobs early.** First time running a script / config / cluster combination? Check actively within the first few minutes rather than passively waiting. Look for: progress rate sane (a common silent failure is CPU fallback when GPU was expected), device count matches what you asked for, no immediate OOM / mount errors / auth failures. Notifiers fire on completion or timeout — they don't tell you the run spent 4 hours on CPU.
- **Parallel sky sweeps.** When evaluating a grid of independent snakemake targets (e.g. every training-step checkpoint of one model arm), prefer launching one sky cluster per target over running the whole DAG on a single big cluster. Each cluster `--down`s on idle, parallelism scales with AWS capacity, and a failure in one target doesn't block the others. The canonical helper is `snakemake/analysis/evals_v2/sky/parallel_sweep.sh` — it takes snakemake target paths as args, derives one cluster per target, and waits for all to finish. The pattern relies on `run.yaml` exposing `$SNAKEMAKE_ARGS` so each cluster runs `snakemake -- <target>` and produces exactly that one parquet. Heads up: bursting >~24 `g5.xlarge` into `us-east-2` typically saturates AZs — sky reports `ResourcesUnavailableError` on the late arrivals. Re-running the helper with just the failed targets after the earlier clusters `--down` is usually enough; sky has no cross-region fallback for AWS-pinned tasks.

### Type Annotations
- Type-annotate all function parameters and return values in every project's `src/` package.
- Use Python 3.11+ syntax (`list[str]`, `X | None`); reach for `typing` only for constructs that still require it.

## Autonomy Boundaries

- Never push to `main` without explicit user approval.
- Never close or merge a PR without explicit user approval.
- Agents may close a `bug`, `task`, or `experiment` issue after its completion criteria are met and its body and final comment reflect the outcome.
- Never close a `research-question` issue without explicit user approval.

## GitHub Communication

### Writing style and workflow skills

Use `.agents/skills/writing-style/` for non-trivial MarinDNA-authored prose, including commit messages and GitHub PR, issue, and comment text. Assume readers understand genomics and ML broadly but were not present for the work or investigation.

Use `file-issue` to create bugs, tasks, and bounded experiment issues. Use `run-research` for multi-session research, with `task-logbook` as its detailed append-only record. Use `maintain-research-question` for human-declared questions that synthesize evidence across experiments.

The Marin-derived skills are pinned to [`marin-community/marin`](https://github.com/marin-community/marin/tree/836eab32a78a9c6421300573a4571f2dc2682de4/.agents/skills) at commit `836eab32a78a9c6421300573a4571f2dc2682de4`, vendored on `2026-08-12`.

- Vendored unchanged: `background-research`, `update-docs`, `wandb-reporting`, and the complete `writing-style` directory.
- Adapted `file-issue`: target `Open-Athena/marin-dna`; apply MarinDNA's kind/topic taxonomy, including EDA-to-experiment routing and bounded-experiment authorization; require a link-free experiment TL;DR, useful experiment visuals, commit-pinned code links, and bidirectional research-question links; route research-question creation to the human-gated local skill.
- Adapted `run-research`: follow MarinDNA's permanent-branch and merge-scope rules; allow bounded experiments inside human-authorized research; synchronize research-question links; replace unavailable Marin-only cross-references with local skills.
- Adapted `task-logbook`: use MarinDNA's production-commit rules and require `🤖` on every agent comment.
- Adapted `task-snapshot`: use the MarinDNA repository in the pinned-link example.
- Adapted `babysit-job`: combine Iris monitoring with MarinDNA's `marin-experiment` launch workflow and remove references to unavailable Marin-only skills.
- Local: `maintain-research-question` encodes MarinDNA's human gate and cross-experiment synthesis.

Treat unchanged vendored files as read-only. Keep local policy in `AGENTS.md` or a separate local skill. When refreshing an adapted skill, reapply every behavior described above and review its upstream diff before replacing the local copy.

For `background-research`, use the current issue, PRs, permanent experiment branches, `.agents/logbooks/`, MarinDNA code and pipeline docs, W&B, and external literature as the local search surface. Skip upstream-only paths that do not exist here, including `docs/reports/`, `docs/experiments/`, `.agents/projects/`, and `lib/`. Ignore its design-doc mode unless a compatible design-doc workflow is available.

A monthly Codex scheduled task outside the repository should compare these skills with current Marin `main` and scan the upstream skill catalog for newly relevant workflows. It should open a draft update PR when vendored content changed and report new-skill candidates for human triage; it should not add skills automatically. The repository stores the pinned source and deviations, not scheduling infrastructure.

**Taxonomy rollout gate:** do not merge a change that activates this taxonomy until the `task` label exists and an issue-migration dry run shows that the coordinated cutover will leave every issue with exactly one Kind. Apply the audited relabeling immediately after merge, then verify the invariant before filing new issues under this workflow.

- When an agent creates a PR or issue, add the `agent-generated` label.
- Agent comments on PRs/issues must begin with `🤖`.
- **Keep the body current.** Treat an issue or PR body as the entry point for a cold reader and comments as the chronological record. Update the body when the current scope, result, decision, or important links change; do not make readers reconstruct current state from the comment thread.
- **Lead long-running work with a link-free summary.** Put a short plain-text TL;DR at the top of long-running experiment and research-question issues. For a non-trivial PR, make the opening paragraph a current summary without requiring a separate heading. Put citations and artifact links in later detail.
- **Use visual communication when it materially speeds up understanding.** Prefer Mermaid diagrams for pipeline stages, dependencies, experiment designs, and comparisons because GitHub renders them natively in issues and PRs; use tables or plots when they communicate the result better. Keep visuals concise, accurate, and labeled, and ensure the surrounding text still states the key takeaway. Do not add decorative or redundant visuals.
- **Always reference code with commit-pinned permalinks.** Whenever an issue, PR, or comment points at code, link it as a commit-pinned GitHub permalink (`blob/<sha>/path#Lx-Ly`), never a bare path or branch link. If the code is a one-off, commit it on its permanent branch before linking it.
- **Branch names.** Worktree harnesses may auto-prefix branches with an agent name and a random slug (e.g. `claude/happy-bose-180d63`). Before opening a PR, rename the branch with `git branch -m` so the branch list is scannable. Use a stable, lowercase `<agent-name>` that identifies the agent (e.g. `codex` or `claude`):
  - With an existing issue: `<agent-name>/issue-<issue-number>-<short-kebab-summary>` (e.g. `codex/issue-187-readme-revamp`).
  - Otherwise: `<agent-name>/<short-kebab-summary>` (e.g. `codex/readme-resources`).
- **Sub-issues.** Use GitHub's native sub-issue metadata for parent/child relationships: `gh api -X POST repos/{owner}/{repo}/issues/{parent}/sub_issues -f sub_issue_id={child_id}`. Reserve sub-issues for decomposing one engineering work item into parts, never for research-question to experiment relationships.
- **Don't put `fixes #N` in PR titles.** Issue-closing keywords (`fixes #131`, `closes #131`, `resolves #131`) belong in the PR body.
- **HuggingFace uploads.** When uploading anything to HuggingFace under `marin-dna/*`, include a README with a commit-pinned permalink to the producing pipeline or training script, a one- or two-sentence contents/provenance description, and the tags `biology, genomics, dna`. Draft the README for user review before pushing to HuggingFace.
- **Collapse large content.** Wrap logs longer than 40 lines, large tables, and code dumps in `<details><summary>…</summary>…</details>`. Keep the main claim outside the collapsed block.
- **Verify rendering.** After posting non-trivial Markdown, re-fetch the issue, comment, or PR body and check line breaks, indentation, lists, code blocks, and details blocks. Correct publishing glitches.

### Issue taxonomy

Every issue gets exactly one **Kind** label, zero or more **Topic** labels, and any applicable metadata labels. Kind selects the body structure and lifecycle. Topic describes the affected system or research area. Keep the label set minimal.

- **Kind, pick exactly one.** `bug`: broken behavior or a regression. `task`: an improvement, feature, refactor, build, documentation change, or other concrete work item. `experiment`: one bounded unit of research with a hypothesis or goal. `research-question`: a durable human-declared question synthesized across multiple experiments.
- **Topic, zero or more.** `infrastructure`: tooling, pipelines, CI, migrations, compute plumbing, and dataset or evaluation builds. `evals`: the evaluation apparatus itself, including an evaluation dataset, scoring protocol, or metric; measuring a model does not by itself add this topic. `data`: training-data construction, including genome projection, region labeling, and filtering; evaluation datasets use `evals`. `modeling`: architecture, objective, tokenization, loss weighting, or context. `hyperparameter-optimization`: optimizer, schedule, weight decay, learning rate, model size, or compute scaling. `baselines`: reference models such as Evo2, GPN-Star, AlphaGenome, conservation, or ChromBPNet. `interpretation`: UMAP, nucleotide-dependency maps, SAEs, or TF-MoDISco. `communication` and `documentation` are topics for dissemination and documentation work.
- **Metadata, zero or more.** `agent-generated`: created by an agent; agents always add it. `marin`: the change belongs upstream in Marin or Levanter. Priority labels and `epic` are metadata; `epic` is only for engineering decomposition.
- **Classify by outcome.** Building a training dataset or evaluation harness is `task` + `infrastructure` plus the relevant topic. Running a bounded analysis to learn from data or results is `experiment`, whether its starting point is a falsifiable hypothesis or an exploratory goal. Do not create an `eda` kind or mode field.
- **Treat preregistration as practice.** Fix the hypothesis or goal, design, primary metric, and stop criteria in the issue or logbook before execution, then record later deviations. Preregistration is not a separate label or experiment mode.
- **Human-gate research questions.** Agents may propose candidates but may create a `research-question` issue only after a human explicitly declares or approves its exact scope. Agents may create bounded `experiment` issues without per-issue approval when they remain inside research the human already authorized.
- **Keep research links bidirectional.** The `research-question` to `experiment` relationship is many-to-many and uses ordinary `#N` references. A research-question body must list every related experiment under `Related experiments`; each experiment body must list every question it informs under `Links` -> `Research questions`. Update and verify both bodies together. Do not use sub-issue metadata for this relationship.
