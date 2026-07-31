# Project Guidelines

## Project Overview

**MarinDNA** is a framework for developing genomic language models (gLMs).

## Domain Conventions

- **Coordinate system.** The codebase consistently uses 0-based, half-open intervals for all genomic coordinates. Assume this everywhere; call out any deviation explicitly. Conversions to/from 1-based closed formats (GTF, VCF, SAM) happen at the tool boundary, not inside our code.
- **Canonical public human genome.** Prefer the authenticated S3 mirror at `s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/` whenever the caller has access and runs in the same AWS region as the bucket; it offers lower and more predictable latency for byte-range queries. For credential-free, public, or non-AWS consumers, use `marin-dna/human-genome` pinned to revision `11b9433582981bb929af333bc6422f10a8fd71b4`: uncompressed FASTA for `pyfaidx` + `fsspec` HTTP queries, or BGZF plus its indexes for HTSlib/samtools and full downloads. Never migrate already-materialized inputs merely to standardize their source. Both mirrors contain the Ensembl 115 GRCh38 soft-masked primary assembly with Ensembl sequence names (`1`...`MT`); coordinates are 0-based, half-open, and other `hg38` variants are not interchangeable.

## Research Code Values

This is research code. Prioritize **reproducibility** and **correctness** over architectural elegance.

- **Put Python logic in `src/marin_dna/` so pytest can reach it.** Even pipeline-specific functions belong in the library — the goal is testability, not a polished shared API. Inline Python in Snakemake rules (`run:` blocks in `Snakefile`/`.smk` files) should be thin glue calling into `src/marin_dna/`. Don't add `.py` script files under `snakemake/` (no `workflow/scripts/`) — all Python logic goes in the library.
- **Duplication beats premature abstraction *within* the library.** The "testable home" rule governs *entry* into `src/marin_dna/` — move logic in freely, even if similar code already exists elsewhere. A separate, weaker rule governs *deduplication*: only merge two similar functions into one shared helper when the shape has stabilized and they're genuinely doing the same thing. Until then, two near-copies in two pipeline modules is better than a premature abstraction coupling unrelated experiments.
- **Modularity is a means, not a goal.** Don't refactor for reuse that may never come. Straight-line code that reads top-to-bottom is often preferable to layered abstractions.
- **Test aggressively.** Every non-trivial function in `src/marin_dna/` should have tests — that's the whole reason logic lives there. For pipelines, add sanity checks on outputs (row counts, value ranges, coordinate invariants) rather than trusting that "it ran".
- **Assert defensively, everywhere.** Use `assert` liberally for invariants that *should* hold: coordinate bounds, dataframe shapes, no NaNs where none are expected, set membership, monotonicity, matching lengths between parallel arrays. A loud failure near the bug is worth far more than a silently corrupted result feeding into training.
- **Fail fast on silent-corruption risks.** Bioinformatics is full of off-by-one errors, strand mix-ups, and reference-build mismatches. When a result could be quietly wrong, prefer a check that crashes over a comment saying "this should be correct".
- **No premature generalizations.** If asked to implement a specific backend, dataset, or model variant, stick to that. Don't generalize to related use-cases on your own — offer the option, but only expand the scope when explicitly told to.
- **Stay in scope.** Don't remove or rewrite unrelated code in other pipelines or library modules while working on a task. Unrelated experiments may depend on exact current behavior.

## Code Structure

The codebase has five main components:

1. **Python Library** (`src/marin_dna/`) - Python logic for all pipelines lives here, including pipeline-specific modules. See **Research Code Values** above for why, and for how Snakemake rules should relate to it.

2. **Pipelines** (`snakemake/`) - Data processing workflows implemented in Snakemake
   - Read the pipeline's README before working on it — each `snakemake/<pipeline>/` has its own. If you change pipeline behaviour, update the README in the same PR so the next human or agent can onboard from it.
   - Always dry-run first (`-n` / `--dry-run`) before any real invocation.
   - Stop before reruns of steps the changes you made for this task did not intentionally touch. If the dry-run shows Snakemake planning to rerun an upstream or unrelated step — retriggered by a timestamp change, an unrelated code edit, `--rerun-triggers` defaults, etc. — stop and ask before running. Default assumption: such reruns are unintended and potentially expensive (training, genome downloads, large bedtools jobs).
   - Invoke as `uv run snakemake …` from the repo root, not bare `snakemake`.
   - Put pipeline-wide defaults (`cores`, `use-conda`, `default-storage-provider`, etc.) in the pipeline's `workflow/profiles/default/config.yaml`, not on the CLI. Snakemake auto-loads that profile, so every invocation picks them up.

3. **Experiments** - Marin-launched training/eval scripts. Each experiment is a **self-contained directory on its own branch** (its own `pyproject.toml` with marin in *base* deps + a `launch.py`), **not merged to `main`** (see "What gets merged to `main`" above) — cite it from its tracking issue via commit-pinned permalinks. Full setup, launch flow, and hard-won lessons live in the **`marin-experiment` skill** (`.agents/skills/marin-experiment/`).
   - **wandb run names.** Set run names that include `dna-exp<N>` (`<N>` = the experiment number from the issue) so runs filter by experiment.

4. **Plots** (`plots/`) - Self-contained Python scripts that turn pipeline metric parquets into figures. One file per recipe; outputs to gitignored `plots/output/<recipe>/`.
   - Load parquets from S3 with `polars.read_parquet("s3://…")` — native object_store support, no fsspec/s3fs needed.
   - Emit both `figure.svg` and `figure.png` to the output dir. SVG is the artifact to upload to GitHub (PR/issue/gist embeds); PNG is the local-iteration format — agents can `Read` it back to visually sanity-check (the `Read` tool renders raster but not SVG) and PNGs also render inline in agent conversations.
   - **Prefer seaborn's figure-level functions** (`relplot`/`catplot`/`displot`) over `sns.lineplot`/`scatterplot` drawn onto hand-built `plt.subplots` axes — seaborn bolted onto manual matplotlib axes reads as neither (missing labels, inconsistent fonts, off title placement). The figure-level call owns faceting, axis labels, facet titles, legend placement, and font scaling; set labels/titles via `g.set_axis_labels`/`g.set_titles` and the suptitle via `g.figure.suptitle`. Drop to manual `plt` axes only when a figure-level call genuinely can't express the layout (e.g. a custom per-facet SE band — add it by iterating `g.axes_dict`). For a **numeric hue**, seaborn auto-sorts it: use a continuous palette + `hue_norm` (e.g. `palette="viridis", hue_norm=(0, 0.65)`), not a dict palette (which forces categorical treatment), and don't pass `hue_order`.
   - **Error bars for standard error get no caps.** Draw ±1 SE (or any dispersion indicator) capless — `matplotlib.errorbar(..., capsize=0)`, which is the default. Caps read as a *bounded interval* with defined endpoints (a CI or range) and misrepresent an SE bar, so reserve them for genuine CIs/ranges. Keep the SE meaning explicit in the label (e.g. `"error bars = ±1 SE (bootstrap)"`).
   - **Two non-level-comparable metrics go on independent twin y-axes.** When two series aren't level-comparable by construction (different scale / matching / aggregation — e.g. two benchmarks, or matched-pair vs unmatched AUPRC), give each its own `ax.twinx()` autoscaled to its own range, color-code each axis to its line, and footnote that the axes are independent. A shared y-axis implies a level comparison we don't intend — the reader should compare shapes/trends, never the vertical gap between the lines.

5. **Scripts** (`scripts/`) - One-off, investigation, and reproduction scripts (analysis, ad-hoc evals, uploads, debugging). **Tracked**, so they can be committed and **permalinked** from issues/PRs. Put any one-off script you might re-run, reference, or cite for reproduction here — *not* in gitignored `scratch/`. Group by issue when it helps (`scripts/issue<N>_*.py` or `scripts/issue<N>/`). Reserve `scratch/` (gitignored) for ephemeral data/artifacts only — checkpoint downloads, intermediate parquets, dumps — never for code you'll point at. This still applies to library logic: anything reusable or worth a test belongs in `src/marin_dna/` (see **Research Code Values**); `scripts/` is for the genuinely one-off.

### What gets merged to `main`

`main` is the **reusable-core framework**: the library kernel, training-data construction, our-model evaluation (`evals_v2`), and `dashboard`/`docs`. Open a PR to `main` only for that core.

- **Assume non-core work never merges.** Experiments, one-off analyses, competitor baselines, and dead-ends stay on their own branches — the aim isn't to prune them out of `main` later, it's to never entangle them in the first place. Keep them self-contained (`experiments/`, `scripts/`), not woven into the library. Lifecycle complement to **Stay in scope** and **No premature generalizations** above.
- **A branch is a permanent reference — merging isn't.** Commit and push freely: a commit-pinned permalink to an unmerged branch is all you need to cite a result or reproduce an experiment from its tracking issue. Nothing has to land on `main` to stay reachable.

## Development Practices

- **Package management**: Use `uv` for Python dependencies
- **Bioinformatics tools**: Use Conda for external CLI tools (bedtools, twoBitToFa, etc.)
- **Testing**: Run `uv run pytest` before committing
- **Code quality**: Pre-commit hooks enforce ruff formatting and linting
- **Documentation**: Before merging a PR, make sure all the relevant READMEs are updated. READMEs describe how to run or use a thing, not what was found — experimental results (tables, leaderboards, key findings, per-genome stats, etc.) belong in the GitHub issue tracking that work, not in any README. Results drift; READMEs shouldn't.
- **Markdown source formatting**: Do not hard-wrap prose to a fixed column width. Keep each paragraph or list item on one source line and let editors/renderers wrap it visually; use manual line breaks only when Markdown structure requires them.
- **Where to run.** For quick work (small data, smoke tests, dev iteration), run locally on the current node — but first check system load (`uptime` / `cat /proc/loadavg`); multiple agent sessions share this small instance. Be careful about parallelizing local subprocesses: it has crashed the instance more than once (requiring reboot). Cap parallel jobs conservatively (rule of thumb: `nproc/2` or less). For heavy work (training, large-scale evals, anything GPU-bound), launch on SkyPilot. Always confirm with the user before launching SkyPilot resources — they're not free.
- **Babysit new jobs early.** First time running a script / config / cluster combination? Check actively within the first few minutes rather than passively waiting. Look for: progress rate sane (a common silent failure is CPU fallback when GPU was expected), device count matches what you asked for, no immediate OOM / mount errors / auth failures. Notifiers fire on completion or timeout — they don't tell you the run spent 4 hours on CPU.
- **Parallel sky sweeps.** When evaluating a grid of independent snakemake targets (e.g. every training-step checkpoint of one model arm), prefer launching one sky cluster per target over running the whole DAG on a single big cluster. Each cluster `--down`s on idle, parallelism scales with AWS capacity, and a failure in one target doesn't block the others. The canonical helper is `snakemake/analysis/evals_v2/sky/parallel_sweep.sh` — it takes snakemake target paths as args, derives one cluster per target, and waits for all to finish. The pattern relies on `run.yaml` exposing `$SNAKEMAKE_ARGS` so each cluster runs `snakemake -- <target>` and produces exactly that one parquet. Heads up: bursting >~24 `g5.xlarge` into `us-east-2` typically saturates AZs — sky reports `ResourcesUnavailableError` on the late arrivals. Re-running the helper with just the failed targets after the earlier clusters `--down` is usually enough; sky has no cross-region fallback for AWS-pinned tasks.

### Type Annotations
- Type-annotate all function parameters and return values in `src/marin_dna/`.
- Use Python 3.11+ syntax (`list[str]`, `X | None`); reach for `typing` only for constructs that still require it.

## Autonomy Boundaries

- Never push to `main` without explicit user approval.
- Never close or merge PRs/issues without explicit user approval.

## GitHub Communication

- When an agent creates a PR or issue, add the `agent-generated` label.
- Agent comments on PRs/issues must begin with `🤖`.
- **Use visual communication when it materially speeds up understanding.** Prefer Mermaid diagrams for pipeline stages, dependencies, experiment designs, and comparisons because GitHub renders them natively in issues and PRs; use tables or plots when they communicate the result better. Keep visuals concise, accurate, and labeled, and ensure the surrounding text still states the key takeaway. Do not add decorative or redundant visuals.
- **Always reference code with commit-pinned permalinks.** Whenever an issue, PR, or comment points at code — a function that backs a claim, a script that reproduces a result, the line a reader should run — link it as a commit-pinned GitHub permalink (`blob/<sha>/path#Lx-Ly`), never a bare path or branch link (branches move; the reference rots). If the code isn't committed yet (e.g. a one-off you wrote in `scratch/`), move it to tracked `scripts/`, commit + push, *then* link it. This is the default for **all** GH posts, not just `agent-research`.
- For iterative investigations the user wants tracked in their own issue, use the `agent-research` skill — issue body is the living doc, comments are the append-only log with commit-pinned permalinks to code.
- **Branch names.** Worktree harnesses may auto-prefix branches with an agent name and a random slug (e.g. `claude/happy-bose-180d63`). Before opening a PR, rename the branch with `git branch -m` so the branch list is scannable. Use a stable, lowercase `<agent-name>` that identifies the agent (e.g. `codex` or `claude`):
  - With an existing issue: `<agent-name>/issue-<issue-number>-<short-kebab-summary>` (e.g. `codex/issue-187-readme-revamp`).
  - Otherwise: `<agent-name>/<short-kebab-summary>` (e.g. `codex/readme-resources`).
- **Sub-issues.** Use GitHub's native sub-issue metadata for parent/child relationships — `gh api -X POST repos/{owner}/{repo}/issues/{parent}/sub_issues -f sub_issue_id={child_id}` — not free-text references in the issue body. The metadata renders in the UI and is queryable; body references drift.
- **Don't put `fixes #N` in PR titles.** Issue-closing keywords (`fixes #131`, `closes #131`, `resolves #131`) belong in the PR *body* — that's where GitHub's auto-close picks them up just the same. Titles should describe the change itself, not the metadata.
- **HuggingFace uploads.** When uploading anything to HuggingFace under `marin-dna/*` (datasets *or* models), include a README that contains: (a) a commit-pinned permalink to the snakemake pipeline (for datasets) or training script (for models) that produced it, (b) a 1–2 sentence description of contents/provenance, (c) the minimal tag set `biology, genomics, dna`. Draft the README content for user review *before* pushing to HF.
- **Collapse large content.** When posting issues, comments, or PRs that include logs (>40 lines), large tables, or code dumps, wrap the content in `<details><summary>…</summary>…</details>`. Easier for humans to scan; agents still read the full body.
- **Verify rendering.** After posting any issue, comment, or PR with non-trivial markdown (tables, lists, code blocks, multi-paragraph bodies), re-fetch the body (`gh issue view`, `gh pr view`, or `gh api`) and check for broken line breaks, dropped indentation, missing blank lines around lists/code blocks, or other rendering glitches. HEREDOC-passed bodies through `gh` can introduce stray whitespace; if so, fix with `gh issue edit` / `gh pr edit`.

### Issue labels

Every issue gets **exactly one Type** label + **any number of Area** labels, plus meta labels. Type = what *kind* of work; Area = what *part* of the project. The two axes are orthogonal and compose. Keep it minimal — one Type, usually one Area (sometimes none, occasionally two); don't stack every plausible Area. Agents always add `agent-generated`.

- **Type (pick exactly one).** `research-question` — a durable, revisitable question pursued across many experiments over time (a north-star; rare). `experiment` — one preregistered, scoped run with the hypothesis/goal fixed *before* running. `eda` — a bounded exploratory analysis of data/results with no preregistered hypothesis (where most `agent-research` investigations land). `infrastructure` — tooling, pipelines, CI, migrations, compute plumbing; this includes **builds** — a new eval dataset or a training-data pipeline is `infrastructure` + an Area, *not* an `experiment`. `bug` — something is broken.
- **Area (zero or more).** `evals` — work on the evaluation **apparatus itself**: a new eval dataset, scoring protocol, or metric. It is **not** applied just because a model was measured — a training run reporting VEP AUPRC is `experiment` + `modeling`/`data`, not `evals`; and eval datasets (incl. neutral-site/cLLR scoring support) are `evals`, never `data`. `data` — **training-data** construction (genome projection, region labeling, filtering). `modeling` — core gLM recipe (architecture, objective, tokenization, loss weighting, context). `hyperparameter-optimization` — optimizer/schedule/weight-decay/LR sweeps, and **scaling** — model size / compute (training a bigger model is HPO, not `modeling`). `baselines` — reference models (Evo2, GPN-Star, AlphaGenome, conservation, ChromBPNet). `interpretation` — UMAP, nucleotide-dependency maps, SAEs, TF-MoDISco.
- **Meta (zero or more).** `agent-generated` — created by an agent (agents always add this). `marin` — the change really belongs upstream in marin/levanter.
- **Picking a Type.** Preregistered hypothesis + a defined run → `experiment`; durable question that outlives any single run → `research-question`; looking at data/results with no hypothesis yet → `eda`; building or fixing plumbing (incl. dataset/eval builds) → `infrastructure`/`bug`.
- **`research-question` ↔ `experiment` is many-to-many, NOT parent/child.** An experiment *references* the question(s) it informs with a plain `#N` in its body; a research-question's body *curates links* to the experiments as they accrue. One experiment may touch several questions. Reserve GitHub sub-issue metadata for decomposing a single work item into parts (e.g. a 4-part pipeline build), never for research-question → experiment.
- **`epic` is for engineering decomposition only** — a build split into parts. Don't use it to organize research; that's what Area labels + `research-question` hubs are for.
- **`agent-research` is a working method, not a label.** It produces an `eda` (usually) or `research-question` issue plus the relevant Area label(s); don't invent a workflow label.
