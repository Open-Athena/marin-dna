# Genomic Language Model Optimization editing workspace

> **Staging only: this branch and every child editing PR are intentionally
> permanent and unmerged. Do not target or merge this workspace into MarinDNA
> `main`.**

This directory is the single-file editing and local-review surface created for
[MarinDNA issue #373](https://github.com/Open-Athena/marin-dna/issues/373).
It imports the existing website PR without editorial changes, renders it with
the real pinned Open Athena website, and exports the integrated article back to
the website's exact paths.

## Pinned provenance

| Input | Immutable pin | Purpose |
| --- | --- | --- |
| Original blog source | [`eric-czech/marin-dna-post-202606@2abef91`](https://github.com/eric-czech/marin-dna-post-202606/commit/2abef91b37a16fde9c9cdf1cfa0046942442b97f) | Source commit whose article and 11 SVG Git objects were imported by the website PR |
| Website PR #59 import | [`Open-Athena/open-athena.github.io@4cd00c9`](https://github.com/Open-Athena/open-athena.github.io/commit/4cd00c970816c0caedaca570252ca390b6f61b67) | Exact initial Markdown and asset snapshot |
| Website renderer | [`Open-Athena/open-athena.github.io@4cd00c9`](https://github.com/Open-Athena/open-athena.github.io/commit/4cd00c970816c0caedaca570252ca390b6f61b67) | Actual build code, templates, CSS, fonts, and site-wide static assets used by the PR preview |

The independent SHA-256 import manifest is in
[`baseline.sha256`](baseline.sha256), and all pins and website paths are in
[`workspace.toml`](workspace.toml). The baseline verifier proves that every
canonical imported byte also exists at the pinned website PR commit before it
runs the real renderer. [`renderer.uv.lock`](renderer.uv.lock) freezes the
website dependencies, including Mistune 3.2.1 from the original source's
pre-preview lock; the website commit itself did not contain a lockfile. Builds
use `uv --frozen`.

## Canonical content

There is exactly one canonical article:

```text
blog/genomic-lm-optimization/content/blog/genomic-lm-optimization.md
```

Its blog-specific assets live at:

```text
blog/genomic-lm-optimization/static/assets/images/blog/genomic-lm-optimization/
```

Edit that Markdown file directly. Sections may be added, deleted, renamed, or
reordered; do not split it into section files.

## One-command preview

Prerequisites are Git, `uv`, and read access to
`Open-Athena/open-athena.github.io`. From the MarinDNA repository root, run:

```bash
uv run --no-project python src/marin_dna/blog_workspace.py preview
```

Then open
[http://127.0.0.1:8765/blog/genomic-lm-optimization/](http://127.0.0.1:8765/blog/genomic-lm-optimization/).
The command clones the website at the renderer pin into an isolated temporary
directory, overlays the canonical article/assets, runs the website's own build,
writes only the rendered gLM article and its referenced CSS/fonts/assets to the
ignored `.preview/` directory, and serves it. It watches the canonical Markdown
and asset directory, debounces saves, rebuilds with the pinned renderer, and
automatically reloads every connected browser after a successful build. A
failed rebuild is reported in the terminal while the last successful preview
remains available.

Other website pages and articles are neither copied into `.preview/` nor
available from the local server. Pass `--no-watch` to serve one build without
watching or browser reload. Stop the server with Ctrl-C. There is intentionally
no hosted preview.

Useful non-serving commands are:

```bash
# Fast local checks: missing referenced assets and footnote invariants.
uv run --no-project python src/marin_dna/blog_workspace.py validate

# Build the real site into the ignored .preview/ directory.
uv run --no-project python src/marin_dna/blog_workspace.py build

# Baseline-only proof of byte identity plus a real renderer build.
# This is expected to fail after intentional editorial changes.
uv run --no-project python src/marin_dna/blog_workspace.py verify-import
```

Validation fails loudly for missing local article assets, duplicate footnote
definitions, undefined footnotes, a renderer checkout that does not resolve to
the pinned commit, a changed baseline manifest, missing built CSS/fonts/assets,
or any website build failure.

## Branch and PR topology

The permanent integration branch is
`claude/issue-373-blog-staging`. The immutable tag
`issue-373-blog-staging-base` marks the common starting commit for every
parallel editing branch.

Start each child branch and worktree from that tag, never from MarinDNA
`main` or from a moving integration tip:

```bash
git fetch origin --tags
git worktree add ../marin-dna-blog-<topic> -b claude/issue-373-<topic> issue-373-blog-staging-base
```

Open each editing PR with base branch
`claude/issue-373-blog-staging`. A child PR may edit overlapping sections in
the one Markdown file. Accept useful changes independently; reconcile accepted
overlaps in a later integration PR targeting the same staging branch. Reviewers
can check out any child PR and run the one-command preview above.

## Deterministic final export

To materialize an inspectable export tree under this workspace:

```bash
uv run --no-project python src/marin_dna/blog_workspace.py export
```

That produces only the exact website-relative article and blog-asset paths
under the ignored `export/open-athena.github.io/` directory. To transfer the
integrated version directly into a checkout of website PR #59:

```bash
uv run --no-project python src/marin_dna/blog_workspace.py export \
  --destination /absolute/path/to/open-athena.github.io
```

The export atomically replaces the article and replaces only the exact
`static/assets/images/blog/genomic-lm-optimization/` directory, removing stale
blog assets without touching other website files. Run the website repository's
own tests or build afterward, then commit the exported paths there for the
final hand-off tracked by
[MarinDNA issue #367](https://github.com/Open-Athena/marin-dna/issues/367).
There is no concatenation or copy-paste step.
