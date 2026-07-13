# Blog figures — "Genomic Language Model Optimization" (epic #361)

Recipes that redo the post's VEP figures on the **new** eval and broaden them
across the **dataset × method** matrix — {Mendelian, SGE} × {zero-shot LLR,
linear probe}. They reuse Eric's matplotlib style (`_style/`, vendored) and read
evals_v2 metrics from S3, so the redone figures sit seamlessly next to the
untouched Figures 1–4. See #361 for the figure × world matrix and sub-issues.

## Running

From the repo root (recipes are modules so they can import the shared style):

```bash
uv run python -m plots.blog.figure11_leaderboard_heatmap
```

Outputs land in gitignored `plots/output/blog/` as `<name>.{svg,png,pdf}` — the
**SVG** is the blog artifact; the **PNG** is for local eyeballing. One file per
`(figure, world)` cell, e.g. `figure11_leaderboard_heatmap__mendelian_llr.svg`.

## Data

- **Registered leaderboard models** (Fig 11) →
  `marin_dna.pipelines.evals.leaderboard` (`normalized_rows` /
  `probe_normalized_rows` / `sge_normalized_rows`).
- **Unregistered intermediate checkpoints** (the scaling ladder / mixture
  lineage behind Figs 5–10, which aren't in `dashboard/models.yaml`) →
  `marin_dna.pipelines.evals.blog_metrics` (direct-S3 read by run-name/step).

If S3 reads fail for lack of credentials, set `BOLINAS_S3_ANON=1` to read the
public bucket prefix anonymously.

Fig 6 uses the validation loss for the training-data region mapped to each
variant type (CDS / Upstream / Downstream), not the global validation loss.
