# Current blog figure recipes

This directory contains the current `evals_v2`-backed plotting recipes for the
[Genomic Language Model Optimization](../../blog/genomic-lm-optimization/)
article. The imported historical plotting bundle remains under
[`genomic_lm_optimization/`](genomic_lm_optimization/).

## Mendelian leaderboard heatmaps

Figure 18 (zero-shot) and the collapsed Figure 19 (linear probe) follow the
canonical scope in [issue #370](https://github.com/Open-Athena/marin-dna/issues/370):

- zero-shot: the six headline models, each using its canonical protocol;
- probe: the four probe-capable models that overlap the zero-shot panel.

Run from the repository root:

```bash
uv run python -m plots.blog.figure11_leaderboard_heatmap
```

The recipe reads current `evals_v2` Mendelian metrics, fails if the expected
model sets are incomplete, and writes SVG, PNG, and PDF files to
`plots/output/blog/`. To copy the generated SVGs into the article after review:

```bash
uv run --project plots/blog/genomic_lm_optimization \
  python plots/blog/genomic_lm_optimization/src/sync_blog_assets.py \
  figure11_leaderboard_heatmap__mendelian_llr \
  figure11_leaderboard_heatmap__mendelian_probe
```

The shared heatmap renderer was ported from the commit-pinned Figure 11 code in
[`73dbfa7`](https://github.com/Open-Athena/marin-dna/tree/73dbfa7/plots/blog).
The article assets reproduce the collaborator-review figures pinned in issue
#370, with the original blog's reader-facing label `MarinDNA (1B/m5.1)` restored
in place of the internal run identifier `exp135-1B-m5.1`.
