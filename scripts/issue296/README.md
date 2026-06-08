# issue #296 — stratified LL gap (per-token validation loss, sliced by genomic annotation)

One-off analysis for [Open-Athena/marin-dna#296](https://github.com/Open-Athena/marin-dna/issues/296):
generalize the functional/non-functional **LL gap** (#274/#8) into a re-sliceable
per-token loss field, and test whether the conserved/non-conserved gap **restricted
to the 1st/2nd codon positions** predicts missense VEP AUPRC better than the vanilla
all-token gap (the #279 missense-degrades-with-scale phenomenon). Findings live in
the issue (body = current state, comments = iteration log) — not here.

## Pipeline (two stages, decoupled)

**Stage 1 — cache per-token loss (GPU, once per checkpoint).** One FWD pass over
`bolinas-dna/zoonomia-v1-val_cds` (Ensembl r115 CDS, 20 bp splice flank); emit
`−log p` for every base, long-format, keyed by genomic coordinate. Reuses the
un-summed `compute_ll_clm` kernel (`marin_dna.model.scoring.compute_per_token_ll_clm`)
and `marin_dna.pipelines.evals.per_token_loss.compute_hf_per_token_loss`.

- `cache_per_token_loss.py` — orchestration (load ckpt → run → parquet; optional
  bit-exact `compute_hf_ll_gap` equivalence re-check, skippable with `--skip-ll-gap-check`).
- `sky/cache.yaml` — runs it on one A10G; loops over `MODELS` (val_cds only with
  `RUN_GENOMES_V5=0`). Checkpoints are pulled from the evals_v2-staged S3 exports.

**Stage 2 — annotate + measure (CPU, iterate freely).** Join the cached loss to
per-position annotations from the Ensembl r115 GTF (canonical transcripts) and
compute the conserved/non-conserved LL gap + mean loss within arbitrary strata.

- `annotate_and_measure.py` — codon position (per-segment GTF phase), 4-fold
  degeneracy, canonical 2 bp splice donor/acceptor (`GT`/`AG`) + broad ≤20 bp
  splice window, UTR, gene strand; writes per-stratum + by-codon + by-strand parquets.
  Reusable helpers (tested) in `marin_dna.pipelines.evals.per_token_annotate`.
- `scaling_analysis.py` — collect per-model gaps/losses + per-model VEP AUPRC
  (evals_v2 `mendelian_traits`, `minus_llr_avg`); correlate each stratum vs class AUPRC.
- `diag_splice_distance.py` — loss vs distance-into-intron diagnostic (splice dilution).
- `plot_regions.py` / `plot_gap_global_vs_task.py` / `plot_codon12_loss.py` —
  the issue figures (→ `plots/output/issue296/`).

## Artifacts (durable)

Caches + metrics: **`s3://oa-bolinas/analysis/issue296/`** (~255 MB) — `per_token/`
(per-token loss caches, ~28 MB/model), `stage2/` (per-stratum / by-codon / by-strand
metrics), `scaling_*.parquet` (cross-model aggregates), `ll_gap_check/`, `README.md`.

Code is on branch `claude/issue-296-stratified-ll-gap` (may never merge — keep the
branch so the issue's commit-pinned permalinks resolve).
