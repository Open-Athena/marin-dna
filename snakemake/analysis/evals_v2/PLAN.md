# Simplify evals v2 inference and metrics

## Goal

Use one VEP score rule and one metrics rule.
New VEP score parquets include pooled allele embeddings by default.
Matched-pair metric parquets retain their existing AUPRC row structure and gain Group SMD columns.
Existing S3 artifacts remain valid historical outputs and are not backfilled or recomputed as part of this change.

## Configuration

- Set `inference.return_embeddings: true`, `inference.torch_compile: true`, and `inference.bf16: true` in `config/config.yaml`.
- Keep these three semantic settings global.
- Reject `return_embeddings`, `torch_compile`, and `bf16` in checkpoint entries so a checkpoint cannot silently override the run-wide behavior.
- Pass `inference.bf16` through VEP scoring, LL-gap inference, embedding UMAP inference, and GPU runtime validation.
- Keep `inference.batch_size: 128` and `inference.eval_accumulation_steps: null` as fallbacks.
- Continue to configure `batch_size` and `eval_accumulation_steps` per checkpoint because their useful values depend on model size, context length, and embedding width.
- Write the execution values explicitly on checkpoint entries rather than deriving them from checkpoint names at runtime.
- Use the prior embedding runs as the initial execution matrix: models below 1B at 255/256 bp use batch 128 with no accumulation override; 1B models at 255/256 bp use batch 64 with accumulation every 8 steps; 2B and 4B models at 255/256 bp use batch 16 with accumulation every 8 steps; existing 512 bp checkpoints keep batch 64 and gain accumulation every 8 steps.
- Keep checkpoint-specific execution knobs out of Snakemake `params` because they do not change the intended output schema or values.
- Add one config comment stating that score parquets produced before this change may lack `emb_ref` and `emb_alt`.

## Canonical score bundle

- Keep `compute_scores` as the only VEP inference rule.
- Keep the existing output path `results/scores/{model}/{dataset}.parquet`.
- Keep the existing variant columns and scalar atoms `llr_fwd`, `llr_rc`, `jsd_fwd`, and `jsd_rc`.
- Always add `emb_ref` and `emb_alt` to newly computed scores.
- Preserve the current embedding contract: entire-window mean pooling, special tokens excluded, FWD/RC averaging in FP32, and Float16 parquet storage.
- Do not use `return_embeddings` as a migration trigger that makes existing score parquets stale.
- Apply the embedding output contract when a score output is missing or an explicit rerun is requested.
- Do not launch a bulk score backfill or recomputation.
- Allow zero-shot metrics to consume legacy score parquets because AUPRC and Group SMD need only the scalar score atoms.
- Require recomputation before a probe consumes a legacy score parquet without embeddings.

## Canonical metrics bundle

- Keep `compute_metrics` as the only VEP metrics rule.
- Keep the existing output path `results/metrics/{model}/{dataset}.parquet`.
- Preserve one matched-pair row per `(score_type, subset)`, including `_global_` and `_macro_avg_` rows.
- Preserve `value` and `se` as the AUPRC estimate and cluster-bootstrap standard error.
- Preserve the existing `n_groups`, `n_rows`, `model`, `dataset`, and `split` columns.
- Add Group SMD to each matched-pair row with `group_smd_value`, `group_smd_se`, `group_smd_ci_low`, `group_smd_ci_high`, `group_smd_confidence_level`, `group_smd_available`, `group_smd_unavailable_reason`, `group_smd_uncertainty_method`, `group_smd_n_bootstrap`, and `group_smd_n_bootstrap_valid`.
- Compute Group SMD for direct subset and `_global_` rows from one positive-minus-mean-negative gap per match group.
- Mark Group SMD unavailable on `_macro_avg_` rows because averaging subset-standardized effects defines a different statistic.
- Reuse joint match-group bootstrap draws across score columns while computing uncertainty, then discard the draws after the summary columns are produced.
- Leave QTL and SGE metric schemas and behavior unchanged.
- Leave leaderboard loading unchanged.
  It continues reading the existing AUPRC `value` and `se` columns and ignores the added Group SMD columns.
- Treat legacy metrics parquets without Group SMD columns as valid AUPRC-only outputs.
- Apply the enriched metrics schema when a metrics output is missing or an explicit rerun is requested.
- Do not rewrite existing metrics parquets solely to add Group SMD columns.

## Remove obsolete paths

- Delete `config/overlays/return_embeddings.yaml`.
- Delete the `compute_grouped_vep_scores` and `compute_grouped_vep_report` rules and the `grouped_vep_metrics` aggregate target.
- Remove the grouped rule include from `workflow/Snakefile`.
- Remove `results/grouped_vep_scores/`, `results/grouped_vep_metrics/`, and `results/grouped_vep_bootstrap/` from the active workflow contract.
- Remove grouped-only path constants and tests that assert those paths.
- Stop persisting aligned bootstrap sidecars.
- Leave already-written grouped S3 artifacts untouched.
- Remove overlay-specific instructions and `--rerun-triggers mtime` workarounds from the README, probe rule, Sky configuration, and probe diagnostics.

## Maintained interfaces

- Add a `bf16: bool` argument to `compute_variant_scores`, `compute_hf_ll_gap`, and `compute_region_embeddings`.
- Map that argument to Hugging Face `bf16_full_eval`.
- Add a checkpoint-aware helper for `eval_accumulation_steps`, parallel to `get_model_batch_size`.
- Keep the scoring kernel's embedding layout and numerical reductions unchanged.
- Refactor the grouped VEP metric helper to enrich the existing AUPRC table instead of returning duplicate long-form metric rows for AUPRC and Group SMD.

## Verification

- Test that global `return_embeddings`, `torch_compile`, and `bf16` values reach inference and that checkpoint entries cannot override them.
- Test checkpoint-specific batch size and accumulation resolution, including fallback behavior and invalid values.
- Test that newly computed VEP scores contain finite, equally sized `emb_ref` and `emb_alt` vectors with Float16 storage.
- Preserve tests for FP32 pooling and FWD/RC averaging.
- Test that the AUPRC columns and row order are unchanged after Group SMD columns are added.
- Test Group SMD values, uncertainty columns, direct-scope failure modes, and the unavailable `_macro_avg_` contract.
- Test that leaderboard normalization produces identical output when the source parquet contains the added Group SMD columns.
- Test that legacy AUPRC-only metric parquets remain readable.
- Extend GPU runtime validation to exercise embeddings with BF16 and compilation enabled while retaining scalar-score comparison against the legacy baseline.
- From `snakemake/analysis/evals_v2`, run `uv sync --locked --group dev` and `uv run --locked pytest`.
- Dry-run the default metrics target, probes, LL-gap, and embedding UMAP with `uv run --locked snakemake -n` and confirm existing outputs are not selected solely because the new optional columns exist.
- Use local test fixtures for verification and do not submit inference or evaluation jobs or mutate S3 artifacts.
- Do not launch paid GPU validation without explicit approval.

## Acceptance criteria

- A newly computed VEP score parquet contains scalar scores plus `emb_ref` and `emb_alt`.
- The default run uses BF16 and `torch.compile` from global configuration.
- Checkpoint entries control only execution sizing, not semantic inference output.
- Matched-pair metrics contain the previous AUPRC rows and values plus Group SMD columns.
- The leaderboard renders the same AUPRC rows without a loader change.
- No active rule produces or consumes a grouped-only score, metric, or bootstrap path.
- Legacy score and metric parquets remain usable for the capabilities their schemas contain.
- No bulk recomputation or backfill is launched as part of the implementation.
