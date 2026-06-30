# Evo 2 baseline (issue #131)

SkyPilot tasks for Evo 2 inference. Project sky conventions apply: `us-east-2`, `project=dna` label, reuse the cluster across iterations with `sky exec`.

## `sky/eval_matched_pair.yaml` — Evo2 on matched-pair leaderboards (#161 mendelian, #162 complex)

Produces per-variant score-bundle parquets (`llr`, `minus_llr`, `abs_llr`, `next_token_jsd_mean` × `{avg, _fwd, _rev}`) and PairwiseAccuracy ± SE metrics parquets for `evo2_1b_base`, `evo2_7b`, `evo2_40b` on `bolinas-dna/evals_mendelian_traits` (default) or `evals_complex_traits` (set `DATASET=complex_traits`), at 8192-bp context, FWD+RC averaged. Uses the joint-bundle pattern from PR #184; Evo2's kernel is a no-KV-cache twin in `scripts/evo2_eval/_evo2_scoring.py`.

**One GH200 (96 GB) fits all three models.** Vortex does its own placement; no `torchrun`, no HF DDP. Reuse the cluster across the sweep via `sky exec` — avoids redundant Docker pulls and genome downloads, and skips GH200 capacity-out risk on re-launch.

```bash
# Smoke test first — verify both FWD and RC passes complete on a small set.
sky launch -c evo2-eval scripts/evo2_eval/sky/eval_matched_pair.yaml \
  --gpus GH200:1 --env MODEL=evo2_1b_base --env LIMIT=50

# Full mendelian sweep on the same cluster.
sky exec evo2-eval scripts/evo2_eval/sky/eval_matched_pair.yaml \
  --env MODEL=evo2_1b_base
sky exec evo2-eval scripts/evo2_eval/sky/eval_matched_pair.yaml \
  --env MODEL=evo2_7b
sky exec evo2-eval scripts/evo2_eval/sky/eval_matched_pair.yaml \
  --env MODEL=evo2_40b

# Complex sweep on the same cluster (smaller dataset, ~10% the time).
sky exec evo2-eval scripts/evo2_eval/sky/eval_matched_pair.yaml \
  --env MODEL=evo2_1b_base --env DATASET=complex_traits
sky exec evo2-eval scripts/evo2_eval/sky/eval_matched_pair.yaml \
  --env MODEL=evo2_7b --env DATASET=complex_traits
sky exec evo2-eval scripts/evo2_eval/sky/eval_matched_pair.yaml \
  --env MODEL=evo2_40b --env DATASET=complex_traits

# Pull results (both datasets).
rsync -avz sky-evo2-eval:/workspace/results/evo2_mendelian_traits/ \
  results/evo2_mendelian_traits/
rsync -avz sky-evo2-eval:/workspace/results/evo2_complex_traits/ \
  results/evo2_complex_traits/
```

Extra envs the YAML threads through to the eval script:

- `DATASET_REVISION=<sha>` — pins the HF dataset to a specific commit / tag / branch (default: HEAD).
- `SKIP_METRICS=1` — writes only the scores parquet and skips `PairwiseAccuracy`. Use when the dataset isn't 1:1 paired (e.g. the 1:9 design in `evals_mendelian_traits` post-#194).
- `RETURN_EMBEDDINGS=1` — also fold the entire-window mean-pooled, both-allele, FWD+RC-averaged last-layer embeddings into the scores parquet as `emb_ref`/`emb_alt` `list[f16]` columns (issue #131; the Evo2 analog of the gLM #325 bundle). `EMB_LAYER=norm` (default) picks the Evo2 module to pool — `norm` is the post-final-norm last-layer state, the HF `last_hidden_state` analog; validated against `evo2.model.named_modules()` and fails loud with candidates if absent. **Requires RC on** (don't combine with `NO_RC_AVG`) and makes the forward heavier (it materializes a `[2·bs, 8192, D]` hidden state) — **halve `BATCH_SIZE`**, especially `evo2_40b` (`D=8192`, already at bs=1). The scores columns are byte-identical to a non-embedding run; the embeddings just ride along. Feed the resulting parquet to `train_variant_probe.py` (below).

### Metrics — AUPRC + cluster-bootstrap SE (post-PR-#194)

The post-#194 datasets are 1:9 matched, so PairwiseAccuracy no longer applies — run inference with `SKIP_METRICS=1`, then compute AUPRC post-hoc from the per-variant scores parquet:

```bash
# Each model is independent — run in parallel (`&` + `wait`) if you're
# in a hurry; the script is single-threaded but each invocation pegs
# one core for ~30 s at n_bootstrap=1000.
for m in evo2_1b_base evo2_7b evo2_40b; do
  uv run python scripts/evo2_eval/compute_auprc_metrics.py \
    --input results/evo2_mendelian_traits/${m}_train.parquet \
    --output results/evo2_mendelian_traits/mendelian_${m}_train_metrics.parquet \
    --model "$m" \
    --dataset mendelian_traits
done
```

This mirrors `snakemake/analysis/evals_v2/workflow/rules/metrics.smk` (PR #195) but in script form, since evo2 isn't wired into the evals_v2 pipeline. Output schema matches `compute_auprc_metrics` + `[model, dataset, split]` — the same shape the dashboard's `marin_dna`-family parquets use, so the metrics flow straight into `src/marin_dna/pipelines/evals/leaderboard.py`'s `family: evo2` resolver. Re-uploading means: upload via `gh-upload-asset`, bump `EVO2_METRICS_GIST_COMMIT` in `leaderboard.py`, rebuild the dashboard.

### Supervised linear probe on frozen embeddings (#131)

`train_variant_probe.py` trains the frozen-embedding linear probe on an Evo2 bundle that was scored with `RETURN_EMBEDDINGS=1`, reusing the exact library code — **no new logistic regression or metric**:

- `marin_dna.pipelines.evals.variant_probe.run_subset_probes` (#320) — the per-subset, nested leave-one-chromosome-out L2 probe over the in-bundle `emb_ref`/`emb_alt`, emitting a LOOC `probe_score` per variant.
- `marin_dna.pipelines.evals.metrics.per_chrom_weighted_ap` (#331) — the TraitGym per-chromosome-weighted AUPRC (primary), scored on `probe_score` and on the zero-shot LLR baseline through the same metric, plus the global match-group-bootstrap AUPRC (secondary).

CPU-only (sklearn on cached embeddings) — run it on the dev box / a CPU instance after rsyncing the bundle, like `compute_auprc_metrics.py`. The feature combo and LLR baseline are dataset-derived (directional `concat_ref_delta` + `minus_llr` for mendelian/sge; symmetric `sum_absdiff` + `abs_llr` for complex/caqtl/dsqtl), overridable with `--feature-combo` / `--baseline-col`.

```bash
for m in evo2_1b_base evo2_7b evo2_40b; do
  uv run python scripts/evo2_eval/train_variant_probe.py \
    --input results/evo2_mendelian_traits/${m}_train.parquet \
    --output-predictions results/evo2_mendelian_traits/${m}_train_probe.parquet \
    --output-metrics results/evo2_mendelian_traits/${m}_train_probe_metrics.parquet \
    --model "$m" --dataset mendelian_traits
done
```

This is the same protocol the gLMs run through the evals_v2 `probe`/metrics rules, so Evo2 vs the in-house gLMs is an apples-to-apples probe comparison (same feature rule, nested-LOOC CV, and headline metric). The `--emb-layer` of the scoring step is the one Evo2-specific knob — `norm` (last layer) keeps protocol parity; sweep mid-network layers (`--emb-layer blocks.<i>...`) only if the last layer underperforms.

### Tear down

```bash
sky down evo2-eval
```

## `sky/ll_gap.yaml` — LL-gap eval on functional vs non-functional tokens

Mean log-likelihood on phyloP-functional (uppercase) target tokens minus mean LL on non-functional (lowercase) target tokens. Positive gap = the model finds functional bases easier to predict — a self-supervised proxy for "captures functional/non-functional sequence structure" (biofoundation PR #18; issue #131 follow-ups #1-#6).

Default dataset: `bolinas-dna/genomes-v5-validation-intervals-v5_255_255` (16,384 × 255-bp CDS). Override `--env DATASET=...` to score the same gap on promoter (`v1_255_255`), 3'-UTR (`v15_255_255`), or enhancer (`v30_255_255`) regions — those are the variants surveyed in the issue follow-ups.

```bash
# Launch + first model on H100 (1B/7B fit).
sky launch -c evo2-llgap scripts/evo2_eval/sky/ll_gap.yaml \
  --env MODEL=evo2_1b_base

# Subsequent models on the same H100 cluster.
sky exec evo2-llgap scripts/evo2_eval/sky/ll_gap.yaml --env MODEL=evo2_7b_base

# 40B OOMs on 80 GB H100 at model construction. Use a GH200 cluster (96 GB).
sky launch -c evo2-llgap-big scripts/evo2_eval/sky/ll_gap.yaml \
  --gpus GH200:1 --env MODEL=evo2_40b

sky down evo2-llgap evo2-llgap-big
```

Per-row LL sums + token counts (not means — means break for all-upper or all-lower rows) written to `results/evo2_ll_gap/{model}__{dataset_tag}__n{limit}.parquet`.

## Insights from running issue #131 baselines (Apr 2026)

### Hardware that worked

| Hardware | Source | Notes |
|---|---|---|
| Lambda `gpu_2x_h100_sxm5` (2×H100 80GB SXM5) | us-south-2 | 1B+7B via `torchrun --nproc_per_node=2` on TraitGym v2 baseline. 40B requires Vortex sharding + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. $8.38/hr OD. |
| Lambda `gpu_1x_gh200` (GH200, 96 GB HBM3, ARM64) | us-east-3 | 40B fits on one GPU, no sharding, simpler path. Fits 1B/7B comfortably as well — current Mendelian sweep runs all three on one GH200. $2.29/hr. **Host is aarch64** — nvcr.io/nvidia/pytorch:25.04-py3 has an arm64 variant (auto-selected by docker). |

### Hardware that failed / was unavailable

- **AWS H100** (p5.4xlarge, p5.48xlarge, p5en.48xlarge, p5e.48xlarge): capacity-out across all zones in every region for ~12 hours. Quotas were fine (L-417A185B = 192 vCPUs OD in us-east-2); issue is pure AWS hardware supply.
- **Lambda `gpu_1x_b200_sxm6`** and `gpu_1x_h100_pcie` / `sxm5`: intermittently capacity-out. Lambda's capacity fluctuates minute-to-minute; `--retry-until-up` works but could take hours.
- **40B on single H100** via data-parallel `torchrun --nproc_per_node=2`: clean OOM on first real-inference batch (model + activations use ~78.5 GB on 80 GB H100 at bs=1, 8192 ctx). Needs ≥96 GB to fit single-GPU.

### Evo2 + HF Trainer pitfalls

1. **`auto_find_batch_size=True` is a no-op for `Trainer.predict()`**. It only wraps the train loop. For inference, tune manually — `find_max_batch_size` in `marin_dna.pipelines.evals.evo2` does OOM-descent halving from a `start` seed.

2. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is required for 40B on sharded 2×H100.** Without it, the failure masquerades as a vortex rotary `AttributeError: 'NoneType' object has no attribute 'shape'` (sin_k is None). With it, the same bs=1 run completes cleanly. Classic OOM-in-disguise. (Not needed on single-GPU GH200, but kept in the yaml as a safe default.)

3. **HF Trainer puts inputs on `cuda:0`, but Vortex may shard the embedding onto another device.** `_Evo2QuackModel` wraps `model.forward` with a small adapter that moves `input_ids` to the embedding's device on entry and moves logits back to the caller's device on exit. No-op on single-GPU.

4. **Running evo2's self-test on a sharded model leaks state** that corrupts the next real inference (hits the same rotary-sin=None error we saw pre-expandable_segments). `SKIP_SELF_TEST=1` is the clean workaround when sharding. On single-GPU GH200 the self-test runs cleanly.

5. **Self-test numerics differ slightly across GPU generations.** Upstream eps is `1e-3`; on GH200 our `evo2_7b_base` test gave loss 0.354 vs expected 0.352 (diff 0.002, fails eps strictly but within rounding).

6. **`sky cancel` does NOT propagate into `sudo docker run` children.** Added a `trap` in the YAML's `run:` block that `sudo docker rm -f $CONTAINER_NAME` on EXIT/INT/TERM so cancellations actually free the GPU. Container is explicitly named via `--name evo2-eval-$MODEL-$$` for this.

7. **`sky launch --retry-until-up` can respawn a cluster even after `sky down`** if the underlying bash process is still alive somewhere. Always verify with `sky status` after `sky down` and reap any ghosts immediately — we lost ~$3 to a ghost `evo2-big` that provisioned in australia-east-1 from a forgotten retry loop.

8. **`sky launch` vs `sky exec`:** `sky launch` on the same cluster rechecks/reruns `setup:`. `sky exec` only runs `run:`. If you've edited setup env vars, `sky launch`. If only the run command changed, `sky exec` (and pass `--gpus` to match the cluster's accelerators, or it will reject with a resource mismatch).

9. **Biofoundation install dropped (May 2026).** After PR #182 vendored the CLM subset into `marin_dna.model.*`, the variant-scoring path no longer needs biofoundation. The current Mendelian yaml's Docker image only installs `evo2`.

### Throughput summary (8192 ctx, 24,530 TraitGym v2 variants — single-strand)

| Model | Config | Rate | Full run |
|---|---|---|---|
| 1B | torchrun n=2 bs=8 on 2×H100 | 12.5 v/s | 34m |
| 7B | torchrun n=2 bs=8 on 2×H100 | ~3.3 v/s | 2h 3m |
| 7b_base | python bs=auto on 1×GH200 | ~1.7 v/s | ~4h |
| 40B | python bs=1 on 1×GH200 | 0.35 v/s | ~19h |
| 40B | python bs=2 sharded 2×H100 | 0.23 v/s | ~30h (didn't run) |

For the Mendelian sweep (~9820 variants × 2 strands with `rc_avg=True`), scale these rates by ~2× for the RC pass and ~0.4× for the smaller dataset.
