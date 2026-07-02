# issue #354 — exp135-1B-m5.1 GH200 inference-cost benchmark

Steady-state variant-scoring throughput + `$/1k` variants + peak VRAM for
`exp135-1B-m5.1` on a GH200, like-for-like with the Evo 2 cost table in
[#131](https://github.com/Open-Athena/marin-dna/issues/131#issuecomment-4869179889).
Results (the cost table) go in issue #354 — **not** here.

## What it measures

The **real** evals_v2 scoring path (`run_variant_score_bundle`) with the
production inference config, nothing stripped: **FWD+RC**, **bf16**,
**`torch_compile=True`**, and **`return_embeddings=True`** — the ~2×-heavier
`output_hidden_states` forward, which is what #131 measured for Evo 2 (so the
cost is like-for-like). Pooled embeddings are stored **f16** (our standard); that
cast is post-forward and moves neither throughput nor peak VRAM. exp135 pools
over 256 tokens, so f16 is sufficient — unlike Evo 2's 8192-token pool (see #354).

`inference_cost.py` does three things:

1. **Compile validation** — scores a tiny subset eager vs compiled (embeddings
   on) and reports `max |Δllr|` / `|Δjsd|`. The #318 embedding overlay left
   `torch.compile` + `return_embeddings` unvalidated; a crash means it's broken,
   a small delta is the expected float-reduction noise.
2. **Batch-size sweep** — one timed FWD+RC+embeddings pass per `--batch-sizes`
   value with a per-batch `TimingCallback`. Steady-state sec/strand-batch =
   median of inter-step diffs (drops the compile batch + the FWD→RC boundary).
   `variants/hr = 3600 · B / (2 · sec_per_strand_batch)` (the 2 = both strands
   per unique variant, matching Evo 2's FWD+RC accounting).
   `$/1k = 1000 / (variants/hr) · price_per_hr` (default `$2.29`, the #131 GH200 rate).
3. **Scores dump** — `sge_scores.parquet` (llr/jsd atoms + f16 emb_ref/emb_alt +
   variant columns) for the offline SGE-AUPRC regression check.

> **Context caveat (the headline):** exp135 runs at its native **256-token**
> context; the #131 Evo 2 numbers are at **8192**. Each is the model's real
> operating point — this is *as-deployed cost*, not a context-controlled speed test.

## Run

```bash
# 1. Stage the canonical GRCh38 to the dev box (has S3 creds) for file_mount.
scripts/issue354/stage_genome.sh

# 2. Cheap x86 validation first (compile+embeddings + end-to-end; A10G 24 GB
#    only fits a small batch with embeddings on, so sweep just batch_size=64):
sky launch scripts/issue354/run.yaml -c dna-exp135-val \
    --gpus A10G:1 --infra aws/us-east-2 --use-spot \
    --env HF_TOKEN=$HF_TOKEN --env BATCH_SIZES=64

# 3. The real benchmark on a GH200 ($2.29/hr, Lambda — same as #131):
sky launch scripts/issue354/run.yaml -c dna-exp135-cost --retry-until-up \
    --env HF_TOKEN=$HF_TOKEN

# 4. Fetch outputs, then tear down.
rsync -Pa dna-exp135-cost:~/out/ ./issue354_out/
sky down dna-exp135-val dna-exp135-cost
```

`HF_TOKEN` is required (the SGE dataset is private). GH200 is Lambda-only; if
capacity pins one region, re-launch `--retry-until-up` or pin `region:` under
`resources`.

## AUPRC regression check (offline, dev box)

The benchmark run doesn't touch S3. Confirm scoring didn't regress by deriving
the SGE AUPRC from `issue354_out/sge_scores.parquet` (`minus_llr` protocol) and
comparing it to the existing official cell
(`s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/mix-v0.9-p1B-i24-exp135-m5.1-step-59158/sge.parquet`)
via `marin_dna.pipelines.evals.metrics.compute_sge_metrics`.

## Files

- `inference_cost.py` — the benchmark (GPU-only; `bf16_full_eval` errors on CPU).
- `run.yaml` — SkyPilot task (GH200 default; A10G via `--gpus`/`--infra` override).
- `stage_genome.sh` — S3 → dev-box genome staging for the `file_mount`.
