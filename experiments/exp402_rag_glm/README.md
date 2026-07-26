# exp402: fixed-layout ortholog-RAG gLM

This permanent research-branch experiment trains the first model for issue
[#402](https://github.com/Open-Athena/marin-dna/issues/402). It consumes the
immutable `bolinas-dna/zoonomia-rag-v1-v1` revision, tokenizes the 2,048-token
documents with uniform causal loss, and trains a scratch 45.9M-parameter Qwen3
model for 999,948,288 tokens.

The eight-token character tokenizer is vendored in [`tokenizer/`](tokenizer/).
This keeps the exact `[PAD]`, `[UNK]`, `[BOS]`, `[SEQ]`, `a`, `c`, `g`, `t`
mapping inside the experiment workspace copied to every Iris/Zephyr worker.
The launcher checks SHA-256 digests for all three tokenizer files and includes
those digests in the tokenized-cache fingerprint.

## Frozen recipe

| Item | Value |
| --- | --- |
| Model | Qwen3, hidden 640, MLP 2560, 7 layers, 5 query/KV heads |
| Parameters | 45.9M at vocabulary size 8 |
| Context | 2,048 tokens |
| Batch | 64 documents = 131,072 tokens/step |
| Steps | 7,629 |
| Tokens | 999,948,288 |
| Initialization | Scratch |
| Accelerator | One `v6e-4`, with `v5p-8` fallback |
| Online eval | Mendelian RAG harness every 1,000 steps by default; optional offline-only fallback |
| HF exports | Every 1,000 steps and at final step 7,629 |
| W&B | group `dna-exp402-v1`, run `dna-exp402-rag-h640-p46M-1B` |

AdamH is transferred from the Complete(d)-inspired reference at the actual
batch and one-billion-token horizon. The pinned values are:

```text
learning_rate = 0.008293207887305696
adam_lr       = 0.0010372270725352284
epsilon       = 1.1700427342623003e-08
beta1         = 0.9
beta2         = 0.9999
max_grad_norm = 0.1
warmup        = 0.1
linear decay  = 0.2
```

The existing 46M/256-token checkpoint is deliberately not loaded: expanding
both its context and vocabulary would add a second experimental variable to
this first operational run. The geometry, not the weights, is reused.

The online harness is `mendelian_traits_rag_255`, backed by the immutable
`marin-dna/evals_mendelian_traits_rag_harness_255_v1` revision
`9acedb683463477f34745af30a63a289873008a4`. Each example shares its 1,920-token
retrieval prefix across the reference and alternate branches through Levanter's
paged KV cache. Fixed batches of 16 reduce device dispatch overhead; raw
nucleotide log-likelihood ratios are aggregated by the existing matched-variant
AUPRC metric. The experiment pins Marin's exact `lm-eval` fork because the
current Marin wheel advertises that integration but omits the dependency from
its built metadata.

If online harness initialization blocks training, launch with
`EXP402_ONLINE_EVAL=0`. This changes no model, data, optimizer, or training
horizon setting; it only omits the in-process harness. The launcher always
exports Hugging Face checkpoints at steps 1,000, 2,000, … under the run's
`hf/step-<N>` directory so the same frozen Mendelian, Complex, and SGE datasets
can be scored independently with `scripts/issue402_score_rag_hf.py`.

## Validate

From this directory:

```bash
uv lock
uv sync
uv run pytest -q
uv run ruff check launch.py test_launch.py
uv run python launch.py --version 2026.07.24
```

The last command only prints the lowered artifact plan. It does not tokenize or
train without `--run`.

## Launch on Iris

Run from this directory with `WANDB_API_KEY` available:

```bash
uv run iris --cluster=marin job run \
  --no-wait --user ubuntu --job-name dna-exp402-rag-h640-p46m-1b \
  --cpu 1 --memory 2g --region us-east5 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \
  -e EXP402_ONLINE_EVAL 1 \
  -- python launch.py --version 2026.07.24 --run
```

The immutable tokenized-cache artifact is built first. Re-launches reuse that
cache and the rolling checkpoint. The launcher uses current Marin's lazy
artifact API; no removed `ExecutorStep` compatibility layer is involved.

## Offline evaluation and frozen probe

`scripts/issue402_offline_eval_sky.yaml` scores one exported checkpoint on the
frozen Mendelian, Complex, and SGE harnesses. The checkpoint sweep uses one
queued Sky job per 1,000-step export so it gets early signals without holding
training open on the in-process harness.

The exports use Transformers 5's Qwen3 config schema: Llama-3 scaling and
`rope_theta=500000` are stored under `rope_parameters`. The repository-wide
offline environment intentionally remains on Transformers 4.57, so all offline
loads must go through `load_rag_model_config_hf` in
`src/marin_dna/pipelines/rag_glm/offline_eval.py`; it translates those values to
Transformers 4's `rope_scaling` and top-level `rope_theta` before model
construction. Calling `AutoModelForCausalLM.from_pretrained` directly in that
environment silently falls back to unscaled RoPE with theta 10,000 and produces
invalid metrics.

The final Mendelian representation probe is a separate Sky task,
`scripts/issue402_probe_sky.yaml`. It pools exactly the final human segment at
0-based half-open token coordinates `[1793, 2048)` (255 tokens), averages the
forward/reverse-complement allele embeddings in float32, and uses
`[emb_ref, emb_alt - emb_ref]`. The frozen classifier protocol is
`StandardScaler` plus L2 logistic regression, outer leave-one-chromosome-out CV,
inner five-fold chromosome-grouped tuning over `C = logspace(-12, 4, 17)`, with
four CPU workers. Probe and zero-shot likelihood AUPRC are computed on the same
prediction rows with the existing per-chromosome-weighted metric.

`online_eval_100m.py` keeps online-eval debugging off the training critical
path. It loads the completed 104M Levanter checkpoint, limits the frozen
Mendelian task to an even number of rows so strand pairs stay together, and
runs the same custom paged-cache method as the in-training callback. It does not
update model weights. The full 1,000-step curves and raw scores continue to come
from the independent offline scorer. Per-sample online logging is disabled
because the zero-shot custom task constructs requests directly and intentionally
has no `doc_to_target`; the aggregate online metric does not call that method.
The experiment-local compatibility shim excludes lm-eval's unused multimodal
HF adapter under Transformers 5; the text-only Levanter adapter and evaluator
remain loaded and are asserted in tests.

`scripts/issue402_parity_eval_sky.yaml` scores the exact first 1,842 rows of the
frozen training split with the independent HF implementation. This is the
smallest deterministic prefix that has both target classes in every represented
annotation subset; it contains 921 complete forward/reverse-complement pairs.
The scorer asserts those pairs and writes the row-selection policy into the
output manifest.

Run the 1,842-row parity smoke directly on one TPU worker:

```bash
uv run iris --cluster=marin job run --no-wait --user ubuntu \
  --job-name dna-exp402-online-parity-p104m-final-smoke1 \
  --enable-extra-resources --tpu v6e-4 --preemptible \
  --extra tpu \
  --cpu 16 --memory 56g --disk 100g --region us-east5 --timeout 1800 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e EXP402_ONLINE_CHECKPOINT_PATH "$EXP402_ONLINE_CHECKPOINT_PATH" \
  -e EXP402_ONLINE_MAX_EXAMPLES 1842 \
  -e EXP402_ONLINE_RUN_ID dna-exp402-online-parity-p104M-final-smoke1 \
  -- python online_eval_100m.py
```

## Gated 103.8M scale rung

After the completed 45.9M run was operationally healthy, the approved scale
rung in `launch_100m.py` froze a single modest size step: Qwen3 hidden 768, MLP
3072, 11 layers, and 6 query/KV heads (103,838,976 parameters at vocabulary size
8). It reuses the exact token cache, batch 64, 999,948,288-token horizon,
transferred AdamH values, TPU policy, and 1,000-step HF export cadence from the
46M run. Launch it with
`EXP402_ONLINE_EVAL=0`; the already-running offline scorer provides the requested
early evaluation without coupling training progress to harness initialization.
The 104M task requests 56 GB host RAM because v6e workers expose 60.2 GB
allocatable; the model and accelerator configuration are otherwise unchanged.

## Fresh 30,000-step follow-up

The 1B-token loss curves were still falling at their final checkpoints, so the
approved follow-up trains **both** model sizes from scratch for 30,000 optimizer
steps (3,932,160,000 tokens each):

| Launcher | Parameters | W&B run |
| --- | ---: | --- |
| `launch_30k.py` | 45.9M | `dna-exp402-rag-h640-p46M-30K-scratch` |
| `launch_100m_30k.py` | 103.8M | `dna-exp402-rag-h768-p104M-30K-scratch` |

Both launchers reuse the immutable corpus and token cache, batch 64, context
2,048, and the same Complete(d)-inspired AdamH transfer function. The transfer
is re-resolved at the actual 3.932B-token horizon. Its schedule is 10% warmup,
70% stable learning rate, then 20% linear decay to zero.

Validation loss, permanent native checkpoints (including optimizer state), and
Hugging Face exports all occur every 1,000 steps. This explicitly fixes the
first runs' retention mistake: those launchers exported HF weights every 1,000
steps but inherited Marin's `keep=[]`, so only the final native optimizer-state
checkpoint survived.

Both launchers request 56 GB of host RAM so their v6e-4 tasks fit below the
worker's 60.2 GB allocatable limit.

Validate and lower the two plans from this directory:

```bash
uv run pytest -q
uv run ruff check launch_30k.py launch_100m_30k.py \
  test_launch_30k.py test_launch_100m_30k.py
EXP402_ONLINE_EVAL=0 uv run python launch_30k.py --version 2026.07.26
EXP402_ONLINE_EVAL=0 uv run python launch_100m_30k.py --version 2026.07.26
```

Launch both with `EXP402_ONLINE_EVAL=0`; the frozen Mendelian, Complex, and SGE
tasks are scored offline at every 1,000-step HF export. This keeps the blocked
in-process harness off the training critical path while preserving the
requested early-signal cadence.

After exports exist, launch their offline evaluations in bounded spot-only
batches from the repository root:

```bash
CODE_REVISION=$(git rev-parse HEAD) MAX_PARALLEL=4 \
  scripts/issue402_30k_eval_sweep.sh \
  46m:1000 104m:1000 46m:2000 104m:2000
```

The dispatcher checks that each export exists, skips checkpoints whose three
benchmark metric parquets are already complete, and writes one local log per
Sky cluster. It never selects on-demand instances; failed spot runs can be
passed to the same command again after inspecting their logs.

## Superseded full-state continuation to 60,000 steps

The later optimizer-batch audit superseded these continuations: their global
batch contained 16 times fewer tokens than the historical 256-token reference
recipe. Both were stopped after their already-saved native and Hugging Face
checkpoints were verified. The retained artifacts are diagnostic only; this
section preserves how that superseded run was constructed.

Because validation loss was still descending at step 30k, `launch_60k.py` and
`launch_100m_60k.py` extend both model sizes to 60,000 total steps. They restore
the permanent native step-24,000 checkpoints with model weights, optimizer
moments, global step, RNG, and data-loader offset intact. W&B records the full
plateau LR at step 24,000 and the first reduction at step 24,001, so this is the
last checkpoint before any decay.

The continuations deliberately retain `OPTIMIZER_30K` rather than re-resolving
the token-transfer hyperparameters: changing its learning rates would introduce
a discontinuity against the restored optimizer state. Raising the trainer
horizon to 60k keeps the LR flat through step 48k and applies the same 20%
linear-decay fraction over steps 48k--60k.

Native optimizer-state checkpoints and HF exports remain on the 1,000-step
cadence. Validation loss remains every 1,000 steps, while the expensive frozen
offline VEP suites run only every 5,000 steps. Levanter names the terminal HF
export `step-59999`; the native terminal checkpoint is `step-60000`.

Validate and lower the continuation plans from this directory:

```bash
uv run pytest -q test_launch_60k.py test_launch_100m_60k.py
uv run ruff check launch_60k.py launch_100m_60k.py \
  test_launch_60k.py test_launch_100m_60k.py
EXP402_ONLINE_EVAL=0 uv run python launch_60k.py --version 2026.07.26
EXP402_ONLINE_EVAL=0 uv run python launch_100m_60k.py --version 2026.07.26
```

The commands above only print the plans. Submit each plan through an Iris
coordinator; do **not** add `--run` to a launcher invoked directly on the local
shared node, because without an Iris client context Marin falls back to its
local executor and attempts to rebuild the frozen cache locally.

```bash
WANDB_API_KEY=$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')

EXP402_ONLINE_EVAL=0 uv run iris --cluster=marin job run \
  --no-wait --user ubuntu --job-name dna-exp402-rag-h640-p46m-60k \
  --cpu 1 --memory 2g --region us-east5 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \
  -e EXP402_ONLINE_EVAL 0 -- \
  python launch_60k.py --version 2026.07.26 --run --max-concurrent 2

EXP402_ONLINE_EVAL=0 uv run iris --cluster=marin job run \
  --no-wait --user ubuntu --job-name dna-exp402-rag-h768-p104m-60k \
  --cpu 1 --memory 2g --region us-east5 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \
  -e EXP402_ONLINE_EVAL 0 -- \
  python launch_100m_60k.py --version 2026.07.26 --run --max-concurrent 2
```

Evaluate the 5k checkpoints from the repository root:

```bash
CODE_REVISION=$(git rev-parse HEAD) MAX_PARALLEL=4 \
  scripts/issue402_60k_eval_sweep.sh \
  46m:25000 104m:25000 46m:30000 104m:30000
```

The dispatcher is spot-only by default. If every `us-east-2` availability zone
rejects the A10G spot request, set `ALLOW_ON_DEMAND=1` to retain the task YAML's
spot-first policy while permitting its on-demand fallback. This keeps the
fallback explicit and leaves `MAX_PARALLEL` as the concurrency bound.

## Replacement 2M-token-batch scratch runs

The replacement experiment matches the historical per-region optimizer batch
in tokens while retaining the 2,048-token RAG documents:

| Item | Value |
| --- | --- |
| Launchers | `launch_large_batch_30k.py`, `launch_100m_large_batch_30k.py` |
| Parameters | 45.9M and 103.8M |
| Global batch | 1,024 documents × 2,048 = 2,097,152 tokens/update |
| Device geometry | 4 TPU chips; 256 documents/chip (46M), 128 documents/chip/microstep (104M) |
| Accumulation | None for 46M; two exact microsteps for 104M |
| Updates | 30,000 from scratch |
| Tokens/model | 62,914,560,000 |
| Schedule | 10% warmup, 70% stable, 20% linear decay to zero |
| Native checkpoints | Permanent every 1,000 updates, including optimizer state |
| HF exports | Every 1,000 updates and terminal `step-29999` |
| Validation loss | Every 1,000 updates |
| Offline VEP | Mendelian, Complex, and SGE every 5,000 updates plus final |

AdamH is re-resolved for the actual batch and 62.9B-token horizon. The pinned
values are:

```text
learning_rate = 0.009575405934753806
adam_lr       = 0.0005230681221568245
epsilon       = 2.3201566843642267e-08
beta1         = 0.9
beta2         = 0.9984011994401821
max_grad_norm = 0.1
warmup        = 0.1
linear decay  = 0.2
```

The accumulation setting has an explicit measurement gate. With no
microbatching, Levanter resolves the global batch to 256 documents per chip.
The 46M full-batch smoke completed two optimizer updates; the 104M full-batch
HLO did not compile because it requested 31.66 GiB from a 31.25 GiB HBM
device. The original two-step timing was not a steady-state measurement:
Levanter compiles train-step variants during both early updates. Its timing
implementation does establish that loading and hooks are separate from
`throughput/duration`; after the initial 20--22 second cache fill, measured
batch handoff was below one millisecond. Six-step benchmarks, with validation
limited to one final out-of-step batch, will select the production geometry
from warmed updates 3--6. The completed medians were 1.712 seconds/update
(1.225M tokens/s, 12.88% MFU) for 46M at PDP=256 with no accumulation and
3.449 seconds/update (610k tokens/s, 13.82% MFU) for 104M at PDP=128 with two
exact microsteps. Median loader handoff was 0.00032 and 0.00556 seconds,
respectively, confirming that data loading is not the steady-state bottleneck.

Validate and lower all production and accumulation-smoke plans from this
directory:

```bash
uv run pytest -q test_launch_large_batch_30k.py \
  test_launch_100m_large_batch_30k.py
uv run ruff check launch_large_batch_30k.py \
  launch_100m_large_batch_30k.py \
  launch_large_batch_pdp256_benchmark.py \
  launch_100m_large_batch_pdp128_benchmark.py \
  test_launch_large_batch_30k.py \
  test_launch_100m_large_batch_30k.py
EXP402_ONLINE_EVAL=0 uv run python launch_large_batch_30k.py \
  --version 2026.07.26.5
EXP402_ONLINE_EVAL=0 uv run python launch_100m_large_batch_30k.py \
  --version 2026.07.26.5
```

Lowering prints the plans and does not run them. Actual execution must go
through Iris; never add `--run` to a launcher invoked directly on the shared
node.

```bash
uv run iris --cluster=marin job run \
  --no-wait --user ubuntu --job-name dna-exp402-rag-h640-p46m-b2m-30k \
  --cpu 1 --memory 2g --region us-east5 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \
  -e EXP402_ONLINE_EVAL 0 -- \
  python launch_large_batch_30k.py --version 2026.07.26.5 \
  --run --max-concurrent 2

uv run iris --cluster=marin job run \
  --no-wait --user ubuntu --job-name dna-exp402-rag-h768-p104m-b2m-30k \
  --cpu 1 --memory 2g --region us-east5 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \
  -e EXP402_ONLINE_EVAL 0 -- \
  python launch_100m_large_batch_30k.py --version 2026.07.26.5 \
  --run --max-concurrent 2
```

Launch only exports that exist, in bounded batches, from the repository root:

```bash
CODE_REVISION=$(git rev-parse HEAD) MAX_PARALLEL=4 \
  scripts/issue402_large_batch_30k_eval_sweep.sh \
  46m:5000 104m:5000 46m:10000 104m:10000
```

The dispatcher accepts 5k, 10k, 15k, 20k, 25k, and final 29,999. It requires
an exact clean HEAD, checks each HF export before launch, skips complete output
triples, uses spot-only Sky resources by default, and retains the explicit
`ALLOW_ON_DEMAND=1` capacity fallback. Plot final curves against both optimizer
step and cumulative training tokens so their scale remains legible beside the
earlier 131,072-token/update runs.
