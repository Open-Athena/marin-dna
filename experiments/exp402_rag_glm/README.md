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

The final Mendelian representation probe is a separate Sky task,
`scripts/issue402_probe_sky.yaml`. It pools exactly the final human segment at
0-based half-open token coordinates `[1793, 2048)` (255 tokens), averages the
forward/reverse-complement allele embeddings in float32, and uses
`[emb_ref, emb_alt - emb_ref]`. The frozen classifier protocol is
`StandardScaler` plus L2 logistic regression, outer leave-one-chromosome-out CV,
inner five-fold chromosome-grouped tuning over `C = logspace(-12, 4, 17)`, with
four CPU workers. Probe and zero-shot likelihood AUPRC are computed on the same
prediction rows with the existing per-chromosome-weighted metric.

## Gated 103.8M scale rung

After the completed 45.9M run showed checkpoint-1,000 AUPRC above the fixed
prevalence references on Mendelian, Complex, and SGE, `launch_100m.py` freezes a
single modest size step: Qwen3 hidden 768, MLP 3072, 11 layers, and 6 query/KV
heads (103,838,976 parameters at vocabulary size 8). It reuses the exact token
cache, batch 64, 999,948,288-token horizon, transferred AdamH values, TPU policy,
and 1,000-step HF export cadence from the 46M run. Launch it with
`EXP402_ONLINE_EVAL=0`; the already-running offline scorer provides the requested
early evaluation without coupling training progress to harness initialization.
