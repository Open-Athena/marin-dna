# Issue 517 functional specialists

This permanent experiment-branch project trains five annotation-first 0.25B DNA specialists for issue #517.
It is an experiment, not a merge proposal.

The arms are CDS, 3′ UTR, TSS region, ncRNA, and enhancer.
Their anchors come from complete Ensembl GRCh38 release 115 annotation across all qualifying transcripts, without RefSeq and without an Ensembl-canonical-only filter.
Enhancers are ENCODE SCREEN Registry V4 dELS and pELS elements.
Coordinates are GRCh38 0-based half-open inside the workflow.

Every arm uses the same Qwen3-like 0.25B geometry, character-plus-BOS tokenizer, case-aware loss, optimizer, seed 0, 1,024-sequence per-device microbatch, and 5,000-step schedule.
The global batch is 8,192 sequences of 256 tokens, or 10,485,760,000 token presentations per arm.
Hugging Face exports and retained native checkpoints are written every 500 steps.

The project is independently locked to Python 3.12 and Marin source commit `53b5b33041f742c7f4991223b0085e41ece4c458`.
The immutable full-data workflow producer is commit `e42a4ea1eca760219e0add91004b45cac59b19c9`.
The vendored tokenizer is byte-identical to `marin-dna/tokenizer-char-bos` revision `a73e9d9ee636f722b4c378703c9e2997857809b2` and is hash-checked before graph construction.

## Dataset boundary

Training reads only public Hugging Face datasets at immutable 40-character revisions.
It never trains from S3.
S3 is workflow-owned producer storage and is outside the training input contract.

The launcher pins the five anonymously verified public datasets at these immutable Hub revisions:

- `marin-dna/functional-cds` at `eb6bc7737c7f546870020a4e3d4c7a2a20d4c92c`
- `marin-dna/functional-utr3` at `790ec0ade6df6dce8e597058fc819dcf13f2eed1`
- `marin-dna/functional-tss` at `90f596e35b9d0a79e3f7a7c889581158472694eb`
- `marin-dna/functional-ncrna` at `ecb7e9480be5e2c18db59b3544a0c61e23fc2a2f`
- `marin-dna/functional-enhancer` at `07fac22abf6d158b8a155150d8aa49e813e6125e`

The launcher rejects any missing, mutable, or malformed revision before graph construction.

## Verify

```bash
uv sync --python /usr/bin/python3.12 --locked --group dev
uv run --python /usr/bin/python3.12 --locked pytest
```

Before the first data-bearing launch, run the tokenizer preflight on a real Iris child worker:

```bash
uv run --python /usr/bin/python3.12 --locked iris --cluster=marin job run \
  --no-wait --job-name exp517-tokenizer-worker-preflight \
  --cpu 1 --memory 2G --region us-east5 --extra=tpu \
  -e MARIN_PREFIX gs://marin-us-east5/MarinDNA/exp517_functional_specialists \
  -- python -m exp517_functional_specialists.tokenizer_preflight \
  --version 2026.08.24 --run
```

## Training launch

CDS was launched first and verified through immutable Hub download, complete tokenization, real TPU optimizer steps, and W&B telemetry.
At explicit user direction, the other four independent arms were launched before the originally planned step-500 checkpoint gate.
The authenticated account can read the `open-athena` W&B entity but cannot create model runs there after the organization migration, so this experiment uses its existing writable `gonzalobenegas/marin` project.

```bash
uv run --python /usr/bin/python3.12 --locked iris --cluster=marin job run \
  --no-wait --job-name exp517-cds \
  --cpu 1 --memory 2G --region us-east5 --extra=tpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_ENTITY gonzalobenegas \
  -e WANDB_PROJECT marin \
  -e MARIN_PREFIX gs://marin-us-east5/MarinDNA/exp517_functional_specialists \
  -e EXP517_TPU_REGION us-east5 \
  -e EXP517_TPU_VARIANT v5p-8 \
  -e EXP517_TPU_RAM 56g \
  -e EXP517_TPU_PREEMPTIBLE true \
  -e EXP517_ARM cds \
  -e UV_PROJECT /app \
  -- bash -lc 'cd /app && uv sync --locked --extra tpu && \
  exec uv run --locked python -m exp517_functional_specialists.experiment \
  --version 2026.08.24 --run'
```

Use the same command with `utr3`, `tss_region`, `ncrna`, and `enhancer`; verify each arm's Hub revision, token cache, TPU allocation, checkpoint export, and W&B run independently.
Each arm has an independent run ID, checkpoint root, tokenized cache, and W&B run.

## Evaluation boundary

Evaluate only the development split unless the user separately authorizes held-out access.
Remove complete mature-miRNA groups before every metric, table, plot, or model-selection decision.
The primary statistic is AUPRC, with paired joint bootstrap uncertainty for whether each home specialist ranks first.
Do not read, generate, publish, or summarize even-autosome or chromosome-Y held-out labels, predictions, or metrics from this branch.

The dedicated `snakemake/analysis/evals_v2/config/issue517.yaml` config registers five arms at steps 500, 1,000, 1,500, 2,000, 2,500, 3,000, 3,500, 4,000, 4,500, and the terminal step 4,999.
Its 110 development cells comprise the complete 50-cell Mendelian and 50-cell Complex Traits trajectories plus the 10-cell CDS-only SGE trajectory.
It pins all three development datasets, enables complete-group mature-miRNA exclusion only for Mendelian metrics, and does not register a held-out dataset.
The same workflow jointly resamples Mendelian match groups across all five arms, reports `P(home ranks first)` for each of the eight preregistered subsets, and applies the two-consecutive-checkpoint 95% persistence rule.

After all checkpoints exist, launch the evaluation workflow with:

```bash
sky launch snakemake/analysis/evals_v2/sky/run.yaml \
  -c evals-v2-exp517 \
  --env SNAKEMAKE_ARGS="--configfile config/issue517.yaml" \
  --down
```
