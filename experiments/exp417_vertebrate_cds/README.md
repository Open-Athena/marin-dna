# exp417: projected vertebrate CDS sanity check

This permanent research-branch experiment answers the training sanity check in
[#417](https://github.com/Open-Athena/marin-dna/issues/417): compare a projected
CDS corpus restricted to the Zoonomia mammals with the same corpus plus the
non-mammalian MultiZ projections. Both models train from scratch and differ
only in the source dataset.

Sequence case is semantic source repeat masking, not a conservation
annotation. The human anchor cohort itself remains selected by the pipeline's
pinned phyloP conservation filter; that filter never rewrites emitted
characters or case. Uppercase bases receive loss weight 1.0 and lowercase
repeat-masked bases receive loss weight 0.01. The same case-aware format is
applied to the Hugging Face `train` and `validation` splits; the tokenizer
lowercases token identities only after the loss-weight array has been derived

## Frozen matched recipe

| Item | Value |
| --- | --- |
| Arms | `cds_mammals_only` vs `cds` (combined vertebrates) |
| Model | Qwen3, hidden 1152, MLP 4608, 12 layers, 9 query/KV heads |
| Context | 255 projected bases plus BOS = 256 tokens |
| Batch | 8,192 sequences = 2,097,152 tokens/step |
| Steps | 5,000 |
| Tokens per arm | 10,485,760,000 |
| Initialization | Scratch, seed 0 |
| Loss weighting | uppercase 1.0; lowercase repeat mask 0.01 |
| Validation | each dataset's held-out `validation` split, every 500 steps |
| Native checkpoints | optimizer state retained every 500 steps |
| Hugging Face exports | every 500 steps |
| Online VEP eval | none; frozen VEP scoring runs offline |
| Accelerator | one `v6e-4`; no higher-cost fallback |
| W&B | group `dna-exp417-v1` |

The standard Adam optimizer is the exact exp353 recipe: learning rate
0.00430097, betas 0.66756/0.952222, epsilon 6.77142e-15, global gradient norm
0.995188, weight decay 0.1, 10% warmup, 70% stable phase, and 20% linear decay
to zero. Current Marin retains that optimizer implementation under
`levanter.optim.config.AdamConfig`; this experiment does not substitute AdamH.

The seven-token character+BOS tokenizer is vendored in
[`tokenizer/`](tokenizer/). It was copied from
`marin-dna/tokenizer-char-bos` revision
`a73e9d9ee636f722b4c378703c9e2997857809b2`; the launcher verifies SHA-256
digests before constructing either artifact graph.

## Runtime and approval envelope

The four completed exp353 runs used this exact 5,000-step, 8,192-sequence
recipe on `v6e-4` and recorded 31,408–32,847 seconds (8.72–9.12 hours) of
W&B runtime. Budget 10 TPU hours per arm, or 20 TPU-node hours total.

As of 2026-08-01, [Google's public TPU pricing](https://cloud.google.com/tpu/pricing)
lists Trillium/v6e in `us-east5` at $2.70 per chip-hour on demand or $1.35
under DWS Flex. A `v6e-4` has four chips, so the conservative public-list
envelope for both arms is $216 on demand or $108 under Flex. The launcher has
no `v5p-8` fallback, preventing an availability retry from silently selecting
a materially more expensive TPU. Request approval against a rounded **$220
training ceiling**; Iris/TRC credits may reduce the actual billed amount.
Tokenization and the separately auto-downing A10G offline evaluation are not
included in that TPU ceiling.

## Dataset revision gate

The dataset cards and uploads are reviewed before publication. Until their
immutable Hugging Face commits exist, the launcher deliberately refuses to
lower or run. Supply both 40-character revisions:

```bash
export EXP417_CDS_MAMMALS_REVISION=<hf-commit>
export EXP417_CDS_COMBINED_REVISION=<hf-commit>
```

The planned repositories are:

```text
mammals_only         bolinas-dna/vertebrate-v1-cds_mammals_only
combined_vertebrates bolinas-dna/vertebrate-v1-cds
```

After publication, replace the environment gate with the reviewed immutable
commits in `launch.py` before a paid run.

## Validate and lower

From this directory:

```bash
uv lock
uv sync
uv run pytest -q
uv run ruff check launch.py test_launch.py
uv run python launch.py --version 2026.08.01
```

The final command only prints the two lowered artifact plans. It does not
tokenize or train without `--run`.

To lower one arm at a time:

```bash
EXP417_ARMS=mammals_only uv run python launch.py --version 2026.08.01
EXP417_ARMS=combined_vertebrates uv run python launch.py --version 2026.08.01
```

## Launch on Iris

Launch the two arms as independent Iris jobs so tokenization, provisioning, and
retries are isolated. A paid launch requires explicit approval.

```bash
uv run iris --cluster=marin job run \
  --no-wait --user ubuntu --job-name dna-exp417-cds-mammals-only \
  --cpu 1 --memory 2g --region us-east5 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \
  -e EXP417_ARMS mammals_only \
  -e EXP417_CDS_MAMMALS_REVISION "$EXP417_CDS_MAMMALS_REVISION" \
  -e EXP417_CDS_COMBINED_REVISION "$EXP417_CDS_COMBINED_REVISION" \
  -- python launch.py --version 2026.08.01 --run
```

Repeat with job name `dna-exp417-cds-combined-vertebrates` and
`EXP417_ARMS=combined_vertebrates`. The immutable token cache is built before
training and reused after a retry. Validation loss, native optimizer-state
checkpoints, and reloadable Hugging Face exports all use the 500-step cadence.

## Frozen offline VEP evaluation

The terminal `step-4999` exports are evaluated once on the held-out `test`
split with the current `evals_v2` zero-shot harness. The experiment-local
[`evals.yaml`](evals.yaml) restricts the DAG to the two matched checkpoints
and three coding-relevant benchmarks:

- Mendelian traits: signed FWD/RC-averaged LLR, overall and consequence-level
  matched-pair AUPRC with cluster-bootstrap uncertainty.
- Complex traits: absolute FWD/RC-averaged LLR with the same matched-pair
  aggregate and consequence-level reporting.
- SGE: signed FWD/RC-averaged LLR, per-accession and missense/splicing AUPRC
  with bootstrap uncertainty.

The expected immutable exports are:

```text
gs://marin-us-east5/checkpoints/dna-exp417-cds-mammals-only-p255m-b2m-5k/2026.08.01/hf/step-4999
gs://marin-us-east5/checkpoints/dna-exp417-cds-combined-vertebrates-p255m-b2m-5k/2026.08.01/hf/step-4999
```

Dry-run from `snakemake/analysis/evals_v2/` before launching any GPU:

```bash
uv run --project ../../.. snakemake --workflow-profile none --dry-run \
  --configfile ../../../experiments/exp417_vertebrate_cds/evals.yaml -- all
```

After both exports pass their final checkpoint gates, launch one bounded,
auto-downing A10G worker using the existing project task. A paid launch still
requires explicit approval:

```bash
sky launch -c dna417-cds-vep --down \
  snakemake/analysis/evals_v2/sky/run.yaml \
  --env SNAKEMAKE_ARGS="--configfile ../../../experiments/exp417_vertebrate_cds/evals.yaml \
    --default-storage-prefix s3://oa-bolinas/snakemake/analysis/issue417_cds_sanity/2026.08.01/ \
    -- all"
```

The six metric outputs are isolated under:

```text
s3://oa-bolinas/snakemake/analysis/issue417_cds_sanity/2026.08.01/results/metrics/
```

This is intentionally offline: the training graphs have no lm-eval harness,
which avoids changing the matched optimizer/training path and lets the current
VEP pipeline provide all consequence-level tables and uncertainty estimates.
