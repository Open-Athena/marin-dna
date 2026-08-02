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
| TPU worker host RAM | 512 GiB container limit on the fixed 720 GB `v6e-4` VM |
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

## Runtime and compute allocation

The four completed exp353 runs used this exact 5,000-step, 8,192-sequence
recipe on `v6e-4` and recorded 31,408–32,847 seconds (8.72–9.12 hours) of
W&B runtime. Expect roughly 10 TPU hours per arm, or 20 TPU-node hours total.
The user authorized both runs on the free Iris/TRC allocation, so they do not
count against a paid experiment budget. The launcher has no accelerator
fallback: both arms request exactly one `v6e-4` in `us-east5`.

## Immutable datasets

The reviewed datasets and exact Hugging Face revisions are frozen directly in
`launch.py`:

```text
mammals_only         marin-dna/vertebrate-v1-cds_mammals_only @ d2bea760f6416775772699b821b266d3ae87245e
combined_vertebrates marin-dna/vertebrate-v1-cds              @ bfab878078c4ee6c0f47b760f1e5e0577549dc9d
```

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
retries are isolated. Both free Iris/TRC launches were explicitly approved.

```bash
uv run iris --cluster=marin job run \
  --no-wait --user ubuntu --job-name dna-exp417-cds-mammals-only \
  --cpu 1 --memory 2g --region us-east5 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \
  -e EXP417_ARMS mammals_only \
  -- python launch.py --version 2026.08.01 --run
```

Repeat with job name `dna-exp417-cds-combined-vertebrates` and
`EXP417_ARMS=combined_vertebrates`. The immutable token cache is built before
training and reused after a retry. Validation loss, native optimizer-state
checkpoints, and reloadable Hugging Face exports all use the 500-step cadence.

## Launch record

Both arms were submitted at 2026-08-01 23:09 UTC from immutable experiment
commit [`914fcdb`](https://github.com/Open-Athena/marin-dna/tree/914fcdbb0715580496681312d4664af9f7aee699/experiments/exp417_vertebrate_cds). The submitted workspace pins the two dataset revisions listed above.

- [mammals-only Iris job](https://iris.oa.dev/#/job/%2Fubuntu%2Fdna-exp417-cds-mammals-only)
- [original combined-vertebrate Iris job](https://iris.oa.dev/#/job/%2Fubuntu%2Fdna-exp417-cds-combined-vertebrates)
- [failed combined retry `r1`](https://iris.oa.dev/#/job/%2Fubuntu%2Fdna-exp417-cds-combined-vertebrates-r1)
- [combined retry `r2`, stopped after its last checkpoint to migrate to the safe runtime](https://iris.oa.dev/#/job/%2Fubuntu%2Fdna-exp417-cds-combined-vertebrates-r2)
- [safe mammals resume `r2`](https://iris.oa.dev/#/job/%2Fubuntu%2Fdna-exp417-cds-mammals-only-r2)
- [safe combined resume `r3`](https://iris.oa.dev/#/job/%2Fubuntu%2Fdna-exp417-cds-combined-vertebrates-r3)
- mammals host-RAM resume `r3` (submit after this revision is pushed)
- combined host-RAM resume `r4` (submit after this revision is pushed)

Both jobs use the authorized free Iris/TRC allocation. The upload-only
`issue417-hf` EC2 cluster was terminated after the publication artifacts and
verification manifests were preserved in S3; the separate HAL staging cluster
was left untouched.

The original combined arm reached step 159 before a temporary-checkpoint
serialization failure. Its committed step-50 checkpoint was preserved. Retry
`r1` reached the TPU worker but failed before checkpoint restoration because
the resubmission omitted the W&B credential. Retry `r2` was submitted at
2026-08-02 01:17 UTC with the credential restored, the same immutable recipe,
and the same checkpoint path. The mammals-only arm was not relaunched.

At 2026-08-02 01:44 UTC, retry `r2` had restored step 50, committed a fresh
temporary checkpoint at step 115, and advanced through step 163, beyond the
original failure point. The mammals-only arm had committed its native step-500
checkpoint, completed validation, finished the 1.02 GB step-500 Hugging Face
export, and advanced through step 606. Both parent jobs reported zero failures
and zero preemptions at that observation.

At 2026-08-02 02:27 UTC, the child training summaries exposed subsequent
preemptible-capacity events hidden by the still-running parent summaries. The
mammals child reported two preemptions and zero failures; its replacement
worker restored temporary step 822, advanced through step 890, and began
committing temporary step 889. The combined child reported one preemption and
zero failures after reaching step 493; its latest committed temporary checkpoint
was step 438 and it was pending replacement capacity. No arm was manually
relaunched: both retained the same checkpoint paths and frozen recipe.
At 2026-08-02 02:36 UTC, the recovered mammals worker committed temporary
checkpoint step 997 and then failed while starting native checkpoint step 1000.
The direct exception was `RuntimeError: Set changed size during iteration`
inside `asyncio.run` during JAX serialization; W&B's background
`AsyncioManager` then raised the same exception during teardown. The child
ended with one failure and two preemptions, while the complete step-997
checkpoint remained available for a three-step replay.

The combined arm was preempted a second time after committing its native
step-500 checkpoint. Iris restored that checkpoint automatically, resumed from
step 501, and had advanced through step 545 at 2026-08-02 02:50 UTC with zero
failures and two preemptions.

Safe resumes from this revision disable the time-based temporary-checkpoint
policy and set `WANDB_MODE=disabled`, removing both asynchronous runtimes
implicated in the race. Native optimizer-state checkpoints, validation, and
reloadable Hugging Face exports remain aligned every 500 steps. This is an
execution-only deviation: the datasets, model, initialization, optimizer,
batch order and size, seed, objective, loss weights, and terminal evaluation
are unchanged. Existing W&B history is retained through the last online
attempt; later progress remains in Iris logs and the durable training outputs.

Both safe resumes were submitted around 2026-08-02 03:00 UTC from immutable
experiment commit
[`aac3f82`](https://github.com/Open-Athena/marin-dna/tree/aac3f82/experiments/exp417_vertebrate_cds).
The mammals resume restored complete temporary checkpoint step 997, committed
native checkpoint step 1000, completed validation with loss 1.299, finished
the 1.02 GB step-1000 Hugging Face export, and advanced through about step 1020
by 03:08 UTC. This crossed the exact checkpoint boundary at which the previous
worker failed. The combined `r2` job was stopped only after a dry-run confirmed
the exact target and its step-565 temporary checkpoint was durable. Safe resume
`r3` restored step 565, committed native checkpoint step 1000 at 03:46:28 UTC,
completed validation with loss 1.304, and finished its four-file, 972.2 MiB
step-1000 Hugging Face export at 03:46:57 UTC before resuming training. Both
safe child jobs reported zero failures and zero preemptions at the 03:48 UTC
observation.

Both safe workers subsequently exhausted their 56 GiB container memory limit:
the mammals arm at about step 1740 and the combined arm at about step 1400.
Iris reported exit 137 and explicitly identified a container OOM; there were no
preemptions. The complete native checkpoints and four-file Hugging Face exports
at mammals step 1500 (validation loss 1.290) and combined step 1000 (validation
loss 1.304) were independently verified in GCS before recovery. The launcher
now requests 512 GiB of the same `v6e-4` VM's 720 GB host RAM, leaving the TPU,
datasets, model, initialization, optimizer, batch order and size, seed,
objective, loss weights, checkpoint cadence, and terminal evaluation unchanged.

## Frozen offline VEP evaluation

The terminal `step-4999` exports are evaluated once on the held-out `test`
split with the current `evals_v2` zero-shot harness. The experiment-local
[`evals.yaml`](evals.yaml) restricts the DAG to the two matched checkpoints
and two coding-relevant benchmarks:

- Mendelian traits: signed FWD/RC-averaged LLR, overall and consequence-level
  matched-pair AUPRC with cluster-bootstrap uncertainty.
- SGE: signed FWD/RC-averaged LLR, per-accession and missense/splicing AUPRC
  with bootstrap uncertainty.

The expected immutable exports are:

```text
gs://marin-us-east5/checkpoints/dna-exp417-cds-mammals-only-p255m-b2m-5k/2026.08.01/hf/step-4999
gs://marin-us-east5/checkpoints/dna-exp417-cds-combined-vertebrates-p255m-b2m-5k/2026.08.01/hf/step-4999
```

Dry-run from `snakemake/analysis/evals_v2/` before launching any GPU:

```bash
uv run snakemake -n \
  --configfile ../../../experiments/exp417_vertebrate_cds/evals.yaml \
  --default-storage-prefix s3://oa-bolinas/snakemake/analysis/issue417_cds_sanity/2026.08.01/ \
  -- all
```

That exact command succeeded from this experiment branch on 2026-08-02. The
resolved DAG contained only two model downloads, four scoring jobs, four metric
jobs, and the final `all` target (11 jobs total); it did not plan any upstream
or unrelated work.

After both exports pass their final checkpoint gates, launch one bounded,
auto-downing A10G worker using the existing project task. The user has
explicitly authorized EC2/SkyPilot resources for this issue:

```bash
sky launch -c dna417-cds-vep --down \
  snakemake/analysis/evals_v2/sky/run.yaml \
  --env SNAKEMAKE_ARGS="--configfile ../../../experiments/exp417_vertebrate_cds/evals.yaml \
    --default-storage-prefix s3://oa-bolinas/snakemake/analysis/issue417_cds_sanity/2026.08.01/ \
    -- all"
```

The four metric outputs are isolated under:

```text
s3://oa-bolinas/snakemake/analysis/issue417_cds_sanity/2026.08.01/results/metrics/
```

For an unattended but fail-closed handoff, the tracked watcher waits for both
exact four-file terminal exports, rechecks a clean pinned commit, repeats the
dry-run under the shared local-heavy lock and thread caps, refuses to duplicate
an existing Sky cluster, and runs the paired reporter only after all four
metric parquets exist:

```bash
uv run python scripts/issue417_wait_and_launch_eval.py \
  --expected-commit FULL_EXPERIMENT_COMMIT \
  --launch
```

Run it under `setsid`/`nohup` when it must survive an idle agent session.
Its nonblocking lock at `/tmp/marin-dna-issue417-eval-handoff.lock` prevents
two watchers from launching the same evaluation.

After the four metric parquets finish, run the frozen paired summarizer from
the same pushed experiment commit:

```bash
uv run --group genome-s3 python scripts/issue417_summarize_vep.py \
  --experiment-commit FULL_EXPERIMENT_COMMIT
```

It writes two durable artifacts alongside the metrics:

```text
s3://oa-bolinas/snakemake/analysis/issue417_cds_sanity/2026.08.01/results/comparison/summary.json
s3://oa-bolinas/snakemake/analysis/issue417_cds_sanity/2026.08.01/results/comparison/summary.md
```

This is intentionally offline: the training graphs have no lm-eval harness,
which avoids changing the matched optimizer/training path and lets the current
VEP pipeline provide all consequence-level tables and uncertainty estimates.
