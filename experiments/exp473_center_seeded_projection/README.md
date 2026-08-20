# exp473 center-seeded projection training

This permanent-branch experiment trains the four preregistered 0.25B
specialists from issue #473:

| `EXP473_ARM` | Region | Projection policy | Dataset |
|---|---|---|---|
| `cds_full_window` | CDS | full window | established `marin-dna/vertebrate-v1-cds` |
| `cds_center_1` | CDS | center 1 bp | new issue #473 dataset |
| `enhancer_full_window` | enhancer-centered cCRE | full window | new issue #473 dataset |
| `enhancer_center_1` | enhancer-centered cCRE | center 1 bp | new issue #473 dataset |

Every arm uses the same Qwen3-like 0.25B geometry, character-plus-BOS
tokenizer, case-aware loss, optimizer, seed 0, 1,024-sequence per-device
microbatch, and 5,000-step schedule from #417. The global batch is 8,192
sequences of 256 tokens, so each arm sees 10,485,760,000 token presentations.
Hugging Face checkpoints are written every 500 steps. Rolling recovery
checkpoints retain Marin's standard ten-minute cadence, with a native
optimizer-state checkpoint retained every 500 steps.

The project is independently locked to Python 3.12 and Marin source commit
`6bb4d74694fa185cabf20d037f414235e6a12eed`. Its DNA tokenizer and Levanter
format adapter are copied into this project so no root package or evaluation
workflow is a runtime dependency. The vendored tokenizer is byte-identical to
`marin-dna/tokenizer-char-bos` revision
`a73e9d9ee636f722b4c378703c9e2997857809b2`; all three file digests are
verified before constructing an artifact graph, and the model's Hugging Face
export points to that same local tokenizer.

## Dataset revisions

The established full-window CDS revision is pinned in source. The three new
dataset revisions are mandatory 40-character environment variables:

- `EXP473_CENTER1_CDS_REVISION`
- `EXP473_FULLWINDOW_ENHANCER_REVISION`
- `EXP473_CENTER1_ENHANCER_REVISION`

The launcher fails before creating a graph if any selected dataset lacks an
immutable revision. All four arms retain each dataset's ordinary chromosome-18
validation split. Native validation losses across different policy datasets
are not a policy comparison; issue #473 uses its paired intersection views for
that diagnostic.

Before the first data-bearing launch, verify that the vendored tokenizer is
available not only to the coordinator but also to a real Iris child worker:

```bash
MARIN_PREFIX=gs://marin-us-east5/MarinDNA/exp473_center_seeded_projection \
uv run --python /usr/bin/python3.12 --locked iris --cluster=marin job run \
  --no-wait --job-name exp473-tokenizer-worker-preflight \
  --cpu 1 --memory 2G --region us-east5 --extra=tpu \
  -- python -m exp473_center_seeded_projection.tokenizer_preflight \
  --version 2026.08.20 --run
```

## Verify

```bash
uv sync --python /usr/bin/python3.12 --locked --group dev
uv run --python /usr/bin/python3.12 --locked pytest
```

Printing the plan is non-mutating; add `--run` only when launching the approved
arm:

```bash
EXP473_ARM=cds_full_window \
WANDB_API_KEY=test WANDB_ENTITY=test WANDB_PROJECT=marin \
MARIN_PREFIX=gs://marin-us-east5/MarinDNA/exp473_center_seeded_projection \
uv run --python /usr/bin/python3.12 --locked \
  python -m exp473_center_seeded_projection.experiment \
  --version 2026.08.20
```

## Iris launch

Launch one isolated coordinator per arm. Supply the exact new dataset revisions
returned by the fail-closed Hugging Face publication step.

```bash
uv run --python /usr/bin/python3.12 --locked iris --cluster=marin job run \
  --no-wait \
  --job-name exp473-cds-full-window \
  --cpu 1 --memory 2G --region us-east5 --extra=tpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_ENTITY "$WANDB_ENTITY" \
  -e WANDB_PROJECT marin \
  -e MARIN_PREFIX gs://marin-us-east5/MarinDNA/exp473_center_seeded_projection \
  -e EXP473_ARM cds_full_window \
  -e EXP473_CENTER1_CDS_REVISION "$EXP473_CENTER1_CDS_REVISION" \
  -e EXP473_FULLWINDOW_ENHANCER_REVISION "$EXP473_FULLWINDOW_ENHANCER_REVISION" \
  -e EXP473_CENTER1_ENHANCER_REVISION "$EXP473_CENTER1_ENHANCER_REVISION" \
  -- python -m exp473_center_seeded_projection.experiment \
  --version 2026.08.20 --run
```

Repeat with the other three arm keys and distinct job names. The four
coordinators share immutable token-cache identities but use independent
checkpoint and W&B run names. Do not launch additional landmark widths or
seeds from this branch without a new recorded decision and compute approval.

### Paired projection loss

A separate additive workflow evaluates policy-matched checkpoints on the
producer-pinned chromosome-18 intersection views. These inputs are unlabeled
projection sequences; the workflow does not read VEP labels, predictions,
effect measurements, or metrics. It imports the unchanged official causal-LM
scorer from `evals_v2`, reconstructs the training objective with uppercase
weight 1.0 and lowercase weight 0.01, and requires exact row identity before
computing `center_1 - full_window` NLL deltas. Negative deltas favor
`center_1`. Uncertainty uses aligned bootstrap draws over human anchors.

Run this after all four checkpoint roots are final:

```bash
sky launch \
  experiments/exp473_center_seeded_projection/sky/intersection_loss.yaml \
  -c exp473-intersection-loss \
  --env EXP473_EXPERIMENT_COMMIT="$EXP473_EXPERIMENT_COMMIT" \
  --env EXP473_CDS_FULL_WINDOW_CHECKPOINT_ROOT="$CDS_FULL_ROOT" \
  --env EXP473_CDS_CENTER_1_CHECKPOINT_ROOT="$CDS_CENTER_ROOT" \
  --env EXP473_ENHANCER_FULL_WINDOW_CHECKPOINT_ROOT="$ENHANCER_FULL_ROOT" \
  --env EXP473_ENHANCER_CENTER_1_CHECKPOINT_ROOT="$ENHANCER_CENTER_ROOT"
```

The isolated `IntersectionLoss.smk` exposes only four new
`issue_473_intersection_*` rules. Its 40 score cells cover two regions, both
policies, and checkpoints 500 through 5,000. Outputs are written under
`s3://oa-bolinas/snakemake/analysis/evals_v2/results/issue473/<experiment-commit>/intersection_loss/`
and include per-policy point estimates, aligned bootstrap samples, paired
deltas, a Markdown summary, and a SHA-256 manifest. Set `DRY_RUN=1` for a
non-executing graph check.

## Evaluation boundary

Offline checkpoint evaluation uses `snakemake/analysis/evals_v2` development
splits only. Even-autosome and chromosome-Y labels, predictions, and aggregate
metrics remain held out. AUPRC is primary; Group SMD is secondary only where
the registered match-group contract applies. Policy deltas are paired across
the eight Mendelian specialist subsets and reported with uncertainty over the
paired units.

Evaluation is additive. The launch in `sky/evaluate.yaml` calls the unchanged
official `snakemake/analysis/evals_v2` Snakefile with an experiment-generated
config. That config hard-codes `split: train`, pins the three evaluation-dataset
revisions, and gives every issue #473 checkpoint a commit-keyed evaluator name.
The analysis refuses a score bundle unless its matching official metric parquet
records only `train`, preventing a stale shared-path result from silently
crossing the development boundary. CDS checkpoints run Mendelian
+ SGE; enhancer checkpoints run Mendelian + Complex. Every family is scored at
steps 500 through 5,000 in increments of 500.

After training, copy the four exact immutable checkpoint artifact roots from
the successful Iris jobs. Each root must stop before `/hf`; the config generator
appends `/hf/step-{step}` and fails on checkpoint-like inputs.

```bash
sky launch \
  experiments/exp473_center_seeded_projection/sky/evaluate.yaml \
  -c exp473-evaluate \
  --env EXP473_CDS_FULL_WINDOW_CHECKPOINT_ROOT="$CDS_FULL_ROOT" \
  --env EXP473_EXPERIMENT_COMMIT="$EXP473_EXPERIMENT_COMMIT" \
  --env EXP473_CDS_CENTER_1_CHECKPOINT_ROOT="$CDS_CENTER_ROOT" \
  --env EXP473_ENHANCER_FULL_WINDOW_CHECKPOINT_ROOT="$ENHANCER_FULL_ROOT" \
  --env EXP473_ENHANCER_CENTER_1_CHECKPOINT_ROOT="$ENHANCER_CENTER_ROOT"
```

The experiment-local analysis in `analyze_evals.py` asserts exact evaluation-row
identity between policies before computing `center_1 - full_window`. It applies
the same match-group bootstrap draws to both policies and reuses the subset seed
across all checkpoints, producing paired AUPRC and Group SMD trajectory
intervals for all eight registered Mendelian specialist subsets. Group SMD is
the #459 statistic: one positive-minus-mean-negative gap per match group, with
the mean gap divided by the across-group sample SD. Inputs with an incompatible
group contract fail rather than falling back to an ungrouped statistic.

After every official evaluation cell succeeds:

```bash
sky launch \
  experiments/exp473_center_seeded_projection/sky/analyze.yaml \
  -c exp473-analyze \
  --env EXP473_EXPERIMENT_COMMIT="$EXP473_EXPERIMENT_COMMIT"
```

The analysis writes point metrics, aligned bootstrap samples, policy deltas,
the official Complex/SGE tables, plots, a Markdown summary, and a SHA-256
manifest under a commit-keyed S3 prefix. Its additional-seed table only records
the preregistered evidence trigger; it does not launch an unapproved arm.
