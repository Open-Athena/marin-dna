# exp473 center-seeded projection training

This permanent-branch experiment trains three new preregistered 0.25B
specialists and reuses the exact matching CDS full-window model from #417:

| Evaluation arm | Region | Projection policy | Source |
|---|---|---|---|
| `cds_full_window` | CDS | full window | reused #417 checkpoint; not trainable here |
| `cds_center_1` | CDS | center 1 bp | new issue #473 training arm |
| `enhancer_full_window` | enhancer-centered cCRE | full window | new issue #473 training arm |
| `enhancer_center_1` | enhancer-centered cCRE | center 1 bp | new issue #473 training arm |

Every new arm uses the same Qwen3-like 0.25B geometry, character-plus-BOS
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

The established full-window CDS arm is not relaunched. Evaluation pins its
existing checkpoint root to
`gs://marin-us-east5/checkpoints/dna-exp417-cds-combined-vertebrates-p255m-b2m-5k/2026.08.01`.
The three new public, ungated datasets are pinned in source to exact revisions:

- `cds_center_1`: [`marin-dna/vertebrate-v1-issue473-center1-cds`](https://huggingface.co/datasets/marin-dna/vertebrate-v1-issue473-center1-cds/tree/4d9a04ab6c4a6e445345fe35fbe2be41b43e7938) at `4d9a04ab6c4a6e445345fe35fbe2be41b43e7938`
- `enhancer_full_window`: [`marin-dna/vertebrate-v1-issue473-fullwindow-ccre-enhancer-centered`](https://huggingface.co/datasets/marin-dna/vertebrate-v1-issue473-fullwindow-ccre-enhancer-centered/tree/ffb9c63fae72311fb457640af9c8365b84f0edf8) at `ffb9c63fae72311fb457640af9c8365b84f0edf8`
- `enhancer_center_1`: [`marin-dna/vertebrate-v1-issue473-center1-ccre-enhancer-centered`](https://huggingface.co/datasets/marin-dna/vertebrate-v1-issue473-center1-ccre-enhancer-centered/tree/23d1531f63998b5716e7895a74437e0568186bd1) at `23d1531f63998b5716e7895a74437e0568186bd1`

The launcher fails before creating a graph if a selected dataset lacks an
immutable revision. The three new arms retain each dataset's ordinary
chromosome-18 validation split. Native validation losses across different policy datasets
are not a policy comparison; issue #473 uses its paired intersection views for
that diagnostic.

Before the first data-bearing launch, verify that the vendored tokenizer is
available not only to the coordinator but also to a real Iris child worker:

```bash
uv run --python /usr/bin/python3.12 --locked iris --cluster=marin job run \
  --no-wait --job-name exp473-tokenizer-worker-preflight \
  --cpu 1 --memory 2G --region us-east5 --extra=tpu \
  -e MARIN_PREFIX gs://marin-us-east5/MarinDNA/exp473_center_seeded_projection \
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
EXP473_ARM=cds_center_1 \
WANDB_API_KEY=test WANDB_ENTITY=test WANDB_PROJECT=marin \
MARIN_PREFIX=gs://marin-us-east5/MarinDNA/exp473_center_seeded_projection \
uv run --python /usr/bin/python3.12 --locked \
  python -m exp473_center_seeded_projection.experiment \
  --version 2026.08.20
```

## Iris launch

Launch one isolated coordinator per arm. Dataset repositories and revisions are
source-pinned to the public outputs of the fail-closed publication step.

```bash
uv run --python /usr/bin/python3.12 --locked iris --cluster=marin job run \
  --no-wait \
  --job-name exp473-cds-center-1 \
  --cpu 1 --memory 2G --region us-east5 --extra=tpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_ENTITY "$WANDB_ENTITY" \
  -e WANDB_PROJECT marin \
  -e MARIN_PREFIX gs://marin-us-east5/MarinDNA/exp473_center_seeded_projection \
  -e EXP473_TPU_REGION us-east5 \
  -e EXP473_ARM cds_center_1 \
  -- python -m exp473_center_seeded_projection.experiment \
  --version 2026.08.20 --run
```

Repeat with the other two new arm keys and distinct job names. The three
coordinators use independent checkpoint and W&B run names. The fourth
comparison arm is the pinned #417 checkpoint above. Do not launch additional
landmark widths or seeds from this branch without a new recorded decision and
compute approval.

W&B tags fail closed at the service's 64-character limit. Overlong values,
including the two enhancer Hugging Face repository names, retain a readable
prefix plus an eight-character SHA-256 suffix. The cache artifact config keeps
the complete public repository name and exact revision for provenance.

The coordinator's `--region` controls only its CPU task. Set
`EXP473_TPU_REGION` explicitly when the training child must run elsewhere;
allowed values are `us-east5` and `us-central1`, with `us-east5` retained as
the default. `MARIN_PREFIX` must use the matching `marin-us-east5` or
`marin-us-central1` bucket; the launcher fails before graph creation when the
bucket and child region differ. A region migration must terminate the old
coordinator and child before launching a replacement with the same run and
checkpoint identities. A new region uses an additive artifact namespace and
rebuilds the source-pinned cache there rather than rewriting a receipt from a
different region.

### Paired projection loss

A separate additive workflow evaluates policy-matched checkpoints on the
producer-pinned chromosome-18 intersection views. These inputs are unlabeled
projection sequences; the workflow does not read VEP labels, predictions,
effect measurements, or metrics. It imports the unchanged official causal-LM
scorer from `evals_v2`, reconstructs the training objective with uppercase
weight 1.0 and lowercase weight 0.01, and requires exact row identity before
computing `center_1 - full_window` NLL deltas. Negative deltas favor
`center_1`. Uncertainty uses aligned bootstrap draws over human anchors.

Run this after the three new checkpoint roots are final:

```bash
sky launch \
  experiments/exp473_center_seeded_projection/sky/intersection_loss.yaml \
  -c exp473-intersection-loss \
  --env EXP473_EXPERIMENT_COMMIT="$EXP473_EXPERIMENT_COMMIT" \
  --env EXP473_CDS_CENTER_1_CHECKPOINT_ROOT="$CDS_CENTER_ROOT" \
  --env EXP473_ENHANCER_FULL_WINDOW_CHECKPOINT_ROOT="$ENHANCER_FULL_ROOT" \
  --env EXP473_ENHANCER_CENTER_1_CHECKPOINT_ROOT="$ENHANCER_CENTER_ROOT"
```

The isolated `IntersectionLoss.smk` exposes only four new
`issue_473_intersection_*` rules. Its 36 score cells cover two regions, both
policies, and the common #417 trajectory at steps 1,000 through 4,500 plus
terminal step 4,999. Outputs are written under
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
the nine common checkpoints: steps 1,000 through 4,500 in increments of 500,
plus terminal step 4,999. The #417 root and these available directories were
verified directly before launch.

After training, copy the three exact immutable checkpoint artifact roots from
the successful Iris jobs. Each root must stop before `/hf`; the config generator
appends `/hf/step-{step}` and fails on checkpoint-like inputs. The CDS
full-window root is pinned in source and cannot be overridden by an environment
variable.

```bash
sky launch \
  experiments/exp473_center_seeded_projection/sky/evaluate.yaml \
  -c exp473-evaluate \
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
