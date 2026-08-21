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
  --no-wait --no-sync \
  --job-name exp473-cds-center-1 \
  --cpu 1 --memory 2G --region us-east5 --extra=tpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_ENTITY "$WANDB_ENTITY" \
  -e WANDB_PROJECT marin \
  -e MARIN_PREFIX gs://marin-us-east5/MarinDNA/exp473_center_seeded_projection \
  -e EXP473_TPU_REGION us-east5 \
  -e EXP473_TPU_VARIANT v5p-8 \
  -e EXP473_TPU_RAM 56g \
  -e EXP473_TPU_PREEMPTIBLE true \
  -e EXP473_ARM cds_center_1 \
  -e UV_PROJECT /app \
  -- bash -lc 'cd /app && uv sync --locked --extra tpu && \
  exec uv run --locked python -m exp473_center_seeded_projection.experiment \
  --version 2026.08.20 --run'
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
allowed values are `us-east5`, `us-east1`, `us-central1`, and `europe-west4`, with
`us-east5` retained as the default. `EXP473_TPU_VARIANT` defaults to
`v5p-8`; east5 also permits `v6e-4`, and a comma-separated
`v5p-8,v6e-4` requests either compatible single-VM topology from the
scheduler. East1 permits only `v6e-4` and uses the `marin-us-east1` bucket.
Central1 permits only `v5p-8`; Europe permits `v6e-4` and uses
the `marin-eu-west4` bucket. Europe also permits the bounded single-VM
`v6e-8` and `v5litepod-8` or four-VM `v5litepod-16` recovery topologies when
smaller single-worker slices are exhausted. The global batch,
seed, optimizer, step count, and checkpoint identity remain fixed; Levanter
reshards the existing TensorStore checkpoint across the execution topology.
`EXP473_TPU_RAM` defaults to `56g`; use the bounded `96g` recovery value when
Hugging Face export exceeds the default host-memory limit. Both settings are
runtime-only and do not change the model or checkpoint identity.
`EXP473_TPU_PREEMPTIBLE` accepts only `true` or `false` and defaults to
`true`. A recovery may set it to `false` after repeated preemptions; this
changes only the Iris capacity class and preserves the same data, model,
optimizer, run ID, and checkpoint identity.
`MARIN_PREFIX` must use the matching
`marin-us-east5`, `marin-us-east1`, `marin-us-central1`, or `marin-eu-west4`
bucket; the
launcher fails before graph creation when the bucket and child region differ,
or when a variant is unsupported in the chosen region. A region migration must
terminate the old coordinator and child before launching a replacement with
the same run and checkpoint identities. A new region uses an additive artifact
namespace rather than rewriting a receipt from a different region. Rebuild the
source-pinned cache there, or copy the completed cache data, artifact contract,
and success marker while omitting executor provenance. To resume rather than
restart, copy the latest durable `checkpoints/step-N` directory and, when a
newer temporary checkpoint has already reloaded successfully, copy that exact
step under the new bucket's matching `tmp/ttl=14d/checkpoints-temp`
namespace. Do not copy training executor status files, because they could make
the incomplete training step appear complete.

Use `--no-sync` for the coordinator and sync the locked project explicitly at
its `/app` bundle root. This avoids coupling the experiment lock to the root
workspace's `uv` requirement. The model resolves the vendored tokenizer to an
absolute path inside that bundle, while the tokenized-cache identity retains
the stable project-relative `tokenizer` value. Never run two children against
the same checkpoint root concurrently; stop the old coordinator and child
before a recovery launch.

### Sampled unaligned flanks

The completed named-PSL trace retains exact alignment blocks but predates
separate left- and right-edge flank fields. The standalone additive
`trace_flanks.py` analysis clips those blocks to both the emitted target window
and original human anchor, orients them to the human anchor, and records left,
right, external, and internal unaligned bases. `sky/trace_flanks.yaml` reads
only the 214 retained sampled PSL files and writes a commit-keyed report; it
does not rerun projection or change an established trace rule or output.

```bash
sky launch \
  experiments/exp473_center_seeded_projection/sky/trace_flanks.yaml \
  -c exp473-trace-flanks \
  --env EXP473_ANALYSIS_COMMIT="$EXP473_ANALYSIS_COMMIT"
```

### Paired projection loss

A separate additive workflow evaluates policy-matched checkpoints on the
producer-pinned chromosome-18 intersection views. These inputs are unlabeled
projection sequences; the workflow does not read VEP labels, predictions,
effect measurements, or metrics. It calls the official evals_v2 causal-LM
kernel through the same experiment-local tokenizer compatibility adapter as
the development evaluator, reconstructs the training objective with uppercase
weight 1.0 and lowercase weight 0.01, and requires exact row identity before
computing center_1 - full_window NLL deltas. Negative deltas favor center_1.
Uncertainty uses aligned bootstrap draws over human anchors.

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
every registered Mendelian specialist subset and reported with uncertainty
over the paired units.

Evaluation is additive. The launch in sky/evaluate.yaml uses the isolated
workflow/Evaluation.smk graph and never modifies or includes the maintained
evals_v2 rules. Its experiment-local score rules call the unchanged official
model runner and metric functions. The generated config hard-codes split=train,
pins the three evaluation-dataset revisions, gives every issue #473 checkpoint
a commit-keyed evaluator name, and writes beneath the new
results/issue473/<experiment-commit>/development_eval namespace.

The loader resolves only train.parquet through the Hugging Face file API and
then constructs a one-file parquet dataset. This prevents a repository dataset
builder from materializing the held-out split before selecting train. It also
rejects any row outside odd autosomes and chromosome X. The analysis refuses a
score bundle unless its matching official metric parquet records only train.
Both GPU launchers pin the validated issue-462 Ubuntu 24.04/R595 AMI and run
the existing evals GPU runtime smoke gate before constructing a score DAG.

CDS checkpoints run Mendelian + Complex + SGE; enhancer checkpoints run
Mendelian + Complex. Complex is relevant to both regions: the presented CDS
slices are missense, splicing, and synonymous, while the presented enhancer
slice is distal. Complete metric artifacts retain every registered subset for
audit. Every family is scored at the nine common checkpoints: steps 1,000
through 4,500 in increments of 500, plus terminal step 4,999. The #417 root and
available directories are verified directly before launch.

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
intervals for every registered Mendelian specialist subset. Group SMD is
the #459 statistic: one positive-minus-mean-negative gap per match group, with
the mean gap divided by the across-group sample SD. Inputs with an incompatible
group contract fail rather than falling back to an ungrouped statistic.

After every official evaluation cell succeeds:

```bash
sky launch \
  experiments/exp473_center_seeded_projection/sky/analyze.yaml \
  -c exp473-analyze \
  --env EXP473_EXPERIMENT_COMMIT="$EXP473_EXPERIMENT_COMMIT" \
  --env EXP473_ANALYSIS_COMMIT="$EXP473_ANALYSIS_COMMIT"
```

The analysis writes point metrics, aligned bootstrap samples, policy deltas,
the official Complex/SGE tables, plots, a Markdown summary, and a SHA-256
manifest under an experiment-commit and analysis-commit-keyed S3 prefix. Its
additional-seed table only records the preregistered evidence trigger; it does
not launch an unapproved arm.
