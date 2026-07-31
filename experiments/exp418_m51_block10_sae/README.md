# exp418: m5.1 block-10 SAE decision gate

This self-contained, non-core experiment implements [issue 418](https://github.com/Open-Athena/marin-dna/issues/418), which informs [issue 288](https://github.com/Open-Athena/marin-dna/issues/288). It trains an 8× BatchTopK SAE (`d_sae=15,360`, `K=64`) on the frozen public m5.1 checkpoint at post-block residual 10 (implementation index 9).

The SAE training distribution is the exact, equal, commit-pinned five-way m5.1 mixture. Biological interpretation is deliberately separate: after the SAE passes this engineering gate, feature analysis uses a held-out, coordinate-clean human GRCh38 panel from `s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz`. This experiment does not treat the cross-species training stream as the biological benchmark.

## Reproduce the local configuration gate

From this directory:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run python launch.py --tier wiring --dry-run
uv run python launch.py --tier micro --dry-run
```

The dry runs do not download the 4.18 GiB checkpoint, read live datasets, allocate an accelerator, contact W&B, or write S3 artifacts.

## Fixed stages

| Stage | Activations | Windows | Per source | Optimizer steps |
|---|---:|---:|---:|---:|
| wiring | 1,000,875 | 3,925 | 785 | 785 |
| micro | 5,000,550 | 19,610 | 3,922 | 3,922 |

One optimizer batch is exactly five windows × 255 nucleotide activations = 1,275 activations. Buffer shuffling is disabled, so a batch consumes one window from each source and both budgets end exactly. BOS is present for the language model but excluded from SAE inputs.

The wiring stage saves two checkpoints, reloads the later checkpoint at an empty passthrough-buffer boundary, and makes the final artifact from the resumed runner. The micro stage is a fresh seed-288 run and requires the successful wiring `manifest.json` URI, preventing the larger run from bypassing the gate.

## Direct accelerator launch (requires explicit approval)

This experiment does not use Marin or Iris. The default [`sky.yaml`](sky.yaml) launches a single H100 PCIe directly on Lambda Cloud. Lambda is the first choice for these short gates: its on-demand H100 is non-preemptible and currently listed at $3.29/GPU-hour. AWS EC2 remains a CLI override; SkyPilot's current catalog lists `p5.4xlarge` at $6.88/hour on demand or about $2.81/hour on spot. The small spot discount versus Lambda is not worth adding preemption to the first wiring run. Prices and capacity can change; check [Lambda pricing](https://lambda.ai/pricing) and the dry-run estimate immediately before launch.

The installed SkyPilot client currently reports both providers enabled. Recheck without allocating anything:

```bash
sky check lambda
sky check aws
sky launch --dryrun --yes sky.yaml \
  --infra lambda/europe-central-1 \
  --instance-type gpu_1x_h100_pcie
```

The runtime pins the official PyTorch `2.11.0+cu128` wheel. Lambda's current H100 driver supports CUDA 12.8 but not the CUDA 13.0 wheel selected by unbounded PyTorch resolution. Remote setup asserts CUDA availability, exactly one GPU, and the device name before the job starts.

Do not remove `--dryrun` until the user explicitly approves paid accelerator use. For an approved wiring run, run from this directory with `WANDB_API_KEY` set and the experiment branch committed and pushed:

```bash
EXP418_COMMIT=$(git rev-parse HEAD)
EXP418_RUN_ID="dna-exp418-wiring-seed288-${EXP418_COMMIT:0:12}"

sky launch sky.yaml \
  --cluster exp418-lambda \
  --infra lambda/europe-central-1 \
  --instance-type gpu_1x_h100_pcie \
  --env TIER=wiring \
  --env RUN_ID="$EXP418_RUN_ID" \
  --env COMPUTE_PROVIDER=lambda \
  --env COMPUTE_INSTANCE_TYPE=gpu_1x_h100_pcie \
  --env SKYPILOT_CLUSTER_NAME=exp418-lambda \
  --env EXPERIMENT_COMMIT="$EXP418_COMMIT" \
  --secret WANDB_API_KEY
```

The task has a 30-minute idle **autodown**, not autostop: Lambda VMs cannot be stopped and restarted. Babysit the first run while `sky launch` streams logs. When it finishes, copy the complete artifact before the countdown expires, inspect the manifest, and then terminate immediately:

```bash
mkdir -p ../../scratch/issue418
rsync -Pavz \
  "exp418-lambda:~/sky_workdir/artifacts/${EXP418_RUN_ID}/" \
  "../../scratch/issue418/${EXP418_RUN_ID}/"
uv run python -m json.tool \
  "../../scratch/issue418/${EXP418_RUN_ID}/manifest.json"
sky down exp418-lambda
```

After verifying the local artifact, persist it from this AWS-authenticated machine at a reviewed S3 prefix. This step is intentionally local because the available AWS credential is an EC2 IAM role and should not be assumed portable to Lambda:

```bash
aws s3 cp --recursive \
  "../../scratch/issue418/${EXP418_RUN_ID}/" \
  "s3://oa-bolinas/experiments/exp418/${EXP418_RUN_ID}/"
```

Only if that wiring `manifest.json` reports `engineering_gate_passed: true`, launch the fresh micro-run with the stable manifest URI. Use a new run ID and cluster name:

```bash
EXP418_COMMIT=$(git rev-parse HEAD)
EXP418_WIRING_URI="s3://oa-bolinas/experiments/exp418/<wiring-run-id>/manifest.json"
EXP418_RUN_ID="dna-exp418-micro-seed288-${EXP418_COMMIT:0:12}"

sky launch sky.yaml \
  --cluster exp418-lambda-micro \
  --infra lambda/europe-central-1 \
  --instance-type gpu_1x_h100_pcie \
  --env TIER=micro \
  --env RUN_ID="$EXP418_RUN_ID" \
  --env WIRING_MANIFEST_URI="$EXP418_WIRING_URI" \
  --env COMPUTE_PROVIDER=lambda \
  --env COMPUTE_INSTANCE_TYPE=gpu_1x_h100_pcie \
  --env SKYPILOT_CLUSTER_NAME=exp418-lambda-micro \
  --env EXPERIMENT_COMMIT="$EXP418_COMMIT" \
  --secret WANDB_API_KEY
```

For an EC2 fallback, replace the Lambda resource flags with `--infra aws/us-east-2 --instance-type p5.4xlarge --no-use-spot`, and record `COMPUTE_PROVIDER=aws` and `COMPUTE_INSTANCE_TYPE=p5.4xlarge`. Use `--use-spot` only after explicitly accepting preemption. Artifact copying and teardown are otherwise identical.

The launcher also supports direct object-store upload through `--artifact-prefix` or `ARTIFACT_PREFIX`, but the provider-neutral default is `--no-upload` plus `rsync`. It refuses to overwrite an existing local or remote run.

## Produced evidence

Each run records the dependency lock/config, exact model/data/tokenizer revisions, hook shape and bitwise logit-invariance result, finite-state checks, checkpoint-resume evidence, throughput and accelerator hours, reconstruction and next-base loss metrics, BatchTopK versus exported JumpReLU L0, threshold support mismatch, held-out inactive-feature fraction, feature-use concentration, and SHA-256 hashes for the final SAE weights/config.

`RESULTS.md` is a concise issue-ready summary. `manifest.json` and `metrics.json` retain the complete machine-readable evidence. A passing engineering result is not, by itself, evidence of biological interpretation.
