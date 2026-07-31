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

## Accelerator launch (requires explicit approval)

Do not run either command until the user explicitly approves paid accelerator use. Run from this directory with the experiment branch committed and pushed. The public Hugging Face checkpoint means CoreWeave does not need GCS access.

```bash
uv run iris --cluster=marin job run --no-wait \
  --target-cluster cw-rno2a \
  --job-name exp418-wiring \
  --enable-extra-resources \
  --gpu H100 --cpu 16 --memory 128g --disk 100g \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e MARIN_PREFIX s3://marin-us-east-02a/marin \
  -e HF_HUB_DOWNLOAD_TIMEOUT 120 \
  -e UV_LOCK_TIMEOUT 7200 \
  -e EXPERIMENT_COMMIT "$(git rev-parse HEAD)" \
  -- python launch.py --tier wiring
```

After the wiring `manifest.json` reports `engineering_gate_passed: true`, launch the fresh micro-run with its stable S3 URI:

```bash
uv run iris --cluster=marin job run --no-wait \
  --target-cluster cw-rno2a \
  --job-name exp418-micro \
  --enable-extra-resources \
  --gpu H100 --cpu 16 --memory 128g --disk 100g \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e MARIN_PREFIX s3://marin-us-east-02a/marin \
  -e HF_HUB_DOWNLOAD_TIMEOUT 120 \
  -e UV_LOCK_TIMEOUT 7200 \
  -e EXPERIMENT_COMMIT "$(git rev-parse HEAD)" \
  -- python launch.py --tier micro \
  --wiring-manifest-uri s3://marin-us-east-02a/marin/experiments/exp418/<wiring-run-id>/manifest.json
```

Artifacts land below the explicitly supplied S3-backed `MARIN_PREFIX` at `experiments/exp418/<run-id>/`. The launcher refuses to overwrite an existing local or S3 run.

## Produced evidence

Each run records the dependency lock/config, exact model/data/tokenizer revisions, hook shape and bitwise logit-invariance result, finite-state checks, checkpoint-resume evidence, throughput and accelerator hours, reconstruction and next-base loss metrics, BatchTopK versus exported JumpReLU L0, threshold support mismatch, held-out inactive-feature fraction, feature-use concentration, and SHA-256 hashes for the final SAE weights/config.

`RESULTS.md` is a concise issue-ready summary. `manifest.json` and `metrics.json` retain the complete machine-readable evidence. A passing engineering result is not, by itself, evidence of biological interpretation.
