# Issue 517 functional specialists

This permanent experiment-branch project trains five annotation-first 0.25B DNA specialists for issue #517.
It is an experiment, not a merge proposal.

The project also contains a separate entry point for the follow-up six-arm GPN-Star-P uniform-grid experiment.
That experiment does not replace or resume the original five annotation-first runs.

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

## GPN-Star-P uniform-grid training

The follow-up experiment partitions every projected window that passed the at-least-51-of-255 GPN-Star-P conservation filter into six exhaustive, mutually exclusive arms.
The arms are CDS, 3′ UTR, protein-coding TSS/5′ UTR, ncRNA exon, issue-326 Arm A enhancer, and the GPN-constrained unassigned background remainder.

The six public datasets are pinned at these immutable Hugging Face revisions:

- `marin-dna/gpn-star-p-uniform-v1-cds` at `4c722c74e4616d8cbf8bce55844ec26da7fc516f`
- `marin-dna/gpn-star-p-uniform-v1-utr3` at `42ac7aed4565d0ec2800c9d8e2b1829daec274bd`
- `marin-dna/gpn-star-p-uniform-v1-tss-utr5` at `c2fdcf05d24856f004be303470183e5fc39188b9`
- `marin-dna/gpn-star-p-uniform-v1-ncrna-exon` at `c5cea96abe3ae84dafdb52967b1168a269e01f43`
- `marin-dna/gpn-star-p-uniform-v1-enhancer-arm-a` at `243210a0d93d93423b42e817d82d0abc3de37ef8`
- `marin-dna/gpn-star-p-uniform-v1-background` at `24f9ccb7cdc7c242d2ce88783e25db5597466543`

The launcher rejects the `UNPUBLISHED` placeholder, malformed revisions, non-MarinDNA repositories, and any S3 training input.
It otherwise reuses the original experiment's fixed model, tokenizer, loss, optimizer, seed, batch, schedule, checkpoint cadence, and bounded TPU resource policy.

Each arm receives 40,960,000 sequence presentations, or 10,485,760,000 tokens including BOS.
Because the arm sizes differ, effective row epochs range from about 0.31 for enhancer Arm A to 2.80 for TSS/5′ UTR.
This exposure difference must accompany every comparison of the resulting specialists.

| arm | source rows | reverse-complemented train rows | effective row epochs |
| --- | ---: | ---: | ---: |
| CDS | 35,517,702 | 71,002,636 | 0.577 |
| 3′ UTR | 10,285,758 | 20,538,748 | 1.994 |
| TSS/5′ UTR | 7,341,817 | 14,650,866 | 2.796 |
| ncRNA exon | 10,029,795 | 20,026,822 | 2.045 |
| enhancer Arm A | 65,750,304 | 131,467,840 | 0.312 |
| background | 38,681,813 | 77,330,858 | 0.530 |

After anonymous Hub verification, launch each arm with the following command, substituting its arm key and job name:

```bash
uv run --python /usr/bin/python3.12 --locked iris --cluster=marin job run \
  --no-wait --job-name exp517-gpn-uniform-cds-flex \
  --cpu 1 --memory 2G --region us-east5 --extra=tpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_ENTITY gonzalobenegas \
  -e WANDB_PROJECT marin \
  -e MARIN_PREFIX gs://marin-us-east5/MarinDNA/exp517_gpn_uniform_specialists \
  -e EXP517_TPU_REGION us-east5 \
  -e EXP517_TPU_VARIANT v5p-8,v6e-4 \
  -e EXP517_TPU_RAM 56g \
  -e EXP517_TPU_PREEMPTIBLE true \
  -e EXP517_GPN_ARM cds \
  -e UV_PROJECT /app \
  -- bash -lc 'cd /app && uv sync --locked --extra tpu && \
  exec uv run --locked python -m exp517_functional_specialists.gpn_uniform_experiment \
  --version 2026.08.26 --run'
```

The six authorized arms may launch together on preemptible TPUs.
The ordered `v5p-8,v6e-4` alternatives retain the original v5p preference while allowing the historically validated v6e-4/PDP-1024 route when v5p capacity is unavailable.
Iris supplies `WANDB_API_KEY` to the coordinator process, and Fray inherits it when the coordinator submits each TPU worker.
The launcher validates that the key exists but deliberately excludes its value from the durable Marin artifact graph.
Verify immutable Hub download, complete tokenization, real TPU optimizer steps, W&B telemetry, and checkpoint creation independently for every arm.
Each GPN arm has an independent run ID, checkpoint root, tokenized cache, and W&B run.

## Strict phyloP-selector control

The strict control changes only the conservation selector from GPN-Star-P to `hg38.phyloP447way`.
It retains the center-1 projector, 107 mammal HAL targets, 28 nonmammal chain targets, six exhaustive arm assignments, model, tokenizer, loss, optimizer, seed, batch, sequence length, 5,000-step schedule, and checkpoint cadence.
The phyloP threshold is score at least 2.2162 in at least 51 of 255 source bases.
The background arm contains every selector-passing window not assigned to the other five arms.

The immutable publication producer is commit `fbc8968b14415b2722e7bcc4afaf95051acd6638`.
The six anonymously verified public datasets are pinned at these Hub revisions:

- `marin-dna/phylop-uniform-v1-cds` at `452a5a3538f22630c3dea94d441ac30216bb28ea`
- `marin-dna/phylop-uniform-v1-utr3` at `2b73d5d9ebda34a361536db5e3d2697b6a1b1d6c`
- `marin-dna/phylop-uniform-v1-tss-utr5` at `5134205d86cd03e7833843d99e947e43e7aa11ac`
- `marin-dna/phylop-uniform-v1-ncrna-exon` at `54667e7bb49505f463afc147676e880a30c11d89`
- `marin-dna/phylop-uniform-v1-enhancer-arm-a` at `6f879b3747330e2c92e1402ead55cda6621f50ff`
- `marin-dna/phylop-uniform-v1-background` at `7d84519dccb4286622a14642a82a4f045d93a42c`

Each repository has 64 train shards and one validation shard.
Launch the six authorized arms together by substituting the arm key and job name below:

```bash
uv run --python /usr/bin/python3.12 --locked \
  iris --cluster=marin job run \
  --no-wait --user gonzalo --priority interactive \
  --job-name exp517-phylop-uniform-cds-alt-d001 \
  --cpu 1 --memory 2G --region us-east5 --extra=tpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_ENTITY gonzalobenegas \
  -e WANDB_PROJECT marin \
  -e MARIN_PREFIX gs://marin-us-east5/MarinDNA/exp517_phylop_uniform_specialists \
  -e EXP517_TPU_REGION us-east5 \
  -e EXP517_TPU_VARIANT v5p-8,v6e-4 \
  -e EXP517_TPU_RAM 56g \
  -e EXP517_TPU_PREEMPTIBLE true \
  -e EXP517_PHYLOP_ARM cds \
  -e UV_PROJECT /app \
  -- bash -lc 'cd /app && uv sync --locked --extra tpu && \
  exec uv run --locked python -m exp517_functional_specialists.phylop_uniform_experiment \
  --version 2026.08.27 --run'
```

## Strict phyloP enhancer order-exposure control

This one-trial control changes the taxonomic sampling unit of the strict-phyloP Arm A enhancer dataset from one non-human species per family to one sequence source per represented NCBI order across the complete dataset.
Human is the sole Primates source.
The 39 non-human targets cover 18 mammalian and 21 non-mammalian orders.

The public dataset is pinned at immutable Hugging Face revision `6a592fffcdd155d19e6c8e0986eab606aab19606` of `marin-dna/phylop-uniform-v1-enhancer-arm-a-vertebrate-order`.
Its 7,876,044 original rows become 15,719,320 training rows after the fixed validation holdout and reverse-complement augmentation.
The unchanged 40,960,000-sequence schedule therefore presents about 2.606 effective row epochs, compared with about 0.514 for the family-deduplicated strict-phyloP enhancer control.

The run has a distinct token cache, checkpoint root, W&B identity, and durable sweep-state prefix.
Launch it on one preemptible ordered-alternative TPU request:

```bash
uv run --python /usr/bin/python3.12 --locked \
  iris --cluster=marin job run \
  --no-wait --user gonzalo --priority interactive \
  --job-name exp517-phylop-enhancer-order-d001 \
  --cpu 1 --memory 2G --region us-east5 --extra=tpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_ENTITY gonzalobenegas \
  -e WANDB_PROJECT marin \
  -e MARIN_PREFIX gs://marin-us-east5/MarinDNA/exp517_phylop_enhancer_order \
  -e EXP517_TPU_REGION us-east5 \
  -e EXP517_TPU_VARIANT v5p-8,v6e-4 \
  -e EXP517_TPU_RAM 56g \
  -e EXP517_TPU_PREEMPTIBLE true \
  -e UV_PROJECT /app \
  -- bash -lc 'cd /app && uv sync --locked --extra tpu && \
  exec uv run --locked python -m exp517_functional_specialists.phylop_enhancer_order_experiment \
  --version 2026.09.04 --run'
```

Completion requires W&B `run_progress >= 1` and a reachable terminal step-4,999 checkpoint.
Run development VEP through the offline `evals_v2` workflow after the terminal checkpoint is available.
Do not register held-out even-autosome or chromosome-Y evaluation data.

## Single-H100 validation

The CoreWeave validation keeps all TPU production workflows live and uses a distinct CDS smoke identity.
It tokenizes 16,384 rows from the same immutable public CDS dataset into cluster-local S3, then runs three optimizer steps on one preemptible H100.
The first attempt uses the complete 8,192-sequence global batch as one per-device microbatch.
Reduce the per-device microbatch to 4,096, 2,048, or 1,024 only after a verified H100 OOM.

Select the production H100 peer from a current Iris capacity snapshot and use batch priority:

```bash
uv run --python /usr/bin/python3.12 --locked \
  iris --cluster=marin job run \
  --no-wait --no-sync --user gonzalo --priority batch --preemptible \
  --target-cluster cw-us-east-02a \
  --job-name exp517-phylop-cds-h100-pdp8192-smoke-d001 \
  --cpu 1 --memory 2G --disk 9G \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_ENTITY gonzalobenegas \
  -e WANDB_PROJECT marin \
  -e EXP517_H100_CLUSTER cw-us-east-02a \
  -e EXP517_H100_SMOKE_PDP 8192 \
  -e UV_PROJECT /app/experiments/exp517_functional_specialists/h100_smoke \
  -- bash -lc 'cd /tmp && UV_TOOL_DIR=/tmp/uv-tools UV_TOOL_BIN_DIR=/tmp/uv-bin uv tool install uv==0.11.31 && cd /app/experiments/exp517_functional_specialists && exec /tmp/uv-bin/uv run --project /app/experiments/exp517_functional_specialists/h100_smoke --locked python -m exp517_functional_specialists.phylop_uniform_h100_smoke --version 2026.08.27 --run'
```

The CoreWeave cluster supplies its S3 `MARIN_PREFIX` and object-store credentials.
The thin `h100_smoke` environment keeps the established TPU lock unchanged and resolves the H100 child with its GPU-only `gpu` extra.
The coordinator disables Iris auto-sync, installs the repository-pinned uv as an isolated tool, and runs from the experiment root so the vendored tokenizer resolves identically on every production peer.
The top-level Iris job owns the CoreWeave peer selection; child jobs inherit that controller and must not attempt to federate back to the same peer.

The measured calibration result is a verified OOM at per-device parallelism 8,192 and 4,096, followed by a successful 3/3-step run at 2,048.
The first full 2,048 run then reached step 5 before a later `jit__train_step` requested another 52.65 GiB and OOMed.
The retry therefore uses per-device parallelism 1,024 while preserving global batch 8,192 through additional accumulation.

The full path tokenizes the complete immutable selected arm into CoreWeave-local S3 and restores the exact 5,000-step schedule and 500-step checkpoint cadence.
Run its tokenization-only mode first on one explicitly sized preemptible CoreWeave CPU task; it uses 16 local Zephyr workers because the pinned Fray CPU actor path does not attach a uv environment on the production peers.
After that immutable cache completes, launch the small coordinator below and it will skip tokenization and submit only the H100 child.
Select the production H100 peer from a fresh capacity snapshot, then launch the CDS arm with batch-priority preemptible capacity:

```bash
uv run --python /usr/bin/python3.12 --locked \
  iris --cluster=marin job run \
  --no-wait --no-sync --user gonzalo --priority batch --preemptible \
  --target-cluster cw-rno2a \
  --job-name exp517-phylop-cds-h100-pdp1024-full-d001 \
  --cpu 1 --memory 2G --disk 9G \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_ENTITY gonzalobenegas \
  -e WANDB_PROJECT marin \
  -e EXP517_H100_CLUSTER cw-rno2a \
  -e EXP517_H100_ARM cds \
  -e EXP517_H100_PDP 1024 \
  -e UV_PROJECT /app/experiments/exp517_functional_specialists/h100_smoke \
  -- bash -lc 'cd /tmp && UV_TOOL_DIR=/tmp/uv-tools UV_TOOL_BIN_DIR=/tmp/uv-bin uv tool install uv==0.11.31 && cd /app/experiments/exp517_functional_specialists && exec /tmp/uv-bin/uv run --project /app/experiments/exp517_functional_specialists/h100_smoke --locked python -m exp517_functional_specialists.phylop_uniform_h100_experiment --version 2026.08.27 --run'
```

Keep the existing TPU workflows alive until the full CDS H100 trajectory and checkpoint behavior have been compared.

## Evaluation boundary

Evaluate only the development split unless the user separately authorizes held-out access.
Remove complete mature-miRNA groups from Mendelian metrics, tables, plots, and model-selection decisions; Complex Traits has no mature-miRNA subset.
The primary statistic is AUPRC, with paired joint bootstrap uncertainty for whether each home specialist ranks first.
Do not read, generate, publish, or summarize even-autosome or chromosome-Y held-out labels, predictions, or metrics from this branch.

The dedicated `snakemake/analysis/evals_v2/config/issue517.yaml` config registers only the terminal step 4,999 checkpoint for each of the five arms.
Its 11 development cells comprise five Mendelian cells, five Complex Traits cells, and one CDS-only SGE cell.
It pins all three development datasets, enables complete-group mature-miRNA exclusion for Mendelian metrics, and does not register a held-out dataset.
The same workflow jointly resamples Mendelian match groups across all five arms and reports terminal `P(home ranks first)` for each of the eight preregistered subsets.
A single checkpoint cannot establish the previously planned two-consecutive-checkpoint persistence criterion.

Launch each model-dataset cell as soon as the terminal checkpoint for that arm exists.
Use one target per Sky cluster so each GPU runs one inference cell.
CDS has Mendelian Traits, Complex Traits, and SGE targets.
Each other arm has Mendelian and Complex Traits targets.

For example, launch the CDS Mendelian cell with:

```bash
sky launch snakemake/analysis/evals_v2/sky/run.yaml -c evals-v2-exp517-cds-mendelian --env SNAKEMAKE_ARGS="--configfile config/issue517.yaml -- results/metrics/exp517-cds-step-4999/mendelian_traits.parquet" --down
```

After all five Mendelian score cells exist, launch the joint terminal home-rank outputs:

```bash
sky launch snakemake/analysis/evals_v2/sky/run.yaml -c evals-v2-exp517-home-rank --env SNAKEMAKE_ARGS="--configfile config/issue517.yaml -- results/analysis/home_rank_trajectory.parquet results/analysis/home_rank_persistence.parquet" --down
```
