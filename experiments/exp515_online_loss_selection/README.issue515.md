# Issue 515: online loss-based token selection

This independent locked project vendors `Open-Athena/glm-experiments` at commit `b46cf87c2926201473797f9b00c13e1781c16403` and adds the experiment registered in MarinDNA issue #515.
The original upstream layout is retained, while maintained package code lives under `src/glm_experiments/`.

CSV files under each run directory are authoritative.
W&B is optional and an initialization failure falls back to CSV without stopping training.

## No-GPU preflight

Run the bounded source-case audit before any paid launch:

```bash
uv sync --locked --group dev
uv run --locked exp515-audit \
  --output ../../.agents/artifacts/515-online-loss-selection/case-distribution-audit.json
uv run --locked pytest tests/test_selection.py tests/test_exp515.py
```

The audit streams 128 fixed-seed examples from each of the 108 represented species.
It records lowercase-base fraction, completely lowercase sequence fraction, eligible-target quantiles, and uppercase/lowercase run-length quantiles.
The paid runner refuses to start training if this committed audit requests the preregistered RefSeq fallback.

## Evaluation endpoint

The primary endpoint is the `tss_proximal` subset of `marin-dna/evals_mendelian_traits` at revision `4aed58e50c5dea0b878a665007af2ef9e5108e9f`.
The pinned train split contains 2,050 TSS-proximal variants in 205 matched groups, with 205 positive variants, on odd autosomes and chromosome X.
The source `pos` field is VCF-like and 1-based; reference-window extraction converts it to 0-based, half-open coordinates at the FASTA boundary.
Per-variant scores and summary JSON are written for the bridge, every midpoint, and every endpoint.

## Paid launch

The launch must reference a clean, pushed, full commit SHA:

```bash
uv run --locked python launch.py --commit "$COMMIT" --execute --retry-until-up
```

The Sky task provisions exactly one Lambda A100, forwards Google and AWS credentials as secrets, runs the targeted remote tests, executes the canary and gated matrix, uploads curated immutable artifacts below `s3://oa-bolinas/issues/515/online-loss-selection/v1/`, and shuts the instance down.
The original Zoonomia protocol used the earlier USD 28 compute stop and USD 30 all-in cap.

## Exp58 CDS gate

The immediate next run starts from the `exp58-animals` step-1,000 Hugging Face export and uses its pinned animal-CDS training corpus.
This older checkpoint consumes 256 nucleotide tokens with no BOS token.
A row therefore contributes 255 causal next-token targets.

The run performs one shared 100-step uniform bridge with linear warmup from 1e-5 to 1e-3.
It then forks seven arms from the complete bridge checkpoint and trains every arm for exactly 100 additional optimizer steps at constant 1e-3.
The five online arms are uniform, random 50%, current-student low 50%, middle 50%, and high 50%.
The two frozen-teacher arms use the matching final step-16,999 export: pure full-distribution `KL(teacher || student)` at temperature 1, and hard-label CE on the lowest-loss 50% of eligible tokens ranked by teacher NLL.
All arms exclude lowercase repeat targets and see the same 204,800 post-bridge input windows.

The primary gate endpoint pools the `missense_variant` and `splicing` subsets of the pinned Mendelian train split.
It contains 8,990 variants in 899 groups, with one positive and nine matched negatives per group.
Missense and splicing AUPRC are also reported separately.
Synonymous variants are excluded by user decision.
The run stops after all seven 100-step arm checkpoints are evaluated and waits for a continuation decision.

Launch this gate with:

```bash
uv run --locked python launch.py \
  --commit "$COMMIT" \
  --run-id "$CDS_RUN_ID" \
  --instance-start-unix "$ORIGINAL_INSTANCE_START" \
  --cds-gate \
  --execute
```

## Queued RefSeq restart

The queued screen is an explicit corpus restart using marin-dna/genomes-v5-genome_set-animals-intervals-v1_255_128 at revision d93209847b02a0c9be5c03591a0a5e56ee09c35d.
It reloads the same source model weights but creates fresh deterministic data, AdamW, scheduler, and run records.
The shared 100-step bridge warms from 1e-5 to 1e-3, and every arm holds 1e-3 thereafter.
All five arms are evaluated after 250 continuation steps.
Uniform always continues to 500.
The four nonuniform arms continue unless a one-sided paired match-group permutation test says that their midpoint AUPRC is worse than bridge after Holm familywise correction at alpha 0.05.

Launch this amended screen with a new run ID:

```bash
uv run --locked python launch.py \
  --commit "$COMMIT" \
  --run-id "$REFSEQ_RUN_ID" \
  --instance-start-unix "$ORIGINAL_INSTANCE_START" \
  --refseq-screen \
  --execute
```

The exact RefSeq plan records its lowercase soft-masked base fraction and checksum.
The runner stops estimated GPU compute at USD 48 under the USD 50 all-in cap.


## Resume and data contract

The training plan stores fixed-width 255- or 256-byte sequences and uint16 corpus IDs in checksumed files.
Every arm consumes the same row interval.
Each full Lightning checkpoint records the plan checksum, exact next sample ID, effective batch, model state, optimizer state, scheduler state, selector RNG, and Python, NumPy, PyTorch, and CUDA RNG states.
An intentional bridge-to-arm fork resets only the arm selector stream; a same-arm resume restores it exactly.
An explicit `--resume-from-bridge` repair mode validates the passing smoke test, registered canary and bridge metadata, and step-100 checkpoint before skipping those completed phases.
When a launch resumes on the same instance, `--instance-start-unix` preserves the original provider allocation time for the hard budget guard.
An explicit `--publish-only` repair mode archives any stale pre-completion failure record and retries immutable S3 publication without rerunning training or evaluation.

All genomic coordinates are 0-based and half-open internally.
The newer Mendelian dataset's VCF-like 1-based `pos` field is converted only when extracting the 255-bp reference window.
