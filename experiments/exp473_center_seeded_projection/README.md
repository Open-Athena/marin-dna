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
tokenizer, case-aware loss, optimizer, seed 0, and 5,000-step schedule. The
global batch is 8,192 sequences of 256 tokens, so each arm sees
10,485,760,000 token presentations. Hugging Face checkpoints are written every
500 steps. Rolling recovery checkpoints retain Marin's standard ten-minute
cadence.

The project is independently locked to Python 3.12 and Marin source commit
`6bb4d74694fa185cabf20d037f414235e6a12eed`. Its DNA tokenizer and Levanter
format adapter are copied into this project so no root package or evaluation
workflow is a runtime dependency.

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

## Evaluation boundary

Offline checkpoint evaluation uses `snakemake/analysis/evals_v2` development
splits only. Even-autosome and chromosome-Y labels, predictions, and aggregate
metrics remain held out. AUPRC is primary; Group SMD is secondary only where
the registered match-group contract applies. Policy deltas are paired across
the eight Mendelian specialist subsets and reported with uncertainty over the
paired units.
