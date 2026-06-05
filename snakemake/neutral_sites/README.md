# neutral_sites — GPN-Star neutral-site set for mutation-rate calibration (#267)

Builds a reusable, **model-independent** parquet of high-confidence neutral
`(chrom, pos, ref)` sites on GRCh38, replicating the GPN-Star neutral
definition. The per-model calibration step in
[`snakemake/analysis/evals_v2`](../analysis/evals_v2/) consumes this artifact to
build per-checkpoint `llr_neutral_mean` / `entropy_neutral_mean` tables for
calibrated LLR / entropy (cLLR) scoring. See
[issue #267](https://github.com/Open-Athena/marin-dna/issues/267).

Kept as its own pipeline (not folded into `evals_v2`) because it's run-once,
model-independent, and needs a genome-data toolchain (bedtools + UCSC liftOver,
via conda) alien to evals_v2's GPU-inference env — the same separation as
`snakemake/evals` (builds the eval datasets) vs `evals_v2` (consumes them).

## What it does

Replicates GPN-Star's hg38 calibration inputs exactly
([songlab-cal/gpn `calibration.smk`](https://github.com/songlab-cal/gpn/blob/20abe5df998872a47ec08d40c11ab472012ccca5/analysis/gpn-star/train_and_eval/workflow/rules/calibration.smk)):

1. **Ancestral repeats** — download UCSC RepeatMasker for hg38 (target) and mm39
   (outgroup) + the `mm39.hg38.rbest` chain; `liftOver` the mm39 repeats into
   hg38; keep hg38 repeats overlapping a lifted mm39 repeat (`bedtools
   intersect -u`). `Simple_repeat` / `Low_complexity` are excluded in
   `parse_rmsk`.
2. **Low-constraint sites** — scan the vertebrate 100-way `phyloP` + `phastCons`
   bigWigs; keep bases with `|phyloP| < 0.1 AND phastCons == 0` (NaN excluded).
3. **Neutral set** — `bedtools intersect` (1) ∩ (2), then enumerate every base
   to `(chrom, pos, ref)`, keeping `ACGT` refs only.

### Two deliberate choices vs the reference (flagged in #267)

- **`Simple_repeat` / `Low_complexity` are actually excluded.** GPN's awk emits
  `swScore` in the column it then filters against the class strings, so its
  exclusion is a silent no-op; `parse_rmsk` filters `repClass` correctly (these
  are poor neutral proxies). One-line revert if exact bug-for-bug parity is
  wanted.
- **Coordinates.** Everything through the liftOver/bedtools steps stays in
  BED-land (UCSC `chr`-prefixed, 0-based half-open). `enumerate_positions` is
  the single boundary back: it strips `chr` (our Ensembl reference uses bare
  names) and emits **1-based** `pos`, matching the HF eval datasets so the same
  scoring path consumes neutral sites and eval variants alike.

## Output

`results/neutral_sites.parquet` — columns `[chrom, pos, ref]` (bare chrom,
1-based pos, `ref` ∈ {A,C,G,T}), autosomes + X + Y.

Pin destination (uploaded at the end of a run; consumed by `evals_v2`):
`s3://oa-bolinas/snakemake/neutral_sites/results/neutral_sites.parquet`.

## Where to run

CPU-only but **heavy I/O**: it downloads ~15 GB of bigWigs plus rmsk/chain and
scans the bigWigs genome-wide (single-threaded). Budget the disk and ~tens of
minutes. Not worth a GPU. The bigWigs/rmsk/chain are kept **local** (no S3
storage churn) — only the small final parquet is pinned. The enumerate step is
RAM-bound (it materializes every neutral base), so run it on a memory-optimized
node. [`sky/run.yaml`](sky/run.yaml) provisions one (`r6id.2xlarge`, 64 GB) and
pins the parquet to S3.

## Setup

```bash
# bedtools + liftOver come from the conda env (use-conda in the default profile).
# The reference is read from S3 by pyfaidx, so install the genome-s3 group:
uv sync --frozen --group genome-s3
```

## Usage

```bash
cd snakemake/neutral_sites

# Inspect the DAG.
uv run --group genome-s3 snakemake -n

# Run (conda envs are built on first use).
uv run --group genome-s3 snakemake

# Pin the artifact for evals_v2 to consume.
aws s3 cp results/neutral_sites.parquet \
  s3://oa-bolinas/snakemake/neutral_sites/results/neutral_sites.parquet
```

### On SkyPilot (the intended path)

```bash
# Provisions a memory-optimized CPU node, runs the pipeline, sanity-checks,
# and pins the parquet to s3://oa-bolinas/snakemake/neutral_sites/...
sky launch snakemake/neutral_sites/sky/run.yaml -c neutral-sites
sky down neutral-sites   # when done
```

## Library

Rules are thin glue around `marin_dna.pipelines.neutral_sites.sites`:

- `parse_rmsk` — UCSC `rmsk.txt.gz` → repeat-interval BED frame.
- `contiguous_runs` / `neutral_mask` / `scan_neutral_intervals` —
  low-constraint sites from phyloP + phastCons.
- `enumerate_positions` — neutral intervals → per-base `(chrom, pos, ref)`.

Tests: [`tests/pipelines/neutral_sites/test_sites.py`](../../tests/pipelines/neutral_sites/test_sites.py).
