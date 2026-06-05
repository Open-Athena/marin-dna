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

## Calibration subsampling (#267 / #270)

For mutation-rate calibration every neutral site is scored against its 3 non-ref
alts and binned by pentanucleotide. Two **additive** rules (downstream of the
pinned `neutral_sites.parquet`, which is left untouched) prepare a reduced set so
the per-checkpoint scoring in `evals_v2` need not cover all 5.94M × 3 ≈ 18M
variants:

- `annotate_pentanuc` → `results/neutral_sites_pentanuc.parquet`
  `[chrom, pos, ref, pentanuc]` — the central 5-mer (window `[pos-2, pos+2]`,
  variant-centered) read from the genome; model-independent.
- `subsample_neutral` → `results/subsampled/neutral_sites_n{n}.parquet` — at most
  `n` sites **per 5-mer**, seeded by `subsample_seed`. `n` is a path wildcard so
  several caps coexist and `evals_v2` pins one by revision.

**The subsample unit is sites per 5-mer, not per `(5-mer, alt)` bin.** A site's
three alt-bins share its sites, so `n` sites/5-mer yield `n` observations in
*each* of its 3 bins, at `3n` variant-scorings per 5-mer → ≈ `3072·n`
variants/checkpoint (minus depletion: 208/1024 five-mers have < 1000 sites). The
convergence pilot (issue #270) found `n ≈ 1000` gives a per-bin mean-LLR
SE ≈ 0.03; `n = 100` is the cheap floor (SE ≈ 0.10). The set is kept site-level /
alt-agnostic so the LLR path (3 alts) and the entropy atom (#269, 4-allele
marginal) consume one artifact.

Pin destination (consumed by `evals_v2`'s `compute_llr_neutral_mean` rule —
[`snakemake/analysis/evals_v2`](../analysis/evals_v2/), `snakemake calibration`):
`s3://oa-bolinas/snakemake/neutral_sites/results/subsampled/neutral_sites_n{n}.parquet`.

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

# Build + pin a subsampled set (cap = n sites per 5-mer; n is a path wildcard).
uv run --group genome-s3 snakemake results/subsampled/neutral_sites_n100.parquet
aws s3 cp results/subsampled/neutral_sites_n100.parquet \
  s3://oa-bolinas/snakemake/neutral_sites/results/subsampled/neutral_sites_n100.parquet
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
- `annotate_pentanucleotide` — neutral sites → `+ pentanuc` (central 5-mer; one
  read per chromosome, asserts 5-mer center == `ref`).
- `subsample_per_context` — keep at most `n` sites per 5-mer, seeded + deterministic.

Tests: [`tests/pipelines/neutral_sites/test_sites.py`](../../tests/pipelines/neutral_sites/test_sites.py).
