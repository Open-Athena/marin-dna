# Center-seeded projection experiment (#473)

This experiment compares the established 255 bp full-window projection policy
with centered 1, 17, 33, 65, and 129 bp source landmarks. It is additive to the
#417 pipeline: existing rules, helpers, configuration, targets, and output
contracts are unchanged.

All genomic coordinates are 0-based and half-open. Each request retains the
original 255 bp human source anchor as its identity and stores the submitted
landmark separately as `projection_start` and `projection_end`. For
`center_1`, source anchor `[s, s + 255)` submits `[s + 127, s + 128)`.

## Review-only request artifacts

The default smoke tier uses the checked-in smoke anchors. Build a
credential-free plan with:

```bash
uv run --locked snakemake -n \
  --profile workflow/profiles/default \
  --default-storage-provider none \
  issue_473_request_artifacts
```

The target creates a policy manifest, one request Parquet per policy, and one
HAL BED6 per policy under the producer-keyed
`results/.../smoke/experiments/473/` namespace. It does not launch HAL,
read a MAF, or start remote compute.

## Wider pilot sample

With `tier=full`, request artifacts depend on a new scored catalog and pilot
sample under `results/.../full/experiments/473/pilot/`. The sample contains
up to 10,000 anchors from each of the five canonical functional-region labels.
Within each region, anchors are assigned to five equal-count conservation-score
quantiles using score and query name as deterministic tie-breakers. The sampler
first selects at least one anchor from every observed source-chromosome by
quantile stratum, then water-fills the remaining regional budget. SHA-256
ordering with seed 473 chooses anchors within each stratum.

Inspect the full plan without consulting S3:

```bash
uv run --locked snakemake -n \
  --profile workflow/profiles/default \
  --default-storage-provider none \
  issue_473_request_artifacts \
  --config tier=full
```

Do not execute the full sample, projection, or any paid remote job without
explicit approval.

## Projection smoke plan

The additive `issue_473_projection_smoke` target projects every policy through
the existing HAL and MultiZ inputs, extracts 255 bp sequences, writes the
standard per-policy QC artifacts, and compares policy recovery against
`full_window`. Its outputs remain isolated under
`results/.../smoke/experiments/473/projection/`.

Resolve the complete smoke DAG without consulting S3 or executing jobs with:

```bash
uv run --locked snakemake -n \
  --profile workflow/profiles/default \
  --default-storage-provider none \
  issue_473_projection_smoke
```

This plan stages the shared HAL and smoke-cohort MAF inputs when executed.
Do not run it locally or launch remote compute without explicit approval.
