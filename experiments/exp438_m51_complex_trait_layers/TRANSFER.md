# Feature 1662 untouched-test transfer

This extension was frozen in issue #438 before the official `test.parquet` was
downloaded or inspected.

- Candidate: block-19/25M feature 1662.
- Primary target: `label` within `missense_variant`.
- Secondary target: overall `label`.
- Responses: FWD and RC `abs(alt_activation - ref_activation)`, separately.
- Statistics: two-sided Welch and Mann–Whitney with positive effect required.
- Correction: BH across the four orientation × target pairs separately for
  each statistic.
- Strict primary success: both missense orientations have positive
  standardized/rank-biserial effects and q<0.05 by both tests.
- AUPRC is descriptive. Mean/max reducers and VEP covariates are excluded from
  confirmation.

The pinned official test object has 10,000 variants in 1,000 exact 1:9 groups,
including 2,040 missense variants and 204 positives. It is 1,036,640 bytes with
SHA-256
`4bc355e8a39ce310d792d5fb0293ef01a7fc6306eef6415dae01f81133520ab6`.

`sky.transfer.yaml` stages only the block-19 SAE, the indexed Ensembl-115
reference, and the pinned test object; extracts only feature 1662; applies the
four-test analysis; creates a hash-complete manifest; and verifies an empty S3
sync dry-run. The durable destination is:

```text
s3://oa-bolinas/experiments/exp438/retrieval/dna-exp438-feature1662-test-r1/
```
