# ELS/PLS focal-coordinate follow-up

This no-inference follow-up to issue #440 overlays the focal genomic coordinate
of every archived window with the frozen ENCODE cCRE table. It tests ELS
(`pELS` or `dELS`) and PLS as two separate one-versus-rest contrasts, retains
the original subtype, and keeps FWD and RC as separate primary views. The
analysis validates the panel, extraction, and cCRE digests before running
complete-dictionary Welch/Mann–Whitney/BH tests.

Launch the modest AWS CPU task from the repository root:

```bash
eval "$(aws configure export-credentials --format env)"

sky launch -c dna-exp440-ccre-subtype-associations \
  experiments/exp440_m51_sae_reference_segmentation/sky.ccre.yaml \
  --env EXPERIMENT_COMMIT=<40-character-analysis-commit> \
  --env EXTRACTION_MANIFEST_SHA256=<64-character-sha256> \
  --env AWS_ACCESS_KEY_ID \
  --env AWS_SECRET_ACCESS_KEY \
  --env AWS_SESSION_TOKEN \
  -y
```

Outputs are uploaded below
`s3://oa-bolinas/experiments/exp440/retrieval/dna-exp440-ccre-subtype-associations-seed288-r1/`,
with `manifest.json` uploaded last.

Experimental findings belong in issue #440 rather than this runbook.
