# Issue #473 publication v3

This additive workflow supersedes the earlier issue-473 publication launchers
without changing them. Each immutable producer input is first retrieved once
from S3 into a retained, commit-keyed local snapshot. Shuffling, card
generation, and artifact validation all consume that snapshot, avoiding any
dependency on repeated implicit retrieval by the Snakemake storage plugin.

Build and validate without writing to Hugging Face:

```bash
sky launch -d -c issue-473-hf \
  snakemake/vertebrate_projection_dataset/sky/issue_473_hf_v3.yaml \
  --env PIPELINE_COMMIT_SHA=<exact-publication-commit> \
  --env DRY_RUN=0
```

After reviewing the three local cards and archived manifest, reuse the worker
for the explicit private upload target:

```bash
sky exec -d -c issue-473-hf \
  snakemake/vertebrate_projection_dataset/sky/issue_473_hf_v3.yaml \
  --env PIPELINE_COMMIT_SHA=<exact-publication-commit> \
  --env TARGET=issue_473_v3_all_hf \
  --env DRY_RUN=0 \
  --env ALLOW_HF_UPLOAD=1
```
