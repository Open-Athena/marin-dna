# Issue #473 publication v2

This additive recovery workflow supersedes the original issue-473 publication
launcher without changing it. Snakemake's default storage provider removes a
remote output's local copy after uploading it, so the original artifact
validator could not read the three generated dataset cards. V2 stages cards
and the validation manifest as explicit local files, archives the validated
manifest separately to S3, and retains the local tree for private upload.

Build and validate without writing to Hugging Face:

```bash
sky launch -d -c issue-473-hf \
  snakemake/vertebrate_projection_dataset/sky/issue_473_hf_v2.yaml \
  --env PIPELINE_COMMIT_SHA=<exact-publication-commit> \
  --env DRY_RUN=0
```

After reviewing the three local cards and archived manifest, reuse the worker
for the explicit private upload target:

```bash
sky exec -d -c issue-473-hf \
  snakemake/vertebrate_projection_dataset/sky/issue_473_hf_v2.yaml \
  --env PIPELINE_COMMIT_SHA=<exact-publication-commit> \
  --env TARGET=issue_473_v2_all_hf \
  --env DRY_RUN=0 \
  --env ALLOW_HF_UPLOAD=1
```
