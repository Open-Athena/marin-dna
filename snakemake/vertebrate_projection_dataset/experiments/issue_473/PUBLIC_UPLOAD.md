# Issue #473 public upload

Issue #473 datasets are always published as public Hugging Face dataset
repositories. The additive upload-only workflow consumes the exact retained
artifact tree pinned in `config/issue_473_public_upload.yaml`. It refuses a
pre-existing private repository and verifies public visibility both before and
after uploading the validated file tree.

Run only on the retained v3 publication worker after reviewing the cards and
manifest. First run the default dry-run; it cannot upload:

```bash
sky exec -d -c issue-473-hf \
  snakemake/vertebrate_projection_dataset/sky/issue_473_hf_public_upload.yaml \
  --env PIPELINE_COMMIT_SHA=<exact-uploader-commit>
```

After the dry-run succeeds, make the public upload explicit:

```bash
sky exec -d -c issue-473-hf \
  snakemake/vertebrate_projection_dataset/sky/issue_473_hf_public_upload.yaml \
  --env PIPELINE_COMMIT_SHA=<exact-uploader-commit> \
  --env DRY_RUN=0 \
  --env ALLOW_PUBLIC_HF_UPLOAD=1
```
