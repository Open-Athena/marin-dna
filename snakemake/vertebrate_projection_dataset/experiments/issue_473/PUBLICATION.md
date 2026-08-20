# Issue #473 publication

The three new Hugging Face datasets are built by the standalone additive
workflow `workflow/Issue473Publication.smk`. It reads only the immutable full
projection namespace pinned in `config/issue_473_publication.yaml`; it neither
includes nor edits established projection, split, card, or upload rules.

The publication set is:

- `marin-dna/vertebrate-v1-issue473-center1-cds`
- `marin-dna/vertebrate-v1-issue473-center1-ccre-enhancer-centered`
- `marin-dna/vertebrate-v1-issue473-fullwindow-ccre-enhancer-centered`

The already-published #417 full-window CDS dataset remains the fourth matched
training arm and is not duplicated.

## Build and review

Commit the complete publication recipe before execution. Build and validate
the draft cards and JSONL.zst trees without writing to Hugging Face:

```bash
sky launch -d -c issue-473-hf \
  snakemake/vertebrate_projection_dataset/sky/issue_473_hf.yaml \
  --env PIPELINE_COMMIT_SHA=<exact-publication-commit> \
  --env DRY_RUN=0
```

The artifact manifest rejects a producer-receipt mismatch, source/shard row
count mismatch, imbalanced shards, malformed boundary records, split leakage,
sequence-length violations, unexpected files, and content-hash drift. Review
all three generated `README.md` cards and the full projection QC/manual sample
before upload.

## Upload

After the generated cards and QC pass review, reuse the retained high-memory
worker and opt in to the only externally mutating target:

```bash
sky exec -d -c issue-473-hf \
  snakemake/vertebrate_projection_dataset/sky/issue_473_hf.yaml \
  --env PIPELINE_COMMIT_SHA=<exact-publication-commit> \
  --env TARGET=issue_473_all_hf \
  --env DRY_RUN=0 \
  --env ALLOW_HF_UPLOAD=1
```

Each repository is checked before upload for unexpected paths and checked
after upload for its exact file tree, LFS sizes and SHA-256 hashes, and a
byte-identical card at the resulting immutable revision.
The retained local `publication/upload.done/<dataset>` receipts record each
repository ID and exact 40-character Hub revision. Capture those three
revisions from the retained worker before termination and pin them in the four-
arm training recipe.
