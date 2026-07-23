# Issue 389 release workspace

This directory contains the tracked, reproducible release materials for
[issue #389](https://github.com/Open-Athena/marin-dna/issues/389). The branch is
permanent and intentionally unmerged; these one-off release artifacts are not
intended for MarinDNA `main`.

## Human-review gate

No Hugging Face repository or Collection may be created until a human approves
[`model_card.md`](model_card.md). The Collection copy is in
[`collection_description.md`](collection_description.md).

## Files

- `model_card.md` — reviewed as the future model-repository `README.md`.
- `collection_description.md` — exact public Collection description.
- `manifest.json` — source, destination, model, tokenizer, dataset, Collection,
  and smoke-test expectations.
- `verify_release.py` — fail-fast source and public-release verification.

The release repository adds the five checkpoint files from the evals_v2 S3
cache, maps `model_card.md` to `README.md`, and copies the repository-root
Apache-2.0 `LICENSE`.

## Source verification

The canonical checkpoint is on GCS and the transfer copy is cached by evals_v2
on S3. Stage only the five model/tokenizer files in gitignored scratch space:

```bash
mkdir -p scratch/issue389/source_checkpoint
aws s3 sync \
  s3://oa-bolinas/snakemake/analysis/evals_v2/results/checkpoints/mix-v0.9-p1B-i24-exp135-m5.1-step-59158/ \
  scratch/issue389/source_checkpoint/ \
  --exclude .snakemake_timestamp
```

Then verify cloud metadata, local hashes and structure, dataset revisions, and
the deterministic source inference:

```bash
uv run python scripts/issue389/verify_release.py \
  --check-cloud \
  --check-datasets \
  --checkpoint-dir scratch/issue389/source_checkpoint \
  --run-inference
```

Post-publication commands will be added after the required model-card review.
