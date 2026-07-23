# Issue 389 release workspace

This directory contains the tracked, reproducible release materials for
[issue #389](https://github.com/Open-Athena/marin-dna/issues/389). The branch is
permanent and intentionally unmerged; these one-off release artifacts are not
intended for MarinDNA `main`.

## Human-review gate

[`model_card.md`](model_card.md) was approved by `gonzalobenegas` before
publication on 2026-07-23. The Collection copy is in
[`collection_description.md`](collection_description.md), and the approval
record is pinned in `manifest.json`.

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

## Public release verification

The manifest pins the public model revision and Collection slug. Verify the
anonymous public download, exact release hashes, deterministic inference, all
dataset revisions, model-card metadata, and ordered Collection contents:

```bash
uv run python scripts/issue389/verify_release.py \
  --check-cloud \
  --check-datasets \
  --hub \
  --hub-dir scratch/issue389/public_checkpoint \
  --run-inference
```
