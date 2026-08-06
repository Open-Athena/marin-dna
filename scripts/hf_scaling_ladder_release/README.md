# Scaling-ladder Hugging Face release

This directory contains the tracked release materials for the eight final MarinDNA v0.5 parameter-scaling checkpoints. The release is intentionally kept on a permanent, unmerged branch because it is a one-off publication record rather than reusable MarinDNA core.

## Human-review gate

No Hugging Face write may happen until a human approves all nine rendered model cards (m5.1 plus the eight scaling checkpoints). The publisher computes a SHA-256 review digest over those nine README files and requires that exact digest through `--approved-review-digest`, preventing publication after an unreviewed copy change.

Proposed public repositories:

| Size | Repository |
|---|---|
| 46M | `marin-dna/marin-dna-scaling-v0.5-h640-p46M` |
| 76M | `marin-dna/marin-dna-scaling-v0.5-h768-p76M` |
| 128M | `marin-dna/marin-dna-scaling-v0.5-h896-p128M` |
| 255M | `marin-dna/marin-dna-scaling-v0.5-h1152-p255M` |
| 476M | `marin-dna/marin-dna-scaling-v0.5-h1408-p476M` |
| 1B | `marin-dna/marin-dna-scaling-v0.5-h1920-p1B` |
| 2B | `marin-dna/marin-dna-scaling-v0.5-h2432-p2B` |
| 4B | `marin-dna/marin-dna-scaling-v0.5-h2944-p4B` |

Each scaling repository receives only its final step-215573 checkpoint, the complete three-file tokenizer bundle, the reviewed model card as `README.md`, and the repository-root Apache-2.0 `LICENSE`. The publisher also updates the existing m5.1 README. It only adds and reorders Collection items; it asserts that the Collection title and description remain unchanged.

## Sources

The canonical exports live under `gs://marin-us-east5/checkpoints/…/hf/step-215573`. The upload reads the byte-identical evals_v2 caches under `s3://oa-bolinas/snakemake/analysis/evals_v2/results/checkpoints/…`, which avoids a GCS-to-local-to-Hub transfer and is the same material used by the offline evaluation pipeline. The manifest pins each S3 object's size, ETag, and full-object CRC64NVME checksum.

The checkpoints total 33,578,382,278 bytes. Because the current node cannot stage the whole ladder at once, `release.py` downloads, hashes, and uploads one file at a time through a temporary directory. Repositories remain private while incomplete and are made public only after their source and Hub inventories match.

## Review

```bash
uv run python scripts/hf_scaling_ladder_release/release.py review-digest
uv run python scripts/hf_scaling_ladder_release/release.py verify-source
```

## Publish after approval

```bash
uv run python scripts/hf_scaling_ladder_release/release.py publish \
  --approved-review-digest <reviewed-sha256>
```

The publisher is idempotent: existing matching files are reused, while any source or destination mismatch fails loudly. It writes `release_state.json` with per-file SHA-256 values and final Hugging Face revisions.

## Public verification

```bash
uv run python scripts/hf_scaling_ladder_release/release.py verify-public
```

Public verification checks anonymous access, exact repository inventories and LFS SHA-256 values, card metadata, tokenizer behavior, configuration/parameter invariants, unchanged Collection metadata and exact item ordering, and a full representative model load. The original evals_v2 runs already loaded and scored all eight source checkpoints; the verifier also checks every sharded index and tensor-byte total.
