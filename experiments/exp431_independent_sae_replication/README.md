# exp431: independent SAE semantic replication

This permanent, unmerged experiment branch implements [issue 431](https://github.com/Open-Athena/marin-dna/issues/431), which informs [issue 288](https://github.com/Open-Athena/marin-dna/issues/288).

The experiment asks whether the causal acceptor, donor, stop-creation, and synonymous/codon-degeneracy responses found in #429 reappear when the SAE dictionary changes. Feature IDs are allowed to permute or split. Candidate selection uses discovery/validation contexts only and reads the response-independent test contexts once.

## Inputs

- Reference dictionary: #418 block-10/5M seed-288 SAE.
- Existing independent-recipe dictionary: #426 block-10/5M seed-288 SAE, trained on an exactly balanced FWD/RC stream with folded scalar activation normalization.
- Fresh independent-seed dictionary: the same #426 block-10/5M recipe with seed 289; training support is added in the second stage of this experiment.
- Frozen #429 22,528-row consequence panel and Ensembl-115 GRCh38 reference artifacts.

All model inputs are 255 bp plus BOS. Dataset variant positions remain 1-based until the FASTA boundary; all constructed genomic intervals are 0-based half-open.

## Existing-dictionary stage

Run from this directory. Replace `COMMIT` with the pushed commit SHA on `codex/issue-431-independent-sae-replication`.

### 1. Freeze decoder candidates locally

```bash
EXPERIMENT_COMMIT=COMMIT uv run python decoder_neighbors.py \
  --reference-sae ../../scratch/issue418/dna-exp418-micro-seed288-e816ec343c30/sae \
  --candidate-sae ../../scratch/issue426/retrieval/dna-exp426-layer-budget-seed288-r5/models/block10-5m \
  --dictionary-name rc-balanced-normalized-seed288 \
  --output-dir ../../scratch/issue431/decoder-existing \
  --top-k 32
```

This compares normalized decoder directions before biological test scoring and stores the top 32 positive-cosine candidates per frozen #429 query.

### 2. Build response-independent split panels on CPU

```bash
sky launch -c exp431-cpu sky.panels.yaml \
  --env EXPERIMENT_COMMIT=COMMIT -y
```

Retrieve the completed artifact:

```bash
sky rsync-down exp431-cpu \
  '$HOME/exp431-artifacts/panels/' \
  ../../scratch/issue431/retrieval/panels/
```

The builder takes the lowest 64 frozen sample hashes per class independently within discovery, validation, and test blocks after unambiguous transcript annotation. It never ranks a source context by SAE response.

### 3. Extract candidate profiles on one A10G

```bash
sky launch -c exp431-gpu-existing sky.extract-existing.yaml \
  --env EXPERIMENT_COMMIT=COMMIT \
  --env DICTIONARY_NAME=rc-balanced-normalized-seed288 -y
```

Retrieve and then terminate the GPU:

```bash
sky rsync-down exp431-gpu-existing \
  '$HOME/exp431-artifacts/existing/' \
  ../../scratch/issue431/retrieval/existing/
sky down exp431-gpu-existing -y
```

The three split extractions run sequentially on the same GPU. Base-model inference is bf16; SAE encoding is fp32; FWD and RC 31-position profiles remain separate.

### 4. Select on discovery/validation and test once

```bash
sky launch -c exp431-cpu sky.analyze-existing.yaml \
  --env EXPERIMENT_COMMIT=COMMIT \
  --env DICTIONARY_NAME=rc-balanced-normalized-seed288 -y
sky rsync-down exp431-cpu \
  '$HOME/exp431-artifacts/analysis-existing/' \
  ../../scratch/issue431/retrieval/analysis-existing/
sky down exp431-cpu -y
```

Discovery retains 16 feature/view candidates per concept; validation requires the same effect sign and chooses the largest absolute effect; test reports a 2,000-sample equal-context bootstrap interval. FWD/RC component intervals and mutation-position/codon profiles are retained.

## Validation

Local checks are bounded to two threads under the shared-host lock:

```bash
flock -n /tmp/marin-dna-local-heavy.lock \
  env POLARS_MAX_THREADS=2 RAYON_NUM_THREADS=2 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  uv run pytest -q
uv run ruff check .
uv lock --check
```

Every artifact directory contains a hash-complete manifest. Paid resources must be terminated after verified retrieval; no experiment output is committed to Git.
