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
rsync -av --protect-args \
  exp431-cpu:/home/ubuntu/exp431-artifacts/panels/ \
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
rsync -av --protect-args \
  exp431-gpu-existing:/home/ubuntu/exp431-artifacts/existing/ \
  ../../scratch/issue431/retrieval/existing/
sky down exp431-gpu-existing -y
```

The three split extractions run sequentially on the same GPU. Base-model inference is bf16; SAE encoding is fp32; FWD and RC 31-position profiles remain separate.

### 4. Select on discovery/validation and test once

```bash
sky launch -c exp431-cpu sky.analyze-existing.yaml \
  --env EXPERIMENT_COMMIT=COMMIT \
  --env DICTIONARY_NAME=rc-balanced-normalized-seed288 -y
rsync -av --protect-args \
  exp431-cpu:/home/ubuntu/exp431-artifacts/analysis-existing/ \
  ../../scratch/issue431/retrieval/analysis-existing/
sky down exp431-cpu -y
```

Discovery retains 16 feature/view candidates per concept; validation requires the same effect sign and chooses the largest absolute effect; test reports a 2,000-sample equal-context bootstrap interval. FWD/RC component intervals and mutation-position/codon profiles are retained.

### 5. Whole-dictionary synonymous sensitivity

The registered geometry-constrained synonymous candidate failed its held-out test, so the mandatory sensitivity searches all 15,360 features. Discovery and validation are one job that cannot see the test panel; only the frozen selection artifact is retrieved before a separate test job is launched.

~~~bash
sky launch -c exp431-fresh-lambda sky.synonymous-search.yaml \
  --env EXPERIMENT_COMMIT=COMMIT \
  --env DICTIONARY_NAME=rc-balanced-normalized-seed288 -y
rsync -av --protect-args \
  exp431-fresh-lambda:/home/ubuntu/exp431-artifacts/synonymous-existing/selection/ \
  ../../scratch/issue431/retrieval/synonymous-existing/selection/

sky launch -c exp431-fresh-lambda sky.synonymous-test.yaml \
  --env EXPERIMENT_COMMIT=COMMIT \
  --env DICTIONARY_NAME=rc-balanced-normalized-seed288 -y
rsync -av --protect-args \
  exp431-fresh-lambda:/home/ubuntu/exp431-artifacts/synonymous-existing/test-analysis/ \
  ../../scratch/issue431/retrieval/synonymous-existing/test-analysis/
~~~

The test command mounts only the test panel, the already frozen selection, and the candidate SAE. Full-dictionary activation arrays stay on the remote instance; only manifests, selection tables, and final test artifacts are retrieved.

## Fresh seed-289 stage

The preregistered fresh arm is reported block 10 at the exact 5,000,550-activation endpoint. `train_seed289.py` narrows the successful #426 path to that one hook and one budget while retaining its optimizer, BatchTopK hyperparameters, 100-batch scalar normalization estimate, pinned data order, exact FWD/RC balance, and export diagnostics. It emits no auxiliary layer or 25M arms. The gLM is bf16, the SAE is fp32, and two LLM batches are prefetched. The tested eager path remains mandatory because #426 found that `torch.compile` dropped the dynamic multi-hook cache.

Launch only after the candidate-extraction A10G has been terminated:

```bash
sky launch --dryrun -c exp431-fresh-lambda sky.train-fresh.yaml \
  --env EXPERIMENT_COMMIT=COMMIT
sky launch -c exp431-fresh-lambda sky.train-fresh.yaml \
  --env EXPERIMENT_COMMIT=COMMIT -y
```

Retrieve and verify the hash-complete run, but keep the H100 warm for the frozen analysis:

```bash
rsync -av --protect-args \
  exp431-fresh-lambda:/home/ubuntu/exp431-artifacts/dna-exp431-fresh-seed289/ \
  ../../scratch/issue431/retrieval/fresh-seed289/
```

### Fresh candidate transfer and held-out test

After the fresh training manifest is independently verified, freeze decoder candidates and run the unchanged three-split analysis on the same H100:

~~~bash
sky launch --dryrun -y -c exp431-fresh-lambda sky.analyze-fresh.yaml \
  --env EXPERIMENT_COMMIT=COMMIT \
  --env DICTIONARY_NAME=rc-balanced-normalized-seed289
sky launch -c exp431-fresh-lambda sky.analyze-fresh.yaml \
  --env EXPERIMENT_COMMIT=COMMIT \
  --env DICTIONARY_NAME=rc-balanced-normalized-seed289 -y
rsync -av --protect-args \
  exp431-fresh-lambda:/home/ubuntu/exp431-artifacts/decoder-fresh/ \
  ../../scratch/issue431/retrieval/decoder-fresh/
rsync -av --protect-args \
  exp431-fresh-lambda:/home/ubuntu/exp431-artifacts/fresh-extraction/ \
  ../../scratch/issue431/retrieval/fresh-extraction/
rsync -av --protect-args \
  exp431-fresh-lambda:/home/ubuntu/exp431-artifacts/analysis-fresh/ \
  ../../scratch/issue431/retrieval/analysis-fresh/
~~~

If any geometry-constrained concept fails, complete its registered whole-dictionary sensitivity before terminating the H100. Otherwise terminate it immediately after all retrieved manifests pass independent hash checks.

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
