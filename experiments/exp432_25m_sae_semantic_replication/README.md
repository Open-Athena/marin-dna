# exp432: 25M SAE causal semantic replication

This permanent, unmerged experiment branch implements [issue 432](https://github.com/Open-Athena/marin-dna/issues/432), which informs [issue 288](https://github.com/Open-Athena/marin-dna/issues/288).

The experiment reuses the exact response-independent #431 discovery, validation, and test panels to ask whether causal acceptor, donor, stop-creation, and synonymous/codon-degeneracy responses persist in the existing #426 block-10/25M SAE. No SAE is trained. Feature IDs may permute or split.

All model inputs are 255 bp plus BOS. Dataset variant positions remain 1-based until the FASTA boundary; all constructed genomic intervals are 0-based half-open. Base-model inference is bf16, exported JumpReLU encoding is fp32, and FWD/RC 31-position profiles remain separate before frozen-view aggregation.

## Primary analysis

Replace `COMMIT` with the pushed commit SHA on `codex/issue-432-25m-sae-replication`:

~~~bash
sky launch --dryrun -y -c exp431-fresh-lambda sky.analyze-25m.yaml \
  --env EXPERIMENT_COMMIT=COMMIT \
  --env DICTIONARY_NAME=rc-balanced-normalized-seed288-25m
sky launch -c exp431-fresh-lambda sky.analyze-25m.yaml \
  --env EXPERIMENT_COMMIT=COMMIT \
  --env DICTIONARY_NAME=rc-balanced-normalized-seed288-25m -y
rsync -av --protect-args \
  exp431-fresh-lambda:/home/ubuntu/exp432-artifacts/ \
  ../../scratch/issue432/retrieval/
~~~

Decoder candidates are frozen before biological scoring. Discovery retains 16 feature/view candidates per concept; validation requires the same sign and freezes one; test reads that feature/view once with 2,000 equal-context bootstraps.

## Whole-dictionary sensitivity

If any geometry-constrained concept fails, run discovery/validation without a test mount, retrieve and independently verify the selection, then run the test-only task:

~~~bash
sky launch -c exp431-fresh-lambda sky.synonymous-search.yaml \
  --env EXPERIMENT_COMMIT=COMMIT \
  --env DICTIONARY_NAME=rc-balanced-normalized-seed288-25m -y
rsync -av --protect-args \
  exp431-fresh-lambda:/home/ubuntu/exp432-artifacts/synonymous-25m/selection/ \
  ../../scratch/issue432/retrieval/synonymous-25m/selection/

sky launch -c exp431-fresh-lambda sky.synonymous-test.yaml \
  --env EXPERIMENT_COMMIT=COMMIT \
  --env DICTIONARY_NAME=rc-balanced-normalized-seed288-25m -y
~~~

The current sensitivity implementation is for synonymous/codon degeneracy because that is the registered failure from #431. A different failed concept requires an explicit task update before test access.

## Validation

Run bounded local checks from this directory:

~~~bash
flock -n /tmp/marin-dna-local-heavy.lock \
  env POLARS_MAX_THREADS=2 RAYON_NUM_THREADS=2 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv lock --check
~~~

Every accepted artifact directory contains a hash-complete manifest. Paid compute is terminated after verified retrieval.
