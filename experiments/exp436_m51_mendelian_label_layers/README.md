# Experiment 436: Mendelian label across SAE layers

This permanent, unmerged experiment implements the protocol in [issue #436](https://github.com/Open-Athena/marin-dna/issues/436) and informs [research question #288](https://github.com/Open-Athena/marin-dna/issues/288).

## Scientific design

The first/middle/final m5.1 panel is reported blocks 1, 10, and 19. Issue #426 already trained and verified block-10 and block-19 SAEs at exact 5,000,550- and 25,000,200-activation checkpoints. This experiment reuses those artifacts and trains only the missing block-1 trajectory under the same seed-288, 8× BatchTopK K=64, 50/50 FWD/RC, per-layer-normalized recipe.

The primary scientific work tests Mendelian `label` against every activation-supported SAE feature, then distinguishes marginal feature association from information distributed across the full sparse code and from signal lost during SAE reconstruction. Layers, budgets, FWD/RC orientations, focal/pooled responses, and statistical families remain separately reported.

## Block-1 training

The copied #426 training path preserves the exact data revisions, budgets, optimizer, normalization sample, bf16 language-model / fp32 SAE boundary, and two-batch LLM prefetch. The language model remains eager because the pinned Qwen/SAELens hook cache failed the real `torch.compile` smoke in #426; this is the known-correct comparable recipe. Later Mendelian extraction uses the standardized compiled prediction path when its hooks pass the real smoke test.

From this directory:

```bash
uv lock
uv sync --frozen
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python train.py --dry-run --no-compile
sky launch -d --dryrun sky.train.yaml
```

After pushing the exact experiment commit:

```bash
sky launch -d -c exp436-lambda \
  --env EXPERIMENT_COMMIT=<40-character-commit> \
  sky.train.yaml
```
 The managed task stages both inference exports and a hash-complete manifest under `retrieval/<run-id>/`. Retrieve and independently verify every listed byte count and SHA-256 before terminating `exp436-lambda`.

## Existing SAE inputs

The verified local #426 artifacts are intentionally not committed. Their expected weight hashes are:

| arm | SHA-256 |
|---|---|
| block10-5m | `dacde7e27d8ff20eb1ca52497f8b76494a4b92fc8ee607cfff8bcd38604267a0` |
| block10-25m | `606b81e2cc34ad7225de0fbaf5e673e688c4f990fc748cb59223316893e826b6` |
| block19-5m | `a35abcd7d8b9098b3574bff1270cd177117b687ade5845471403900b46f00971` |
| block19-25m | `e4f10ba59f10be943dbdc33f469f986f598c5e34fcba42577efad27717231533` |

Analysis results belong in issue #436; this README remains a reproduction runbook.

## Paired focal extraction

`extract_focal.py` runs each reference/alternate pair in both forward and
reverse-complement orientation. It captures the focal hidden state from blocks
1, 10, and 19 in one model pass, then applies the 5M and 25M SAE for each layer.
Each arm/orientation is written as a sparse union over features active in either
allele, preserving `ref_activation`, `alt_activation`, and signed `delta`.

After the block-1 trajectory has been retrieved and independently hash-checked:

```bash
uv run pytest tests/test_extract_focal.py
uv run ruff check extract_focal.py tests/test_extract_focal.py
uv run ruff format --check extract_focal.py tests/test_extract_focal.py
sky launch -d --dryrun sky.extract.yaml
sky launch -d -c exp436-lambda \
  --env EXPERIMENT_COMMIT=<40-character-commit> \
  sky.extract.yaml
```

The task stages hash-complete sparse Parquets under
`retrieval/dna-exp436-mendelian-focal-seed288-r1/`. Retrieve and verify the
manifest before analysis or cluster termination.

## Focal association inventory

The focal scan tests every response-supported feature against Mendelian
`label` overall and within every adequately supported predefined subset. The
primary variant contrasts are `abs(delta)` and signed `delta`; reference-only
and alternate-only activation scans are contextual controls. Welch and
Mann–Whitney families receive separate within-family BH correction, while raw
and sign-reversed AUPRC remain descriptive.

```bash
uv run pytest tests/test_analyze_focal.py
uv run ruff check analyze_focal.py tests/test_analyze_focal.py
uv run ruff format --check analyze_focal.py tests/test_analyze_focal.py
sky launch -d -c exp436-lambda --dryrun sky.analyze.yaml
sky launch -d -c exp436-lambda \
  --env EXPERIMENT_COMMIT=<40-character-commit> \
  sky.analyze.yaml
```

The task writes one complete Parquet per declared hypothesis family plus a
compact `top_hits.parquet`, results manifest, and hashes. It uses the warm H100
node's CPUs; no second paid instance is required.

## Summaries and figures

After retrieving and hash-verifying the complete focal association run, produce
the target, robustness, strand, checkpoint-lineage, and recurrent-feature
summaries with bounded local threads:

```bash
flock -n /tmp/marin-dna-local-heavy.lock \
  env EXPERIMENT_COMMIT=<40-character-commit> \
  RUN_ID=dna-exp436-mendelian-focal-summary-seed288-r1 \
  POLARS_MAX_THREADS=2 RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  uv run --no-dev python summarize_focal.py \
    --associations-root <verified-association-root> \
    --output-dir <new-summary-output>
```

The summarizer emits both SVG and PNG figures. FWD/RC remain separate in every
inferential family; overlap is summarized only after those results exist.
