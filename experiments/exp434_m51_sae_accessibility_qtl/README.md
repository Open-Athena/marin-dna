# Experiment 434: accessibility-QTL SAE features

This permanent, unmerged experiment implements the protocol in [issue #434](https://github.com/Open-Athena/marin-dna/issues/434) and informs [research question #288](https://github.com/Open-Athena/marin-dna/issues/288). Results belong in those issues; this file is the reproduction runbook.

## Scientific design

For each official QTL variant, the extractor computes the paired signed response

```text
delta = SAE(ALT) - SAE(REF)
```

at the focal nucleotide. Reported blocks 1, 10, and 19 are captured in one m5.1 forward pass and then encoded by separate 25M-activation JumpReLU SAEs. Forward and reverse-complement orientations remain separate. The language model runs in bf16 and the SAEs encode in fp32. `torch.compile` is disabled because the pinned multi-hook `HookedProxyLM` path is incompatible with compilation; the shared three-layer forward avoids three separate model passes.

Causality uses `abs(delta)` over positives and negatives. Direction uses signed `delta` against the official signed experimental effect among causal positives. Every eligible feature is tested, with Benjamini–Hochberg correction within each declared layer × dataset × orientation × outcome × statistic family.

## Fast signed-direction pilot

The first stage uses only the smaller dsQTL dataset's 559 official causal positives. This validates coordinate conversion, ALT sign orientation, sparse paired extraction, Pearson/Spearman inference, and within-family BH before the full 106,372-variant scan. It is the direction slice of the frozen full protocol, not a separate feature-selection step.

The CPU task uses the canonical uncompressed Ensembl release-115 GRCh38 FASTA and its `.fai` from S3. It converts the official 1-based positions exactly once at the FASTA boundary and materializes 255-bp reference/alternate windows. The panel is stored at:

```text
s3://oa-bolinas/experiments/exp434/inputs/dna-exp434-dsqtl-positive-panel-r1/
```

After committing and pushing the exact experiment branch:

```bash
sky launch -d --dryrun experiments/exp434_m51_sae_accessibility_qtl/sky.panel-pilot.yaml
sky launch -d -c exp434-qtl-panel \
  --env EXPERIMENT_COMMIT=<40-character-commit> \
  experiments/exp434_m51_sae_accessibility_qtl/sky.panel-pilot.yaml
```

Once the panel manifest and hashes are verified, launch the joint GPU extraction and small association analysis:

```bash
sky launch -d --dryrun experiments/exp434_m51_sae_accessibility_qtl/sky.direction-pilot.yaml
sky launch -d -c exp434-qtl-gpu \
  --env EXPERIMENT_COMMIT=<40-character-commit> \
  --env AWS_ACCESS_KEY_ID \
  --env AWS_SECRET_ACCESS_KEY \
  --env AWS_SESSION_TOKEN \
  experiments/exp434_m51_sae_accessibility_qtl/sky.direction-pilot.yaml
```

The GPU task stages hash-complete outputs under:

```text
s3://oa-bolinas/experiments/exp434/retrieval/dna-exp434-dsqtl-direction-pilot-extraction-r1/
s3://oa-bolinas/experiments/exp434/retrieval/dna-exp434-dsqtl-direction-pilot-associations-r1/
```

## Post-hoc feature-1829 interpretation

The pilot's strongest block-19 feature is followed up with a bounded mechanism
pass on the same 559 positive dsQTLs. `interpret_feature1829.py` selects the 32
most-active contexts in each orientation using activation alone (one most-active
allele per variant), saturates the 16 model-input positions ending at the focal
allele with A/C/G/T, and subtracts the decoded feature-only contribution before
the frozen final norm and LM head. The QTL effect is carried only for the
separately reported robustness diagnostics; it is never used to select contexts.
This is explicitly post-hoc and does not alter the discovery statistics.

```bash
sky launch -d --dryrun \
  experiments/exp434_m51_sae_accessibility_qtl/sky.feature1829.yaml
sky launch -d -c exp434-feature1829 \
  --env EXPERIMENT_COMMIT=<40-character-commit> \
  --env AWS_ACCESS_KEY_ID \
  --env AWS_SECRET_ACCESS_KEY \
  --env AWS_SESSION_TOKEN \
  experiments/exp434_m51_sae_accessibility_qtl/sky.feature1829.yaml
```

The hash-complete result is stored at:

```text
s3://oa-bolinas/experiments/exp434/retrieval/dna-exp434-feature1829-interpretation-r1/
```

## Local verification

Use bounded threads on the shared workstation:

```bash
flock -n /tmp/marin-dna-local-heavy.lock \
  env POLARS_MAX_THREADS=2 RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  uv run --project experiments/exp434_m51_sae_accessibility_qtl pytest -q \
  experiments/exp434_m51_sae_accessibility_qtl/tests

uv run --project experiments/exp434_m51_sae_accessibility_qtl ruff check \
  experiments/exp434_m51_sae_accessibility_qtl
uv run --project experiments/exp434_m51_sae_accessibility_qtl ruff format --check \
  experiments/exp434_m51_sae_accessibility_qtl
```

## Full scan

`build_panel.py`, `extract_focal.py`, and `analyze.py` also implement the frozen full caQTL + dsQTL scan. The full run materializes all official variants, uses all variants for binary causality, restricts signed direction correlations to official positives, and reports the pre-existing #421 candidate IDs only within the exact same SAE layer/dictionary/orientation. Its launch configuration should be added only after the pilot's sign and provenance checks pass.
