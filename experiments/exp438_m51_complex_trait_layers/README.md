# Experiment 438: complex-trait label across SAE layers

This permanent, unmerged experiment implements the protocol in [issue #438](https://github.com/Open-Athena/marin-dna/issues/438) and informs [research question #288](https://github.com/Open-Athena/marin-dna/issues/288).

The fixed feature-1662 mechanistic follow-up has its own runbook in
[`SATURATION.md`](SATURATION.md), including its separate immutable-design and
final-result S3 prefixes.

## Scientific design

The experiment asks whether paired reference/alternate SAE responses associate with the official complex-trait causal label, both overall and within each predefined subset. It reports the preregistered first/middle/final panel—m5.1 blocks 1, 10, and 19—separately. The primary response is focal `abs(alt_activation - ref_activation)`; signed delta is a sensitivity analysis. Forward and reverse-complement orientations also remain separate.

Every response-supported feature is tested. Welch and Mann–Whitney tests report standardized mean difference and rank-biserial effect sizes. BH correction is separate within each layer × orientation × response family across the overall target and six subsets with at least 30 positives. Splicing and synonymous subsets retain descriptive effects and AUPRC but are excluded from BH families because each has fewer than 30 positives. AUPRC is descriptive, and the class prevalence is fixed at 10% overall and within every subset.

The qualitative hypothesis is that the middle layer may carry more complex-trait signal than the final layer because these variants can perturb local regulatory biology without the stronger natural-selection signature common in Mendelian disease. The code does not select a layer post hoc; all three are reported.

## Pinned inputs

- Official panel: `marin-dna/evals_complex_traits`, train split, revision `22f86a89c65cb8f3007ac3cc2739f40efefa4340`; 11,630 rows, 1,163 matched groups, SHA-256 `d65fcee2740317451c41e5df6d4dd52cddf33847578b7565075e38d74d9865e3`.
- Model: `marin-dna/marin-dna-exp135-m5.1`, revision `c0676b2012b8b9c526deb26ff517f6b92b6d375d`.
- SAEs: 15,360-feature JumpReLU dictionaries at exactly 25,000,200 training activations for blocks 1, 10, and 19.
- Reference: Ensembl release 115 GRCh38 soft-masked primary assembly under `s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/`.

The block-1 SAE is staged from `s3://oa-bolinas/experiments/exp436/retrieval/`; block-10 and block-19 are staged from `s3://oa-bolinas/experiments/exp426/retrieval/`. Their internal metadata and every required file hash are checked before inference.

## Inference and compute

`extract_focal.py` constructs a 255-bp reference and alternate window with one explicit VCF-position-to-0-based conversion. It captures the focal hidden state from all three layers in one model pass and then applies the corresponding SAE. Sparse outputs retain reference activation, alternate activation, and signed delta for every feature active in either allele.

The frozen language model runs in bf16 and the SAEs in fp32. The hook-based prediction path is intentionally eager: the pinned Qwen/SAELens hook cache failed the real `torch.compile` smoke in the earlier layer experiment, so compile is not a valid comparable optimization here. Batch size is 64. The panel is small enough that a DataLoader worker pool would add orchestration without removing the serial indexed-FASTA boundary; the CPU-heavy association stage instead uses the 16-vCPU A10G host with bounded numerical-library threads.

## Storage

S3 is the durable source of truth. The EC2 root disk is a disposable staging cache for the reference, three SAE checkpoints, model cache, and run outputs. A completed run is hash-inventoried and uploaded to:

```text
s3://oa-bolinas/experiments/exp438/retrieval/<run-id>/
```

The archive contains the byte-identical panel, panel manifest, sparse extraction, complete association Parquets, summary tables, SVG/PNG plots, and nested plus top-level SHA-256 manifests. Only compact summaries and figures need to be retrieved locally.

## Validate and launch

From the repository root:

```bash
uv lock --project experiments/exp438_m51_complex_trait_layers
uv sync --project experiments/exp438_m51_complex_trait_layers --frozen
uv run --project experiments/exp438_m51_complex_trait_layers \
  pytest -q experiments/exp438_m51_complex_trait_layers/tests
uv run --project experiments/exp438_m51_complex_trait_layers \
  ruff check experiments/exp438_m51_complex_trait_layers
uv run --project experiments/exp438_m51_complex_trait_layers \
  ruff format --check experiments/exp438_m51_complex_trait_layers
sky launch -d --dryrun experiments/exp438_m51_complex_trait_layers/sky.run.yaml
```

After committing and pushing the exact experiment branch:

```bash
sky launch -d -c exp438-a10g \
  --env EXPERIMENT_COMMIT=<40-character-commit> \
  experiments/exp438_m51_complex_trait_layers/sky.run.yaml
```

Babysit the initial checkpoint/reference downloads and the first extraction batches. After completion, independently compare the S3 inventory and archive manifest before terminating the cluster. Findings and plots belong in issue #438 rather than this runbook.
