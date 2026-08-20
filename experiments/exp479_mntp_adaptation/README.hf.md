---
license: apache-2.0
library_name: transformers
tags:
  - biology
  - genomics
  - dna
  - masked-language-modeling
base_model: marin-dna/marin-dna-exp135-m5.1
---

# MarinDNA m5.1 MNTP adaptation pilot — private negative-result staging

This card records the completed transferred-MNTP checkpoint from issue #479. Checkpoints remain private across `marin-dna/marin-dna-exp479-mntp-m5.1` and `gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover`. The model is not approved for public release or production use; the pilot's primary VEP results were negative.

## Model

The model starts from [`marin-dna/marin-dna-exp135-m5.1`](https://huggingface.co/marin-dna/marin-dna-exp135-m5.1/tree/a73a5dcfb3d64b8941e7e7596c6e88ef77db3e7a), removes the causal attention restriction, adds `[MASK]` to the nucleotide vocabulary, and trains all parameters for 1,000 masked-next-token-prediction steps. A target nucleotide at input position `i` is replaced with `[MASK]` and predicted from output position `i - 1`.

Producing code: [`97a6e3c5`](https://github.com/Open-Athena/marin-dna/tree/97a6e3c50080005ad4f93f2206c4155b8f5cb7b9/experiments/exp479_mntp_adaptation); integrity audit: [`issue-479-mntp-pilot-audited-result`](https://github.com/Open-Athena/marin-dna/tree/issue-479-mntp-pilot-audited-result/experiments/exp479_mntp_adaptation)

Compact result snapshot: [`issue-479-mntp-pilot-audited-result`](https://github.com/Open-Athena/marin-dna/tree/issue-479-mntp-pilot-audited-result/.agents/artifacts/479-mntp-adaptation)

Direct W&B runs: [checkpoint audit](https://wandb.ai/gonzalobenegas/marin/runs/gavkgtmf), [stability audit](https://wandb.ai/gonzalobenegas/marin/runs/q67hbkp4), and [final dependency](https://wandb.ai/gonzalobenegas/marin/runs/yl5sgffn)

## Training data and objective

The pilot uniformly sampled five public m5.1 components: CDS, upstream, downstream, enhancer, and ncRNA. Dataset revisions are pinned in the producing code. Each sequence sampled one mask probability from `Uniform(0, 1)`; every selected A/C/G/T token was replaced with `[MASK]`. The loss averaged selected-token cross-entropy per sequence before averaging sequences, with weight 1 for uppercase bases and 0.01 for lowercase bases.

Training used one Lambda GH200 96 GB, bf16 mixed precision, batch 64, no gradient accumulation, and one seed. The 1,000-step 10%/70%/20% warmup/stable/cooldown schedule exposed the model to 16,384,000 model tokens and 16,320,000 nucleotide bases. Full Lightning state was preserved every 100 steps; step 800 is the pre-cooldown recovery point.

## Results

The pilot was technically valid: all actual-checkpoint smoke tests passed and transferred MNTP, scratch MNTP, and causal continuation completed with finite loss, gradients, and optimizer state.

| Step-1,000 metric | Transferred MNTP | Scratch MNTP |
|---|---:|---:|
| Pooled MNTP validation loss | 0.397270 | 0.399543 |
| Single-mask validation loss | 0.310077 | 0.313152 |
| Pooled nucleotide accuracy | 0.334408 | 0.333749 |
| Single-mask nucleotide accuracy | 0.418750 | 0.396875 |

Transferred MNTP acquired bilateral context dependence. On matched VEP probe sequences its left/right L1 response was 0.02007/0.01988, versus 0.02581/0.01280 for full attention without adaptation and 0.16545/0 for source CLM. It exceeded the no-adaptation control only on the right, so the strict control criterion was not met.

| Odd-autosome/X primary endpoint | Source CLM FWD+RC | Transferred MNTP FWD | Scratch MNTP FWD | Continued CLM FWD+RC | Transferred FWD+RC |
|---|---:|---:|---:|---:|---:|
| Mendelian macro AUPRC | 0.3951 | 0.1151 | 0.1112 | 0.3064 | 0.1152 |
| Complex-trait global AUPRC | 0.1342 | 0.1003 | 0.1018 | 0.1188 | 0.0996 |
| SGE accession/consequence macro AUPRC | 0.3577 | 0.1427 | 0.1378 | 0.3052 | 0.1429 |

No task passed the preregistered single-orientation inference gate because transferred FWD did not exceed source CLM FWD+RC. Complete-flank ablations confirmed bilateral score dependence, and ±64-base window shifts were stable, but neither changed the downstream conclusion.

The integrity audit found no checkpoint, replay, coordinate, tokenizer, readout-shift, or shared-loss-path bug. Source save/reload and replayed CLM step 400 were bit-exact across all 51,623 odd/X variants and both strands. Continued-CLM validation and AUPRC degradation began gradually after the first ten steps, consistent with destructive optimization rather than serialization. Its gradient norms were mild and had no post-warmup spikes.

The original dependency diagnostic did contain a batch-shape bug: a batch-one baseline was compared with batch-1,020 substitutions, exposing BF16 kernel differences. Corrected same-call maps at the step-1,000 checkpoints show bilateral dependency for both MNTP arms and an exactly zero future-context triangle for continued CLM. Transferred MNTP's past/future mean dependency was 0.05314/0.05334, scratch MNTP's was 0.03056/0.02917, and continued CLM's was 0.12510/0.

The final conservative listed-price estimate, including failures and all audit attempts, was $24.7340 against the $50 cap. The 10,000-step extension is not proposed.

## Intended use and limitations

This checkpoint is an experimental, full-attention DNA model for research on causal-to-bidirectional conversion. It is not the autoregressive m5.1 generation checkpoint and should not be substituted for it. It is not recommended for variant-effect prediction, clinical use, or production inference.

The evidence is one seed and 1,000 adaptation steps. Small validation-loss differences have no replication uncertainty. Direct VEP regressed substantially relative to the source checkpoint. Corrected dependency maps and context dependence are mechanistic diagnostics, not proof of useful representations. A negative pilot does not establish whether ordinary MLM, longer adaptation, different layer/pooling choices, or supervised chromatin-accessibility training would help.

All labeled development and diagnostics were restricted to odd-numbered autosomes and chromosome X. No even-autosome or chromosome-Y labels, predictions, effect measurements, or aggregate metrics were accessed. The preregistered HBA1 dependency map used unlabeled chromosome-16 reference sequence only. Per-variant scores and model weights remain private.
