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

# MarinDNA m5.1 MNTP adaptation pilot — private staging

This is the human-review draft for the private `marin-dna/marin-dna-exp479-mntp-m5.1` staging repository. The pilot may upload this reviewed draft privately so a self-terminating Lambda job can preserve resumable checkpoints. Do not make the repository public until the producing commit, run links, result table, and limitations below are finalized and reviewed again.

## Model

The model starts from [`marin-dna/marin-dna-exp135-m5.1`](https://huggingface.co/marin-dna/marin-dna-exp135-m5.1/tree/a73a5dcfb3d64b8941e7e7596c6e88ef77db3e7a), removes the causal attention restriction, adds `[MASK]` to the nucleotide vocabulary, and trains all parameters for 1,000 masked-next-token-prediction steps. A target nucleotide at input position `i` is replaced with `[MASK]` and predicted from output position `i - 1`.

Producing code: `<commit-pinned experiments/exp479_mntp_adaptation link>`

W&B report: `<report link>`

## Training data and objective

The pilot uniformly samples five public m5.1 components: CDS, upstream, downstream, enhancer, and ncRNA. Dataset revisions are pinned in the producing code. Each sequence samples one mask probability from `Uniform(0, 1)`; every selected A/C/G/T token is replaced with `[MASK]`. The loss averages selected-token cross-entropy per sequence before averaging sequences, with weight 1 for uppercase bases and 0.01 for lowercase bases.

## Results

`<Add the registered transferred-versus-scratch MNTP, bidirectionality, odd-autosome/X VEP, and single-orientation/FWD+RC results.>`

## Intended use and limitations

This checkpoint is an experimental bidirectional DNA representation model. It is not the autoregressive m5.1 generation checkpoint. The pilot uses one seed and 1,000 adaptation steps. The staging repository is incomplete while the run is active and must not be used as a released model. Labeled development and diagnostics exclude even-numbered autosomes and chromosome Y. A negative pilot result does not determine whether longer adaptation or supervised chromatin-accessibility training would help.
