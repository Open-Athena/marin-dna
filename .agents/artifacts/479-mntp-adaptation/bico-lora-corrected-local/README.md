# Corrected BICO LoRA causal gate

This compact evidence replaces the invalid causal-baseline row in W&B run `t37n0upf` with the previously verified standard causal source readout on the same 640-target validation plan.
The source and BICO artifacts share validation-plan SHA-256 `35542611d71102479f3d07dc6565350120d1d89944e5a93f88efb641ece7e3ba`.
All 640 sample IDs, components, target indices, and repeat-mask indicators match exactly.
The source used `[UNK]` and the BICO candidate used `[PAD]` at the selected input token, but that input is one position after the causal source readout and is therefore invisible to the causal baseline.

The corrected gate fails at every retained checkpoint.
At step 1,000, BICO LoRA CE is `1.282991` versus causal `1.051000`, and accuracy is `0.409375` versus causal `0.509375`.
The paired step-1,000 CE delta is `+0.231991` with 95% interval `[+0.186284, +0.278412]`.
The paired step-1,000 accuracy delta is `-0.100000` with interval `[-0.143750, -0.057813]`.

The full-attention training trajectory remains valid and improves progressively.
This local correction does not establish fresh-process adapter reload parity because the attempted A10G audit was blocked pending explicit authorization to read and publish private W&B artifacts.
No checkpoint was deleted or modified.
