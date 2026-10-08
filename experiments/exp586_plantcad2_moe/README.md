# exp586: d768 MoE DNA adaptation

Experiment [#586](https://github.com/Open-Athena/marin-dna/issues/586) repeats experiment 582 with the `d768` Hero scaling-ladder model: approximately 1.6B total parameters and 61M active parameters per token.
It compares the permanent language-pretrained checkpoint with the same architecture initialized from scratch.

## Training contract

- Use separate 8,192-base examples without packing and independently reverse-complement each training occurrence with 50% probability.
- Pretrained training retains the 128,256-entry language vocabulary but forces one existing token per DNA character.
- Scratch training uses the historical eight-entry DNA tokenizer, including EOS in the vocabulary without inserting EOS into examples.
- Train for exactly 216,158,699,520 input tokens: ten corpus epochs and 412,290 updates at global batch 64.
- Match the Hero text-run schedule: linear warmup for 4,122 updates, then immediate linear decay to 5% of peak through the end of training.
- Save a permanent checkpoint at update 4,123, after the one exact peak-LR update, and another at update 412,290.
- Use the Hero MuonH/Adam heuristic evaluated at the actual training budget, width 768, and 524,288 tokens per global update.
- Use zero weight decay and pooled EP8 with receiver capacity 32 and sender transport capacity 8.

The six seed-zero trials use pretrained LR multipliers 0.5, 1, and 2, and scratch multipliers 1, 3, and 10.
Their W&B IDs follow `exp586-plantcad2-d768-{pretrained|scratch}-lrm{multiplier}-seed0-v1`.
Checkpoints are stored under `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/checkpoints/<run-id>/2026.10.08/checkpoints/`.

## Runtime

The project uses the coherent Marin `0.2.149.dev37775727576` package set published on 2026-10-08 and vendors the required Hero experiment modules from Marin commit `33efc10f13319ae88d4bd71d0b3451b2971615c1`.
The permanent language source is `s3://marin-us-east-02a/marin/grug/rav-ladder-d768-v2/2026.08.18/checkpoints/step-11420`.

```bash
uv sync --locked --extra cpu
uv run --locked --extra cpu pytest
```

An individual production driver is selected explicitly:

```bash
python -m exp586_moe.train --condition pretrained --lr-multiplier 1 --cluster cw-rno2a --nodes 2
```

CoreWeave placement is operational and does not change the W&B or checkpoint identity.
Every GPU request uses whole eight-H100 nodes at batch priority.
