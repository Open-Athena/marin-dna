# exp586: d768 MoE DNA adaptation

Experiment [#586](https://github.com/Open-Athena/marin-dna/issues/586) repeats experiment 582 with the `d768` Hero scaling-ladder model: approximately 1.6B total parameters and 61M active parameters per token.
It compares the permanent language-pretrained checkpoint with the same architecture initialized from scratch.

## Training contract

- Use separate 8,192-base examples without packing and independently reverse-complement each training occurrence with 50% probability.
- Pretrained training retains the 128,256-entry language vocabulary but forces one existing token per DNA character.
- Scratch training uses the historical eight-entry DNA tokenizer, including EOS in the vocabulary without inserting EOS into examples.
- Train for exactly 216,158,699,520 input tokens: ten corpus epochs and 412,290 updates at global batch 64.
- Match the Hero text-run schedule: linear warmup for 4,122 updates, then immediate linear decay to 5% of peak through the end of training.
- Keep eight permanent checkpoints: update 4,123 at peak LR, six evenly spaced token milestones, and update 412,290 at the end.
- Use the Hero MuonH/Adam heuristic evaluated at the actual training budget, width 768, and 524,288 tokens per global update.
- Use zero weight decay and EP8: pooled-wave routing with receiver capacity 32 and sender transport capacity 8 for pretrained runs, and ring routing with capacity 32 for scratch runs.

The six seed-zero trials use pretrained LR multipliers 0.5, 1, and 2, and scratch multipliers 1, 3, and 10.
Their W&B IDs follow `exp586-plantcad2-d768-{pretrained|scratch}-lrm{multiplier}-seed0-v1`.
Checkpoints are stored under `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/checkpoints/<run-id>/2026.10.08/checkpoints/`.

## Runtime

This final-evaluation checkout uses the coherent Marin `0.2.224.dev40027069267` package set required by the current Iris controller and vendors the checkpoint's Hero model modules.
The training snapshot used Marin `0.2.201.dev39243773500`; final-checkpoint loss replay and independent probability/feature comparisons validate the evaluation runtime transition.
The permanent language source is `s3://marin-us-east-02a/marin/grug/rav-ladder-d768-v2/2026.08.18/checkpoints/step-11420`.

```bash
uv sync --locked --extra cpu
uv run --locked --extra cpu pytest
```

An individual production driver is selected explicitly:

```bash
python -m exp586_moe.train --condition pretrained --lr-multiplier 1 --cluster cw-rno2a --nodes 2
```

## Pretrained 1x continuation

Stage `s02` restores the complete evaluated pretrained 1x `step-412290` state and continues its absolute data and optimizer clocks to exactly 562,022,055,936 input tokens.
It linearly rewarms from 5% to the original peak learning rates over 6,596 updates, then immediately decays linearly to the same 5% endpoint at update 1,071,972.
The optimizer heuristic remains bound to the original 216,158,699,520-token stage so only the two injected learning-rate scalars change after restore.
Stage `s02` has a separate W&B identity and checkpoint root and retains eight continuation checkpoints, including the rewarm peak and final state.

```bash
python -m exp586_moe.train --condition pretrained --lr-multiplier 1 --stage s02 --cluster cw-rno2a --nodes 8
```

## Fresh pretrained 1T run

Stage `t1` starts a new DNA token clock from the same permanent `d768` language checkpoint and full tokenizer as the original pretrained condition.
It trains for 1,000,000,192,512 input tokens at global batch 64, recomputes the Hero optimizer heuristic for that horizon, warms up for 1% of the run, and linearly decays to 5% of peak.
Its separate W&B and checkpoint identity retains 12 permanent checkpoints, including the peak and final state.

```bash
python -m exp586_moe.train --condition pretrained --lr-multiplier 1 --stage t1 --cluster cw-rno2a --nodes 8
```

CoreWeave placement is operational and does not change the W&B or checkpoint identity.
Every GPU request uses whole eight-H100 nodes at batch priority.

## Isolated data-parallel experiments

`data_parallel.py` builds a mesh with one complete model and optimizer replica per GPU and selects the existing local `scatter` MoE backend.
The experiment also keeps Muon orthogonalization local, with gradient averaging and the existing global histogram router reductions as its only cross-replica operations.
No production launcher selects this path.

`dp_profile.py` restores a fixed independent checkpoint into a bounded experiment identity.
Batch-eight paired checks compare against two EP controls; batch-64 checks use the actual 524,288-token production batch on one eight-H100 node.
Profiles preload inputs and separately verify model/optimizer replication, collective operations, numerical differences, and temporary checkpoint replay.
The authorized 64-H100 comparison uses `--nodes 8 --batch-size 64 --verification --ep-reference live --end-to-end` to compare against the live EP configuration on the same allocation.
It separately measures prefetched training with metrics, production validation, and temporary checkpoint overhead.
`--end-to-end-updates 150 --end-to-end-order balanced` runs both backend orders for 300 updates per backend; both include per-update checkpoint-decision coordination and finite-metric checks.
See [the production-size results](dp_production_results.md) for throughput relative to the live EP runtime, variability, and correctness evidence.
`dp_control.py`, `dp_reconcile.py`, and [dp_operations.md](dp_operations.md) own the isolated submission and persistence boundary.
They never use the live run or its checkpoint owner.

## Evaluation

Final PlantCAD2 evaluation follows the appendable workflow in [`exp586_evaluation_operations.md`](exp586_evaluation_operations.md).
It reuses the exp582 benchmark protocol and historical results while binding every new artifact to an exact exp586 checkpoint, tokenizer, input manifest, and evaluation wheel.
The final 1T evaluation is governed by [`exp586_1t_evaluation_operations.md`](exp586_1t_evaluation_operations.md), including its frozen scorer and disjoint execution parts.

Additional frozen classification probes are governed by [`exp586_probe_operations.md`](exp586_probe_operations.md).
`python -m exp586_moe.classification_features` extracts final flank states for conservation/core tasks and real-token allele means for SV without projecting the full vocabulary.
Its `--gate` mode checks repeats, batch composition, withheld-base and outcome counterfactuals, independent full-state reduction, and exact ordered-FP32 expert-reference parity before production extraction.
Inputs and feature chunks are immutable and resumable; classifier fitting belongs to the PlantCAD2 evaluation project.
