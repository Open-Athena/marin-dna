# exp582 Final Evaluation Operations

> This document governs evaluation of the two selected final exp582 checkpoints and the untouched d1536 language checkpoint used as a negative control.

## Scope

Evaluate the final update-21,440 checkpoints for pretrained 1× and random 2×, plus the original d1536 language checkpoint with no DNA training.
Do not run inference for the other exp582 trials or historical comparison models.
Read historical values from their published artifacts.
Report final results only on `eric-czech/plantcad2` PR #2; do not automatically update MarinDNA issue #582.

The pretrained checkpoint is `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/checkpoints/exp582-plantcad2-d1536-pretrained-lrm1-seed0-v1/2026.10.06/checkpoints/short-linear-v2/step-21440`, with raw metadata SHA-256 `3f175a36575f0fc55ecf0dc5fc0bd9ce4b22923f54f31f5bfa26c60322c44b88` and canonical metadata digest `c882f2cedb62266fc33f5dc60d0b83741b72fb506892baee99a70c7cd7efa4d6`.
The random checkpoint is `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/checkpoints/exp582-plantcad2-d1536-random-lrm2-seed0-v1/2026.10.06/checkpoints/short-linear-v2/step-21440`, with raw metadata SHA-256 `8e4249e1b37a4d37ab272c513ff5aa7831cbb990865fb7384236bd40d5a8e6f7` and canonical metadata digest `4a1ddbd00f0e204a662c20b431ec68dc72d398ce77022318755758b1df98c347`.
The language-only control is `s3://marin-us-east-02a/marin/grug/rav-ladder-d1536/2026.08.18/checkpoints/step-15128`, with raw metadata SHA-256 `128b0e2779b943fbef139132d4add4961e9e12cda3358b65951169940aa556fe` and canonical metadata digest `7198833f20fd9d1f442e6d0d9ef9738e634d674ff225ed276e7b56d71cf60227`.
It uses the same character-bound language tokenizer as the DNA-trained pretrained condition at inference. The checkpoint predates tokenizer fingerprints, so the code permits the missing fingerprint only for this exact URI and both pinned metadata digests.

## Execution

Use CoreWeave H100s and refresh fleet availability before every admission or placement change.
Every Iris root and child uses batch priority and user `eczech`.
Use W&B project `eric-czech/marin` as the source of evaluation progress, and use Iris only for exact dispatch liveness and concrete startup failures.
Stop if direct CoreWeave S3 access fails.

Run one bounded checkpoint pilot per evaluated model before scaling the evaluation.
Each pilot must verify checkpoint and tokenizer bindings, exact input hashes, deterministic repeats, batch invariance, direct ACGT projection parity, strand and coordinate alignment, output checksums, and exact reduction compatibility with the PlantCAD2 PR #20 protocols.
Use separate W&B identities and artifact roots for pilots so their values cannot be mistaken for publishable results.

The working SQLite database is `scratch/exp582-final-eval/exp582_eval.sqlite`.
Its immutable backup owner is `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/operations/evaluation/`.
Publish a verified immutable backup before and after every external state change.
The durable owner for publishable evaluation artifacts is `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/evaluations/final-v1/`.

## Evaluation Products

The primary leaderboard uses the exact 20-task PR #20 protocol and the natural forward/reverse-complement aggregation.
Also evaluate the full maize allele-frequency task and the frozen whole-window and variant-token probes.
Test task-independent bidirectional inference methods on held-out genomic base reconstruction before presenting them on task-labelled data.
Include two-sided product-of-experts scoring, directional feature concatenation, and context-length sensitivity; treat cyclic or sandwich prompts as an ablation that must pass the intrinsic reconstruction check.

Compare against previously published MarinDNA 0.22T, 0.39T, and 0.56T results, PlantCAD2 Small and Large, PlantCAD2.5 where available, and Evo where available.
Present results in the style of PlantCAD2 PR #20: one focused figure and a compact table per result, short interpretation, and committed machine-readable JSON or TSV.
Include a model-family overview, per-task deltas, allele-frequency and context plots, and probe and inference-method plots.

## Completion

A pilot completes only when W&B reaches `run_progress = 1`, its manifest and every output chunk are directly readable from CoreWeave S3, and local reduction reproduces the expected input coverage without duplicates.
A publishable evaluation completes only after all three evaluated checkpoints have complete manifests, all reductions and plots are reproducible from committed code and machine-readable artifacts, and an independent review finds no unresolved correctness issue.
