---
topic: exp586-torch-forward
issue: https://github.com/Open-Athena/marin-dna/pull/590
description: Restricted Torch forward-only benchmark and ablations
author: eric-czech
---

# exp586 Torch forward benchmark logbook

## 2026-11-06 — Initial protocol

The user authorized a single-H100 forward benchmark on a whole eight-H100 CoreWeave node at batch priority with user eczech.
Keep the allocation running for later iteration and stop experimental work for review after reporting results in chat.
Do not update PR #590 or other GitHub artifacts.
The implementation is isolated on `codex/exp586-torch-forward`, based on PR head `8b0b7997c4fd4c3c5a63377c047b4e52c103d8ff`.
The exact initial configuration, protocol, ablations and durable owner are in `experiments/exp586_plantcad2_moe/forward_benchmark.md`.
The raw fixed-fixture inputs were recovered from the original content-addressed PlantCAD source and the preserved native-JAX reference, then saved under the new durable experiment prefix.
Reno's initial availability was stale and east had no complete free node; Reno subsequently refreshed and showed 34 free H100s.
