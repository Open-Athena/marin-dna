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

## 2026-11-06 — Placement and replicated ablations

The retained allocation is `/eczech/exp586-torch-forward-20261106-a4` in `cw-rno2a`, submitted with `--priority batch --user eczech`, one whole eight-H100 node, and only GPU 0 exposed to measurement processes.
Earlier roots a1, a2 and a3 were cancelled before benchmark execution because CPU or host-memory reservations prevented admission despite free GPUs.
The retained job reserves one CPU and 32 GiB host RAM; `cpu.max` is `max 100000`, so the reservation is not a one-core CPU limit.
The main task sleeps indefinitely to retain the allocation after the benchmark worker finishes.
Preemptions were recovered within the existing root, restoring completed S3 results and running only unfinished configurations.
Each case is a separate process, and every completed result and log is immediately uploaded to the experiment's native S3 prefix.
The final ablation block used the same GPU UUID `GPU-b5e2efea-3217-4717-94ab-4a1fe6a348e9` throughout all 15 cases.

The downloaded HF model is `plantcad/marin-dna-exp586-1.5B-60M-216B` at revision `4130f71edcd117a7375af1ac78723d9a9173533f`, corresponding to native checkpoint step 412290 and 216,158,699,520 input tokens.
Its LFS SHA-256 values were verified directly after download.
The 32-window native-JAX reference is unchanged, and every successfully timed case passes the existing probability/NLL/accuracy gate.
This is tolerance-based parity, not bitwise equality.
Timing rotates device-resident inputs across the 512-window fixture with no tokenizer worker or prefetch running concurrently.

The runtime source starts from commit `ccd6f6ab2894598112c5423b78016d02d816b8e2` with `forward_torch.py` and `forward_sweep.py` overlaid from commit `0719a077`.
The final overlay adds memory telemetry and prevents resumed capacity screens from running above a previously recorded OOM boundary.
The original source archives, overlays, explicit JSON plans, input hashes, package versions, GPU identity, CUDA-event samples and all failed cases are retained under the durable experiment prefix.
CPU verification completed with 129 tests passed and 3 skipped.

At batch 64, three fresh-process repetitions per configuration give these median forward throughputs:

| Configuration | DNA tokens/second | Change from control |
| --- | ---: | ---: |
| Restricted control | 1,066,139 | — |
| `TORCHINDUCTOR_FREEZING=0` | 1,057,933 | -0.77% |
| Separate shared-expert projections | 1,054,806 | -1.06% |
| `sconv_backend=conv1d` | 852,224 | -20.06% |
| Custom `specialized_topk=True` | 1,237,890 | +16.11% |

These are replicated single-H100 forward-only results for complete 8,192-base windows, excluding compilation, warmup, tokenization and input/output transfers.
The custom top-k kernel remains excluded from the restricted baseline.
The maximum fitting power of two is batch 256; batch 304 also fits, while 320 and 512 OOM during the complete compile/validate/warmup protocol.
Fine batch tuning and same-GPU confirmation are recorded in the next entry.
No GitHub artifacts were updated.
