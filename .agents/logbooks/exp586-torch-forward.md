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


## 2026-11-06 23:43 UTC — Final confirmation and handoff

Source before this result entry is local commit `689179c2`; the benchmark implementation overlay remains `0719a077`.
The exact final trial list is `plans/batch-confirmation.json` in the native experiment S3 prefix.
It runs three repetitions each at batch 96 and 256 in randomized order, eight warmups and 32 timed forwards, using the same device and baseline flags as the ablations.
The batch-96 median is 736.010 ms / 1,068,507 DNA tokens/second, and batch 256 is 1979.807 ms / 1,059,271 DNA tokens/second, a 0.87% gain.
The fine sweep shows a broad plateau; do not interpret 96 as an exact optimum across inputs or hardware.
There are 49 successful timed cases, all passing the existing native-JAX gate, plus the two recorded capacity OOM cases.
The final aggregation verifies a single model revision, checkpoint identity, weight-index/tokenizer/reference/fixture hashes and GPU UUID across the replicated comparisons, and confirms exactly one configuration flag changes per ablation.

`examples/restricted_torch_logits.py` was smoke-tested with `/app/inference-venv/bin/python /app/examples/restricted_torch_logits.py --model-dir /app/forward-artifacts/hf-bf16`.
It produced `A/C/G/T logits: (1, 8192, 4)`.
The final process observation found no active benchmark workers, and all eight H100s reported zero device memory and zero GPU utilization.
Iris reports the exact retained root running, zero failures and one preemption; its preemption diagnostic is stale relative to productive execution and the final successful task exec.
The sequential successor uses a Linux `pidfd_open(2)` exit event after this Python build lacked the `os.pidfd_open` wrapper; no GPU trial was retried for that helper correction.
Correction to the CPU reservation record above: the launch requested one CPU, while the Iris task-description snapshot reports `cpu_millicores=4000`; the cgroup CPU limit remains unlimited (`max 100000`).

Final environment: Python 3.12.13, Torch 2.14.1+cu132, CUDA 13.2, vLLM 0.0.0.dev20261013+marin.ac3b5ea5e851.cu132, Triton 3.8.0, NVIDIA driver 595.71.05.
CPU verification is 129 passed, 3 skipped; scoped pre-commit checks passed after formatting the new constructor condition.
Final result tables are in `experiments/exp586_plantcad2_moe/forward_benchmark.md`.
Machine-readable summary, all raw per-process results/logs, explicit plans, input/provenance hashes, final source archive and a verified checksum manifest remain owned by `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/inference-optimization/forward-only/2026-11-06-r1`.
Experimental work stops for user review, with the original batch-priority/eczech allocation kept up and no new jobs or GitHub updates.
