# exp517 Strict phyloP Enhancer Order-Control Operations

> This document governs the one-trial enhancer order-exposure control managed with the `run-training-sweep-trc` skill.
> Read it in full at the start of every heartbeat before inspecting code, SQLite, W&B, or Iris.

## Invariants

The taxonomic sampling unit is the experimental variable relative to the completed strict-phyloP Arm A enhancer run.
The selector, uniform Arm A anchors, center-1 projection, model, tokenizer, optimizer, seed, global batch, schedule, and checkpoint cadence remain fixed.
Human is the sole Primates source, and every other represented NCBI order contributes one non-human source species.
The run has one active regional execution and one checkpoint writer at a time.
No held-out VEP data may be read or registered.

## Sweep Definition

The one-trial entry point and immutable input are defined in [`phylop_enhancer_order_experiment.py`](src/exp517_functional_specialists/phylop_enhancer_order_experiment.py).
The public training input is Hugging Face revision `6a592fffcdd155d19e6c8e0986eab606aab19606` of `marin-dna/phylop-uniform-v1-enhancer-arm-a-vertebrate-order`.
The token cache and checkpoint resolve below the region-local `MARIN_PREFIX`.
A same-region replacement resumes from the same regional checkpoint root.
No cross-region relocation is approved.

## Operator Choices

The absolute sweep deadline is 2026-09-10 17:00 UTC.
The deadline stops the whole one-trial sweep; recovery timing is controlled separately by `reslice_after`, `restart_after`, and `relocate_after`.
Use one regional replica.
The approved scope is preemptible TPU capacity in `us-east5`, limited to `v5p-8` and `v6e-4`, with a ceiling of eight submitted TPU chips.
Cross-family reslicing is allowed between these two explicitly approved TPU variants, and the actual family is reproducibility lineage.
Use the existing `interactive` priority band and Iris user `gonzalo`.
Do not change priority, region, TPU families, replication, or the chip limit without operator approval.

The durable sweep-state owner is `gs://marin-us-east5/MarinDNA/exp517_phylop_enhancer_order/sweep_state/`.
Upload each consistent SQLite backup to a new immutable timestamped key with `gcloud storage cp` and verify its byte count and SHA-256 against the helper output.
To recover, list immutable backups, download the newest candidate, verify its recorded size and SHA-256, run the helper integrity check, and fall back newest-to-oldest until one validates.

## Operating Policy

Use `heartbeat_every=30m`, `reslice_after=1h`, `restart_after=3h`, and `relocate_after=3d`.
Use `pending_target_limit=1`.
The launch requests ordered alternatives `v5p-8,v6e-4`; Iris chooses the actual eligible slice.
Protect progress within `reslice_after`.
Iris handles ordinary preemption; do not replace a progressing or merely preempted job.
Completion requires `run_progress >= 1` and a reachable terminal step-4,999 checkpoint.
If a late checkpoint write alone exhausts 56 GiB of coordinator RAM, preserve the same regional checkpoint root, W&B identity, and scientific configuration and retry with 96 GiB of coordinator RAM.

| Region | Bucket | Slice | Chips | State | Reason |
| --- | --- | --- | ---: | --- | --- |
| `us-east5` | `marin-us-east5` | `v5p-8` | 8 | `eligible` | — |
| `us-east5` | `marin-us-east5` | `v6e-4` | 4 | `eligible` | — |

## Change Record

None.
