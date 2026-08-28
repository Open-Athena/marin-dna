# exp517 Strict phyloP Sweep Operations

> This document governs the strict phyloP-control training sweep managed with the `run-training-sweep-trc` skill.
> Read it in full at the start of every heartbeat before inspecting code, SQLite, W&B, or Iris.

## Invariants

The conservation selector is the only experimental variable relative to the six GPN-Star-P uniform-grid specialists.
Training semantics and immutable inputs are owned by [`phylop_uniform_experiment.py`](src/exp517_functional_specialists/phylop_uniform_experiment.py).
Each logical arm has one active regional run and one checkpoint writer at a time.
No held-out VEP data may be read or registered.

## Sweep Definition

The training entry point and six-arm catalog are defined in [`phylop_uniform_experiment.py`](src/exp517_functional_specialists/phylop_uniform_experiment.py).
Select exactly one trial with `EXP517_PHYLOP_ARM`.
All data inputs are public immutable Hugging Face revisions, and all token caches and checkpoints resolve below the region-local `MARIN_PREFIX`.
Same-region replacement resumes from the same regional checkpoint root.
No cross-region relocation is approved for this sweep.

## Operator Choices

The absolute sweep deadline is 2026-09-10 17:00 UTC.
The deadline stops the entire sweep; recovery timing is controlled separately by `reslice_after`, `restart_after`, and `relocate_after`.
Use one regional replica per trial and launch all six trials without a canary.
The approved scope is preemptible TPU capacity in `us-east5`, limited to `v5p-8` and `v6e-4`, with an overall ceiling of 48 submitted TPU chips.
The operator explicitly allows either TPU family; record the actual family used by every run as reproducibility lineage.
Use the existing `interactive` priority band and Iris user `gonzalo`.
Do not change priority, region, TPU families, replication, or the chip limit without operator approval.

The durable sweep-state owner is `gs://marin-us-east5/MarinDNA/exp517_phylop_uniform_specialists/sweep_state/`.
Upload each consistent SQLite backup to a new immutable timestamped key with `gcloud storage cp` and verify its byte count and SHA-256 against the helper output.
To recover, list immutable backups, download the newest candidate, verify its recorded size and SHA-256, run the helper integrity check, and fall back newest-to-oldest until one validates.

## Operating Policy

Use `heartbeat_every=30m`, `reslice_after=1h`, `restart_after=3h`, and `relocate_after=3d`.
Use `pending_target_limit=6` for the single ordered-alternative request so the operator's six-simultaneous-launch decision is preserved.
The launch requests ordered alternatives `v5p-8,v6e-4`; Iris chooses the actual eligible slice, and the observed physical allocation must be recorded with the dispatch.
Protect progress within `reslice_after`.
Iris handles ordinary preemption; do not replace a progressing or merely preempted job.
Classify isolated versus systemic failures from the full W&B fleet before recovery.
Completion requires `run_progress >= 1` and a reachable terminal step-4,999 checkpoint.
If a late checkpoint write alone exhausts coordinator host memory, preserve the same regional checkpoint root, W&B identity, and scientific configuration and retry that logical run with 96 GiB coordinator RAM.
Treat this host-memory-only retry as execution recovery, not a new scientific trial.

| Region | Bucket | Slice | Chips | State | Reason |
| --- | --- | --- | ---: | --- | --- |
| `us-east5` | `marin-us-east5` | `v5p-8` | 8 | `eligible` | — |
| `us-east5` | `marin-us-east5` | `v6e-4` | 4 | `eligible` | — |

## Change Record

- 2026-08-28: ncRNA exon and background exhausted 56 GiB coordinator RAM while writing late checkpoints after W&B steps 4,984 and 4,998.
  Their same-identity recovery coordinators now request 96 GiB while retaining preemptible `v6e-4` children and the original regional checkpoint roots.
