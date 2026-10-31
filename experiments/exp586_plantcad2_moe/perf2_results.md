# Exp586 throughput experiments, round 2

Two useful changes gave about **25% higher compute throughput** in a paired eight-H100 benchmark: an equivalent input-embedding lookup and guarded removal of zero expert rows from GEMMs. The current 1T training run has not been modified. All profiling jobs have ended, within the six-hour limit.

## What changes

- **Input lookup:** select the same existing lowercase character embedding rows using a one-hot matrix product. Its standard autodiff avoids expensive repeated-index gradient atomics. The full 128,256-token vocabulary, output softmax, parameter arrays, and optimizer updates remain intact. Other token IDs fall back to the original lookup.
- **Expert padding:** use a 2,048-row expert GEMM only when every received expert row fits, then restore the original output shape with zeros. Larger loads use the original computation. EP8, three waves, receiver capacity 32, sender capacity 8, dispatch, routing, and drop accounting are retained.

No LR, schedule, training precision, data, augmentation, context, or optimizer changes are proposed. The faster paths currently exist only in the isolated profiling checkout.

## Initial screening

All runs used one eight-H100 node in `cw-us-east-02a`, batch priority, and the required Iris account. Each restored the same frozen DNA checkpoint at update 693,582, including FP32 master weights, optimizer state, and router state. The node processes eight separate 8K examples per update: node zero's slice of the original 64-example stream. This proxy omits communication across eight nodes.

Timings are updates 30–100, with one-batch prefetch and a five-step trace collected earlier in the run. All times below are milliseconds. Means retain stalls and fallbacks.

| Setting | Mean step | Median step | Decision |
| --- | ---: | ---: | --- |
| Existing Combined runtime | 390.2 | 390.3 | Reference |
| Save-MoE rematerialization | 396.0 | 394.1 | No benefit; backend lacks relevant save tags |
| No outer rematerialization | 399.7 | 399.9 | No benefit; peak allocation more than doubled |
| One transport wave | 403.1 | 403.4 | No benefit |
| Layer-loop unroll 2 | 395.0 | 395.6 | No benefit; did not test unroll 4 |
| Compact input gather | 408.6 | 409.5 | Atomic contention remained |
| Batched host metric transfers | 391.0 | 391.5 | Exact metrics, marginal benefit |
| One-hot input lookup | 348.0 | 347.6 | Useful |
| Guarded expert padding | 350.8 | 344.1 | Useful |
| Both useful changes | 315.6 | 304.5 | Selected for validation |
| One host process for eight GPUs | 1035.3 | 408.3 | No advantage; large stalls |
| Extra expert row tiers | 616.3 | 346.4 | Not selected; slower and large stalls |

The single-process and extra-tier results include substantial late stalls; not all of those stalls have been attributed to their configurations. Neither showed an advantage sufficient to pursue.

## Correctness evidence

The selected combination completed two 200-update repeats alongside two unchanged controls on the same node, using identical examples, token/LR clocks, starting weights, and optimizer state. Each repeat uses the same 200 batches; pooling gives 400 paired comparisons, not a single 400-update trajectory.

| Loss difference from control A | Optimized candidate | Unchanged control B |
| --- | ---: | ---: |
| First repeat RMSE | 0.00068485 | 0.00074005 |
| Second repeat RMSE | 0.00075776 | 0.00065520 |
| Pooled RMSE | 0.00072223 | 0.00069891 |
| Pooled mean absolute difference | 0.00050787 | 0.00049081 |
| Pooled signed mean difference | +0.00006267 | +0.00007350 |

Both conditions had zero token drops throughout both comparisons. Initial state values matched exactly and first-update forward loss matched exactly. Shapes, dtypes, and partitioning were also checked. At update 200, FP32 master relative L2 differences were 0.01224 versus 0.01261 in the first repeat and 0.01299 versus 0.01233 in the second. Optimizer relative L2 differences were 7.43e-7 versus 7.14e-7 and 7.67e-7 versus 7.75e-7. Active input-embedding rows and their Adam moments were checked separately. Numerical differences are comparable to ordinary GPU nondeterminism, rather than bitwise identity.

The full CPU suite passed 100 tests; nine device-dependent tests were skipped there and exercised separately on eight CPU devices. The device oracles cover lookup forward/gradient equivalence, arbitrary-token fallback, balanced/crowded expert loads, forced overflow, all-padding inputs, row-tier fallbacks, and existing context/sharding restoration behavior. A source AST comparison confirmed that the copied backend's dispatch and wave logic differs only at the expert-compute hook.

[200-update validation in W&B](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-perf2-onehot-trim-a1-verify).

## Repeat benchmarks and limits

Repeat benchmarks preloaded the same examples and disabled tracing to isolate compute from input and profiler stalls. A standalone baseline still showed runtime jitter: 490.4 ms mean and 414.2 ms median. That separate allocation is not used to claim a speedup.

The final paired benchmark used the same allocation and preloaded batches for the reference and candidate. After control A established data readiness, control B and the candidate alternated execution order. Three independent training states remained resident. Timings cover updates 30–200, or 171 paired observations.

| Paired mean step time | Seconds |
| --- | ---: |
| Unchanged reference B | 0.380540 |
| One-hot lookup + guarded padding | 0.303414 |

This is **25.4% higher compute throughput**, or **20.3% less time per step** in this paired setup. The earlier standalone profile with the normal prefetch path measured a consistent 24% step-throughput gain. The paired job's total iteration time includes three training trajectories and correctness work, so it is not a standalone training rate.

[Final paired benchmark and second validation in W&B](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-perf2-onehot-trim-a1-verify-bench).

Measured gains apply to this one-node setup. Communication and the slowest of eight training nodes can change the gain, especially when a node takes the full-size fallback. Validation on 64 H100s and checkpoint replay remain necessary before deployment. The original job, run identity, checkpoints, and monitoring session remain untouched.

## Reproduction

Code is in `experiments/exp586_plantcad2_moe/src/exp586_moe/perf2.py`, `perf2_pooled.py`, `perf2_control.py`, `perf2_reconcile.py`, and `perf2_report.py`, plus optional hooks in the isolated `model.py`. `perf2_operations.md` defines the operating constraints and six-hour deadline (2026-10-31 18:07:14 UTC). All profiling identities contain `-perf2-` and belong to W&B group `exp586-plantcad2-moe-perf2`.

`scratch/exp586_perf2/wandb-results.json` at the repository root contains downloaded summaries and complete histories. The SQLite ledger and verified immutable CW S3 backups record dispatches and completion. Every successful profile has a verified temporary `result.json`; no profile wrote a training checkpoint. One early compact-gather attempt failed at startup on explicit sharding, was corrected with an eight-device regression oracle, and was replaced by a fresh attempt identity. Pre-commit and ledger integrity checks passed. The original checkout remains at its starting commit with only its pre-existing dependency-file changes.
