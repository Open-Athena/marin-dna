# exp586 second-round throughput profiling

This isolated sweep uses the repository `run-training-sweep-cw` skill.
Read this document before each decision pass.

## Invariants

The live 1T training job, its monitoring session, checkpoint namespace, and source checkout must not be changed by this sweep.
Use the separate `codex/exp586-perf2` checkout.
Use one whole eight-H100 node at a time, batch priority, and Iris user `eczech` for every submission.
Do not publish GitHub updates.
All profiles use separate W&B identities and seven-day temporary artifacts.
Keep the full vocabulary, input tokens, model weights, optimizer math, and routing capacities unchanged.
Any candidate affecting numerical behavior requires comparison to ordinary control nondeterminism.

## Definition

The experiment catalog, pinned source checkpoint, bounded updates, and deadline guards are defined in `src/exp586_moe/perf2.py`.
The baseline is the validated Combined runtime on a single-node proxy, with eight examples and EP8.
It omits production data-parallel communication and does not establish the eventual 64-H100 gain.
Profile completion requires the expected update count, a verified temporary `result.json`, and a terminal Iris root; these bounded profiles deliberately produce no training checkpoint.

## Operator choices

The hard deadline is 2026-10-31T18:07:14Z, six hours after work began.
Stop profiling at the deadline; no further runs are authorized after it.
Only H100s in `cw-rno2a` and `cw-us-east-02a` are eligible.
Keep exactly one active dispatch and one pending target at most.
Remote compute was explicitly authorized by the user.
The local SQLite ledger is `scratch/exp586_perf2/perf2.sqlite`.
Immutable ledger backups belong under `s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp586_plantcad2_moe/perf2-20261031/operations/` using `<UTC>-<uuid>.sqlite` keys.
Use the skill's backup helper, upload through authenticated CoreWeave S3, verify byte size and SHA-256, and restore the newest valid snapshot if the local database is missing.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| cw-us-east-02a | H100 | 1 | 8 | eligible | Prior first-round profiling proved the identical GPU shape and inputs |
| cw-rno2a | H100 | 1 | 8 | eligible | Prior exp586 validation proved the identical GPU shape and shared inputs |

## Operation

Inspect W&B first; use Iris only for submission, exact-job liveness, and concrete fast-failure diagnosis.
Refresh the skill's v3 utilization tool before every submission.
Each job is limited to 45 minutes and the global deadline, whichever is earlier.
Check short profiles as needed for startup and completion, without periodic production-monitoring work.
Do not blindly retry failures; diagnose first.
Do not deploy candidates: review results with the user first.

## Change record

2026-10-31: `combined`, `save-moe`, `no-remat`, `waves1`, and `unroll2` ran to their bounded endpoints. The three rematerialization/transport alternatives and loop unrolling did not beat the baseline, so they are not deployment candidates. `no-remat` also substantially increased peak allocation.

The full-vocabulary input-gradient scatter and reduction are promising trace findings. `compact-lookup` retains all model/optimizer arrays and the full output softmax; it gathers a small input table with ordinary JAX autodiff and falls back to the existing lookup for any other token. Its first attempt failed before updates because an intermediate gather needed explicit output sharding. The corrected implementation has an eight-CPU-device forward/gradient oracle with explicitly sharded input IDs.

The optional `--verification` mode retains two unchanged controls for an entire trajectory. It compares loss means, absolute differences, RMSE, grouped optimizer/model/router state, and dropped assignments. Its timings include competing trajectories and must not be presented as standalone throughput measurements.

The plain compact gather did not improve performance. The trace showed that repeated-character gradient atomics remained costly. The one-hot lookup uses standard matrix multiplication for the same input rows and avoids those atomics; the full softmax, vocabulary, weights, and optimizer stay intact. Other IDs use the original lookup.

Guarded expert-row trimming leaves dispatch and all capacity limits intact. It uses smaller expert GEMMs only when every received row fits, then pads the unused output back to the original shape. Larger loads use the original computation. CPU EP oracles compare the public backend's outputs, gradients, and drop counts under balanced loads, crowded experts, forced overflow, and all-padding inputs. Tiered row sizes are a performance candidate for reducing occasional full-size fallbacks.

Use mean steady-state step and iteration times for final throughput claims. Guarded fallbacks make the medians more optimistic than the wall-clock average. These are one-node, eight-example proxies; no result establishes production data-parallel throughput or replaces a later production-topology equivalence check.

Later trials showed substantial runtime stalls, including a repeat baseline with no active tracing and data already preloaded. The final paired comparison therefore measures control B and the candidate on the same allocation, with identical preloaded batches and alternating execution order after control A. Report its relative compute timings as a paired experiment with three resident states, not as a standalone end-to-end training rate. Benchmark mode is confined to this profiling harness and never changes the current 1T job's input pipeline.

The completed experiments support the one-hot input lookup plus the single guarded expert-row size as the preferred candidate. Detailed measurements and limits are in [perf2_results.md](perf2_results.md). Keep the existing process count, rematerialization, transport waves, precision, and capacity settings. Further validation on 64 H100s or deployment requires the user's review; this sweep's approved compute scope remains one eight-H100 node. No further profiling dispatch is planned for this round.
