# Exp586: initial one-node data-parallel feasibility

The subsequent [64-H100 comparison against the live EP runtime](dp_production_results.md) is the relevant throughput report for production decisions.
This document preserves the initial one-node feasibility evidence.

Full-replica data parallelism works on H100s without EP or FSDP.
In the paired eight-GPU comparison it improved compute throughput by **56.3%** over the round-2 EP implementation.
The actual 64-example global batch also fits on one eight-H100 node and reached **956,892 tokens/s**.
The live 1T run and its monitoring agent were not modified or interrupted.

## Implementation

The model, FP32 master weights, complete optimizer state, and router state are replicated on every GPU.
Persistent state is **15.568 GiB per GPU**: 2.965 GiB BF16 parameters, 5.930 GiB FP32 masters, and 6.673 GiB optimizer state, plus small counters/router arrays.
Assertions verify that every state array is fully replicated before and after training.

The initial mesh uses `replica_dcn=8`, with `data=context=expert=model=1`.
Examples are partitioned across replicas; the existing JAX gradient of the global mean loss supplies gradient averaging.
The expert computation uses Levanter's existing local `scatter` backend: sort assignments, grouped matrix multiplication, then weighted scatter-add.
Every GPU has all 384 experts; tokens never travel to another GPU to reach an expert.
The local backend has no capacity truncation and processes every valid assignment.
Padding remains excluded.

The existing Muon optimizer initially partitioned temporary Newton–Schulz work over the replica axis even though its persistent state was replicated.
The strict candidate removes that fallback and performs the same five-step orthogonalization locally, retaining the existing QuACK symmetric GEMMs for expert matrices.
A fully replicated local computation scope prevents size-one logical axes from leaking into contraction annotations.
Compiled HLO checks reject synchronous and asynchronous all-to-all, all-gather, and reduce-scatter operations.
Gradient and global router-statistics reductions remain.

The global histogram router retains its 10,000 bins and global quantile calculation.
It does not become a separate router per replica.
The CPU oracle gives identical histogram thresholds for the same margins partitioned across one or eight replicas.

The full 128,256-token vocabulary, existing character tokenization, output softmax, architecture, attention, 8K context, augmentation, learning rates, schedule, optimizer equations, and data order are retained.
Round 2's input lookup optimization remains enabled; expert-buffer trimming is unnecessary for the local backend.
On one node, batch 64 means eight separate 8K examples per GPU, with **524,288 input tokens per update** and no example packing.
The production-size comparison instead uses one such example per GPU on 64 H100s.

The adapter and replication checks are in [data_parallel.py](src/exp586_moe/data_parallel.py).
The optimizer changes are in [grugmuon_hero.py](src/experiments/grug/moe_hero_ep/grugmuon_hero.py).
No production launcher imports the new adapter.

## Throughput

All initial jobs in this section used one eight-H100 node in East at batch priority with the required Iris account.
Timings exclude compilation, input loading, metric materialization, checkpointing, and Python LR injection.
Inputs were preloaded.
Means include stalls and cover updates 10–100, giving 91 timed updates.

| Implementation | GPUs | Global batch | Mean step | Median step | Input tokens/s from mean step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Round-2 EP reference | 8 | 8 | 314.687 ms | 302.346 ms | 208,258 |
| Fully local DP | 8 | 8 | 201.327 ms | 193.059 ms | 325,521 |
| Fully local DP, actual production batch | 8 | 64 | 547.907 ms | 527.508 ms | 956,892 |

The batch-eight result is a paired comparison on the same allocation, with two unchanged EP controls and one DP trajectory resident.
The candidate and timed reference alternate execution order.
The speedup is `314.686883 / 201.326516 - 1 = 56.31%`, or 36.02% less step time.
That benchmark's total iteration time includes correctness work and all three trajectories.

Batch-eight tests retain the earlier profiling protocol: take the first eight examples from each production batch, with a 64-example reference cursor and LR-clock stride.
They are comparisons on an eight-example proxy, not production continuations.
The batch-64 test uses the actual production examples, update size, data cursor, and token/LR clocks.
Its 100 updates process 52,428,800 input tokens.

[Paired DP/EP benchmark](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-dp-scatter-localopt-a2-verify).
[Production-batch DP validation](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-dp-scatter-globalbatch-a1-b64).

## Scientific equivalence

The useful contract is identical model equations, global objective, optimizer equations, input stream, clocks, and state values at restoration, while allowing different floating-point accumulation and different sparse kernels.
Bitwise update equality is not required.
The EP comparison is conditional on zero reference drops; DP always processes all valid assignments, including any that EP would have dropped under overflow.

Every comparison restores the same immutable DNA checkpoint, including all FP32 masters, optimizer moments, router state, and counters:

`s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/checkpoints/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1/2026.10.20/checkpoints/step-693582`

Its metadata checksum is `ad4827f1160660c264712a99cfea4e6a9df9f699e3bc4d405a745bf86d2ddfe5`.
The source scientific binding is `6a5273b56f94f5da995a882b44208678dbbb8baf8871a4c3ccc37c286f222899` and tokenizer binding is `d1cf7a63f3e0e0778a7ef137020e6f39d04be1e6da14fe22d6283eb36cddae5e`.
All initial state-array differences against the independently restored EP reference were exactly zero.
Only the temporary profiling namespace receives new checkpoints; their metadata explicitly marks them as experimental and unsuitable for production continuation.

The paired 100-update comparison produced:

| Difference from unchanged control A | DP candidate | Unchanged control B |
| --- | ---: | ---: |
| Loss RMSE | 0.000536730 | 0.000534564 |
| Loss mean absolute difference | 0.000356421 | 0.000355735 |
| Loss signed mean difference | −0.000068839 | −0.000118896 |
| FP32 master relative L2 at update 100 | 0.00665053 | 0.00626700 |
| Router-state relative L2 at update 100 | 0.219571 | 0.210518 |

The first DP forward loss differed by −0.000055671 from the EP reference.
Both unchanged EP controls had identical first forward loss.
All trajectories had zero drops.
The subsequent numerical differences are comparable to the unchanged controls; this is not proof of identical long-term learning or downstream quality.

For the real global batch, the original EP run already logged the next 100 production updates after the same source checkpoint.
Comparison against those recorded losses for updates 693,583–693,682 gives **RMSE 0.000124344**, MAE 0.000090911, and signed mean difference +0.000029058.
The first loss differs by +0.000052929 and the last by −0.000383377.
This is a historical comparison across different allocations, not another paired control experiment.
No baseline training was rerun.

## Restoration and tests

Both successful validation jobs wrote and restored a temporary checkpoint with **zero difference in every state array** and exact token/data/optimizer clocks.
The full-batch test also executed the same following update directly and after restoration.
The forward-loss difference was **0.000019312**; FP32-master relative L2 after that update was 0.000025522.
Optimizer counts remained exact and all state remained replicated.
These checkpoints preserve complete state rather than reconstructing a fresh optimizer.

The owning project's default CPU suite passed **101 tests** with 16 device-dependent skips.
Eight DP-specific tests passed separately on eight CPU devices, covering local-vs-EP outputs and gradients under balanced/crowded/all-padding assignments, full model/optimizer replication, unsupported-backend rejection, histogram partition invariance, and local optimizer math/collectives.
The optimizer oracle uses tight equality for well-conditioned matrices and direction/norm tolerances for random BF16 Newton–Schulz iterations; actual H100 trajectory differences are calibrated against unchanged controls above.
Pre-commit and SQLite integrity checks passed.

## Operational record and limits

The initial phase had four bounded dispatches: an initial successful ten-update smoke with temporary optimizer work partitioning, one diagnosed startup failure, the successful 100-update strict-DP comparison, and the successful 100-update batch-64 validation.
The initial smoke's 133 ms final step is not used as a strict-DP benchmark: its compiled optimizer contained six asynchronous all-gathers.
The failed strict attempt exposed an explicit-sharding contraction error, fixed before the successful attempt.
This initial phase used no larger GPU gang, alternative optimizer, altered attention, precision reduction, vocabulary pruning, or scaling test.

All four initial roots and both later production-size comparison roots are terminal; the final ledger has zero active dispatches and zero unreconciled intents.
No profiling allocation remains active.
No GitHub update or push was made for this isolated experiment.

The one-node results alone do not establish 64-GPU performance.
Replicating the experts changes the gradient-communication volume and topology across nodes; the subsequent production-size comparison measures that directly.
The full-batch result establishes feasibility and matching token accounting on one node.
The tests do not establish long-horizon quality equivalence.

Raw summaries and complete bounded histories are in `scratch/exp586_dp/wandb-results.json` at the checkout root.
The historical reference and comparison are `production-reference-losses.json` and `production-loss-comparison.json` in the same directory.
The SQLite ledger and immutable verified backups are described in [dp_operations.md](dp_operations.md).
Runtime source fingerprints, full configurations, and compressed compiled HLO are recorded in W&B and temporary S3 artifacts.
The final source adds a type annotation to the local optimizer helper after GPU validation; its computation is unchanged.
