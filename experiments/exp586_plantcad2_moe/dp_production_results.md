# Exp586: data parallelism versus the live EP runtime

On 64 H100s with the real global batch, the DP candidate improved measured training-loop throughput by **22–73%** over the currently deployed EP configuration in two bounded comparisons.
The longer confirmation measured **2.059M versus 1.187M input tokens/s**, a 73.46% throughput increase and 42.35% reduction in time for the measured updates.
The spread between trials and execution orders matters; these are short benchmark measurements, not an observed whole-run speedup or a promise of 73% sustained improvement.
The live run, its checkpoint owner, and its monitoring session were not changed.

## Reference and candidate

The reference reproduces the live `combined-v1-20261031` EP runtime: 64 H100s, EP8 inside each node, FP32 masters and optimizer state on GPU, prefetch depth one, and global batch 64 × 8,192 = 524,288 input tokens per update.
W&B snapshots before and after the experiments verified the same live runtime ID and code fingerprint `9b6a26dae3c8e5dc497fefe1475d00f482fedefab5d35f7ae0380b4b691a68f3`.
The live run advanced from update 807,738 to 811,143 during the read-only observations.

The DP candidate has the full model, masters, optimizer, and router state on every GPU, with `replica_dcn=64` and `data=context=expert=model=1`.
Its local `scatter` experts do not drop assignments.
It retains the previously validated character-input lookup optimization; the reference uses the live runtime's original lookup and pooled expert kernels.
This measures the complete DP candidate against deployed EP, rather than isolating the effect of removing EP from every other optimization.
It does not establish an advantage over an EP runtime that has not yet been deployed.

Both backends used the same eight physical H100 nodes in Reno within each comparison, identical restored arrays, the same 1T-horizon optimizer and token LR clocks, and identical examples, augmentation, masks, and loss normalization.
The immutable source checkpoint and scientific/tokenizer bindings are documented in [the initial feasibility report](dp_results.md#scientific-equivalence).
All remote jobs used batch priority and the required account.

## Training-loop measurements

| Measurement | Updates executed per backend | Updates timed per backend | EP tokens/s | DP tokens/s | Throughput increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial production-size pass | 60 | 55 | 1,206,295 | 1,468,792 | 21.76% |
| Longer confirmation, both orders | 300 | 290 | 1,186,757 | 2,058,599 | 73.46% |

Each run first restored two EP controls and the DP candidate and executed 100 matched updates for numerical comparison.
The subsequent training-loop measurements use normal data loading and one-batch prefetch, LR injection, synchronous training, router metric materialization, and W&B logging.
The confirmation additionally includes the production loop's per-step finite-metric checks, phase summary update, and checkpoint-decision host synchronization, which the initial pass omitted.
Compilation and the first five updates of each timed window are excluded.
Throughput uses total input tokens divided by the complete measured loop wall time, including stalls and metric/logging work.
Scheduled evaluation and checkpoint writes are timed separately below.
The relevant W&B summary fields are `benchmark/dp/wall_tokens_per_second` and `benchmark/ep/wall_tokens_per_second`; the ordinary `throughput/*` history belongs to the preloaded correctness benchmark.

The confirmation runs EP then DP for 150 updates each, advances both states along the same stream, then runs DP followed by EP for another 150 updates each.
The first order measured 2.310M versus 1.140M tokens/s; the reversed order measured 1.856M versus 1.237M.
Aggregation adds equal token counts and wall times rather than averaging throughput values.
These differences are consistent with input/cache and host variability; loading time was not separately instrumented, so that attribution is an inference.
Both backends share the process-local input reader during a comparison, making cache/order effects especially relevant.

The confirmation's EP mean training-step time was 0.429s, close to the live run's usual approximately 0.414s.
A separate 2,200-update live W&B window measured 999,529 input tokens/s over 1,153.45 wall seconds, including real pauses.
That window is context, not a denominator for claiming the DP benchmark's measured speedup.
The live run's full wall time and the bounded training-loop measurements contain different overheads.

The preloaded compute-only confirmation measured 0.2269s for DP and 0.4242s for EP.
That 87.0% compute-only increase is not the headline training-loop result.

## Evaluation, checkpointing, and correctness

| Confirmation measurement | EP | DP |
| --- | ---: | ---: |
| Warm validation, 512 fixed examples | 0.821s | 0.304s |
| Held-out loss | 0.99629948 | 0.99615495 |
| Native checkpoint write | 70.99s | 75.64s |

The initial pass's checkpoint writes were 75.69s for EP and 75.92s for DP.
These are native checkpoint writes into the isolated temporary namespace; production pointer publication and pruning are not included.
Production normally evaluates every 2,000 updates and saves recovery checkpoints every 900 seconds, with additional permanent milestones.
The test measures those operations separately rather than observing a complete 15-minute save cycle for each backend.

Every restored state array initially matched exactly, including FP32 masters, optimizer moments and counts, pending router state, and token/data clocks.
The confirmation's 100-update DP loss RMSE versus control A was **0.000152716**, versus **0.000115021** between unchanged EP controls A and B.
Mean absolute differences were 0.000110895 and 0.000082597, respectively; signed mean differences were +0.000047062 and +0.000002303.
The candidate difference is small but exceeds the control's in this repetition; do not describe it as bitwise equivalence or proof of identical long-horizon quality.
The initial pass's corresponding RMSE values were 0.000120725 and 0.000117639.
All measured training and validation paths had zero drops.

Both passes saved and restored the complete DP state with zero array differences and exact clocks.
The confirmation's following-update replay loss difference was **0.000008404**; optimizer counts remained exact.
FP32 master relative L2 after that update was 0.000027031.
Direct object-store reads verified both backend checkpoint metadata at updates 693,742 and 693,982 for the initial and confirmation passes, respectively.
Compiled DP executables contained no all-to-all, all-gather, or reduce-scatter operations, and all model/optimizer arrays remained fully replicated.

The owning project's final suite passed **107 tests**, with 16 device-dependent skips.
Pre-commit checks passed.
Both new Iris roots are terminal and their W&B runs finished with complete expected histories; no isolated GPU allocation remains active.
No GitHub changes or pushes were made.

## Reproduction and artifacts

The first pass used local code commit `02d371e5`; the longer confirmation used `0e5ab647`.
The project retains its locked Marin 0.2.201.dev39243773500 and JAX 0.11.1 dependency set.
The entry point, model adapter, launch/persistence boundary, and policy are [dp_profile.py](src/exp586_moe/dp_profile.py), [data_parallel.py](src/exp586_moe/data_parallel.py), [dp_control.py](src/exp586_moe/dp_control.py), and [dp_operations.md](dp_operations.md).

- [Initial 64-H100 comparison](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-dp-scatter-prod64-a1-b64-verify-n8-live)
- [Longer counterbalanced confirmation](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-dp-scatter-prod64-balanced-a2-b64-verify-n8-live)
- [Unmodified live training run](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1)

Raw histories, summaries, checkpoint metadata, and baseline snapshots are in `scratch/exp586_dp/production-comparison-a1.json`, `production-comparison-balanced-a2.json`, `live-production-window-64gpu.json`, and `live-baseline-{before,after}-64gpu.json` at the checkout root.
Result artifacts, compressed HLO, temporary checkpoints, and immutable SQLite backups are under `s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp586_plantcad2_moe/dp-20261031/`.
The final immutable ledger backup is `operations/20261031T163803Z-eef3c94340274f00ab7594d2bc1f97c6.sqlite`, SHA-256 `14c8240e98f4a4707ec38b2e6eff3946bb5e7f9f7c8a85706aa256aba2e02eb6`.
