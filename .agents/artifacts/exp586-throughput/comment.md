🤖 Throughput optimizations for the 1T d768 pretrained run.

| Round | Optimization | Throughput increase¹ | Correctness observations |
| --- | --- | ---: | --- |
| 1: GPU state and input overlap | Keep FP32 master weights and optimizer state on GPUs | +16.0% alone | Same precision and optimizer equations |
| 1 | Prefetch one batch during GPU computation | +25.5% alone | Identical examples, augmentation, and batch order |
| 1 | Both changes together | **+75.0%** | 100 updates on 64 H100s: loss RMSE 0.000105; unchanged-control RMSE 0.000130 |
| 2: Embedding and expert kernels | Use a small one-hot matrix product for input lookup | +12.1% alone | Same embedding vectors; gradient summation order changes |
| 2 | Skip unused expert-buffer rows, with a full-size fallback | +11.2% alone | Forward, gradient, overflow, and fallback tests pass |
| 2 | Both changes together | **+25.4% over round 1** | Two 200-update checks on 8 H100s: loss RMSE 0.000722; unchanged-control RMSE 0.000699; zero drops |
| 1 + 2 | Combined projection | **2.20× baseline throughput** | Product of separately measured gains; not a measured cumulative training result |

¹ Gains use ratios of mean step times from eight-H100 benchmarks: round 1 versus CPU offload with synchronous input; round 2 versus round 1 together. Individual gains do not add. This post will be updated for subsequent rounds.

<details>
<summary>Round 1: keep master weights and optimizer state on the GPUs</summary>

The previous training step transferred FP32 master weights and optimizer state from pinned CPU memory to the GPU, applied the update, then transferred them back. `MasterParamMode.FP32_DEVICE` keeps both resident on the GPUs and retains a separate BF16 compute copy. This removes transfers while preserving FP32 updates, all optimizer moments, and checkpoint contents. It uses more GPU memory.

- Before: [master-weight and optimizer transfers](https://github.com/Open-Athena/marin-dna/blob/1d35d888db4ff4542a164284409b1fa55fc7b9fb/experiments/exp586_plantcad2_moe/src/experiments/grug/moe_hero_ep/train.py#L855-L867).
- After: [device-resident FP32 update path](https://github.com/Open-Athena/marin-dna/blob/1bcfc4f309a642694e421c9e7636845d23438dff/experiments/exp586_plantcad2_moe/src/experiments/grug/moe_hero_ep/train.py#L951-L981) and [runtime selection](https://github.com/Open-Athena/marin-dna/blob/1bcfc4f309a642694e421c9e7636845d23438dff/experiments/exp586_plantcad2_moe/src/exp586_moe/train.py#L950-L965).
- Evidence: [isolated GPU-state benchmark](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-profile-b8-fp32device).

</details>

<details>
<summary>Round 1: prepare the next batch while the GPUs train</summary>

Previously, each iteration waited for `distributed_batch` before starting GPU computation. A single-worker `ThreadPoolExecutor` now prepares the next batch concurrently, including loading, tokenization, augmentation, and device transfer. Prefetch depth is one; example indices still advance by exactly one global batch. Tokenization and augmentation are unchanged.

- Before: [synchronous batch construction](https://github.com/Open-Athena/marin-dna/blob/1d35d888db4ff4542a164284409b1fa55fc7b9fb/experiments/exp586_plantcad2_moe/src/exp586_moe/train.py#L812-L827).
- After: [one-batch prefetch iterator](https://github.com/Open-Athena/marin-dna/blob/1bcfc4f309a642694e421c9e7636845d23438dff/experiments/exp586_plantcad2_moe/src/exp586_moe/train.py#L994-L1038).
- Evidence: [prefetch alone](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-profile-b8-prefetch), [both round-1 changes](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-profile-b8-combined-r2), and [100-update full-batch correctness and checkpoint validation](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-equivalence-combined-v1).

</details>

<details>
<summary>Round 2: select the same input embeddings with a small matrix product</summary>

Repeated DNA characters cause the ordinary embedding lookup's backward pass to accumulate many gradients into the same rows. The faster path first selects the 26 existing lowercase-character rows from the full embedding table, maps token IDs to those rows, then computes `one_hot(ids, 26) @ table`. Standard JAX autodiff aggregates gradients through a matrix product and reduces the small table before scattering back into the full gradient.

The full 128,256-token input table, optimizer state, and output softmax remain intact. Inputs containing other token IDs fall back to the ordinary lookup. No vocabulary or tokenizer change is involved.

- Before: [ordinary indexed lookup](https://github.com/Open-Athena/marin-dna/blob/f96971ac50fe71a61ae02a077dd5b833a1285aec/experiments/exp586_plantcad2_moe/src/experiments/grug/moe_hero_ep/model.py#L150-L164).
- After: [small one-hot lookup and arbitrary-token fallback](https://github.com/Open-Athena/marin-dna/blob/603ff2238ed164ddd34b5fe444230a8e75ea2acb/experiments/exp586_plantcad2_moe/src/experiments/grug/moe_hero_ep/model.py#L155-L193) and [model call site](https://github.com/Open-Athena/marin-dna/blob/603ff2238ed164ddd34b5fe444230a8e75ea2acb/experiments/exp586_plantcad2_moe/src/experiments/grug/moe_hero_ep/model.py#L1307-L1315).
- Evidence: [lookup benchmark](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-perf2-onehot-lookup-a1) and [forward/gradient/fallback tests](https://github.com/Open-Athena/marin-dna/blob/603ff2238ed164ddd34b5fe444230a8e75ea2acb/experiments/exp586_plantcad2_moe/tests/test_perf2_lookup.py).

</details>

<details>
<summary>Round 2: avoid expert matrix multiplications on unused padding</summary>

The pooled EP backend applies each expert's feed-forward network to its full padded receiver buffer. The faster path checks the highest row occupied by any kept assignment. If all occupied rows fit within the first 2,048 rows per expert, it calls the same `_compute_pooled` function on that prefix and pads the output back to the original shape. Larger loads use the full original computation.

Only unused trailing rows are skipped. EP8, three transport waves, sender/receiver capacity factors 8/32, routing, dispatch, combine, and drop accounting remain unchanged. The 2,048-row threshold refers to expert buffers; sequence length remains 8,192.

- Before: [full-buffer expert matrix products in the pinned Levanter dependency](https://github.com/marin-community/marin/blob/55e96dc7cbcb00f11c7d4d265acbf1ce0ce1d4bd/lib/levanter/src/levanter/grug/_moe/ep_fixed_pooled_wave_all_to_all.py#L449-L470).
- After: [guarded prefix computation, padding, and full-size fallback](https://github.com/Open-Athena/marin-dna/blob/603ff2238ed164ddd34b5fe444230a8e75ea2acb/experiments/exp586_plantcad2_moe/src/exp586_moe/perf2_pooled.py#L39-L97) and [model call site](https://github.com/Open-Athena/marin-dna/blob/603ff2238ed164ddd34b5fe444230a8e75ea2acb/experiments/exp586_plantcad2_moe/src/experiments/grug/moe_hero_ep/model.py#L1119-L1125).
- Evidence: [expert-padding benchmark](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-perf2-trim-expert-rows-a1) and [forward/gradient/overflow/fallback tests](https://github.com/Open-Athena/marin-dna/blob/603ff2238ed164ddd34b5fe444230a8e75ea2acb/experiments/exp586_plantcad2_moe/tests/test_perf2_pooled.py).

</details>

<details>
<summary>Measurement and correctness details</summary>

Round-1 screening averaged updates 30–80: 650.0 ms [baseline](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-profile-b8-prodstate-a2), 560.3 ms with GPU state, 517.8 ms with prefetch, and 371.4 ms with both. Its separate 64-H100 validation restored identical weights, optimizer state, router state, and clocks; compared two unchanged controls with the candidate for 100 updates; verified identical batches; and checked exact checkpoint serialization round trips.

Round-2 screening averaged updates 30–100: 390.2 ms baseline, 348.0 ms lookup, and 350.8 ms expert padding. The final paired benchmark averaged 171 observations on the same allocation with preloaded inputs and alternating reference/candidate order: 380.540 → 303.414 ms, giving 25.4% higher compute throughput. Three training states were resident; this is not an end-to-end production throughput measurement.

Two 200-update round-2 comparisons used identical source checkpoints, examples, LR clocks, and optimizer state. Pooled loss MAE was 0.000508 versus 0.000491 between unchanged controls; signed mean difference was +0.0000627 versus +0.0000735. Initial state and first forward loss matched exactly; subsequent differences reflect floating-point accumulation order and were comparable to control nondeterminism. Both repeats had zero drops. These are repeated checks on the same 200 batches, not a single 400-update trajectory or proof of identical long-term learning.

The round-2 suite passed 100 CPU tests, with device-dependent cases exercised separately on eight CPU devices. The one-node round-2 benchmarks omit production communication across nodes, so the cumulative projection needs a direct production measurement.

- [First round-2 correctness check](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-perf2-onehot-trim-a1-verify).
- [Paired benchmark and second correctness check](https://forge.coreweave.com/wandb/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-perf2-onehot-trim-a1-verify-bench).
- [Round-2 report](https://github.com/Open-Athena/marin-dna/blob/603ff2238ed164ddd34b5fe444230a8e75ea2acb/experiments/exp586_plantcad2_moe/perf2_results.md).

</details>
