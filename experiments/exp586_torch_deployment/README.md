# exp586 Torch deployment experiments

This independent project continues the deployment experiment line, separate from PR #590's upper-bound benchmark frozen at `8b0b7997c4fd4c3c5a63377c047b4e52c103d8ff`.
The original project and benchmark entry points are unchanged.
The chronological working log and numerical results remain ignored under `../exp586_plantcad2_moe/scratch/deployment-experiments/`.

The inference core depends on Torch, NumPy, Safetensors, Tokenizers and Hugging Face Hub; it imports no Marin, JAX or W&B packages.
The actual model class is `MarinDNAMoE`, with two inference backend options: `use_vllm=False|True` and `flash_attention_version=2|3`.
`use_vllm=True` selects vLLM attention and built-in BF16 fused experts together, without a serving engine or model-side kernel tuning.
`use_vllm=False` selects standalone attention and native Torch grouped experts, and can run in an environment with vLLM entirely absent.
Provider import failures are reported without substituting a different kernel.
Standalone FA2 and FA3 are separate optional builds.
The benchmark's vLLM dependency is the pinned Marin build `0.0.0.dev20261013+marin.ac3b5ea5e851.cu132`, rather than an unmodified PyPI wheel.

The model uses BF16 checkpoint weights, ordinary Torch top-k, direct Torch short convolution, separate shared experts, native GQA and ordinary rotary caches.
It keeps five ACGTN embedding rows and four A/C/G/T output rows.
Native experts use the documented `torch.nn.functional.grouped_mm` API.
It does not import our bespoke router or convolution kernels.
Graph freezing, combined shared projections, quantization, manual kernel tiles and serving hooks are excluded from the ordinary model's defaults.
Call `prepare_rope(length, device)` outside compilation when changing the request length.

The preserved [Canonical usage](examples/canonical_usage.py) example loads the model directly and calls `torch.compile(model, fullgraph=True)`.
This remains the named canonical usage for `pt-fa2`.
The companion examples deliberately spell out each configuration without a CLI or wrapper:

| Configuration | Canonical usage |
| --- | --- |
| `pt-fa2` | [Standalone FA2 and native Torch experts](examples/canonical_usage.py) |
| `pt-fa3` | [Standalone FA3 and native Torch experts](examples/canonical_usage_fa3.py) |
| `vl-fa2` | [vLLM FA2 and built-in fused experts](examples/canonical_usage_vllm_fa2.py) |
| `vl-fa3` | [vLLM FA3 and built-in fused experts](examples/canonical_usage_vllm_fa3.py) |
| `fast-fa3` | [FA3, vLLM fused experts, project Triton router, and compiler tuning](examples/canonical_usage_fast.py) |

These examples run one window for readability; benchmark throughput uses the batch size stated with each result.
Encoding and compilation stay explicit outside the model.
`fullgraph=True` is our required assertion against graph breaks; it is not Torch's default.
Compiler mode and dynamic-shape policy are otherwise left at their defaults.
The benchmark separately records `mode="reduce-overhead"`, `dynamic=False`, and `TORCHINDUCTOR_FREEZING=1` whenever selected.
A compiler failure stops the campaign for review; it never disables the assertion or falls back to eager execution.
Standard external FA/vLLM operators can remain opaque graph nodes.

`FastMarinDNAMoE` is a separate H100 ceiling experiment with our specialized Triton top-k router and combined shared-expert projections, adapted from the frozen historical source.
It defaults to vLLM fused experts and FA3, and accepts the same two backend options for controlled comparisons.
It does not alter process-global compiler settings during construction.
Freezing, compiler modes and batch tuning are explicit benchmark choices outside the class.
Fast must pass exactly the same numerical gate as the ordinary model; a failed approximate result cannot count toward the accepted ceiling.

The benchmark uses unique complete 8,192-base windows and returns dense four-base logits at every position.
It measures only warmed CUDA-event forwards, with loading, compilation, tokenization and transfers outside timing and no prefetch worker.
Final batch comparisons time whole cyclic fixture periods, weighting every real window equally even for non-power-of-two batches.
The requested iteration count is a minimum and rounds up to a whole period; receipts record the actual event count.
All variants must pass the unchanged fixed 32-window native-JAX probability/NLL/accuracy gate before timing.
The current four ordinary variants compare standalone FA2/FA3 with native Torch experts and vLLM FA2/FA3 with built-in fused experts.
Each trial runs in a fresh process, with isolated compiler caches shared only by matching source and configuration, and records every CUDA-event sample.
The serial campaign tool uses a separate tools environment with Boto3 for artifact publication; cloud clients are not inference-core dependencies.

For CPU validation:

```bash
uv sync --locked --extra cpu --group dev
uv run --locked --extra cpu pytest
```

For CUDA, select `--extra cuda` and install the explicitly selected optional provider using its pinned build recipe.
The benchmark receipt records provider source revisions, dependency versions, build options, resolved functions, GPU identity and all event samples.
