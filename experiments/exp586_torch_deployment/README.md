# exp586 Torch deployment experiments

This independent project continues the deployment experiment line, separate from PR #590's upper-bound benchmark frozen at `8b0b7997c4fd4c3c5a63377c047b4e52c103d8ff`.
The original project and benchmark entry points are unchanged.
The chronological working log and numerical results remain ignored under `../exp586_plantcad2_moe/scratch/deployment-experiments/`.

The inference core depends on Torch, NumPy, Safetensors, Tokenizers and Hugging Face Hub; it imports no Marin, JAX, W&B or vLLM packages.
Optional providers load only when explicitly selected in `RuntimeConfig(attention_provider=..., flash_attention_version=..., moe_backend=...)`.
Provider import failures are reported without substituting a different kernel.
Standalone FA2 and FA3 are separate optional builds; vLLM attention and fused experts are separate optional utility choices, with no serving engine.

The model uses BF16 checkpoint weights, ordinary Torch top-k, direct Torch short convolution, separate shared experts, native GQA and ordinary rotary caches.
It keeps five ACGTN embedding rows and four A/C/G/T output rows.
Native experts use the documented `torch.nn.functional.grouped_mm` API.
It does not import our bespoke router or convolution kernels.
Graph freezing, combined shared projections, quantization, manual kernel tiles and serving hooks are excluded.
Call `prepare_rope(length, device)` outside compilation when changing the request length.

The benchmark uses unique complete 8,192-base windows and returns dense four-base logits at every position.
It measures only warmed CUDA-event forwards, with loading, compilation, tokenization and transfers outside timing and no prefetch worker.
All variants must pass the unchanged fixed 32-window native-JAX probability/NLL/accuracy gate before timing.
The six initial variants compare standalone/vLLM FA2/FA3 with native Torch experts, then vLLM FA2/FA3 with BF16 fused experts.

For CPU validation:

```bash
uv sync --locked --extra cpu --group dev
uv run --locked --extra cpu pytest
```

For CUDA, select `--extra cuda` and install the explicitly selected optional provider using its pinned build recipe.
The benchmark receipt records provider source revisions, dependency versions, build options, resolved functions, GPU identity and all event samples.
