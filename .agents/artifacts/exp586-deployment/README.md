# exp586 Torch deployment comparison

`throughput.svg` and `throughput.png` compare five configurations at batch 64 on one H100 80GB GPU.
Each value is the median of three fresh-process median throughputs; error bars show the minimum-to-maximum range of those three medians.
All samples measure warmed BF16 device forwards over complete 8,192-base windows, returning four raw A/C/G/T logits at every position.
Tokenization, transfers, model loading, compilation and eight warmup passes are outside the timed interval.

`comparison.json` names the 15 plotted trials, exact model/source identities, configurations and aggregation.
`random-input-comparison.json` compares the real fixture against seed-0 uniformly random A/C/G/T inputs using three fresh processes per configuration and input distribution.
`measurements.jsonl` contains those 30 timed trials plus five eager/compiled preflights.
Every reported process passed the fixed 32-window native-JAX gate and the fullgraph compilation assertion.
`usage-smoke.json` records execution of all five literal canonical examples, including finite BF16 outputs of shape `[64, 8192, 4]` and their single-forward timings.
`final-summary.json` records validation counts, numerical-error maxima and random-input sensitivity.

The canonical usage files under `experiments/exp586_torch_deployment/examples/` show explicit precision, vocabulary and implementation choices.
Their preamble is identical across all five files; freezing is selected by the per-example launch command before importing Torch.
Ordinary configurations leave compiler mode and shape policy at defaults while asserting `fullgraph=True`.
Fast additionally enables freezing, static shapes and reduce-overhead mode; its router kernel, combined shared projections, convolution and rotary-cache choices are fixed inside the distinct class.

The source revisions are recorded in `comparison.json`.
`source-equivalence.json` verifies that the sole model-code difference between the measurement and literal-example snapshots is an early rejection of unsupported FP16/native-Torch execution in the pinned CUDA build.
All other model AST nodes are identical, including every BF16 forward implementation.

See `experiments/exp586_torch_deployment/SETUP.md` for environment and full reproduction instructions.
The provider build recipes and provenance in this directory identify the exact upstream FlashAttention source, restricted FA3 build and Marin vLLM distribution.
The standalone environment has no vLLM installed.
The native owner for full receipts and manifests is `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/inference-deployment/2026-11-07-r3`.

To regenerate the compact figure with Matplotlib and NumPy:

```bash
python .agents/artifacts/exp586-deployment/plot_throughput.py
```

These artifacts support https://github.com/Open-Athena/marin-dna/pull/590#issuecomment-6710081437.
