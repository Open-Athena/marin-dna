# exp586 Torch deployment comparison

`throughput.svg` and `throughput.png` compare the five configurations at the same batch size of 64 on one H100 80GB GPU.
Each value is the median of three fresh-process medians; error bars show their minimum-to-maximum range, not a confidence interval.
All samples measure warmed BF16 device forwards over complete 8,192-base windows, with tokenization, transfers, loading, compilation and warmup excluded.
Every plotted case passed the fixed native-JAX correctness gate.

`comparison.json` names the exact 15 plotted trials, configurations, source revisions, and summary statistics.
`measurements.jsonl` preserves all 173 trials from this round: 166 passed and seven expected batch-512 capacity OOMs.
The `stage` field identifies preflight, compiler, batch search, refinement, confirmation, robustness, or uniform-fixture verification.
`final-summary.json` records compiler ablations, corrected batch comparisons, and numerical acceptance results.
The separate historical benchmark remains frozen at commit `8b0b7997c4fd4c3c5a63377c047b4e52c103d8ff`; its timing included tokenization and host output.

The model and API were measured at `81821bd77dfee5e22904ca1cded3717a4ad1e3fd`.
Benchmark-only correction `1fcee555cc4606c547c6210d8d31dd303929fa08` makes non-power-of-two batch comparisons weight every fixture window equally.
The batch-64 figure already used equal weighting: eight pre-encoded batches and 32 timed forwards per process.
Short exploratory screens remain in the raw ledger; the final corrected batch-12/128/240 comparisons are under stage `uniform`.
The best confirmed Fast result is 1,588,435 tokens/s at batch 128; the matched batch-64 figure uses 1,584,105 tokens/s.

To regenerate the figure in an environment with Matplotlib and NumPy:

```bash
python .agents/artifacts/exp586-deployment/plot_throughput.py
```

The preserved named canonical usage is `experiments/exp586_torch_deployment/examples/canonical_usage.py` (`pt-fa2`).
The companion examples spell out `pt-fa3`, `vl-fa2`, `vl-fa3` and `fast-fa3` directly.
The first four use compiler defaults plus the required `fullgraph=True` graph-break assertion.
Fast explicitly enables freezing before importing Torch, then selects static shapes and reduce-overhead mode.
All five require Triton for compiled CUDA execution; only Fast additionally imports this project's custom Triton router source.

Complete native provenance and provider-binary archives are linked from `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/inference-deployment/2026-11-07-r2` and its DEPLOY-002 references.
These files are supporting code and data for a locally drafted PR comment; the comment requires user approval before publication.
