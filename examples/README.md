# MarinDNA examples

## Model inference and zero-shot VEP notebook

[`model_inference_and_vep.py`](model_inference_and_vep.py) is a linear,
code-visible Marimo tutorial for loading the pinned public MarinDNA 1B model,
running a real 255 bp GRCh38 sequence, and scoring the pinned BRCA1
saturation-genome-editing dataset through the existing VEP runner.

The notebook declares its complete environment with PEP 723 metadata. It
requires a CUDA GPU with BF16 support and intentionally does not fall back to
CPU. Launch it from the repository root:

```bash
uv run marimo edit examples/model_inference_and_vep.py
```

GitHub remains the source of truth. A public code-visible notebook link will be
added here when the committed revision has been synced and GPU-verified on
Molab.

## Sequence explorer

The [MarinDNA Sequence Explorer](https://molab.marimo.io/notebooks/nb_MrPpr5xYcN3HGt5tLY86bk/app)
is a task-focused interactive application for submitting a sequence and viewing
polished nucleotide-logo and dependency-map visualizations. Its permanent
branch-only source is pinned at
[`f837d86`](https://github.com/Open-Athena/marin-dna/tree/f837d8600223208427d524ed5efa3bd375ab4afd/apps/sequence_explorer).

The explorer and tutorial are complementary: use the explorer for a focused
sequence-in/figures-out workflow, and use the notebook to inspect, execute,
modify, and learn the complete inference and VEP code path.
