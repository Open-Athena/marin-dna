# MarinDNA examples

## Model inference and zero-shot VEP notebook

[`model_inference_and_vep.py`](model_inference_and_vep.py) is a linear,
code-visible Marimo tutorial for loading the pinned public MarinDNA 1B model,
running the exact 186 bp negative-strand TH interval shown in the GPN-Star
reference panel, and scoring the pinned BRCA1 saturation-genome-editing dataset
with 255 bp contexts through the existing VEP runner.

The notebook declares its complete environment with PEP 723 metadata. It
requires a CUDA GPU with BF16 support and intentionally does not fall back to
CPU. Launch it from the repository root:

```bash
uv run marimo edit examples/model_inference_and_vep.py
```

GitHub remains the source of truth. Open the
[public code-visible Molab notebook](https://molab.marimo.io/github/Open-Athena/marin-dna/blob/7224b7e60349ac8746a4df537185130d2c9e6fd9/examples/model_inference_and_vep.py)
to inspect or execute the GPU-verified committed revision. Choose **Server**,
then **Configure compute → GPU**, before running cells. The GitHub-backed source
preview does not store executed outputs, and WebAssembly cannot provide the
CUDA/native PyTorch runtime this notebook requires.

## Sequence explorer

The [MarinDNA Sequence Explorer](https://molab.marimo.io/notebooks/nb_MrPpr5xYcN3HGt5tLY86bk/app)
is a task-focused interactive application for submitting a sequence and viewing
polished nucleotide-logo and dependency-map visualizations. Its permanent
branch-only source is pinned at
[`f837d86`](https://github.com/Open-Athena/marin-dna/tree/f837d8600223208427d524ed5efa3bd375ab4afd/apps/sequence_explorer).

The explorer and tutorial are complementary: use the explorer for a focused
sequence-in/figures-out workflow, and use the notebook to inspect, execute,
modify, and learn the complete inference and VEP code path.
