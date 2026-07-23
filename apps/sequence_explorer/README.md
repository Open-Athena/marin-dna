---
title: MarinDNA Sequence Explorer
emoji: 🧬
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.49.1
python_version: 3.12
app_file: app.py
fullWidth: true
startup_duration_timeout: 30m
license: apache-2.0
models:
  - bolinas-dna/marin-dna-exp135-m5.1
tags:
  - biology
  - genomics
  - dna
preload_from_hub:
  - bolinas-dna/marin-dna-exp135-m5.1 config.json,model.safetensors,special_tokens_map.json,tokenizer.json,tokenizer_config.json c0676b2012b8b9c526deb26ff517f6b92b6d375d
---

# MarinDNA sequence explorer

This directory is the self-contained Hugging Face Space application for
[Open-Athena/marin-dna issue #387](https://github.com/Open-Athena/marin-dna/issues/387).
It runs the pinned public model
[`bolinas-dna/marin-dna-exp135-m5.1`](https://huggingface.co/bolinas-dna/marin-dna-exp135-m5.1)
at revision `c0676b2012b8b9c526deb26ff517f6b92b6d375d`.

The Gradio layer is intentionally thin. Sequence validation, the
forward/reverse-complement probability logo, and the categorical-Jacobian
dependency map live in `src/marin_dna/` and are installed from a pinned GitHub
revision for deployment.

## Deployment

1. Before mirroring this directory to the Space, replace the branch revision on
   the `marin-dna` line in `requirements.txt` with the immutable Git commit that
   contains the computation code.
2. Mirror this directory to a Gradio Space under `gonzalobenegas`.
3. Select ZeroGPU hardware in the Space settings. The README metadata does not
   assign hardware automatically.
4. Set `SOURCE_REVISION` to the immutable GitHub application commit shown by the
   mirror job.
5. Keep `PROGRESSIVE_MIN_LENGTH` unset until the benchmark below determines a
   threshold. If a sequence-length range has at least 3 seconds of
   dependency-map work after the logo, set it to the shortest measured length
   in that range.

Current Hugging Face documentation says personal accounts need a PRO
subscription to host ZeroGPU Spaces. Confirm the target account's eligibility
before deployment; creating or upgrading a Space is an external account action
and is not done by this repository.

## Benchmark gate

Run 74, 126, and 255 bp inputs on the deployed ZeroGPU Space and record:

- cold-start time;
- warm total runtime;
- time to logo;
- additional time from logo to dependency map;
- peak VRAM.

The UI reports the last four values for each submission. Record cold start from
the Space runtime logs. Tune `NUCLEOTIDE_DEPENDENCY_BATCH_SIZE` if the 255 bp
warm run exceeds 120 seconds. Do not enable CPU fallback or change the
dependency method.

## Local checks

The app loads the 4.5 GB model at module import and expects ZeroGPU's CUDA
emulation, so normal local tests should import `examples.py` and `ui.py`, not
`app.py`. The core computation is covered by:

```bash
uv run pytest tests/model/test_sequence_interpretation.py
```

Install this directory's requirements in an isolated environment to smoke-test
the Gradio layout on a CUDA-capable machine.
