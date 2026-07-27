# MarinDNA sequence explorer

This directory contains the molab-hosted marimo application for
[Open-Athena/marin-dna issue #387](https://github.com/Open-Athena/marin-dna/issues/387).
It runs the pinned public model
[`bolinas-dna/marin-dna-exp135-m5.1`](https://huggingface.co/bolinas-dna/marin-dna-exp135-m5.1)
at revision `c0676b2012b8b9c526deb26ff517f6b92b6d375d`.
The hosted notebook reports and installs the tested application implementation
at Git commit `04f44dbd9b5a5bc4cc172f4caf925d548d4bf911`.

Public app:
https://molab.marimo.io/notebooks/nb_MrPpr5xYcN3HGt5tLY86bk/app

The marimo notebook is intentionally thin. Sequence validation, the
forward/reverse-complement probability logo, the categorical-Jacobian
dependency map, built-in examples, Plotly figures, linked-span conversion, and
in-memory downloads are imported from the installable `marin_dna` package.
The notebook declares its minimal immutable runtime dependencies with PEP 723
metadata. Because molab's public app sandbox does not currently install those
dependencies before evaluating cells, the first cell defensively installs the
same pinned `jaxtyping==0.3.9` dependency before installing the commit-pinned
`marin_dna` source with `uv pip install --no-deps`. The full research
environment includes unrelated genomics and pipeline dependencies that are
unnecessary for this app and may require build tools unavailable in molab.
`requirements.txt` contains only the extra dependencies used for local checks.
A GitHub-synced molab notebook therefore does not depend on sibling files or
install unrelated pipeline dependencies.

## Deployment on molab

molab is currently a public preview and provides its GPU for free subject to
reasonable-use limits. This application must not enable a paid resource or
billable fallback.

1. Push a fully tested application commit and replace `SOURCE_REVISION` in
   `app.py` with that immutable commit.
2. In [molab](https://molab.marimo.io/), create a synced notebook from the
   commit-pinned GitHub URL for `apps/sequence_explorer/app.py`. GitHub remains
   the source of truth.
3. Open the notebook specs menu and attach the free NVIDIA RTX Pro 6000
   Blackwell GPU. Confirm that the application runtime-status callout names the
   GPU and reports CUDA as available. The application will not fall back to CPU
   and reports the failing stage and exception if analysis cannot proceed.
4. Run the notebook, then choose **Run as app** from molab's share menu. Record
   the public app URL in issue #387.
5. Keep `PROGRESSIVE_MIN_LENGTH` unset until the benchmark below establishes a
   threshold. If a sequence-length range has at least 3 seconds of dependency
   work after the logo, set it to the shortest measured length in that range.

A molab session can run for at most 12 hours and is stopped after 90 minutes of
inactivity. A new session reloads the environment and pinned model. If molab's
free GPU becomes unavailable, stop and revisit the platform decision instead
of incurring cost.

## Benchmark gate

Run the built-in 74, 126, and 255 bp examples on the deployed free molab GPU
and record:

- environment/model cold-start time;
- warm total runtime;
- time to logo;
- additional time from logo to dependency map;
- peak VRAM and detected GPU.

The app reports the last four values for each submission. Record complete cold
start externally from opening the ephemeral server through the first result.
Tune `NUCLEOTIDE_DEPENDENCY_BATCH_SIZE` if the 255 bp warm run exceeds 120
seconds. Do not enable CPU fallback or change the dependency method.

## Privacy

molab's terms prohibit sensitive data. Submitted sequences and matrices remain
in session memory; the application does not add persistent caching, analytics,
or sequence logging.

## Local checks

The notebook imports without loading the model because model loading is cached
inside a marimo cell and begins only after a valid submission. Validate the
notebook and focused app tests with:

```bash
uv run --with-requirements apps/sequence_explorer/requirements.txt \
  marimo check apps/sequence_explorer/app.py
uv run --with-requirements apps/sequence_explorer/requirements.txt \
  pytest apps/sequence_explorer/tests
```

The core computation remains covered by:

```bash
uv run pytest tests/model/test_sequence_interpretation.py
```

A complete inference smoke test requires a CUDA GPU and downloads the 4.5 GB
pinned model. Do not run it on CPU.
