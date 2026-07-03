---
name: marin-experiment
description: Set up and launch a marin-launched DNA training experiment on the shared iris cluster. Each experiment is a self-contained directory with its OWN pyproject (marin packages in BASE deps, local lib as a git dep) — the pattern that makes iris/zephyr workers resolve the runtime. Use when creating a new exp<N>, launching training on the marin/iris cluster, packaging an experiment for marin, or debugging launch failures like "No module named 'zephyr'", "marin-iris client is too old", or a TPU job that won't start. NOT for snakemake pipelines (each snakemake/<pipeline>/ has its own README) or one-off analysis scripts (those go in scripts/).
---

# marin-experiment

How to run a marin-launched DNA training/eval experiment on the shared **iris** cluster. Experiments do **not** live on `main` (see AGENTS.md → "What gets merged to `main`") — each is a self-contained directory on its own branch, cited from its tracking issue via commit-pinned permalinks.

## The pattern: one self-contained directory per experiment

Each experiment is its own dir — `experiments/exp<N>_<slug>/` — with its **own `pyproject.toml` + `uv.lock`** and a `launch.py`. The **non-negotiable** rule: **marin packages go in base `dependencies`, not an `extra`.** The iris/zephyr-dispatched worker syncs the shipped project with a plain `uv sync --all-packages --no-group dev` (no `--extra`); if marin is in an extra it isn't installed and the tokenize worker dies with `ModuleNotFoundError: No module named 'zephyr'`. In base deps, the worker gets it. (This mirrors the canonical consumers — see Sources.)

Template `pyproject.toml` (the pins are **interim** — read the callout after it):

```toml
[project]
name = "exp<N>-<slug>"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    # marin packages in BASE deps (NOT an extra) — this is what makes the
    # iris/zephyr worker's plain `uv sync` install them. In an *extra* the
    # tokenize worker fails: `ModuleNotFoundError: No module named 'zephyr'`.
    #
    # PIN these to the NEWEST build that (a) clears the controller freshness floor
    # AND (b) still has the marin.execution symbols your launch.py imports. marin
    # 0.2.32+ removed ExecutorStep / executor_main / this_output_path /
    # ensure_versioned → an old launch.py submits fine then dies on the coordinator
    # with `ImportError: cannot import name 'ExecutorStep'`. The floor advances
    # daily (deploy−14d) so this window shrinks — 0.2.31 below is a CURRENT example
    # (will rot); check #356 for today's build, or migrate launch.py to the new API.
    "marin-core==0.2.31.*",       # NB: PyPI name is marin-core, not `marin`
    "marin-levanter==0.2.31.*",
    "marin-iris==0.2.31.*",
    "marin-zephyr==0.2.31.*",
    "marin-rigging==0.2.31.*",
    "marin-dupekit",              # PyPI name is marin-dupekit, not `dupekit`
    # finelog 06-17: iris (<=0.2.31) imports finelog.client.proxy, which finelog
    # 0.2.12 (06-28+) dropped → "No module named 'finelog.client.proxy'".
    "marin-finelog==0.2.11.dev202606171051",
    "marin-finelog-server==0.2.11.dev202606171051",
    # marin-levanter[lm_eval]'s extra is a URL dep; uv requires URL deps top-level.
    "lm-eval @ git+https://github.com/stanford-crfm/lm-evaluation-harness.git@d5e3391f22cde186c827674d5c3ec7c5f4fe0cab",
    # YOUR library code as a GIT dep (NOT a path) so remote workers clone+install
    # it rather than relying on the unreliable workspace bundle. Pin a SHA for
    # reproducibility, or @main to track. Verify every marin_dna symbol your
    # launch imports exists on the pinned ref.
    "marin_dna @ git+https://github.com/Open-Athena/marin-dna.git@main",
    # httpx<1: with prerelease=allow httpx floats to 1.0.dev*, whose Client()
    # dropped the `timeout` kwarg marin-iris's GCP provider passes → iris crash.
    "httpx>=0.28.1,<1",
]

[project.optional-dependencies]
# List torch/torchvision here as DIRECT deps of the extra (not only in
# [tool.uv.sources]) — uv only applies an index source to a package that is a
# direct dep of the extra. marin-core[tpu] (>=0.2.20) pins torch/torchvision
# ==...+cpu wheels that live ONLY on PyTorch's CPU index; without this listing
# the whole tpu closure silently backtracks to a too-old marin
# (`no version of torchvision==0.26.0+cpu`). This was the load-bearing step.
tpu = [
    "marin-core[tpu]; sys_platform == 'linux'",
    "torch; sys_platform == 'linux'",
    "torchvision; sys_platform == 'linux'",
]

# Iris's remote worker runs `uv sync --all-packages --no-group dev`, which errors
# if the dev group is undefined — define an empty one.
[dependency-groups]
dev = []

[tool.uv]
package = false
fork-strategy = "fewest"
prerelease = "allow"
# Linux-only: workers are always linux; the darwin branch has no libtpu/+cpu
# wheels, so it makes the tpu extra unsatisfiable and derails the whole resolution.
environments = ["sys_platform == 'linux'"]
# Mirrored from marin's pyproject — omitting these makes uv resolution fail the
# same way it does for marin itself (+ pandas<3 to keep parquet dtypes stable).
# transformers<5: fresh marin (>=0.2.32) declares transformers-5, which breaks
# lm-eval@d5e3391 (its hf_vlms uses AutoModelForVision2Seq, renamed in tf5). The
# override keeps uv on the working tf4 line.
override-dependencies = [
    "omegaconf>=2.4.0.dev4", "antlr4-python3-runtime==4.11",
    "python-multipart>=0.0.22", "wheel>=0.46.2",
    "datasets>=3.1.0,<4.0.0", "equinox>=0.11.10", "pandas>=2.0,<3.0",
    "transformers>=4.57,<5",
]
# marin depends on resiliparse>=0.17.2, which only exists on this custom index.
[[tool.uv.index]]
name = "marin-resiliparse"
url = "https://marin-community.github.io/chatnoir-resiliparse/simple"
# marin-core[tpu]'s torch/torchvision ==...+cpu wheels live only here. explicit
# so it doesn't shadow PyPI globally (only the routed torch/torchvision use it).
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
[tool.uv.sources]
resiliparse = { index = "marin-resiliparse" }
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

> ⚠️ **The marin 0.2.x line churns fast, and the controller freshness floor advances DAILY (deploy − 14 days)** — a pin that launches today ages out in days (it moved 06-18 → 06-19 within one session). The template's *shape* is durable (marin in base deps; the tpu-extra torch/torchvision routing; linux-only; transformers<5); the exact versions are not — the `0.2.31` example **will rot**. **Before trusting the versions, get the current build from [#356](https://github.com/Open-Athena/marin-dna/issues/356)** (the live launch-blocker issue, carrying the full working recipe) or [#328](https://github.com/Open-Athena/marin-dna/issues/328). The *durable* fix, once the old-executor-API pin window closes (floor passes the last build with `ExecutorStep` et al.), is **migrating `launch.py` to marin's new `marin.execution` API** rather than chasing an ever-aging pin.

## Launch

Run from **inside the experiment dir**, with an iris tunnel available. **No `--extra marin`** — marin is in base deps, so the coordinator's plain `uv sync` already has it (and `--extra marin` would error).

```bash
cd experiments/exp<N>_<slug>
SWEEP_DATASETS=<dataset> uv run iris --cluster=marin job run \
    --no-wait --user <you> --job-name exp<N>-<arm> \
    --cpu 1 --memory 2g --region us-east5 \
    -e WANDB_API_KEY "$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')" \
    -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \
    -e SWEEP_DATASETS <dataset> \
    -- python launch.py
```

Two things the `launch.py` must declare on **every** `remote(...)` call (workers do **not** inherit the parent's `--extra` or `-e` flags):
- **`pip_dependency_groups=`** — `[]` for CPU/tokenize steps (base already has marin), `["tpu"]` for the TPU-train step (adds `libtpu` via the `tpu` extra). Never `["marin"]` (no such extra in this layout).
- **`env_vars=`** — re-declare `HF_HUB_DOWNLOAD_TIMEOUT`, `UV_LOCK_TIMEOUT`, `WANDB_API_KEY`, etc. on each step; the parent's `-e` flags reach only the coordinator.

**wandb naming:** run name carries `dna-exp<N>` and group `dna-exp<N>-v<ver>`, so runs filter by experiment.

`--cluster=marin` auto-establishes the SSH tunnel to the controller (IAP allows port 22 only — don't `gcloud port-forward`). `--region` is a hint (keep it matched to your data's GCS bucket or you re-tokenize into the new region); the coordinator only needs `--cpu 1 --memory 2g`.

## Babysit (first minutes + the whole run)

- **Watch the launch actively.** A successful *submit* is not a successful *run* — a job can submit fine then **fail on the coordinator with an `ImportError`** (marin API mismatch) *before* dispatching anything, so check `iris job summary` state, not just "did it submit". Confirm the chain: coordinator imports launch.py + builds the DAG (no `ExecutorStep` ImportError); **tokenize worker imports `zephyr` and runs `marin.processing.tokenize`** (the moment of truth for the base-deps setup); the TPU-train step logs `TPU detected` (not CPU fallback) + a wandb run URL + advancing `global_step`/`loss`.
- **Use a detached on-box poller, not a background sleep.** `run_in_background` Bash `sleep` loops get reaped when the session idles; a `setsid nohup bash poller.sh </dev/null &>log &` process reparents to init and survives. Poll `iris job summary` / `job logs` every ~10 min; flag `No accelerator found` / `RESOURCE_EXHAUSTED` / `JOB_STATE_FAILED`.
- **A "failed" job may be complete.** A JAX/TPU **teardown SIGSEGV (exit 139)** often fires *after* the final checkpoint + eval. Before EVER relaunching a `State: failed` job, check the logs for `Finished saving HF-compatible checkpoint to gs://…/hf/step-<final>` and `gsutil ls` that dir (expect `config.json` + `model.safetensors` + tokenizer). If present, it's done — relaunching wastes the whole run.

## Lessons (failure → fix)

Anchored to specific worker-log messages so they survive iris/marin churn.

| Symptom (worker logs) | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'zephyr'` in the tokenize worker's `cloudpickle.loads` | marin is in an **extra**, not base deps. Move `marin-*` into base `dependencies` (see the template). The worker's `--no-group dev` sync (no `--extra`) then installs it. |
| `marin-iris client is too old (build <date>; minimum <floor>)` at submit | The controller's freshness floor (deploy − 14 d) **advances daily**. Bump the marin pin to the newest build that BOTH still has your launch.py's `marin.execution` symbols (ExecutorStep row below) AND resolves the tpu extra (torchvision row below). Not a blind `uv lock --upgrade` — that jumps to a head whose executor API breaks your launch.py. Wheels are on **PyPI** (`marin-core`, not GitHub find-links). |
| `ImportError: cannot import name 'ExecutorStep' from 'marin.execution'` — the job **submits OK then fails on the coordinator** (before any dispatch; `iris job summary` shows the coordinator task errored) | marin **0.2.32+ refactored the executor API** (`ExecutorStep` / `executor_main` / `this_output_path` / `ensure_versioned` removed). Pin marin to the newest build that still has them (`0.2.31`, early July — the closing "old-API window"), or migrate launch.py to the new API. Verify with `uv run python -c "import launch"` before relaunching. |
| `no version of torchvision==0.26.0+cpu` / the tpu closure only resolves at a too-old marin (e.g. 0.2.19) | `marin-core[tpu]` (>=0.2.20) pins torch/torchvision `==…+cpu`, published only on **PyTorch's CPU index**. Add a `pytorch-cpu` **explicit** index AND list `torch`/`torchvision` as **direct deps of the `tpu` extra** (uv only routes an extra's *direct* deps — routing them only in `[tool.uv.sources]` isn't enough). Also set `environments = ["sys_platform == 'linux'"]` (the darwin branch has no such wheels and derails resolution). See template. |
| `error: Extra "cpu" is not defined…` on a `remote(...)` step | `pip_dependency_groups` names a non-existent extra. Use `[]` (marin is base) for CPU/tokenize, `["tpu"]` for TPU-train. |
| `Timeout (Ns) waiting for lock on /uv/cache/.../lm-eval...` | `-e UV_LOCK_TIMEOUT 7200` on `job run` **and** in the step's `env_vars=`. Many zephyr workers share a uv cache; the first build serializes and the 300 s default races. |
| `requests.exceptions.ReadTimeout: huggingface.co ... timeout=10` | `-e HF_HUB_DOWNLOAD_TIMEOUT 120` (also in the step's `env_vars=`). HF's 10 s default is too short for parquet-manifest fetches. |
| `No accelerator found` + repeated `iris: TPU bad-node signature detected` | Switch zone: `--region us-east5` ↔ `--region us-central1` (the two `v5p-preemptible` regions). iris keeps hitting the same bad scale-group under tight capacity; the other pool is fresh. |
| Parent stuck `pending`, `Scheduler: Insufficient CPU (need 1, available 0.05)` | Don't over-constrain with `--zone`; use `--region` so the coordinator lands in any zone — only the TPU child needs the zone-specific accelerator. |
| `CompileTimeHbmOom: Used <N>G of <M>G hbm` on a small-HBM TPU (v6e ≈31 GB/chip) | It's **activations**, not weights. Lower per-chip microbatch via `per_device_parallelism` (= grad accumulation); effective `train_batch_size` is unchanged. `gradient_checkpointing` is already on — not the lever. |
| Spot-TPU job dies `RuntimeError: N step(s) failed`, `preemptions=0` | A GCS-side preemption crashed the step; marin marks it a hard failure and does **not** auto-resume. Just relaunch → resumes from the last checkpoint. Neither TPU type/size nor PDP is in the config hash (checkpoints re-shard on load), so you can **freely re-route to any pool + set PDP to fit that slice's HBM** (`PDP=1024` on v6e-4, `PDP=-1` on v6e-8/v5p-8). |
| Preemptible v5p/v4 queued for hours | v6e/v5e (v5litepod) are usually abundant — re-route (long `pending` is capacity, not a freeze). Keep `--region` matched to your data's bucket. |

## Sources & the fast-churn caveat

marin/iris infra changes **rapidly** — the build/pin choices here rot fast; verify against the current known-good build (tracked in [#328](https://github.com/Open-Athena/marin-dna/issues/328)) before trusting them.

- **Reference consumer repos** (our per-experiment pyproject mirrors theirs): [`marin-community/marin-experiments`](https://github.com/marin-community/marin-experiments) (e.g. `tiny-stories/`) and [`Open-Athena/MarinFold`](https://github.com/Open-Athena/MarinFold).
- **marin Discord** — public read-only mirror at `https://marin-discord.pages.dev/archive.db`; download the SQLite and query it locally for "what's been said about X".
- **[`marin-community/marin`](https://github.com/marin-community/marin)** issues/PRs for upstream changes.

Cite public artifacts (the repos/issues/Discord mirror above), never internal channels.
