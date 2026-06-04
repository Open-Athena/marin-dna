# Marin-launched DNA experiments

Scripts under `experiments/` are launched on the shared marin iris cluster.
Each is a small Python file that calls `marin.execution.executor.executor_main`
to wire together one or more `ExecutorStep`s (tokenize → train → eval).

## Prerequisite: marin install

```bash
uv sync --extra marin --extra tpu
```

The `marin` extra and `aws-cli` group are mutually exclusive (see the
top-level README).

## Launch

`--cluster=marin` resolves the named cluster config and **auto-establishes the
controller tunnel** (SSH-through-IAP) — no manual port-forward needed. Just make
sure `gcloud` is authed first (`gcloud auth login`):

```bash
uv run iris --cluster=marin job run \
  --no-wait \
  --user gonzalo \
  --job-name <descriptive-name> \
  --cpu 1 --memory 2g \
  --extra marin \
  --region us-east5 \
  -e WANDB_API_KEY "$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')" \
  -e HF_HUB_DOWNLOAD_TIMEOUT 120 \
  -e UV_LOCK_TIMEOUT 7200 \
  -- python experiments/<your-script>.py
```

- **Don't** use `gcloud compute start-iap-tunnel … 10000`: the IAP firewall only
  allows port 22, so a direct TCP forward to the controller's port 10000 fails
  with `ERROR: … [4033: 'not authorized']`. `--cluster=marin` handles the tunnel
  for you. If you genuinely need a persistent manual tunnel (e.g. to reuse one
  across many invocations), open it via **SSH**-through-IAP and pass
  `--controller-url`:
  ```bash
  gcloud compute ssh iris-controller-marin --zone=us-central1-a \
    --project=hai-gcp-models --tunnel-through-iap -- -L 10000:localhost:10000 -N
  # then add: --controller-url=http://localhost:10000
  ```
- `--no-wait` returns immediately; follow with `iris job logs <id> -f`.
- `--cpu 1 --memory 2g` is the right coordinator footprint — `executor_main`
  parents spawn TPU sub-tasks via `remote(...)`; the parent stays CPU-only.
  Memory ≥ 4 GB needs `--enable-extra-resources`; you don't want that.
- `--region us-east5` is a hint, not a hard pin — iris path-infers from
  GCS dependencies otherwise, which has bitten us by inferring `us-east1`
  (no v5p there). See gotcha below.

See the upstream [`lib/iris/OPS.md`](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md)
for the full iris CLI surface (`iris job logs`, `iris job stop`, `iris job summary`,
`iris task exec`, etc.) — that's the source of truth and what to consult when
this README drifts.

## wandb run names

Set wandb run names with a `dna-exp<N>` prefix, where `<N>` is the experiment
number from the issue/directory (e.g. `dna-exp42-baseline-vs-tuned`). Lets runs
be filtered by experiment in the wandb UI.

## Lessons we learned the hard way (failure → fix)

These are anchored to specific failure messages so they survive iris/marin
API churn.

| Symptom (in worker logs) | Fix |
|---|---|
| `error: Extra "cpu" is not defined in any project's "optional-dependencies" table` | `pip_dependency_groups=["marin"]` in `remote(...)` of any vendored tokenize step (marin-dna's marin extra installs the cpu-flavored deps transitively). |
| `Timeout (Ns) when waiting for lock on /uv/cache/.../lm-eval...` | `-e UV_LOCK_TIMEOUT 7200` on `iris job run` AND in the tokenize step's `env_vars=`. Many zephyr workers share a uv cache; first build serializes; default 300s isn't enough. |
| `requests.exceptions.ReadTimeout: huggingface.co... read timeout=10` | `-e HF_HUB_DOWNLOAD_TIMEOUT 120` (passed to children via the tokenize step's `env_vars=`). HF's default is 10s; fine for small datasets, too short for parquet-manifest fetches on big ones. |
| `No accelerator found. Please run on a TPU or GPU.` + `iris: TPU bad-node signature detected` (repeated) | Switch zone — `--region us-east5` ↔ `--region us-central1` (the two `v5p-preemptible` regions). iris keeps allocating the same bad scale-group instance under tight capacity; the other zone's pool is fresh. |
| `marin-iris client is too old (build <date>; minimum <floor>). Run uv sync or upgrade marin-iris and retry.` | `uv lock --upgrade-package marin-iris && uv sync --extra marin`. Marin bumps `*-latest` tags daily; the iris controller's freshness check has a 14-day floor. If `marin-finelog` or some other sister-package is missing from releases, see #168's PR thread for the source-mapping workaround. |
| `Failed to download "marin-zephyr==0.99.dev<DATE>"` (404) | `*-latest` rotated. Same `uv lock --upgrade-package` dance. |
| `Build failed with exit_code=1` and `marin-root` pulls in workspace versions of `marin-*` that override find-links | Don't depend on `marin-root` from a consumer. Marin's `experiments/` package ships only with marin-root, which carries `[tool.uv.sources] marin-* = { workspace = true }` — that propagates and clobbers find-links. Vendor any experiments-package symbols you need inline. |
| Parent job stuck `pending` with `Scheduler: Insufficient CPU (need 1 cores, available 0.05 cores)` | Don't over-constrain via `--zone`. Prefer `--region` so the coordinator can land in any zone in that region; only the TPU child needs the zone-specific accelerator. |
| `CompileTimeHbmOom: Used <N>G of <M>G hbm` on a small-HBM TPU (e.g. v6e ≈31 GB/chip) at a large batch | The OOM is **activations**, not weights. Lower the per-chip microbatch via `per_device_parallelism` (= gradient accumulation); the effective `train_batch_size` is unchanged. `gradient_checkpointing` is already on by default — it's *not* the lever. |
| Spot-TPU job dies with `RuntimeError: N step(s) failed`, coordinator exits, `preemptions=0` | A GCS-side TPU preemption crashed the step; marin propagates it as a hard failure and does **not** auto-resume. Relaunch with the identical config → resumes from the last checkpoint. **Don't change `per_device_parallelism` on relaunch** — it's in the executor config hash, so a change orphans the checkpoint (silent restart from scratch). TPU *type/size* is not in the hash (checkpoints re-shard on load), so re-routing pools is fine. |
| Preemptible v5p / v4 queued for hours behind reserved jobs | v6e / v5e (v5litepod) are usually abundant — re-route there (long `pending` is capacity, not a freeze). Keep `--region` matched to your data's GCS bucket, or you re-tokenize into the new region. |

## What an experiment script needs to declare

Two pieces of marin context that aren't obvious from the python:

1. **`pip_dependency_groups` on each `remote(...)` call.** Workers don't
   inherit the parent's `--extra` flag — they each do their own `uv sync` with
   whatever extras the `remote()` requested. Default upstream value is `["cpu"]`
   which marin-dna doesn't define; use `["marin"]` instead until/unless we
   add a `cpu` extra.
2. **`env_vars` on each `remote(...)` call.** Same reason — workers don't
   inherit parent `-e` flags. Bake `HF_HUB_DOWNLOAD_TIMEOUT` and
   `UV_LOCK_TIMEOUT` into the script's `remote()`s; don't rely on
   `iris job run -e ...` reaching tokenize/train workers.

See `experiments/parity/exp179_eval_only.py` for a worked example.
