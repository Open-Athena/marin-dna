#!/usr/bin/env bash
# Dispatch one sky cluster per snakemake target, with autostop + --down.
#
# Usage (invoke from anywhere — the helper cd's to the repo root):
#   snakemake/analysis/evals_v2/sky/parallel_sweep.sh <target> [<target> ...]
#
# Why the cd: `run.yaml` has `workdir: .` (resolves to sky-launch CWD)
# and its run script does `cd snakemake/analysis/evals_v2`, so the synced
# workdir on the cluster must be the repo root or setup fails with
# `No pyproject.toml found ...`. The helper cd's there so the caller
# doesn't have to remember.
#
# Each target is a snakemake output path relative to the pipeline cwd
# (`snakemake/analysis/evals_v2/`), e.g.
#   results/metrics/exp58-mammals-step-1000/mendelian_traits.parquet
# — i.e. the target string is what the cluster's snakemake sees, *not*
# relative to your shell CWD. Cluster name is derived from the target's
# parent dir (the `{model}` wildcard) prefixed with `evals-v2-`, so
# different (model × dataset) targets for the same model would collide
# — pass one target per model per invocation. The pipeline rules
# dispatch correctly because the `SNAKEMAKE_ARGS="-- $target"` knob
# narrows snakemake to exactly that parquet.
#
# Env vars:
#   SKY_STAGGER         Seconds between dispatches (default 3). Bigger
#                       = less pressure on the sky API server and on
#                       AWS RunInstances.
#   SKY_AUTOSTOP_MIN    Idle minutes before `--down` terminates the
#                       cluster (default 1). Bump if the per-cluster
#                       run might need more time for its S3 upload.
#   SKY_LOG_DIR         Per-cluster log dir (default
#                       /tmp/sky_<pwd-basename>). One file per cluster
#                       named `<cluster>.log` for post-hoc triage.
#
# AZ saturation: when bursting >~24 g5.xlarge into us-east-2, expect
# some launches to fail with `ResourcesUnavailableError` — AWS spreads
# capacity unevenly across AZs and sky has no cross-region fallback for
# AWS-pinned tasks. Wait for the successful clusters to `--down`, then
# re-run this script with just the failed target names.
#
# Tally state after the run:
#   grep -l 'Job finished (status: SUCCEEDED)' "$SKY_LOG_DIR"/*.log | wc -l
#   grep -lE 'ResourcesUnavailableError|FAILED' "$SKY_LOG_DIR"/*.log

set -uo pipefail

if [[ $# -eq 0 ]]; then
    sed -n '2,/^$/p' "$0" >&2
    exit 2
fi

here=$(cd "$(dirname "$0")" && pwd)
run_yaml="$here/run.yaml"
[[ -f "$run_yaml" ]] || { echo "missing $run_yaml" >&2; exit 1; }

# Cd to the repo root so `sky launch` syncs the whole repo via
# `workdir: .` (the cluster's run script then does
# `cd snakemake/analysis/evals_v2`). This makes the helper invariant
# to caller cwd. Script lives at
# `snakemake/analysis/evals_v2/sky/parallel_sweep.sh`, so the repo
# root is 4 levels up from `$here`. Sanity-check the resolution
# before relying on it.
repo_root=$(cd "$here/../../../.." && pwd)
[[ -f "$repo_root/pyproject.toml" ]] || {
    echo "[parallel_sweep] expected pyproject.toml at $repo_root — has the helper moved?" >&2
    exit 1
}
cd "$repo_root"

stagger=${SKY_STAGGER:-3}
autostop=${SKY_AUTOSTOP_MIN:-1}
log_dir=${SKY_LOG_DIR:-/tmp/sky_$(basename "$PWD")}
mkdir -p "$log_dir"
echo "[parallel_sweep] log_dir=$log_dir" >&2

pids=()
for target in "$@"; do
    model=$(basename "$(dirname "$target")")
    cluster="evals-v2-${model}"
    sky launch -c "$cluster" \
        --env SNAKEMAKE_ARGS="-- $target" \
        --idle-minutes-to-autostop="$autostop" \
        --down \
        --yes \
        "$run_yaml" \
        > "$log_dir/$cluster.log" 2>&1 &
    pids+=($!)
    sleep "$stagger"
done

echo "[parallel_sweep] dispatched ${#pids[@]} sky launches; waiting…" >&2
wait
echo "[parallel_sweep] all sky clusters reached terminal state" >&2
