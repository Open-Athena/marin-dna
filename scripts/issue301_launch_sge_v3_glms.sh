#!/usr/bin/env bash
# Re-score the 3 #292 gLMs on evals_sge v3 (AUPRC-only, #301), one A10G cluster
# per model. Mirrors snakemake/analysis/evals_v2/sky/parallel_sweep.sh but injects
# `--forcerun results/scores/{model}/sge.parquet` — a fresh cluster has no local
# snakemake metadata and the v2 sge.parquet already exists on S3, so without the
# force snakemake would report "nothing to do" and never re-score on v3.
#
# Usage (from the repo/worktree root):
#   scripts/issue301_launch_sge_v3_glms.sh [<model> ...]
# With no args, launches all 3. Pass model names to re-launch just those (e.g.
# after an AZ-saturation ResourcesUnavailableError on a subset).
#
# Cluster names sanitize the dotted model names (sky cluster names are
# hostname-like). Logs: $SKY_LOG_DIR/<cluster>.log (default /tmp/sky_sge_v3).
set -uo pipefail

DEFAULT_MODELS=(
    mix-v0.9-p1B-i24-exp135-m5.1-step-59158
    exp166-v0.1-p1B-step-27329
    exp13-mixture-proportional-step-26000
)
models=("$@")
[[ ${#models[@]} -eq 0 ]] && models=("${DEFAULT_MODELS[@]}")

here=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$here/.." && pwd)
[[ -f "$repo_root/pyproject.toml" ]] || { echo "no pyproject.toml at $repo_root" >&2; exit 1; }
cd "$repo_root"

run_yaml="snakemake/analysis/evals_v2/sky/run.yaml"
[[ -f "$run_yaml" ]] || { echo "missing $run_yaml" >&2; exit 1; }

stagger=${SKY_STAGGER:-5}
autostop=${SKY_AUTOSTOP_MIN:-5}
log_dir=${SKY_LOG_DIR:-/tmp/sky_sge_v3}
mkdir -p "$log_dir"
echo "[launch] log_dir=$log_dir models=${#models[@]}" >&2

# Short, dot-free cluster aliases keyed by model.
cluster_for() {
    case "$1" in
        mix-v0.9-p1B-i24-exp135-m5.1-step-59158) echo "evals-sge-exp135" ;;
        exp166-v0.1-p1B-step-27329)              echo "evals-sge-exp166" ;;
        exp13-mixture-proportional-step-26000)   echo "evals-sge-exp13mp" ;;
        *) echo "evals-sge-$(echo "$1" | tr '.' '-' | cut -c1-40)" ;;
    esac
}

pids=()
for model in "${models[@]}"; do
    cluster=$(cluster_for "$model")
    args="--forcerun results/scores/$model/sge.parquet -- results/metrics/$model/sge.parquet"
    echo "[launch] $cluster  <-  $model" >&2
    sky launch -c "$cluster" \
        --env SNAKEMAKE_ARGS="$args" \
        --idle-minutes-to-autostop="$autostop" \
        --down \
        --yes \
        "$run_yaml" \
        > "$log_dir/$cluster.log" 2>&1 &
    pids+=($!)
    sleep "$stagger"
done

echo "[launch] dispatched ${#pids[@]} sky launches; waiting…" >&2
wait
echo "[launch] all sky clusters reached terminal state" >&2
