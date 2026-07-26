#!/usr/bin/env bash
# Launch bounded, spot-only offline evaluations for issue #402's 60k continuations.
#
# Usage:
#   CODE_REVISION=<40-char-sha> scripts/issue402_60k_eval_sweep.sh \
#     46m:25000 104m:25000 46m:30000 104m:30000
#
# Allowed cadence: 25k, 30k, ..., 55k, and the final 59,999 HF export.
#
# Environment:
#   MAX_PARALLEL  Maximum simultaneous Sky launch processes (default: 4).
#   LOG_DIR       Local per-cluster logs (default: /tmp/issue402_60k_evals).
#   DRY_RUN       Set to 1 to validate inputs and print commands only.

set -uo pipefail

if [[ $# -eq 0 ]]; then
    sed -n '2,/^$/p' "$0" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
task_yaml="$repo_root/scripts/issue402_offline_eval_sky.yaml"
[[ -f "$task_yaml" ]] || { echo "missing $task_yaml" >&2; exit 1; }

code_revision=${CODE_REVISION:-}
max_parallel=${MAX_PARALLEL:-4}
log_dir=${LOG_DIR:-/tmp/issue402_60k_evals}
dry_run=${DRY_RUN:-0}

[[ "$code_revision" =~ ^[0-9a-f]{40}$ ]] || {
    echo "CODE_REVISION must be a full 40-character lowercase commit SHA" >&2
    exit 2
}
current_revision=$(git -C "$repo_root" rev-parse HEAD)
[[ "$code_revision" == "$current_revision" ]] || {
    echo "CODE_REVISION $code_revision does not match worktree HEAD $current_revision" >&2
    exit 2
}
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || {
    echo "worktree must be clean so Sky sync and CODE_REVISION describe identical code" >&2
    exit 2
}
[[ "$max_parallel" =~ ^[1-9][0-9]*$ ]] || {
    echo "MAX_PARALLEL must be a positive integer" >&2
    exit 2
}
[[ "$dry_run" == 0 || "$dry_run" == 1 ]] || {
    echo "DRY_RUN must be 0 or 1" >&2
    exit 2
}
mkdir -p "$log_dir"

declare -a batch_pids=()
declare -a batch_clusters=()
failures=0
launched=0
skipped=0

wait_batch() {
    local index pid cluster
    for index in "${!batch_pids[@]}"; do
        pid=${batch_pids[$index]}
        cluster=${batch_clusters[$index]}
        if wait "$pid"; then
            echo "[issue402-60k-eval] completed $cluster" >&2
        else
            echo "[issue402-60k-eval] FAILED $cluster (see $log_dir/$cluster.log)" >&2
            failures=$((failures + 1))
        fi
    done
    batch_pids=()
    batch_clusters=()
}

cd "$repo_root"
for spec in "$@"; do
    if [[ ! "$spec" =~ ^(46m|104m):([0-9]+)$ ]]; then
        echo "invalid spec '$spec'; expected 46m:<step> or 104m:<step>" >&2
        exit 2
    fi
    model=${BASH_REMATCH[1]}
    step=${BASH_REMATCH[2]}
    if (( (step < 25000 || step > 55000 || step % 5000 != 0) && step != 59999 )); then
        echo "invalid step '$step'; expected 25000, 30000, ..., 55000, or 59999" >&2
        exit 2
    fi

    if [[ "$model" == 46m ]]; then
        artifact="dna-exp402-rag-h640-p46m-60k-from24k"
    else
        artifact="dna-exp402-rag-h768-p104m-60k-from24k"
    fi
    checkpoint_uri="gs://marin-us-east5/checkpoints/$artifact/2026.07.26/hf/step-$step"
    output_uri="gs://marin-us-east5/evals/$artifact/2026.07.26/step-$step"
    cluster="dna402-60k-${model}-s${step}"

    if [[ "$dry_run" == 0 ]]; then
        if ! gcloud storage ls "$checkpoint_uri/config.json" >/dev/null 2>&1; then
            echo "[issue402-60k-eval] missing export: $checkpoint_uri/config.json" >&2
            failures=$((failures + 1))
            continue
        fi
        output_complete=1
        for benchmark in mendelian_traits complex_traits sge; do
            if ! gcloud storage ls "$output_uri/$benchmark/metrics.parquet" >/dev/null 2>&1; then
                output_complete=0
                break
            fi
        done
        if (( output_complete )); then
            echo "[issue402-60k-eval] skip complete $model step $step" >&2
            skipped=$((skipped + 1))
            continue
        fi
    fi

    command=(
        sky launch --yes --use-spot --down
        -c "$cluster"
        "$task_yaml"
        --env "CHECKPOINT_URI=$checkpoint_uri"
        --env "OUTPUT_URI=$output_uri"
        --env "CODE_REVISION=$code_revision"
    )
    if (( dry_run )); then
        printf '[issue402-60k-eval] DRY RUN:' >&2
        printf ' %q' "${command[@]}" >&2
        printf '\n' >&2
        continue
    fi

    echo "[issue402-60k-eval] launch $model step $step as $cluster" >&2
    "${command[@]}" >"$log_dir/$cluster.log" 2>&1 &
    batch_pids+=("$!")
    batch_clusters+=("$cluster")
    launched=$((launched + 1))
    if (( ${#batch_pids[@]} >= max_parallel )); then
        wait_batch
    fi
done

if (( ${#batch_pids[@]} )); then
    wait_batch
fi

echo "[issue402-60k-eval] launched=$launched skipped=$skipped failures=$failures logs=$log_dir" >&2
(( failures == 0 ))
