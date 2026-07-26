#!/usr/bin/env bash
# Launch bounded, spot-only offline evaluations for issue #402's 30k runs.
#
# Usage:
#   CODE_REVISION=<40-char-sha> scripts/issue402_30k_eval_sweep.sh \
#     46m:2000 104m:2000 46m:3000 104m:3000
#
# Environment:
#   MAX_PARALLEL  Maximum simultaneous Sky launch processes (default: 4).
#   LOG_DIR       Local per-cluster logs (default: /tmp/issue402_30k_evals).
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
log_dir=${LOG_DIR:-/tmp/issue402_30k_evals}
dry_run=${DRY_RUN:-0}

[[ "$code_revision" =~ ^[0-9a-f]{40}$ ]] || {
    echo "CODE_REVISION must be a full 40-character lowercase commit SHA" >&2
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
            echo "[issue402-eval] completed $cluster" >&2
        else
            echo "[issue402-eval] FAILED $cluster (see $log_dir/$cluster.log)" >&2
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
    if (( step != 29999 && step % 1000 != 0 )); then
        echo "invalid step '$step'; expected a 1,000-step export or final step 29999" >&2
        exit 2
    fi
    if (( step < 1000 || step > 29999 )); then
        echo "invalid step '$step'; expected 1000..29999" >&2
        exit 2
    fi

    if [[ "$model" == 46m ]]; then
        artifact="dna-exp402-rag-h640-p46m-30k"
    else
        artifact="dna-exp402-rag-h768-p104m-30k"
    fi
    checkpoint_uri="gs://marin-us-east5/checkpoints/$artifact/2026.07.26/hf/step-$step"
    output_uri="gs://marin-us-east5/evals/$artifact/2026.07.26/step-$step"
    cluster="dna402-${model}-s${step}"

    if [[ "$dry_run" == 0 ]]; then
        if ! gcloud storage ls "$checkpoint_uri/config.json" >/dev/null 2>&1; then
            echo "[issue402-eval] missing export: $checkpoint_uri/config.json" >&2
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
            echo "[issue402-eval] skip complete $model step $step" >&2
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
        printf '[issue402-eval] DRY RUN:' >&2
        printf ' %q' "${command[@]}" >&2
        printf '\n' >&2
        continue
    fi

    echo "[issue402-eval] launch $model step $step as $cluster" >&2
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

echo "[issue402-eval] launched=$launched skipped=$skipped failures=$failures logs=$log_dir" >&2
(( failures == 0 ))
