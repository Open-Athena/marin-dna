#!/usr/bin/env bash
# Launch issue #402's final large-batch behavioral diagnostics.
#
# Usage:
#   CODE_REVISION=<40-char-sha> scripts/issue402_large_batch_final_sanity.sh
#
# The two model-sanity jobs run concurrently on spot A10Gs. They include
# fixed-length BOS/[SEQ] VEP interventions as a guard against special tokens
# being silently ignored. The combined indel-aware attention job starts only
# after both model-sanity jobs complete.
#
# Environment:
#   LOG_DIR  Local Sky logs (default: /tmp/issue402_large_batch_final_sanity).
#   DRY_RUN  Set to 1 to validate and print commands without launching.

set -uo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
sanity_yaml="$repo_root/scripts/issue402_model_sanity_sky.yaml"
indel_yaml="$repo_root/scripts/issue402_indel_attention_sky.yaml"
code_revision=${CODE_REVISION:-}
log_dir=${LOG_DIR:-/tmp/issue402_large_batch_final_sanity}
dry_run=${DRY_RUN:-0}
artifact_version=2026.07.26.5

[[ -f "$sanity_yaml" ]] || { echo "missing $sanity_yaml" >&2; exit 1; }
[[ -f "$indel_yaml" ]] || { echo "missing $indel_yaml" >&2; exit 1; }
[[ "$code_revision" =~ ^[0-9a-f]{40}$ ]] || {
    echo "CODE_REVISION must be a full 40-character lowercase commit SHA" >&2
    exit 2
}
current_revision=$(git -C "$repo_root" rev-parse HEAD)
[[ "$code_revision" == "$current_revision" ]] || {
    echo "CODE_REVISION $code_revision does not match HEAD $current_revision" >&2
    exit 2
}
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || {
    echo "worktree must be clean" >&2
    exit 2
}
[[ "$dry_run" == 0 || "$dry_run" == 1 ]] || {
    echo "DRY_RUN must be 0 or 1" >&2
    exit 2
}

# Sky packages the caller's current directory, so execute from the same clean
# repository root whose commit was validated above.
cd "$repo_root"
mkdir -p "$log_dir"
revision_short=${code_revision:0:7}
checkpoint_46m="gs://marin-us-east5/checkpoints/dna-exp402-rag-h640-p46m-b2m-30k/$artifact_version/hf/step-29999"
checkpoint_104m="gs://marin-us-east5/checkpoints/dna-exp402-rag-h768-p104m-b2m-30k/$artifact_version/hf/step-29999"
output_46m="gs://marin-us-east5/evals/dna-exp402-rag-h640-p46m-b2m-30k/$artifact_version/sanity-$revision_short"
output_104m="gs://marin-us-east5/evals/dna-exp402-rag-h768-p104m-b2m-30k/$artifact_version/sanity-$revision_short"
indel_output="gs://marin-us-east5/evals/issue402-rag-large-batch-30k/$artifact_version/indel-attention-$revision_short"

if (( ! dry_run )); then
    for checkpoint in "$checkpoint_46m" "$checkpoint_104m"; do
        gcloud storage ls "$checkpoint/config.json" >/dev/null
        gcloud storage ls "$checkpoint/model.safetensors" >/dev/null
        gcloud storage ls "$checkpoint/tokenizer.json" >/dev/null
    done
fi

declare -a pids=()
declare -a labels=()
failures=0
for model in 46M 104M; do
    if [[ "$model" == 46M ]]; then
        checkpoint=$checkpoint_46m
        output=$output_46m
        slug=46m
    else
        checkpoint=$checkpoint_104m
        output=$output_104m
        slug=104m
    fi
    if (( ! dry_run )) && gcloud storage ls "$output/manifest.json" >/dev/null 2>&1; then
        echo "[issue402-final-sanity] skip complete $model: $output" >&2
        continue
    fi
    command=(
        sky launch --yes --use-spot --down
        -c "dna402-b2m-sanity-$slug"
        "$sanity_yaml"
        --env "CHECKPOINT_URI=$checkpoint"
        --env "OUTPUT_URI=$output"
        --env "MODEL_LABEL=$model"
        --env "CODE_REVISION=$code_revision"
    )
    if (( dry_run )); then
        printf '[issue402-final-sanity] DRY RUN:' >&2
        printf ' %q' "${command[@]}" >&2
        printf '\n' >&2
    else
        echo "[issue402-final-sanity] launch $model -> $output" >&2
        "${command[@]}" >"$log_dir/sanity-$slug.log" 2>&1 &
        pids+=("$!")
        labels+=("$model")
    fi
done

for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "[issue402-final-sanity] completed ${labels[$index]}" >&2
    else
        echo "[issue402-final-sanity] FAILED ${labels[$index]}" >&2
        failures=$((failures + 1))
    fi
done
(( failures == 0 )) || exit 1

if (( ! dry_run )) && gcloud storage ls "$indel_output/46M/manifest.json" >/dev/null 2>&1 \
    && gcloud storage ls "$indel_output/104M/manifest.json" >/dev/null 2>&1; then
    echo "[issue402-final-sanity] skip complete indel attention: $indel_output" >&2
    exit 0
fi
indel_command=(
    sky launch --yes --use-spot --down
    -c dna402-b2m-indel-attention
    "$indel_yaml"
    --env "CHECKPOINT_46M_URI=$checkpoint_46m"
    --env "CHECKPOINT_104M_URI=$checkpoint_104m"
    --env "OUTPUT_URI=$indel_output"
    --env "CODE_REVISION=$code_revision"
)
if (( dry_run )); then
    printf '[issue402-final-sanity] DRY RUN:' >&2
    printf ' %q' "${indel_command[@]}" >&2
    printf '\n' >&2
    exit 0
fi
"${indel_command[@]}" >"$log_dir/indel-attention.log" 2>&1
