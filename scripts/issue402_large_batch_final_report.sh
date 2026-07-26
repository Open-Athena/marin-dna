#!/usr/bin/env bash
# Rebuild issue #402's complete large-batch report from frozen cloud artifacts.
#
# Usage:
#   scripts/issue402_large_batch_final_report.sh
#   DRY_RUN=1 scripts/issue402_large_batch_final_report.sh

set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
dry_run=${DRY_RUN:-0}
artifact_version=2026.07.26.5
output_root="$repo_root/plots/output"

checkpoint_46m="gs://marin-us-east5/checkpoints/dna-exp402-rag-h640-p46m-b2m-30k/$artifact_version"
checkpoint_104m="gs://marin-us-east5/checkpoints/dna-exp402-rag-h768-p104m-b2m-30k/$artifact_version"
eval_46m="gs://marin-us-east5/evals/dna-exp402-rag-h640-p46m-b2m-30k/$artifact_version"
eval_104m="gs://marin-us-east5/evals/dna-exp402-rag-h768-p104m-b2m-30k/$artifact_version"
final_46m="$eval_46m/step-29999"
final_104m="$eval_104m/step-29999"
phylop_root="gs://marin-us-east5/users/ubuntu/evals/dna-exp402-rag-phylop447m/exact-test-a57a69c"
steps=(5000 10000 15000 20000 25000 29999)
checkpoint_steps=({1000..29000..1000} 29999)

[[ "$dry_run" == 0 || "$dry_run" == 1 ]]

run() {
    if (( dry_run )); then
        printf '[issue402-large-batch-report] DRY RUN:' >&2
        printf ' %q' "$@" >&2
        printf '\n' >&2
    else
        "$@"
    fi
}

if (( ! dry_run )); then
    command -v gcloud >/dev/null
    command -v jq >/dev/null
    command -v uv >/dev/null
    for root in "$checkpoint_46m" "$checkpoint_104m"; do
        for checkpoint_step in "${checkpoint_steps[@]}"; do
            gcloud storage cat "$root/checkpoints/step-$checkpoint_step/metadata.json" \
                | jq -e --argjson expected "$checkpoint_step" \
                    '.step == $expected and .is_temporary == false' >/dev/null
            for filename in config.json model.safetensors tokenizer.json; do
                gcloud storage ls "$root/hf/step-$checkpoint_step/$filename" >/dev/null
            done
        done
    done
fi

cd "$repo_root"
for model in 46m 104m; do
    for step in "${steps[@]}"; do
        run uv run python scripts/issue402_audit_large_batch_eval_bundle.py \
            --model "$model" \
            --step "$step"
    done
done
run uv run python plots/issue402_rag_large_batch_validation_loss.py
run uv run python plots/issue402_rag_batch_size_validation_loss.py
run uv run python scripts/issue402_audit_training_dynamics.py
run uv run python plots/issue402_rag_large_batch_auprc.py \
    --input-46m "$eval_46m" \
    --input-104m "$eval_104m"
run uv run python plots/issue402_rag_subset_auprc.py \
    --input-46m "$final_46m" \
    --input-104m "$final_104m" \
    --output-dir "$output_root/issue402_rag_large_batch_subset_auprc"
run uv run python plots/issue402_rag_vs_phylop_subset_auprc.py \
    --input-46m "$final_46m" \
    --input-104m "$final_104m" \
    --phylop-root "$phylop_root" \
    --output-dir "$output_root/issue402_rag_large_batch_vs_phylop_subset_auprc"
# Full behavioral/attention sanity was already accepted on the frozen path at
# 46M step 5k. Per the updated scope, final large-batch checkpoints do not
# repeat those paid diagnostics.
run uv run python scripts/issue402_audit_final_rc.py \
    --input-46m "$final_46m" \
    --input-104m "$final_104m" \
    --output "$output_root/issue402_rag_large_batch_rc_audit/metrics.parquet"
