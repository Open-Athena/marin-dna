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
revision_short=4566dfb
output_root="$repo_root/plots/output"

eval_46m="gs://marin-us-east5/evals/dna-exp402-rag-h640-p46m-b2m-30k/$artifact_version"
eval_104m="gs://marin-us-east5/evals/dna-exp402-rag-h768-p104m-b2m-30k/$artifact_version"
final_46m="$eval_46m/step-29999"
final_104m="$eval_104m/step-29999"
sanity_46m="$eval_46m/sanity-$revision_short"
sanity_104m="$eval_104m/sanity-$revision_short"
indel_root="gs://marin-us-east5/evals/issue402-rag-large-batch-30k/$artifact_version/indel-attention-$revision_short"
phylop_root="gs://marin-us-east5/users/ubuntu/evals/dna-exp402-rag-phylop447m/exact-test-a57a69c"
steps=(5000 10000 15000 20000 25000 29999)

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
    command -v uv >/dev/null
    for root in "$eval_46m" "$eval_104m"; do
        for step in "${steps[@]}"; do
            for benchmark in mendelian_traits complex_traits sge; do
                gcloud storage ls "$root/step-$step/$benchmark/manifest.json" >/dev/null
                gcloud storage ls "$root/step-$step/$benchmark/metrics.parquet" >/dev/null
            done
        done
    done
    for root in "$sanity_46m" "$sanity_104m"; do
        gcloud storage ls "$root/manifest.json" >/dev/null
        gcloud storage ls "$root/vep_special_token_diagnostics.parquet" >/dev/null
    done
    for model in 46M 104M; do
        gcloud storage ls "$indel_root/$model/manifest.json" >/dev/null
    done
fi

cd "$repo_root"
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
run uv run python plots/issue402_rag_validation_position.py \
    --input-46m "$sanity_46m" \
    --input-104m "$sanity_104m" \
    --output-dir "$output_root/issue402_rag_large_batch_validation_position"
run uv run python plots/issue402_rag_attention_alignment.py \
    --input-46m "$sanity_46m" \
    --input-104m "$sanity_104m" \
    --output-dir "$output_root/issue402_rag_large_batch_attention_alignment"
run uv run python plots/issue402_rag_context_ablation.py \
    --input-46m "$sanity_46m" \
    --input-104m "$sanity_104m" \
    --output-dir "$output_root/issue402_rag_large_batch_context_ablation"
run uv run python plots/issue402_rag_vep_context_ablation.py \
    --eval-46m "$eval_46m" \
    --eval-104m "$eval_104m" \
    --sanity-46m "$sanity_46m" \
    --sanity-104m "$sanity_104m" \
    --include-special-token-controls \
    --output-dir "$output_root/issue402_rag_large_batch_vep_context_ablation"
run uv run python plots/issue402_rag_indel_attention.py \
    --input-root "$indel_root" \
    --output-dir "$output_root/issue402_rag_large_batch_indel_attention"
run uv run python scripts/issue402_audit_final_rc.py \
    --input-46m "$final_46m" \
    --input-104m "$final_104m" \
    --output "$output_root/issue402_rag_large_batch_rc_audit/metrics.parquet"
