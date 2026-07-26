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
frozen_revision=7d9a7c9f5a2f8040af3daadb8c2be10804c211fc
revision_short=${frozen_revision:0:7}
output_root="$repo_root/plots/output"

checkpoint_46m="gs://marin-us-east5/checkpoints/dna-exp402-rag-h640-p46m-b2m-30k/$artifact_version"
checkpoint_104m="gs://marin-us-east5/checkpoints/dna-exp402-rag-h768-p104m-b2m-30k/$artifact_version"
eval_46m="gs://marin-us-east5/evals/dna-exp402-rag-h640-p46m-b2m-30k/$artifact_version"
eval_104m="gs://marin-us-east5/evals/dna-exp402-rag-h768-p104m-b2m-30k/$artifact_version"
final_46m="$eval_46m/step-29999"
final_104m="$eval_104m/step-29999"
sanity_46m="$eval_46m/sanity-$revision_short"
sanity_104m="$eval_104m/sanity-$revision_short"
indel_root="gs://marin-us-east5/evals/issue402-rag-large-batch-30k/$artifact_version/indel-attention-$revision_short"
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
    for root in "$eval_46m" "$eval_104m"; do
        if [[ "$root" == "$eval_46m" ]]; then
            expected_checkpoint_root=$checkpoint_46m
        else
            expected_checkpoint_root=$checkpoint_104m
        fi
        for step in "${steps[@]}"; do
            for benchmark in mendelian_traits complex_traits sge; do
                case "$benchmark" in
                    mendelian_traits)
                        dataset_repo=marin-dna/evals_mendelian_traits_rag_harness_255_v1
                        dataset_revision=9acedb683463477f34745af30a63a289873008a4
                        score_column=minus_llr_avg
                        ;;
                    complex_traits)
                        dataset_repo=marin-dna/evals_complex_traits_rag_harness_255_v1
                        dataset_revision=0252a883f650819a8e1fa22062027daafe956540
                        score_column=abs_llr_avg
                        ;;
                    sge)
                        dataset_repo=marin-dna/evals_sge_rag_harness_255_v1
                        dataset_revision=c20cc58fceb9bc053a55152a89d160f1b070f75d
                        score_column=minus_llr_avg
                        ;;
                esac
                benchmark_root="$root/step-$step/$benchmark"
                for filename in documents.parquet variants.parquet metrics.parquet; do
                    gcloud storage ls "$benchmark_root/$filename" >/dev/null
                done
                gcloud storage cat "$benchmark_root/manifest.json" \
                    | jq -e \
                        --arg benchmark "$benchmark" \
                        --arg model_source "$expected_checkpoint_root/hf/step-$step" \
                        --arg code_revision "$frozen_revision" \
                        --arg dataset_repo "$dataset_repo" \
                        --arg dataset_revision "$dataset_revision" \
                        --arg score_column "$score_column" \
                        '.benchmark == $benchmark
                         and .split == "test"
                         and .model_source == $model_source
                         and .code_revision == $code_revision
                         and .dataset_repo == $dataset_repo
                         and .dataset_revision == $dataset_revision
                         and .score_column == $score_column
                         and .aggregation == "average raw fwd/rc LLR, then apply score transform"
                         and .row_selection == "all"
                         and .batch_size == 16
                         and .n_document_rows > 0
                         and .n_document_rows == (2 * .n_variants)' >/dev/null
            done
        done
    done
    for model in 46M 104M; do
        if [[ "$model" == 46M ]]; then
            root=$sanity_46m
            model_source="$checkpoint_46m/hf/step-29999"
        else
            root=$sanity_104m
            model_source="$checkpoint_104m/hf/step-29999"
        fi
        gcloud storage ls "$root/vep_special_token_diagnostics.parquet" >/dev/null
        gcloud storage cat "$root/manifest.json" \
            | jq -e \
                --arg model "$model" \
                --arg model_source "$model_source" \
                --arg code_revision "$frozen_revision" \
                '.model_label == $model
                 and .model_source == $model_source
                 and .code_revision == $code_revision
                 and .validation_rows == 2048
                 and .ablation_rows == 512
                 and .attention_rows == 4
                 and .n_bootstrap == 1000' >/dev/null
        gcloud storage cat "$indel_root/$model/manifest.json" \
            | jq -e \
                --arg model "$model" \
                --arg model_source "$model_source" \
                --arg code_revision "$frozen_revision" \
                '.model_label == $model
                 and .model_source == $model_source
                 and .code_revision == $code_revision
                 and (.anchor_ids | length) >= 2
                 and .query_stride == 4' >/dev/null
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
