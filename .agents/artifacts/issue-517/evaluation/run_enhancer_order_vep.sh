#!/usr/bin/env bash
# Run on the dedicated short-lived spot A10G; the instance terminates on shutdown.
set -euo pipefail
export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"
export WANDB_DISABLED=true
export POLARS_MAX_THREADS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

finish() {
    local exit_status=$?
    trap - EXIT
    printf 'evaluation_exit_status=%s\n' "$exit_status"
    if command -v aws >/dev/null 2>&1; then
        aws s3 cp /home/ubuntu/issue517-order-vep.log \
            s3://oa-bolinas/snakemake/analysis/evals_v2/results/metadata/exp517-phylop-uniform-enhancer-order-step-4999/20260908-run.log || true
    fi
    sudo shutdown -h now
    exit "$exit_status"
}
trap finish EXIT

cd /home/ubuntu/issue517-order-vep
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/0.11.31/install.sh | sh
fi
if [[ "$(uv --version | awk '{print $2}')" != "0.11.31" ]]; then
    uv self update 0.11.31
fi
if ! command -v gcloud >/dev/null 2>&1; then
    curl -fsSL https://sdk.cloud.google.com -o /tmp/issue517-gcloud-install.sh
    bash /tmp/issue517-gcloud-install.sh --disable-prompts --install-dir="$HOME"
fi

cd snakemake/analysis/evals_v2
uv sync --locked --group dev --group genome-s3
uv run --locked --group genome-s3 evals-gpu-runtime-check \
    --config config/gpu_runtime_validation.yaml smoke
uv run --locked --group genome-s3 pytest

targets=(
    results/metrics/exp517-phylop-uniform-enhancer-order-step-4999/mendelian_traits.parquet
    results/metrics/exp517-phylop-uniform-enhancer-order-step-4999/complex_traits.parquet
)
common=(--profile workflow/profiles/default --cores 4 --rerun-incomplete
    --printshellcmds --configfile config/issue517_enhancer_order.yaml)
uv run --locked --group genome-s3 snakemake "${common[@]}" --dry-run -- "${targets[@]}"
uv run --locked --group genome-s3 snakemake "${common[@]}" -- "${targets[@]}"
