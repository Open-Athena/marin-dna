#!/bin/bash
set -euo pipefail
export PATH=/home/ubuntu/miniforge3/bin:/home/ubuntu/.local/bin:$PATH
export POLARS_MAX_THREADS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
: "${HF_TOKEN:?The approved HF credential must be provided in the runtime environment}"
trap 'result=$?; printf "%s\n" "$result" > /opt/issue550/recovery-exit.txt; date -u +%FT%TZ > /opt/issue550/recovery-end.txt' EXIT
date -u +%FT%TZ > /opt/issue550/recovery-start.txt
printf 'assembly\n' > /opt/issue550/recovery-stage.txt
cd /opt/issue550/producer/snakemake/vertebrate_projection_dataset
test "$(git rev-parse HEAD)" = 6b1593c274a886d20f5c0ddf3712916d446f5fed
producer_options=(--configfile config/rag_issue550/config.yaml --cores 4 --resources mem_mb=110000 --local-storage-prefix /opt/issue550/assets --rerun-triggers code params input --keep-storage-local-copies)
/usr/bin/time -v uv run --locked snakemake rag_all_documents "${producer_options[@]}"
printf 'release_preparation\n' > /opt/issue550/recovery-stage.txt
cd /opt/issue550/publisher/snakemake/vertebrate_projection_dataset
test "$(git rev-parse HEAD)" = 54b6f936467bbc657ae88753a6f24b768bd3363d
uv sync --locked
publisher_options=(--configfile config/rag_issue550/config.yaml config/rag_issue550/publication.yaml --cores 4 --resources mem_mb=24000 hf_uploads=1 --local-storage-prefix /opt/issue550/publication-assets --rerun-triggers code params input --keep-storage-local-copies)
uv run --locked snakemake rag_all_publication_files "${publisher_options[@]}" -n > /opt/issue550/release-ready-dryrun.log 2>&1
/usr/bin/time -v uv run --locked snakemake rag_all_publication_files "${publisher_options[@]}"
printf 'release_audit\n' > /opt/issue550/recovery-stage.txt
/usr/bin/time -v uv run --locked python /opt/issue550/issue550-audit-release.py
printf 'publication\n' > /opt/issue550/recovery-stage.txt
uv run --locked snakemake rag_publish "${publisher_options[@]}" -n > /opt/issue550/hub-ready-dryrun.log 2>&1
/usr/bin/time -v uv run --locked snakemake rag_publish "${publisher_options[@]}"
printf 'manifest_verification\n' > /opt/issue550/recovery-stage.txt
uv run --locked python /opt/issue550/issue550-training-manifest.py
printf 'ready_for_training\n' > /opt/issue550/recovery-stage.txt
