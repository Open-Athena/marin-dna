#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stage_root="$project_root/results/gh200-full-inputs"
temporary_root="$(mktemp -d)"
trap 'rm -rf -- "$temporary_root"' EXIT

canonical_prefix="s3://oa-bolinas/snakemake/analysis/carbon_conditioning_vep/results"
snapshot_prefix="s3://oa-bolinas/snakemake/analysis/carbon_conditioning_vep/snapshots/carbon-conditioning-vep-full-three-arm-20260820"
mkdir -p "$stage_root/preflight" "$stage_root/windows"
mkdir -p "$temporary_root/windows"

aws s3 cp --no-progress \
  "$canonical_prefix/preflight/prompt_grammar.json" \
  "$stage_root/preflight/prompt_grammar.json"
aws s3 cp --no-progress \
  "$canonical_prefix/preflight/prompt_grammar.parquet" \
  "$stage_root/preflight/prompt_grammar.parquet"

(
  cd "$stage_root"
  printf '%s\n' \
    'c09520428a35a1b8d15e03b7afec65039c925ed8556ec04ce509e2735dd1f3b3  preflight/prompt_grammar.json' \
    '3cbaa4ae48043fa1aa1932220717a02cc597c7bbfb8e750acd9b2b18301a6b04  preflight/prompt_grammar.parquet' |
    sha256sum --check --strict -
)

aws s3 cp --no-progress \
  "$snapshot_prefix/SHA256SUMS" \
  "$temporary_root/SHA256SUMS"
aws s3 cp --no-progress \
  "$snapshot_prefix/windows/exclusions.parquet" \
  "$temporary_root/windows/exclusions.parquet"
aws s3 cp --no-progress \
  "$snapshot_prefix/windows/mendelian.parquet" \
  "$temporary_root/windows/mendelian.parquet"

(
  cd "$temporary_root"
  printf '%s\n' \
    '200980cdc6d96dce810b3d5119d7e811fcd615b77437bb512b6a0f0de5bbafe5  SHA256SUMS' |
    sha256sum --check --strict -
  sha256sum --check --strict --ignore-missing SHA256SUMS
)

cp "$temporary_root/windows/exclusions.parquet" \
  "$stage_root/windows/exclusions.parquet"

uv run --project "$project_root" --locked carbon-conditioning-vep stage-windows \
  --config "$project_root/config/full_development.yaml" \
  --source "$temporary_root/windows/mendelian.parquet" \
  --output "$stage_root/windows/mendelian.parquet"

printf 'Staged full-development inputs at %s\n' "$stage_root"
