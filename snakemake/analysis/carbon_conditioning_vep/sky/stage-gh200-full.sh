#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stage_root="$project_root/results/gh200-full-inputs"
temporary_root="$(mktemp -d)"
trap 'rm -rf -- "$temporary_root"' EXIT

artifact_prefix="s3://oa-bolinas/snakemake/analysis/carbon_conditioning_vep/results"
mkdir -p "$stage_root/preflight" "$stage_root/windows"

aws s3 cp --no-progress \
  "$artifact_prefix/preflight/prompt_grammar.json" \
  "$stage_root/preflight/prompt_grammar.json"
aws s3 cp --no-progress \
  "$artifact_prefix/preflight/prompt_grammar.parquet" \
  "$stage_root/preflight/prompt_grammar.parquet"
aws s3 cp --no-progress \
  "$artifact_prefix/windows/exclusions.parquet" \
  "$stage_root/windows/exclusions.parquet"
aws s3 cp --no-progress \
  "$artifact_prefix/windows/mendelian.parquet" \
  "$temporary_root/mendelian.parquet"

uv run --project "$project_root" --locked carbon-conditioning-vep stage-windows \
  --config "$project_root/config/full_development.yaml" \
  --source "$temporary_root/mendelian.parquet" \
  --output "$stage_root/windows/mendelian.parquet"

printf 'Staged full-development inputs at %s\n' "$stage_root"
