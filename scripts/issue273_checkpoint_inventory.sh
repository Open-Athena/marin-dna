#!/usr/bin/env bash
# Inventory of native (Levanter) vs HF-exported checkpoint steps per training run,
# across the marin GCS buckets (project hai-gcp-models). Backs the 2026-07-07 update
# to issue #273 (checkpoint inventory).
#
# For each run dir it prints the step-N present under:
#   checkpoints/step-N/  ← native Levanter (OCDBT); resume-capable, NOT AutoModel-loadable
#   hf/step-N/           ← HF export (config.json + model.safetensors); what evals_v2 consumes
#
# Key finding this reproduces: "no hf/" almost always means "no weights at all", not
# "native weights waiting to be exported" — Levanter prunes the native store to the last
# few steps, and dead-end runs were cleaned (their checkpoints/ holds only eval_metrics.jsonl).
# So for exported runs HF is typically the superset of native; the genuine Levanter-only
# export-wins are a thin set (see #273).
#
# Usage: scripts/issue273_checkpoint_inventory.sh
set -uo pipefail

EAST5=gs://marin-us-east5/checkpoints
CENTRAL1=gs://marin-us-central1/checkpoints

# Compact, sorted list of step-N dirs under a store (native or hf); empty if none.
# `gcloud storage ls` returns dir lines ending in `/`, so extract the step-N token
# directly rather than stripping to the last path segment (which would be empty).
steps() {
  gcloud storage ls "$1" 2>/dev/null | grep -oE 'step-[0-9]+' | sort -V -u | tr '\n' ' ' || true
}

inventory() {  # $1 = full run dir (no trailing slash)
  local d="${1%/}"
  printf '%-62s\n    native: %s\n    hf:     %s\n' \
    "$(basename "$d")" "$(steps "$d/checkpoints/")" "$(steps "$d/hf/")"
}

sweep() {  # $1 = "gs://bucket/checkpoints" prefix, $2 = grep pattern
  local prefix="$1" pat="$2" d
  for d in $(gcloud storage ls "$prefix/" 2>/dev/null | grep -E "$pat" || true); do
    inventory "$d"
  done
}

echo "### mixture sweep — mix-v0.9 lineage (blog Fig 9/10)"
sweep "$EAST5" 'dna-bolinas-mix-v0\.9-p1B-i'

echo "### parameter-scaling ladder — scaling-v0.5 (blog Fig 4-8)"
sweep "$EAST5" 'dna-bolinas-scaling-v0\.5-'

echo "### hyperparameter transfer + Vizier reference (blog Fig 1-3)"
sweep "$CENTRAL1" 'dna-bolinas-transfer-v0\.12|adamhr-3e18-nemotron-vizier'
