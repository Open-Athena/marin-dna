#!/bin/bash
# issue #354 — GH200 region-cycling grabber for run.yaml.
#
# Lambda GH200 is capacity-scarce, and sky won't cycle regions on its own:
# `--retry-until-up` (and `any_of`) reuse the INIT cluster's ORIGINAL region.
# So we explicitly down-then-launch each region via `--infra lambda/<region>`,
# fail-fast, until one has capacity. We STOP the moment a GH200 *provisions*
# (cluster UP) — whether the benchmark then succeeds or fails — so the aarch64
# CUDA-torch step can be babysat rather than a scarce GPU discarded.
#
# Prereqs (dev box): `scripts/issue354/stage_genome.sh` run, HF token at
# ~/.cache/huggingface/token, GCS ADC mounted (see run.yaml). Run detached:
#   nohup bash scripts/issue354/grab_gh200.sh > grab.log 2>&1 &
# On success the cluster is left UP: fetch ~/out and `sky down dna-exp135-cost`.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
TOK=$(cat ~/.cache/huggingface/token)
CLUSTER=${CLUSTER:-dna-exp135-cost}
# Less-contended regions first (us-east-* is everyone's default).
REGIONS=(europe-central-1 europe-south-1 asia-south-1 me-west-1 \
         asia-northeast-1 asia-northeast-2 australia-east-1 \
         us-west-1 us-west-2 us-west-3 us-south-1 us-south-2 us-south-3 \
         us-midwest-1 us-east-2 us-east-3 us-east-1)

ts() { date -u +%H:%M:%S; }
c=0
while true; do
  c=$((c + 1))
  for r in "${REGIONS[@]}"; do
    echo "[$(ts)] cycle=$c region=$r: clearing stale record + launching"
    uv run sky down "$CLUSTER" -y >/dev/null 2>&1 || true
    uv run sky launch scripts/issue354/run.yaml -c "$CLUSTER" \
        --infra "lambda/$r" --env HF_TOKEN="$TOK" -y
    rc=$?
    if uv run sky status "$CLUSTER" 2>/dev/null \
         | grep -qE "$CLUSTER.*(UP|RUNNING)"; then
      echo "[$(ts)] PROVISIONED region=$r (launch rc=$rc) — cluster UP; STOPPING for babysit"
      exit 0
    fi
    echo "[$(ts)] cycle=$c region=$r: no capacity (rc=$rc)"
  done
  echo "[$(ts)] cycle=$c: no GH200 capacity in any region; sleep 120s"
  sleep 120
done
