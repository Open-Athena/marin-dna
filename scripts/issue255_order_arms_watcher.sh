#!/usr/bin/env bash
# Detached relaunch-watcher for the 3 new exp255 order arms (#255) on
# europe-west4 v6e-4. Relaunches an arm on coordinator failure — SIGSEGV
# (transient libtpu/JAX crash, exit 139) and GCS preemption BOTH surface as
# `State: failed, preemptions=0` — resuming from its last 10-min checkpoint
# (resource/TPU-type/region are NOT in the marin output-path hash, so a
# same-SWEEP_DATASETS relaunch loads the existing checkpoint). Exits when all 3
# arms reach step 5000.
#
# Run DETACHED so it survives SSH disconnect / Mac sleep / session idle (the
# lesson from this experiment — a backgrounded Bash probe dies over a long idle):
#   nohup bash scripts/issue255_order_arms_watcher.sh >> scratch/exp255_order_arms_watcher.log 2>&1 &
set -u

WORKTREE=/home/ubuntu/bolinas-dna/.claude/worktrees/infallible-blackwell-4c774c
cd "$WORKTREE" || { echo "FATAL: cannot cd $WORKTREE"; exit 1; }
export PATH="/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin:/snap/bin:$PATH"
WKEY="$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')"

POLL_SECS="${WATCH_POLL_SECS:-600}"
TARGET_STEP=5000
MAX_RELAUNCHES="${WATCH_MAX_RELAUNCHES:-25}"
RELAUNCHES=0

ARMS=(utr3 ncrna tss)
declare -A SWEEP=( [utr3]=v4_utr3_order [ncrna]=v4_ncrna_exon_order [tss]=v4_tss_region_and_utr5_order )
# current job per arm at watcher start (ncrna already on its -r2 relaunch)
declare -A JOB=(
  [utr3]=/gonzalo/exp255-utr3-order-eu-v6e4
  [ncrna]=/gonzalo/exp255-ncrna-order-eu-v6e4-r2
  [tss]=/gonzalo/exp255-tss-order-eu-v6e4
)
declare -A SEQ=( [utr3]=1 [ncrna]=2 [tss]=1 )
declare -A DONE=( [utr3]=0 [ncrna]=0 [tss]=0 )

log(){ echo "[$(TZ=America/New_York date '+%m-%d %H:%M:%S') NY] $*"; }
irisc(){ timeout 150 uv run --no-sync iris --cluster=marin "$@" 2>&1; }
coord_state(){ irisc job summary "$1" | grep -oE 'State: [a-z]+' | awk '{print $2}' | head -1; }

arm_step(){  # $1 = sweep key -> max-step wandb run's _step
  timeout 120 uv run --no-sync python - "$1" <<'PY' 2>/dev/null
import sys, wandb
key = sys.argv[1]
api = wandb.Api(); ent = api.default_entity
rs = [r for r in api.runs(f"{ent}/marin", filters={"group": "dna-exp255-v0.1"}) if key in r.name]
print(int(max(rs, key=lambda x: dict(x.summary).get("_step") or 0).summary.get("_step") or 0) if rs else 0)
PY
}

relaunch(){  # $1 = arm short name
  local a="$1"
  SEQ[$a]=$(( ${SEQ[$a]} + 1 )); RELAUNCHES=$(( RELAUNCHES + 1 ))
  local nj="exp255-${a}-order-eu-v6e4-r${SEQ[$a]}"
  log ">>> RELAUNCH #${RELAUNCHES}: ${a} -> ${nj} (resume from last checkpoint)"
  irisc job run --no-wait --user gonzalo --job-name "$nj" \
    --cpu 1 --memory 2g --extra marin --region europe-west4 \
    -e WANDB_API_KEY "$WKEY" -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \
    -e SWEEP_DATASETS "${SWEEP[$a]}" -e TPU_TYPE v6e-4 -e PDP 1024 -e TPU_RAM 300g \
    -- python experiments/exp255_per_region_order.py | tail -1
  JOB[$a]="/gonzalo/${nj}"
}

log "=== watcher START — arms ${ARMS[*]}; poll ${POLL_SECS}s; target ${TARGET_STEP}; cap ${MAX_RELAUNCHES} ==="
while true; do
  alldone=1
  for a in "${ARMS[@]}"; do
    [ "${DONE[$a]}" = "1" ] && continue
    alldone=0
    cs="$(coord_state "${JOB[$a]}")"; step="$(arm_step "${SWEEP[$a]}")"
    log "poll ${a}: coord=${cs:-UNREACHABLE} step=${step}/${TARGET_STEP} job=${JOB[$a]##*/} relaunches=${RELAUNCHES}/${MAX_RELAUNCHES}"
    if { [ "${step:-0}" -ge "$TARGET_STEP" ] 2>/dev/null; } || [ "${cs:-}" = "succeeded" ]; then
      log "=== ${a} DONE (step ${step}, coord ${cs:-?}) ==="; DONE[$a]=1; continue
    fi
    if [ "${cs:-}" = "failed" ]; then
      if [ "$RELAUNCHES" -ge "$MAX_RELAUNCHES" ]; then
        log "=== STOP: hit relaunch cap ${MAX_RELAUNCHES} — needs a manual look. Exiting. ==="; break 2
      fi
      relaunch "$a"; sleep 90
    fi
  done
  [ "$alldone" = "1" ] && { log "=== ALL ${#ARMS[@]} ARMS DONE. Watcher exiting. ==="; break; }
  sleep "$POLL_SECS"
done
log "watcher exit."
