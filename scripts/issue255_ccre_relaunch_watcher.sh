#!/usr/bin/env bash
# Detached, contention-hardened relaunch-watcher for exp255 ccre_order (#255).
#
# ccre_order keeps getting bounced by the busy-hours us-east5 cluster. This loop
# polls the current ccre coordinator every POLL_SECS and recovers from BOTH
# failure modes we've observed:
#   (1) coordinator `failed`  (GCS v6e preemption: RuntimeError, preemptions=0)
#         -> relaunch on the SAME pool (bad luck, not a pool problem).
#   (2) coordinator `running` but the TPU task stuck `pending` for >=STUCK_POLLS
#       polls (capacity / "Insufficient memory" on a saturated pool)
#         -> stop + relaunch on the OTHER pool (v6e-4 <-> v6e-8).
# Every relaunch resumes from ccre's last checkpoint (resources/TPU-type/PDP/ram
# are NOT in the marin output-path hash). Exits at step 5000 (coordinator
# `succeeded`) or after MAX_RELAUNCHES (safety cap). The point is to ride through
# the contended day and auto-finish ccre whenever a long-enough window opens
# (the original arm got its 2500 steps overnight). Effective batch is unchanged
# across pools (8192); PDP=1024 is correct on both (grad-accum on v6e-4, no-op on
# v6e-8).
#
# Run DETACHED so it survives SSH disconnect / Mac sleep / app close:
#   nohup bash scripts/issue255_ccre_relaunch_watcher.sh >> scratch/exp255_ccre_watcher.log 2>&1 &
set -u

WORKTREE=/home/ubuntu/bolinas-dna/.claude/worktrees/infallible-blackwell-4c774c
cd "$WORKTREE" || { echo "FATAL: cannot cd $WORKTREE"; exit 1; }
export PATH="/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin:/snap/bin:$PATH"

ARM=v4_ccre_non_promoter_order
POOL="${WATCH_POOL:-v6e-4}"                 # pool the current job is on
RUN_SEQ="${WATCH_START_SEQ:-8}"             # global relaunch counter / which -rN to watch
POOL_TAG="${POOL//-/}"                      # v6e-4 -> v6e4 (job-name suffix)
JOB="/gonzalo/exp255-ccre_non_promoter_order-${POOL_TAG}-r${RUN_SEQ}"
POLL_SECS=900
STUCK_POLLS="${WATCH_STUCK_POLLS:-2}"       # consecutive polls of a pending TPU task before switching pool (~30min)
MAX_RELAUNCHES="${WATCH_MAX_RELAUNCHES:-50}"
RELAUNCHES=0
PENDING_POLLS=0
WKEY="$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')"

log(){ echo "[$(TZ=America/New_York date '+%m-%d %H:%M:%S') NY] $*"; }
irisc(){ timeout 150 uv run --no-sync iris --cluster=marin "$@" 2>&1; }
other_pool(){ [ "$1" = "v6e-4" ] && echo "v6e-8" || echo "v6e-4"; }

coord_state(){ irisc job summary "$JOB" | grep -oE 'State: [a-z]+' | awk '{print $2}' | head -1; }

task_state(){                               # prints running / pending / none
  local j="${JOB#/gonzalo/}" out
  out=$(for s in running pending; do irisc job list --state "$s" | grep -F "${j}/checkpoints" | grep -oE '(running|pending)'; done | head -1)
  echo "${out:-none}"
}

ccre_step(){
  timeout 120 uv run --no-sync python - <<'PY' 2>/dev/null
import wandb
api = wandb.Api(); ent = api.default_entity
runs = [r for r in api.runs(f"{ent}/marin", filters={"group": "dna-exp255-v0.1"}) if "ccre" in r.name]
running = [r for r in runs if r.state == "running"]
r = max(running or runs, key=lambda x: dict(x.summary).get("_step") or 0) if runs else None
print(int(dict(r.summary).get("_step") or 0) if r else 0)
PY
}

relaunch(){                                 # $1 = target pool, $2 = reason
  POOL="$1"; POOL_TAG="${POOL//-/}"
  RUN_SEQ=$((RUN_SEQ + 1)); RELAUNCHES=$((RELAUNCHES + 1)); PENDING_POLLS=0
  local nj="exp255-ccre_non_promoter_order-${POOL_TAG}-r${RUN_SEQ}"
  log ">>> RELAUNCH #${RELAUNCHES} [${2}]: ${nj} (${POOL} / PDP=1024 / RAM=40g; resumes from last checkpoint)"
  irisc job run --no-wait --user gonzalo \
    --job-name "$nj" --cpu 1 --memory 2g --extra marin --region us-east5 \
    -e WANDB_API_KEY "$WKEY" -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \
    -e SWEEP_DATASETS "$ARM" -e TPU_TYPE "$POOL" -e PDP 1024 -e TPU_RAM 40g \
    -- python experiments/exp255_per_region_order.py | tail -2
  JOB="/gonzalo/${nj}"
}

log "=== watcher START — watching ${JOB} (pool ${POOL}); poll ${POLL_SECS}s; cap ${MAX_RELAUNCHES}; stuck-switch after ${STUCK_POLLS} polls ==="
while true; do
  cs="$(coord_state)"; ts="$(task_state)"; step="$(ccre_step)"
  log "poll: coord=${cs:-UNREACHABLE} task=${ts} step=${step} pool=${POOL} pending_polls=${PENDING_POLLS} relaunches=${RELAUNCHES}/${MAX_RELAUNCHES}"

  if { [ "${step:-0}" -ge 5000 ] 2>/dev/null; } || [ "${cs:-}" = "succeeded" ]; then
    log "=== DONE: ccre at step ${step} (coordinator ${cs:-?}). Watcher exiting. ==="; break
  fi
  if [ "$RELAUNCHES" -ge "$MAX_RELAUNCHES" ]; then
    log "=== STOP: hit relaunch cap ${MAX_RELAUNCHES} without finishing — needs a manual look. Exiting. ==="; break
  fi

  if [ "${cs:-}" = "failed" ]; then
    relaunch "$POOL" "preempted"; sleep 120; continue
  fi
  if [ "$ts" = "running" ]; then
    PENDING_POLLS=0
  elif [ "$ts" = "pending" ]; then
    PENDING_POLLS=$((PENDING_POLLS + 1))
    if [ "$PENDING_POLLS" -ge "$STUCK_POLLS" ]; then
      log "TPU task stuck pending ${PENDING_POLLS} polls on ${POOL} -> stop + switch pool"
      irisc job stop "$JOB" >/dev/null 2>&1
      relaunch "$(other_pool "$POOL")" "stuck-pending:switch-pool"; sleep 120; continue
    fi
  fi
  # ts=none (booting) or coord pending/unreachable: just wait
  sleep "$POLL_SECS"
done
log "watcher exit."
