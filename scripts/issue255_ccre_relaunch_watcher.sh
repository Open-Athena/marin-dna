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
REGION="${WATCH_REGION:-us-east5}"          # --region for relaunches (picks the data bucket + zone)
NAME_BASE="${WATCH_NAME_BASE:-exp255-ccre_non_promoter_order}"
# Host RAM. 40g schedules under contention but is TOO SMALL for the 0.25B
# training: it OOM-kills (exit 137) around step ~141 (the original/cds ran at
# 300g for all 5000 steps). 40g only "worked" on us-east5 because preemption hit
# each attempt before it grew past 40g. Use 300g where it fits (roomy pools).
TPU_RAM_W="${WATCH_TPU_RAM:-300g}"
POOL_TAG="${POOL//-/}"                      # v6e-4 -> v6e4 (job-name suffix)
JOB="/gonzalo/${NAME_BASE}-${POOL_TAG}-r${RUN_SEQ}"
POLL_SECS=900
STUCK_POLLS="${WATCH_STUCK_POLLS:-2}"       # consecutive polls of a pending TPU task before switching pool (~30min)
MAX_RELAUNCHES="${WATCH_MAX_RELAUNCHES:-50}"
RELAUNCHES=0
PENDING_POLLS=0
# Backoff: during sustained contention (preempted faster than the ~15-20min
# startup, so no durable progress), don't churn a relaunch every ~15min — after
# BACKOFF_AFTER no-progress relaunches, wait BACKOFF_SECS between probe-relaunches
# (~hourly). Snaps back to prompt relaunching the moment a window opens (the ccre
# step jumps by >PROGRESS_MARGIN, i.e. a checkpoint past the resume point).
BACKOFF_AFTER="${WATCH_BACKOFF_AFTER:-3}"
BACKOFF_SECS="${WATCH_BACKOFF_SECS:-2700}"   # 45min probe interval
PROGRESS_MARGIN=150
NO_PROGRESS=0
LAST_PROGRESS_STEP=2500
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
cands = running or runs
if not cands:
    print(0); raise SystemExit
try:                                  # prefer the newest current attempt (avoids stale runs after a region switch)
    r = sorted(cands, key=lambda x: x.created_at or "", reverse=True)[0]
except Exception:
    r = max(cands, key=lambda x: dict(x.summary).get("_step") or 0)
print(int(dict(r.summary).get("_step") or 0))
PY
}

relaunch(){                                 # $1 = target pool, $2 = reason
  POOL="$1"; POOL_TAG="${POOL//-/}"
  RUN_SEQ=$((RUN_SEQ + 1)); RELAUNCHES=$((RELAUNCHES + 1)); PENDING_POLLS=0
  local nj="${NAME_BASE}-${POOL_TAG}-r${RUN_SEQ}"
  log ">>> RELAUNCH #${RELAUNCHES} [${2}]: ${nj} (${POOL} @ ${REGION} / PDP=1024 / RAM=${TPU_RAM_W}; resumes from last checkpoint)"
  irisc job run --no-wait --user gonzalo \
    --job-name "$nj" --cpu 1 --memory 2g --extra marin --region "$REGION" \
    -e WANDB_API_KEY "$WKEY" -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \
    -e SWEEP_DATASETS "$ARM" -e TPU_TYPE "$POOL" -e PDP 1024 -e TPU_RAM "$TPU_RAM_W" \
    -- python experiments/exp255_per_region_order.py | tail -2
  JOB="/gonzalo/${nj}"
}

LAST_PROGRESS_STEP="${WATCH_RESUME_STEP:-$(ccre_step)}"; case "$LAST_PROGRESS_STEP" in ''|*[!0-9]*) LAST_PROGRESS_STEP=0;; esac
log "=== watcher START — watching ${JOB} (pool ${POOL}); poll ${POLL_SECS}s; cap ${MAX_RELAUNCHES}; stuck-switch ${STUCK_POLLS} polls; backoff ${BACKOFF_SECS}s after ${BACKOFF_AFTER} no-progress relaunches (from step ${LAST_PROGRESS_STEP}) ==="
while true; do
  cs="$(coord_state)"; ts="$(task_state)"; step="$(ccre_step)"
  if [ "${step:-0}" -gt "$((LAST_PROGRESS_STEP + PROGRESS_MARGIN))" ] 2>/dev/null; then
    LAST_PROGRESS_STEP="$step"; NO_PROGRESS=0          # a window opened — durable progress, resume normal cadence
  fi
  log "poll: coord=${cs:-UNREACHABLE} task=${ts} step=${step} pool=${POOL} pending=${PENDING_POLLS} no_progress=${NO_PROGRESS} relaunches=${RELAUNCHES}/${MAX_RELAUNCHES}"

  if { [ "${step:-0}" -ge 5000 ] 2>/dev/null; } || [ "${cs:-}" = "succeeded" ]; then
    log "=== DONE: ccre at step ${step} (coordinator ${cs:-?}). Watcher exiting. ==="; break
  fi
  if [ "$RELAUNCHES" -ge "$MAX_RELAUNCHES" ]; then
    log "=== STOP: hit relaunch cap ${MAX_RELAUNCHES} without finishing — needs a manual look. Exiting. ==="; break
  fi

  if [ "${cs:-}" = "failed" ]; then
    NO_PROGRESS=$((NO_PROGRESS + 1))
    if [ "$NO_PROGRESS" -ge "$BACKOFF_AFTER" ]; then
      log "sustained contention: ${NO_PROGRESS} relaunches, no durable progress past step ${LAST_PROGRESS_STEP} — backing off ${BACKOFF_SECS}s before next probe (nothing running meanwhile)"
      sleep "$BACKOFF_SECS"
    fi
    relaunch "$POOL" "preempt/probe"; sleep 120; continue
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
