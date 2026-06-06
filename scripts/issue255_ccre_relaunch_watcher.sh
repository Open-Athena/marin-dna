#!/usr/bin/env bash
# Detached relaunch-watcher for exp255 ccre_order (issue #255).
#
# Polls the ccre coordinator job every POLL_SECS. If it has `failed` (the
# GCS-side v6e preemption signature: failed + preemptions=0), relaunch it on
# v6e-8 / PDP=1024 / TPU_RAM=40g -> it resumes from its last 500-step checkpoint
# (resources/TPU-type/ram/PDP are NOT in the marin executor output-path hash).
# Exits cleanly when ccre reaches step 5000 (coordinator `succeeded`) or after
# MAX_RELAUNCHES (safety cap, so it can't churn forever).
#
# Run DETACHED so it survives SSH disconnect / Mac sleep / desktop-app close
# (a `run_in_background` Bash task is session-scoped and gets reaped on idle):
#   nohup bash scripts/issue255_ccre_relaunch_watcher.sh \
#       > scratch/exp255_ccre_watcher.log 2>&1 &
set -u

WORKTREE=/home/ubuntu/bolinas-dna/.claude/worktrees/infallible-blackwell-4c774c
cd "$WORKTREE" || { echo "FATAL: cannot cd $WORKTREE"; exit 1; }
export PATH="/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin:/snap/bin:$PATH"

ARM=v4_ccre_non_promoter_order
# Pool is env-overridable: v6e-8 us-east5-b went memory-saturated on 2026-06-06
# morning (train task stuck pending "Insufficient memory, available 1.3GB"), so
# we re-routed to v6e-4 (roomy: cds + the original ccre ran there). Defaults below
# target v6e-4; override WATCH_TPU_TYPE / WATCH_JOB_BASE / WATCH_PDP to switch.
JOB_BASE="${WATCH_JOB_BASE:-exp255-ccre_non_promoter_order-v6e4}"
TPU_TYPE_W="${WATCH_TPU_TYPE:-v6e-4}"
PDP_W="${WATCH_PDP:-1024}"
RUN_SEQ="${WATCH_START_SEQ:-7}"             # which -rN to start watching (env-overridable for restarts)
JOB="/gonzalo/${JOB_BASE}-r${RUN_SEQ}"
POLL_SECS=900                               # 15 min: catch a preemption promptly (<=1 checkpoint of loss)
MAX_RELAUNCHES=10
RELAUNCHES=0
WKEY="$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')"

log(){ echo "[$(TZ=America/New_York date '+%m-%d %H:%M:%S') NY] $*"; }
irisc(){ timeout 150 uv run --no-sync iris --cluster=marin "$@" 2>&1; }

ccre_state(){ irisc job summary "$JOB" | grep -oE 'State: [a-z]+' | awk '{print $2}' | head -1; }

ccre_wandb(){                                # prints "<state> <step>"
  timeout 150 uv run --no-sync python - <<'PY' 2>/dev/null
import wandb
api = wandb.Api(); ent = api.default_entity
runs = [r for r in api.runs(f"{ent}/marin", filters={"group": "dna-exp255-v0.1"}) if "ccre" in r.name]
if not runs:
    print("none 0"); raise SystemExit
running = [r for r in runs if r.state == "running"]
r = max(running or runs, key=lambda x: dict(x.summary).get("_step") or 0)
print(r.state, int(dict(r.summary).get("_step") or 0))
PY
}

relaunch(){
  RUN_SEQ=$((RUN_SEQ + 1)); RELAUNCHES=$((RELAUNCHES + 1))
  local nj="${JOB_BASE}-r${RUN_SEQ}"
  log ">>> RELAUNCH #${RELAUNCHES}: ${nj} (${TPU_TYPE_W} / TPU_RAM=40g; resumes from last checkpoint)"
  irisc job run --no-wait --user gonzalo \
    --job-name "$nj" --cpu 1 --memory 2g --extra marin --region us-east5 \
    -e WANDB_API_KEY "$WKEY" -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \
    -e SWEEP_DATASETS "$ARM" -e TPU_TYPE "$TPU_TYPE_W" -e PDP "$PDP_W" -e TPU_RAM 40g \
    -- python experiments/exp255_per_region_order.py | tail -2
  JOB="/gonzalo/${nj}"
}

log "=== watcher START — watching ${JOB}; poll ${POLL_SECS}s; relaunch cap ${MAX_RELAUNCHES} ==="
while true; do
  st="$(ccre_state)"
  read -r ws wstep < <(ccre_wandb)
  log "poll: iris=${st:-UNREACHABLE} wandb=${ws:-NA}@${wstep:-NA} relaunches=${RELAUNCHES}/${MAX_RELAUNCHES}"

  if { [ -n "${wstep:-}" ] && [ "${wstep:-0}" -ge 5000 ] 2>/dev/null; } || [ "${st:-}" = "succeeded" ]; then
    log "=== DONE: ccre at step ${wstep:-?} (coordinator ${st:-?}). Watcher exiting. ==="
    break
  fi
  if [ "${st:-}" = "failed" ]; then
    if [ "$RELAUNCHES" -ge "$MAX_RELAUNCHES" ]; then
      log "=== STOP: hit relaunch cap ${MAX_RELAUNCHES}; ccre keeps failing — needs a manual look. Exiting. ==="
      break
    fi
    relaunch
    sleep 120        # let the new coordinator boot before the next poll (avoids a double-relaunch)
    continue
  fi
  sleep "$POLL_SECS"
done
