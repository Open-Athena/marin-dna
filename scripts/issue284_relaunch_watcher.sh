#!/usr/bin/env bash
# Detached relaunch-watcher for exp284 TSS epoch-reshuffle (#284, PR #285).
#
# v6e is preemptible; marin propagates a GCS-side TPU preemption as
# "RuntimeError: N step(s) failed" (coordinator `failed`, preemptions=0) and does
# NOT auto-resume (see snakemake/.../experiments README "Lessons learned"). This
# loop relaunches on `failed` until training completes (`succeeded`) or
# MAX_RELAUNCHES. Every relaunch RESUMES from exp284's last checkpoint — resources
# / TPU-type / PDP / ram are NOT in the marin output-path hash, and the slice-mix
# data stream is deterministic (SliceMixLmDataConfig: fixed slice seed + trainer
# seed) — so resume is data-consistent. Backs off to ~hourly probes during
# sustained no-progress contention; snaps back the moment a window opens (the
# wandb step advances past the resume point).
#
# Run DETACHED so it survives SSH disconnect / session idle:
#   nohup bash scripts/issue284_relaunch_watcher.sh >> scratch/exp284_watcher.log 2>&1 &
set -u

WORKTREE=/home/ubuntu/bolinas-dna/.claude/worktrees/flamboyant-feistel-bc4aa7
cd "$WORKTREE" || { echo "FATAL: cannot cd $WORKTREE"; exit 1; }
export PATH="/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin:/snap/bin:$PATH"

NAME_BASE="${WATCH_NAME_BASE:-exp284-tss-reshuffle-v6e4}"
REGION="${WATCH_REGION:-europe-west4}"          # picks the marin data bucket + zone (reuse exp255's eu caches)
START_SEQ="${WATCH_START_SEQ:-4}"               # run-3 was the last manual launch; resume-relaunches start at -4
MAX_RELAUNCHES="${WATCH_MAX_RELAUNCHES:-20}"
POLL_SECS="${WATCH_POLL_SECS:-300}"
BACKOFF_AFTER="${WATCH_BACKOFF_AFTER:-3}"       # no-progress relaunches before backing off
BACKOFF_SECS="${WATCH_BACKOFF_SECS:-2700}"      # ~45min probe interval during saturation
PROGRESS_MARGIN=100                             # wandb-step advance that counts as "made progress"
export WANDB_API_KEY="$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')"

log(){ echo "[$(TZ=America/New_York date '+%m-%d %H:%M:%S') NY] $*"; }
irisc(){ timeout 150 uv run --no-sync iris --cluster=marin "$@" 2>&1; }

coord_state(){ irisc job summary "$1" | grep -oE 'State: [a-z]+' | awk '{print $2}' | head -1; }

wandb_step(){
  timeout 120 uv run --no-sync python - <<'PY' 2>/dev/null || echo 0
import wandb
api = wandb.Api(); ent = api.default_entity
runs = list(api.runs(f"{ent}/marin", filters={"group": "dna-exp284-v0.1"}))
running = [r for r in runs if r.state == "running"] or runs
if not running:
    print(0); raise SystemExit
r = sorted(running, key=lambda x: x.created_at or "", reverse=True)[0]
print(int(r.summary.get("_step", 0) or 0))
PY
}

NO_PROGRESS=0
LAST_STEP=0
for SEQ in $(seq "$START_SEQ" "$((START_SEQ + MAX_RELAUNCHES))"); do
  JOB="/gonzalo/${NAME_BASE}-${SEQ}"
  log "launching attempt ${SEQ}: ${JOB} (region=${REGION})"
  irisc job run --no-wait --user gonzalo --job-name "${NAME_BASE}-${SEQ}" \
    --cpu 1 --memory 2g --extra marin --region "$REGION" \
    -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \
    -e TPU_TYPE v6e-4 -e PDP 1024 -e TPU_RAM 300g \
    -- python experiments/exp284_tss_reshuffle.py | tail -2

  # Poll this attempt until it reaches a terminal state.
  while true; do
    sleep "$POLL_SECS"
    ST="$(coord_state "$JOB")"
    log "${JOB} state=${ST:-<none>}"
    case "$ST" in
      succeeded|completed) log "DONE — exp284 training completed."; exit 0 ;;
      failed) log "attempt ${SEQ} failed (preemption) -> relaunch+resume"; break ;;
      *) : ;;  # running / pending / transient empty -> keep polling
    esac
  done

  # Progress-aware backoff: only churn fast relaunches when we're actually
  # banking steps; if a saturated pool keeps preempting before any checkpoint
  # past the resume point, probe ~hourly instead.
  STEP="$(wandb_step)"; STEP="${STEP:-0}"
  if [ "$STEP" -gt "$((LAST_STEP + PROGRESS_MARGIN))" ]; then
    log "progress: wandb step ${LAST_STEP} -> ${STEP}; prompt relaunch"
    NO_PROGRESS=0; LAST_STEP="$STEP"
  else
    NO_PROGRESS=$((NO_PROGRESS + 1))
    log "no durable progress (step ~${STEP}, streak ${NO_PROGRESS})"
  fi
  if [ "$NO_PROGRESS" -ge "$BACKOFF_AFTER" ]; then
    log "sustained contention -> backoff ${BACKOFF_SECS}s before next probe"
    sleep "$BACKOFF_SECS"
  fi
  sleep 20
done
log "hit MAX_RELAUNCHES (${MAX_RELAUNCHES}); stopping — check capacity / re-route manually."
