#!/usr/bin/env bash
# Detached auto-relauncher for exp303's two v6e spot arms. Resumes any arm that
# preempts (the `RuntimeError: step failed` signature → marin hard-fails, no
# auto-resume) until both wandb runs reach state=finished. Resume is free — the
# output path is unchanged, so a relaunch loads the latest levanter checkpoint.
# One-off operational helper for issue #303 — committed under scripts/ for
# reference/reproduction. The live instance ran from the worktree and writes its
# log + lock under the gitignored scratch/ at the worktree root.
set -u
cd /home/ubuntu/bolinas-dna/.claude/worktrees/lucid-hofstadter-11f30e || exit 1
LOG="scratch/exp303_relauncher.log"
LOCK="scratch/exp303_relauncher.lock"
PY=".venv/bin/python"
IRIS=".venv/bin/iris"

# single-instance guard
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "relauncher already running (pid $(cat "$LOCK"))"; exit 0
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

WK="$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')"
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

# tag|weight_decay|wandb_run_id
ARMS=( "wd0p3|0.3|dna-exp303-zoonomia-v1-0p25b-v4_cds-wd0p3-v0.1-6b925b"
       "wd0p5|0.5|dna-exp303-zoonomia-v1-0p25b-v4_cds-wd0p5-v0.1-144dd0" )

declare -A DONE RELA COOL
grace=$(( $(date +%s) + 1200 ))   # 20-min grace: the manual -r2 relaunches are coming up
for a in "${ARMS[@]}"; do t=${a%%|*}; DONE[$t]=0; RELA[$t]=0; COOL[$t]=$grace; done
CAP=20                            # per-arm relaunch cap (runaway backstop)
ENDBY=$(( $(date +%s) + 57600 ))  # 16h overall safety cap

state_step(){
  $PY - "$1" <<'PYEOF' 2>/dev/null
import sys, wandb
try:
    r = wandb.Api().run("gonzalobenegas/marin/" + sys.argv[1])
    print(r.state, int(r.summary.get("_step", -1) or -1))
except Exception:
    print("ERR", -1)
PYEOF
}

relaunch(){
  local t=$1 wd=$2 n=$(( ${RELA[$t]} + 1 )); RELA[$t]=$n
  log "RELAUNCH $t WD=$wd attempt#$n"
  timeout 300 "$IRIS" --cluster=marin job run --no-wait --user gonzalo \
    --job-name "exp303-cds-$t-v6e4-auto$n" --cpu 1 --memory 2g --extra marin --region us-east5 \
    -e WANDB_API_KEY "$WK" -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \
    -e WEIGHT_DECAY "$wd" -e TPU_TYPE v6e-4 -e PDP 1024 \
    -- python experiments/exp303_cds_weight_decay.py >> "$LOG" 2>&1
  COOL[$t]=$(( $(date +%s) + 1500 ))   # 25-min cooldown before this arm can relaunch again
}

log "relauncher up (pid $$) — covering wd0p3,wd0p5 until finished (cap=$CAP/arm, 16h)"
while true; do
  alldone=1
  for a in "${ARMS[@]}"; do
    IFS='|' read -r t wd rid <<< "$a"
    [ "${DONE[$t]}" = 1 ] && continue
    read -r st step < <(state_step "$rid")
    if [ "$st" = "finished" ]; then DONE[$t]=1; log "$t FINISHED (step=$step)"; continue; fi
    alldone=0
    now=$(date +%s)
    case "$st" in
      running|pending) log "$t ok state=$st step=$step" ;;
      failed|crashed|killed)
        if   [ "$now" -lt "${COOL[$t]}" ]; then log "$t state=$st step=$step in-cooldown $(( (${COOL[$t]}-now)/60 ))m";
        elif [ "${RELA[$t]}" -ge "$CAP" ]; then log "$t CAP=$CAP hit, giving up"; DONE[$t]=1;
        else relaunch "$t" "$wd"; fi ;;
      *) log "$t transient state=$st step=$step (skip, no relaunch)" ;;
    esac
  done
  [ "$alldone" = 1 ] && { log "ALL DONE"; break; }
  [ "$(date +%s)" -ge "$ENDBY" ] && { log "16h cap reached, exiting"; break; }
  sleep 600
done
log "relauncher exiting"
