#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 CLUSTER JOB_ID COMMIT_SHA [--down]" >&2
  exit 2
fi

cluster=$1
job_id=$2
commit_sha=$3
down_after_relay=${4:-}
poll_seconds=${ISSUE419_POLL_SECONDS:-300}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ -n $down_after_relay && $down_after_relay != --down ]]; then
  echo "Fourth argument must be --down when provided" >&2
  exit 2
fi
[[ $job_id =~ ^[0-9]+$ ]]
[[ $commit_sha =~ ^[0-9a-f]{40}$ ]]
[[ $poll_seconds =~ ^[1-9][0-9]*$ ]]

while true; do
  if status_output=$(sky logs "$cluster" "$job_id" --status 2>&1); then
    printf '%s\n' "$status_output"
    break
  else
    status_code=$?
    printf '%s\n' "$status_output"
    if [[ $status_code -ne 101 ]]; then
      echo "Sky job did not complete successfully (exit $status_code); preserving cluster" >&2
      exit "$status_code"
    fi
  fi
  sleep "$poll_seconds"
done

"$script_dir/issue419_relay_gh200_release.sh" "$cluster" "$commit_sha"

if [[ $down_after_relay == --down ]]; then
  sky down -y "$cluster"
fi
