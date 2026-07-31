#!/usr/bin/env bash
# Stream a completed issue #419 GH200 release through the AWS controller to S3.

set -euo pipefail

cluster=${1:-chinchilla-logo-gh200-full}
commit_sha=${2:-$(git rev-parse HEAD)}
bucket=${ISSUE419_RELEASE_BUCKET:-oa-bolinas}
prefix="snakemake/analysis/chinchilla_logo/issue419-full/runs/$commit_sha"

if [[ ! $commit_sha =~ ^[0-9a-f]{40}$ ]]; then
  echo "Expected a 40-character lowercase commit SHA, got: $commit_sha" >&2
  exit 2
fi

command -v aws >/dev/null
command -v sky >/dev/null
command -v ssh >/dev/null

# Refresh SkyPilot's generated SSH config, then resolve the remote home instead
# of assuming a provider-specific username or filesystem layout.
sky status "$cluster" >/dev/null
remote_home=$(ssh -o BatchMode=yes "$cluster" 'printf %s "$HOME"')
remote_root="$remote_home/.issue419/full-runs/$commit_sha"
relay_tmp=$(mktemp -d)
cleanup() {
  rm -rf "$relay_tmp"
}
trap cleanup EXIT

ssh -o BatchMode=yes "$cluster" "test -f '$remote_root/COMPLETE.json'"
ssh -o BatchMode=yes "$cluster" "cat '$remote_root/COMPLETE.json'" > "$relay_tmp/COMPLETE.json"
python - "$relay_tmp/COMPLETE.json" "$commit_sha" <<'PY'
import json
import pathlib
import sys

complete_path = pathlib.Path(sys.argv[1])
expected_commit = sys.argv[2]
payload = json.loads(complete_path.read_text())
assert payload == {
    "application_commit": expected_commit,
    "complete": True,
    "s3_relay": "pending",
}
PY

upload_tree() {
  local tree=$1
  local inventory="$relay_tmp/$tree-inventory.tsv"

  ssh -o BatchMode=yes "$cluster" \
    "find '$remote_root/$tree' -type f -printf '%P\\t%s\\n' | sort" > "$inventory"
  test -s "$inventory"

  while IFS=$'\t' read -r relative_path expected_bytes; do
    if [[ ! $relative_path =~ ^[A-Za-z0-9._/-]+$ ]] ||
      [[ $relative_path == /* ]] || [[ $relative_path == *..* ]]; then
      echo "Unsafe remote path in $tree inventory: $relative_path" >&2
      exit 3
    fi
    [[ $expected_bytes =~ ^[0-9]+$ ]]

    remote_path="$remote_root/$tree/$relative_path"
    object_key="$prefix/$tree/$relative_path"
    ssh -o BatchMode=yes "$cluster" "cat '$remote_path'" |
      aws s3 cp - "s3://$bucket/$object_key" \
        --expected-size "$expected_bytes" --only-show-errors

    uploaded_bytes=$(aws s3api head-object \
      --bucket "$bucket" --key "$object_key" \
      --query ContentLength --output text)
    test "$uploaded_bytes" = "$expected_bytes"
  done < "$inventory"

  aws s3 cp "$inventory" "s3://$bucket/$prefix/$tree-inventory.tsv" \
    --content-type text/tab-separated-values --only-show-errors
}

# Upload artifacts before the completion marker. Consumers can therefore use
# COMPLETE.json as an atomic indication that every advertised object exists.
upload_tree release
upload_tree plans
upload_tree logs

python - "$relay_tmp/COMPLETE.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["s3_relay"] = "complete"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
aws s3 cp "$relay_tmp/COMPLETE.json" "s3://$bucket/$prefix/COMPLETE.json" \
  --content-type application/json --only-show-errors

echo "Verified release relay: s3://$bucket/$prefix/"
