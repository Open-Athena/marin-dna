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

upload_release() {
  local inventory="$relay_tmp/release-inventory.tsv"

  ssh -o BatchMode=yes "$cluster" \
    "find '$remote_root/release' -type f -printf '%P\\t%s\\n' | sort" > "$inventory"
  test -s "$inventory"

  while IFS=$'\t' read -r relative_path expected_bytes; do
    if [[ ! $relative_path =~ ^[A-Za-z0-9._/-]+$ ]] ||
      [[ $relative_path == /* ]] || [[ $relative_path == *..* ]]; then
      echo "Unsafe remote path in release inventory: $relative_path" >&2
      exit 3
    fi
    [[ $expected_bytes =~ ^[0-9]+$ ]]

    case $relative_path in
      *.html) content_type=text/html ;;
      *.json) content_type=application/json ;;
      *.txt) content_type=text/plain ;;
      *) content_type=application/octet-stream ;;
    esac

    remote_path="$remote_root/release/$relative_path"
    object_key="$prefix/release/$relative_path"
    ssh -o BatchMode=yes "$cluster" "cat '$remote_path'" |
      aws s3 cp - "s3://$bucket/$object_key" \
        --content-type "$content_type" --expected-size "$expected_bytes" \
        --only-show-errors

    uploaded_bytes=$(aws s3api head-object \
      --bucket "$bucket" --key "$object_key" \
      --query ContentLength --output text)
    test "$uploaded_bytes" = "$expected_bytes"
  done < "$inventory"

  aws s3 cp "$inventory" "s3://$bucket/$prefix/release-inventory.tsv" \
    --content-type text/tab-separated-values --only-show-errors
}

upload_archive() {
  local tree=$1
  local archive="$tree.tar.gz"
  local remote_archive="$remote_root/$archive"

  # Thousands of small plan files would make one SSH connection per object
  # prohibitively slow. Build one deterministic archive on the GH200 node,
  # then stream and verify it without staging it on the controller.
  ssh -o BatchMode=yes "$cluster" \
    "tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner -C '$remote_root' -czf '$remote_archive' '$tree'"
  ssh -o BatchMode=yes "$cluster" \
    "stat -c '%s' '$remote_archive'; sha256sum '$remote_archive'" > "$relay_tmp/$tree-metadata.txt"
  expected_bytes=$(sed -n '1p' "$relay_tmp/$tree-metadata.txt")
  expected_sha256=$(sed -n '2p' "$relay_tmp/$tree-metadata.txt" | cut -d' ' -f1)
  [[ $expected_bytes =~ ^[0-9]+$ ]]
  [[ $expected_sha256 =~ ^[0-9a-f]{64}$ ]]

  ssh -o BatchMode=yes "$cluster" "cat '$remote_archive'" |
    aws s3 cp - "s3://$bucket/$prefix/$archive" \
      --content-type application/gzip --expected-size "$expected_bytes" \
      --only-show-errors
  uploaded_bytes=$(aws s3api head-object \
    --bucket "$bucket" --key "$prefix/$archive" \
    --query ContentLength --output text)
  test "$uploaded_bytes" = "$expected_bytes"
  printf '%s  %s\n' "$expected_sha256" "$archive" |
    aws s3 cp - "s3://$bucket/$prefix/$archive.sha256" \
      --content-type text/plain --only-show-errors
}

# Upload artifacts before the completion marker. Consumers can therefore use
# COMPLETE.json as an atomic indication that every advertised object exists.
upload_release
upload_archive plans
upload_archive logs

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
