#!/usr/bin/env bash
#
# Copy a completed issue #417 result tree to an immutable S3 prefix and verify
# exact relative-path/byte-size parity. The sync is resumable after interruption.

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 RESULTS_DIR s3://BUCKET/PREFIX" >&2
  exit 2
fi

results_dir="$(realpath "$1")"
s3_uri="${2%/}"
if [[ ! -d "$results_dir" ]]; then
  echo "results directory does not exist: $results_dir" >&2
  exit 2
fi
if [[ ! "$s3_uri" =~ ^s3://([^/]+)/(.+)$ ]]; then
  echo "destination must be a non-root S3 prefix: $s3_uri" >&2
  exit 2
fi

bucket="${BASH_REMATCH[1]}"
prefix="${BASH_REMATCH[2]%/}"
inventory_name="artifact_inventory.tsv"
local_inventory="$(mktemp)"
remote_inventory="$(mktemp)"
trap 'rm -f "$local_inventory" "$remote_inventory"' EXIT

find "$results_dir" -type f ! -name "$inventory_name" -printf '%P\t%s\n' \
  | LC_ALL=C sort >"$local_inventory"
local_files="$(wc -l <"$local_inventory")"
local_bytes="$(awk -F '\t' '{total += $2} END {printf "%.0f", total}' "$local_inventory")"
if [[ "$local_files" -eq 0 || "$local_bytes" -eq 0 ]]; then
  echo "refusing to preserve an empty result tree: $results_dir" >&2
  exit 1
fi

aws s3 sync "$results_dir/" "$s3_uri/" \
  --exclude "$inventory_name" \
  --only-show-errors
aws s3 cp "$local_inventory" "$s3_uri/$inventory_name" --only-show-errors

aws s3api list-objects-v2 \
  --bucket "$bucket" \
  --prefix "$prefix/" \
  --query 'Contents[].[Key,Size]' \
  --output text \
  | awk -F '\t' -v root="$prefix/" -v inventory="$inventory_name" '
      $1 != root inventory {
        sub("^" root, "", $1)
        print $1 "\t" $2
      }
    ' \
  | LC_ALL=C sort >"$remote_inventory"

if ! cmp -s "$local_inventory" "$remote_inventory"; then
  echo "S3 inventory differs from the local result tree" >&2
  diff -u "$local_inventory" "$remote_inventory" | head -n 200 >&2 || true
  exit 1
fi

remote_files="$(wc -l <"$remote_inventory")"
remote_bytes="$(awk -F '\t' '{total += $2} END {printf "%.0f", total}' "$remote_inventory")"
if [[ "$local_files" -ne "$remote_files" || "$local_bytes" -ne "$remote_bytes" ]]; then
  echo "S3 file/byte totals differ after exact inventory comparison" >&2
  exit 1
fi

echo "preserved_uri=$s3_uri"
echo "files=$remote_files"
echo "bytes=$remote_bytes"
echo "inventory=$s3_uri/$inventory_name"
