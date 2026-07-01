#!/usr/bin/env bash
# Delete the retired `enhancement` label. Run ONLY after apply_labels.py has
# moved every issue off it (its work is now infrastructure + Area). Deleting a
# label also removes it from any remaining issues, so this is the final step.
# Idempotent: a no-op if the label is already gone.
set -euo pipefail
if gh label list --limit 200 --json name --jq '.[].name' | grep -qx enhancement; then
  gh label delete enhancement --yes
  echo "Deleted 'enhancement'."
else
  echo "'enhancement' already absent — nothing to do."
fi
