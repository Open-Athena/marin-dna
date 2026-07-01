#!/usr/bin/env bash
# Delete the retired `enhancement` label. Run ONLY after apply_labels.py has
# moved every issue off it (its work is now infrastructure + Area). Deleting a
# label also removes it from any remaining issues, so this is the final step.
set -euo pipefail
gh label delete enhancement --yes
echo "Deleted 'enhancement'."
