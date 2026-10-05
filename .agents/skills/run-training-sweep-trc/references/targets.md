# Targets

Translate the operator's plain-English TPU scope into the complete candidate grid before validating placement.
By default, include every 4–16-chip TPU target in all otherwise allowed regions.

Read the current Marin and Iris configuration.
Expand the requested scope into exact `(region, bucket, slice, chips)` rows and remove duplicate rows.
Normalize region aliases such as `euw4` to `europe-west4`; resolve buckets from configuration, never by convention.

Scopes may select purpose, chip count, TPU family, region, or exclusions.
Examples include “training chips only,” “4–16-chip inference TPUs in `euw4`,” and “32+-chip TPUs in any region.”
Resolve terms such as training and inference from current Iris pool definitions, not TPU-family assumptions.
Show the interpretation when it is ambiguous.

Add every candidate to Operations as `unvalidated`.
Only validation or a verified invalid-target result changes eligibility.
Fleet utilization and measured throughput rank eligible targets; they never define or narrow the candidate grid.
