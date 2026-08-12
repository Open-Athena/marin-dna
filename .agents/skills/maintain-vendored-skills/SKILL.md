---
name: maintain-vendored-skills
description: Compare, refresh, and audit Marin-derived skills in MarinDNA. Use when checking the pinned Marin commit, showing exact upstream-to-local modifications, performing the monthly upstream review, updating vendored skills, or triaging newly available Marin skills.
---

# Maintain Vendored Skills

Keep vendor maintenance out of normal task context. Load [references/manifest.json](references/manifest.json) only when comparing or updating Marin-derived skills.

## Show Exact Modifications

1. Obtain a Git checkout of the manifest's pinned Marin commit when possible. For a non-Git archive, retain the commit URL used to fetch it and pass its SHA with `--upstream-commit`; the report marks archive provenance as not independently verified.
2. Run:

   ```bash
   python3 .agents/skills/maintain-vendored-skills/scripts/compare_upstream.py \
     --upstream-root <marin-checkout> \
     --repo-root .
   ```

   For a non-Git archive, add `--upstream-commit <pinned-sha>`. A Git checkout verifies `HEAD` automatically.

3. Use `--output <path>` to save the Markdown report. The report identifies byte-identical skills and includes a unified diff for every adapted skill.
4. Treat a diff in a skill classified as unchanged, or no diff in a skill classified as adapted, as a maintenance decision that requires review.

## Refresh From Upstream

1. Check for an existing vendor-update PR.
2. Compare the pinned commit with current Marin `main`.
3. Replace unchanged vendors byte-for-byte.
4. Start each adapted vendor from the new upstream file, then reapply only the deviations in the manifest.
5. Run the comparison script against the new upstream checkout and inspect every adapted diff.
6. Validate every changed skill with the standard skill validator.
7. Update the manifest commit and `vendored_on` date.
8. Open a draft PR only when content or recorded provenance changed.

Do not add newly discovered skills automatically. Report candidates with their purpose, dependencies, Marin-specific assumptions, and likely local adaptations for human triage.

## Monthly Review

Use one external monthly Codex scheduled task. Compare current Marin `main`, run the exact-delta report, and scan the upstream skill catalog. Keep scheduling infrastructure outside the repository.
