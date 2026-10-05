---
name: maintain-vendored-skills
description: Compare, refresh, and audit externally derived skills in MarinDNA. Use when checking a pinned source commit, showing exact upstream-to-local modifications, performing the monthly upstream review, updating vendored skills, or triaging newly available upstream skills.
---

# Maintain Vendored Skills

Keep vendor maintenance out of normal task context.
Load the source-specific manifest under `references/` only when comparing or updating its vendored skills.

## Preserve The Vendor Boundary

- Treat every skill listed under `unchanged` or `adapted` in a vendor manifest as vendor-owned. Do not edit those skill directories in ordinary feature, documentation, or repository-guidance work.
- Change vendored content only in a dedicated vendor-maintenance task that explicitly owns the upstream comparison and provenance update.
- Put MarinDNA-specific behavior in a local skill that composes with the vendored skill. The local skill may route to the vendor, add repository-specific constraints, or coordinate it with other skills.
- Register composing skills under `local` in the manifest. If composition cannot express a required change, report the limitation and propose a vendor adaptation instead of modifying the vendor implicitly.

## Show Exact Modifications

1. Select the vendor manifest and obtain a clean Git checkout of its pinned commit when possible.
   The comparison rejects uncommitted or untracked files under the upstream skill tree.
   For a non-Git archive, retain the commit URL used to fetch it and pass its SHA with `--upstream-commit`; the report marks archive provenance as not independently verified.
2. Run:

   ```bash
   python3 .agents/skills/maintain-vendored-skills/scripts/compare_upstream.py \
     --manifest .agents/skills/maintain-vendored-skills/references/<source>-manifest.json \
     --upstream-root <upstream-checkout> \
     --repo-root .
   ```

   Omit `--manifest` only for the default Marin [manifest](references/manifest.json).
   Run the comparison once per vendor manifest and matching checkout.

   For a non-Git archive, add `--upstream-commit <pinned-sha>`. A Git checkout verifies `HEAD` automatically.

3. Use `--output <path>` to save the Markdown report.
   The report identifies content-, type-, and executable-mode-identical skills and includes a unified diff or metadata delta for every adapted skill.
4. Treat a diff in a skill classified as unchanged, or no diff in a skill classified as adapted, as a maintenance decision that requires review.

## Refresh From Upstream

1. Check for an existing vendor-update PR.
2. Compare each selected manifest's pinned commit with its current upstream branch.
3. Replace unchanged vendors content-for-content while preserving regular-file, symlink, and executable modes.
4. Start each adapted vendor from the new upstream file, then reapply only the deviations in the manifest.
5. Run the comparison script against the new upstream checkout and inspect every adapted diff.
6. Validate every changed skill with the standard skill validator.
7. Update the manifest commit and `vendored_on` date.
8. Open a draft PR only when content or recorded provenance changed.

Do not add newly discovered skills automatically. Report candidates with their purpose, dependencies, Marin-specific assumptions, and likely local adaptations for human triage.

## Monthly Review

Use one external monthly Codex scheduled task.
Compare each vendored source's current branch, run its exact-delta report, and scan its skill catalog.
Keep scheduling infrastructure outside the repository.
