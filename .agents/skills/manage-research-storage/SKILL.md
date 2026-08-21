---
name: manage-research-storage
description: Choose and document durable ownership and storage for MarinDNA research artifacts. Use when deciding among workflow-native object storage, pipeline-owned S3, issue-owned S3, W&B, or commit-pinned repository storage, or when organizing datasets, checkpoints, predictions, metrics, and figures.
---

# Manage Research Storage

Choose one durable owner before writing an artifact.
Keep provenance with that owner instead of copying the same artifact into several namespaces.
Treat public distribution as a separate decision from storage ownership.

## Choose The Owner

| Artifact | Durable owner and location |
| --- | --- |
| Output of a maintained Snakemake workflow | The workflow's storage profile under `s3://oa-bolinas/snakemake/<pipeline>/...` |
| Checkpoint or native output of a Marin-launched experiment | The experiment's launch configuration and its native object-store path, following `marin-experiment` |
| Durable output of an issue-scoped analysis with no workflow owner | The coordinating issue under `s3://oa-bolinas/issues/<issue-number>/<artifact>/...` |
| Dense metrics and run comparisons | W&B, following `wandb-reporting` |
| SVG referenced by a knowledge-base experiment page | `docs/research/experiments/figures/<issue>/` on `main`, following `maintain-knowledge-base` and `plot-research-results` |
| Static code, tables, or figures small enough for normal Git review and cloning | The permanent task or research branch, linked at an immutable commit |

Do not copy a Snakemake-owned output into an issue prefix merely because an issue consumes it.
Link the pipeline output from the issue instead.
Do not relocate or duplicate a Marin-launched experiment's native checkpoints or outputs into issue-owned S3.
Link their exact native paths from the coordinating issue instead.

## Store Snakemake Outputs

- Let the pipeline's storage profile and rules own the object layout.
- Keep the pipeline name and output identity stable enough for downstream rules to address them.
- Use `develop-snakemake-pipelines` for pipeline-specific layout, execution, and validation.
- Record the producing commit and configuration in the pipeline's normal provenance chain.

## Store Issue-Owned Outputs

- Require a coordinating GitHub issue before creating an issue-owned S3 prefix.
- Use `s3://oa-bolinas/issues/<issue-number>/<artifact>/...` for durable outputs from one-off analyses or investigations that have no maintained workflow owner.
- Give reruns distinct versioned paths when outputs can change.
  Do not silently overwrite an artifact already cited by a result.
- Store or link the producing commit, configuration, input identifiers, schema, and a manifest with file sizes and SHA-256 checksums needed to interpret and reproduce the artifact.
- Publish an immutable artifact link or exact object prefix in the coordinating issue.

## Publish Links

- Use commit-pinned repository URLs for branch artifacts.
- Use exact S3 prefixes, object versions, manifests, or checksums when a result depends on mutable object storage.
- Keep a human-readable entry point in the coordinating issue, PR, experiment page, or question page.
- Use `task-snapshot` before publishing a milestone link that must remain reproducible.

## Compose Existing Skills

- Use `develop-snakemake-pipelines` for pipeline-owned outputs.
- Use `marin-experiment` for the native checkpoints and outputs of Marin-launched experiments.
- Use `wandb-reporting` for W&B runs, reports, and artifacts.
- Use `access-reference-genomes` for reference-genome mirrors and asset compatibility.
- Use `task-snapshot` for immutable task milestones.
- Use `communicate-on-github` to publish durable artifact links and provenance summaries.
- Use `publish-to-hugging-face` when a stored model, dataset, checkpoint, or other reusable artifact needs public distribution.
