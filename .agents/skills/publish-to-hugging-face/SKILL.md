---
name: publish-to-hugging-face
description: Publish MarinDNA models, datasets, checkpoints, and reusable research artifacts on Hugging Face. Use when creating or updating a Hugging Face repository, preparing its model or dataset card, pinning a Hub dependency, or verifying a published artifact.
---

# Publish To Hugging Face

Treat Hugging Face as a public distribution channel, not as the sole provenance or storage owner.
Use `manage-research-storage` first when the producing artifact does not already have a durable owner.

## Require Public Access

- When Hugging Face publication is part of the authorized task, treat creating or updating the public repository as a normal delivery step.
  Do not pause for separate human approval before upload.
- Use only public Hugging Face repositories for every MarinDNA input and output.
- Do not create, upload to, or depend on a private or gated repository.
- Do not publish controlled human data, credentials, or an artifact whose license, consent, or terms prohibit public redistribution.
  If an artifact cannot be public, do not use Hugging Face for it.
- Verify public access without credentials after every creation or update.
- Pin an immutable repository revision in every pipeline, training configuration, evaluation, or document that consumes a Hub artifact.

## Prepare The Repository

1. Select the correct repository type and a stable name under the `marin-dna` organization.
2. Identify the authoritative producing artifact, commit, configuration, license, and intended consumers.
   Retain the exact publishable files under that owner, or retain a deterministic transformation from the authoritative artifact.
   In either case, create and validate a release manifest listing every published path, file size, and SHA-256 checksum before upload.
3. Prepare the README, model card, or dataset card before the first upload or a material metadata change.
4. Include a commit-pinned producing pipeline or training-script link, a concise provenance description, the artifact's intended use and important limitations, and the `biology`, `genomics`, and `dna` tags.
5. Describe the file layout, formats, schemas, assemblies, sequence naming, coordinate conventions, and checksums that consumers need.

## Publish Reproducibly

- Upload an internally consistent artifact set rather than exposing a partially updated repository.
- Preserve versioned producing artifacts in their durable owner when Hugging Face contains a transformed or curated publication copy.
- Record the resulting Hub URL and immutable revision in the coordinating producing issue and, when relevant, the pipeline documentation or accepted experiment record.
- Update consumers to use the immutable revision instead of a moving branch name.

## Verify The Publication

1. Open the public repository and card without Hugging Face credentials.
2. Compare the published file list, sizes, and checksums with the validated release manifest, and check that the expected metadata, tags, and license are present.
3. Download or load a small representative file through the documented public interface.
4. Confirm that the cited revision resolves and that the card's producing links are immutable.
5. Correct any mismatch before announcing the publication.

## Compose Existing Skills

- Use `manage-research-storage` to choose the authoritative owner of producing artifacts.
- Use `develop-snakemake-pipelines` when a pipeline produces or consumes the publication.
- Use `access-reference-genomes` for public reference-genome publications and compatibility checks.
- Use `task-snapshot` to create an immutable producing milestone.
- Use `communicate-on-github` to publish the Hub URL, revision, and provenance summary.
