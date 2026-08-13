---
name: access-reference-genomes
description: Select, retrieve, query, and validate MarinDNA reference-genome assets. Use when choosing between the authenticated S3 mirror and public Hugging Face mirror, selecting FASTA versus BGZF assets, converting query coordinates at an API boundary, or checking assembly and sequence-name compatibility.
---

# Access Reference Genomes

Use the Ensembl release 115 GRCh38 soft-masked primary assembly with Ensembl sequence names (`1` through `MT`) for the canonical public human genome. Treat other `hg38` variants as different references.

## Choose An Existing Mirror

- Prefer `s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/` when the caller has credentials and runs in `us-east-2`. It provides predictable low-latency byte-range access in-region.
- Use `marin-dna/human-genome` at revision `11b9433582981bb929af333bc6422f10a8fd71b4` for credential-free, public, or non-AWS consumers.
- Do not migrate already-materialized inputs solely to standardize their source. Record the source and revision used.

## Choose The Asset Format

- Use the uncompressed FASTA for `pyfaidx` with `fsspec` HTTP range queries.
- Use BGZF plus its indexes for HTSlib or samtools queries and for full downloads.
- Retrieve the matching indexes with indexed assets. Do not mix indexes, contig dictionaries, or sequence files from different builds or revisions.

## Validate Before Use

- Confirm the assembly release, soft-masking, checksum or pinned revision, and Ensembl sequence names.
- Reject silent `chr`-prefix translation or another naming rewrite unless the consuming tool boundary explicitly requires and records it.
- Assert requested contigs exist and coordinates fit their lengths.

## Convert Coordinates At The Boundary

Keep MarinDNA coordinates 0-based and half-open. Convert only for an external API that uses another convention.

- Convert `[start, end)` to the 1-based closed interval `start + 1` through `end` for `pyfaidx.get_seq()` and samtools-style region strings.
- Convert returned intervals back immediately and assert the sequence or interval length equals `end - start`.
- State any different tool-specific convention explicitly.
