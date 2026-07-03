"""HuggingFace dataset card for the issue #353 Arm B (projected-CDS) datasets.

The Arm B ``cds_projection`` flow uploads shards with ``hf upload-large-folder``,
which skips top-level files, so the card is generated + pushed separately (same
pattern as the shared ``training_hf_readme``).
"""

_REPO_BLOB = "https://github.com/Open-Athena/marin-dna/blob"
_PIPELINE_SUBDIR = "snakemake/training_dataset/dataset_creation"

_VERT_WARNING = """## ⚠️ Coverage is vertebrate-weighted

Nucleotide CDS projection reaches vertebrates well but fades across the animal \
tree: **97% of the ~12.6 M retained windows come from the 125 chordates** \
(mammals ~79%, birds/reptiles ~37%, fish ~20%, jawless ~8–10% of human CDS \
windows projected); the **79 invertebrate species (16 phyla) contribute only \
~3.7%**, at a ~2% reach floor. A vertebrate-only subset is at \
`bolinas-dna/animals-cds-proj-v1-vert125`.

"""


def build_cds_projection_readme(
    *,
    cohort: str,
    n_species: int,
    n_rows: int,
    commit_sha: str,
    is_vertebrate_subset: bool,
) -> str:
    """Return the HuggingFace dataset-card markdown for one Arm B cohort.

    Args:
        cohort: cohort tag (e.g. ``"all204"`` / ``"vert125"``), only for the title.
        n_species: number of target species merged into this dataset.
        n_rows: total rows (already reverse-complement-doubled).
        commit_sha: 40-char SHA for the pipeline/species-list permalinks.
        is_vertebrate_subset: drop the vertebrate-weighted warning + point at the
            Chordata species list when this cohort is the vertebrate subset.
    """
    base = f"{_REPO_BLOB}/{commit_sha}/{_PIPELINE_SUBDIR}"
    if is_vertebrate_subset:
        title = "vertebrates (Chordata)"
        scope = f"{n_species} vertebrate species (Chordata, one per order)"
        species_tsv = "animals_order204_chordata.tsv"
        warning = ""
    else:
        title = "all animal orders"
        scope = f"{n_species} animal species (one annotated genome per Metazoan order)"
        species_tsv = "animals_order204.tsv"
        warning = _VERT_WARNING
    return f"""---
tags:
- biology
- genomics
- dna
---

# Animal CDS via human→animal projection — {title}

Human protein-coding (CDS) 255 bp windows projected onto **{scope}** by \
**mmseqs2 nucleotide local alignment** — the projection ("Arm B") side of the \
[issue #353](https://github.com/Open-Athena/marin-dna/issues/353) CDS \
projection-vs-annotation experiment.

## How it was built

The human `v5` CDS intervals (Ensembl coding exons, filtered 20–10,000 bp, \
+20 bp splice flank, expanded ≥256 bp) are tiled into 255 bp windows (269,866 \
windows) and searched against each target genome with mmseqs2 \
(`--search-type 3 --strand 2 --mask-lower-case 1 -s 7.5 --max-accept 1` — the \
v30 operating point). The best hit per window (by bitscore) is kept and \
midpoint-resized to exactly 255 bp. Pipeline: \
[`cds_projection.smk`]({base}/workflow/rules/cds_projection.smk); species list: \
[`{species_tsv}`]({base}/config/{species_tsv}).

{warning}## Schema

| column | type | description |
|---|---|---|
| `id` | str | target locus `chrom:start-end` (0-based half-open, 255 bp) |
| `seq` | str | 255 bp target-genome sequence (+ strand), reverse-complement-augmented |

**{n_rows:,} rows** ({n_rows // 2:,} projected windows × 2 for RC augmentation), \
{n_species} species.
"""
