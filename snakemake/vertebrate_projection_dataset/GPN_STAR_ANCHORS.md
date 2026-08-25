# GPN-Star-P uniform-grid anchors

This runbook records the additive issue #517 recipe selected in [the experiment discussion](https://github.com/Open-Athena/marin-dna/issues/517).
It does not mutate the historical phyloP entry point or its artifacts.

## Selection contract

The source is `songlab/gpn-star-scores`, canonical score set `gpn-star-hg38-p243-200m`, at immutable dataset revision `5c799b2ec6aa089f0caa8294ae72adb4510f81ae`.
`config/gpn_star_p_entropy_manifest.tsv` pins the size and SHA-256 of every chromosome Parquet.
The Parquets expose bare chromosome names and 1-based positions; the scorer converts each position to the internal 0-based interval `[pos - 1, pos)` at the read boundary.

Windows use the historical 255 bp width and 128 bp stride on hg38 primary chromosomes, including chrY.
Undefined reference intervals and incomplete terminal windows are excluded in the same way as the historical grid.
A source base is selected only when `entropy_calibrated < 0.081001`.
Missing source positions are non-passing.
A window passes the 10% audit at 26 selected bases and passes the projection filter at 51 selected bases, corresponding to at least 20% of 255 bases.

The full-tier assertions are:

| audit | expected count |
| --- | ---: |
| uniform windows | 22,948,560 |
| selected source positions | 218,273,080 |
| windows passing 10% | 2,421,580 |
| windows passing 20% | 1,627,410 |

The chrY result is retained with an explicit interpretation caveat because GPN-Star-P and phyloP agree substantially less well there than on the autosomes and chrX.

## Exhaustive six-arm assignment

Assignment occurs only after the GPN 20% filter.
Every one of the 1,627,410 eligible windows must receive exactly one arm.

| arm | rule |
| --- | --- |
| `cds` | direct issue #232 v4 CDS label |
| `utr3` | direct issue #232 v4 3′ UTR label |
| `tss_region_and_utr5` | direct issue #232 v4 protein-coding TSS/5′ UTR label |
| `ncrna_exon` | direct issue #232 v4 ncRNA-exon label |
| `enhancer` | issue #326 Arm A: v4 `ccre_non_promoter` and exactly zero CDS, 3′ UTR, TSS/5′ UTR, and ncRNA-exon fraction |
| `background` | every other window in the GPN 20%-filtered universe |

The background arm includes both old v4-background windows and cCRE-labelled windows rejected by the Arm A zero-overlap rule.
It is therefore a constrained unassigned complement, not an unconstrained genomic background or the issue #232 negative-control arm.
The assignment table records the recipe version, original v4 label, coverage fractions, and reason for every row so Arm B can be evaluated later without repeating projection.

## Durable outputs

The resolved config hash and producer commit key every result below `s3://oa-bolinas/snakemake/vertebrate_projection_dataset/results/gpn-star-p-uniform-v1/`.
The durable catalog outputs are `anchors/catalog.parquet`, `anchors/assignments.parquet`, `anchors/audit/gpn_threshold_summary.json`, and `anchors/audit/assignment_summary.json`.
Downloaded GPN shards, per-chromosome scores, temporary BEDs, and intermediate labels are regenerable `local()` files on EC2 instance-store NVMe and are not uploaded.

## EC2 execution gates

Run every command through `sky/gpn_star_project.yaml`; do not score or project on the shared development node.
Use the same retained worker for the full catalog, smoke projection, and full projection so the large HAL stage can be reused.

After committing the complete recipe, run the locked tests and both Snakemake graph checks on the worker.
Then build the full catalog and verify the four exact selection counts, six nonempty arms, unique assignments, and an assignment total of 1,627,410.
Run the chr18 smoke projection through both HAL and MultiZ and inspect the six-arm QC before launching `all_projection` with `tier=full`.

```bash
sky launch -c issue-517-gpn-project \
  snakemake/vertebrate_projection_dataset/sky/gpn_star_project.yaml \
  --env TIER=full --env TARGET=all --env DRY_RUN=1 \
  --env PIPELINE_COMMIT_SHA="$(git rev-parse HEAD)"

sky exec issue-517-gpn-project \
  snakemake/vertebrate_projection_dataset/sky/gpn_star_project.yaml \
  --env TIER=full --env TARGET=all --env DRY_RUN=0 \
  --env PIPELINE_COMMIT_SHA="$(git rev-parse HEAD)"

sky exec issue-517-gpn-project \
  snakemake/vertebrate_projection_dataset/sky/gpn_star_project.yaml \
  --env TIER=smoke --env TARGET=all_projection --env DRY_RUN=0 \
  --env PIPELINE_COMMIT_SHA="$(git rev-parse HEAD)"

sky exec issue-517-gpn-project \
  snakemake/vertebrate_projection_dataset/sky/gpn_star_project.yaml \
  --env TIER=full --env TARGET=all_projection --env DRY_RUN=0 \
  --env PIPELINE_COMMIT_SHA="$(git rev-parse HEAD)"
```

Terminate the worker only after every required durable output is present in canonical storage and the issue records the catalog and projection QC.
