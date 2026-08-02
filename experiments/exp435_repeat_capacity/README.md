# Repeat-capacity inventory

Issue #435 investigates how much m5.1 SAE feature capacity is associated with
human repeat classes, families, and subfamilies. This directory is an unmerged,
self-contained experiment. Its first stage materializes an annotation-only hg38
RepeatMasker inventory before any model activation is extracted.

## Stage 0: annotation inventory

`prepare_repeat_inventory.py` reads UCSC’s 17-column hg38 `rmsk` table and the
pinned Ensembl release-115 primary-assembly FASTA index. UCSC coordinates are
already 0-based, half-open; only `chr`-prefix normalization occurs. The inventory
retains chromosomes 1–22, X, and Y and writes:

- every normalized primary-chromosome annotation;
- exact all-repeat union coverage per chromosome;
- record count and raw annotated bases at class, class/family, and
  class/family/subfamily resolution;
- source byte counts and SHA-256 values plus a recursive archive manifest.

Category `raw_annotated_bp` is deliberately labeled raw because source records
can overlap. It is used only for outcome-blind feasibility/ranking. The next
panel-construction stage will sample uniformly from the all-repeat union, assign
one primary focal label under the issue’s recorded overlap policy, and preserve
all alternate overlaps.

## Run

The CPU-only Sky task downloads and archives the exact annotation source, runs
the tests and inventory, uploads the result, and verifies S3 synchronization:

```bash
sky launch -y -c exp435-repeat-inventory \
  --env EXPERIMENT_COMMIT=$(git rev-parse HEAD) \
  experiments/exp435_repeat_capacity/sky.inventory.yaml
```

Durable output:
`s3://oa-bolinas/experiments/exp435/retrieval/dna-exp435-repeat-inventory-r1/`.
Local and Sky disks are staging areas only.
