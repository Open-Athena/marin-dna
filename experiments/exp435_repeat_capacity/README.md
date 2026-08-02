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

## Stage 0 run

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

## Stage 1: frozen reference panel

`prepare_reference_panel.py` validates the stage-0 archive and materializes the
outcome-blind 255-bp sequence panel recorded in issue #435: 32,768 uniformly
sampled repeat loci paired to full-window repeat-free controls by chromosome and
repeat-derived GC decile, plus 128 focal loci and 128 other-repeat controls for
each selected class, family, and subfamily. Primary labels resolve overlaps by
highest RepeatMasker score, then lower divergence, longer interval, and stable
annotation ID; all alternate overlaps and sequence-composition covariates are
retained.

```bash
sky launch -y -c exp435-repeat-reference-panel \
  --env EXPERIMENT_COMMIT=$(git rev-parse HEAD) \
  experiments/exp435_repeat_capacity/sky.panel.yaml
```

Durable output:
`s3://oa-bolinas/experiments/exp435/retrieval/dna-exp435-repeat-reference-panel-r1/`.
The full FASTA and inventory are downloaded only to the temporary Sky node; the
archive retains their identities and hashes, not redundant source copies.


## Stage 2: three-layer sparse reference activations

`extract_reference_activations.py` consumes the exact stage-1 S3 archive, not a
resampled panel. Because `contexts.parquet` already contains the validated 255-bp
sequences, this GPU stage does not download the reference FASTA. One shared bf16
gLM forward captures blocks 1, 10, and 19; their pinned 25M SAEs are applied to
the focal token and only nonzero activations are written. FWD and RC remain
separate.

```bash
sky launch -y -d --down -c exp435-repeat-reference-activations \
  --env EXPERIMENT_COMMIT=$(git rev-parse HEAD) \
  experiments/exp435_repeat_capacity/sky.extract.yaml
```

Durable output:
`s3://oa-bolinas/experiments/exp435/retrieval/dna-exp435-repeat-reference-activations-r1/`.
The AWS GPU reads and writes S3 directly; neither the model checkpoints nor the
sparse activation tables are staged on the shared Codex node.


## Stage 3: sparse repeat-capacity associations

`analyze_reference_capacity.py` reads the stage-2 sparse parquets directly from
S3 and tests the frozen repeat, class, family, and subfamily contrast families.
Welch statistics use sparse sums and sums of squares. Mann–Whitney U and both
AUPRC directions are computed exactly from nonzero activations while zero and
equal-value ties are handled analytically. BH correction remains separate by
layer, orientation, hierarchy, and statistic.

After recording the stage-2 archive-manifest hash:

```bash
sky launch -y -d --down -c exp435-repeat-reference-associations \
  --env EXPERIMENT_COMMIT=$(git rev-parse HEAD) \
  --env EXTRACTION_ARCHIVE_SHA256=<sha256> \
  experiments/exp435_repeat_capacity/sky.analyze.yaml
```

Durable output:
`s3://oa-bolinas/experiments/exp435/retrieval/dna-exp435-repeat-reference-associations-r1/`.


## Stage 4: composition and decoder sensitivities

`analyze_reference_sensitivities.py` verifies the exact stage-2 and stage-3
archives, reruns frozen composition/boundary/overlap subsets, and measures
decoder-space redundancy for broad-repeat and category-associated feature sets.
The CPU node downloads the three exact SAE decoder files and validates their
pinned hashes; large inputs remain on ephemeral instance storage.

```bash
sky launch -y -d --down -c exp435-repeat-reference-sensitivities \
  --env EXPERIMENT_COMMIT=$(git rev-parse HEAD) \
  experiments/exp435_repeat_capacity/sky.sensitivity.yaml
```

Durable output:
`s3://oa-bolinas/experiments/exp435/retrieval/dna-exp435-repeat-reference-sensitivities-r1/`.

## Stage 5: outcome-blind paired repeat-variant panel

`prepare_variant_panel.py` joins the exact 16,140-row official Mendelian panel
to the frozen RepeatMasker inventory without retaining `label`. It distinguishes
focal repeat overlap, nonfocal repeat overlap within the 255 bp window, and a
fully repeat-free window; preserves the primary annotation and all overlaps;
and records class/family/subfamily feasibility before any paired activation is
inspected. This metadata stage does not run the gLM and will allow the existing
#436 block-1/10/19 25M paired sparse activations to be reused.

```bash
sky launch -y -d --down -c exp435-repeat-variant-panel --env EXPERIMENT_COMMIT=<40-character-commit> experiments/exp435_repeat_capacity/sky.variant_panel.yaml
```

Durable output:
`s3://oa-bolinas/experiments/exp435/retrieval/dna-exp435-repeat-variant-panel-r1/`.

## Stage 6: paired repeat-variant response

`analyze_variant_deltas.py` reuses #436's existing block-1/10/19 25M paired
sparse activations and transfers only the reference-positive repeat feature sets
from Stage 3. It runs the frozen broad, subset, class, family, subfamily,
unique-overlap, and repeat-interior association families without reading the
Mendelian label. FWD/RC remain separate and are jointly included in each
within-layer BH family.

```bash
sky launch -y -d --down -c exp435-repeat-variant-deltas --env EXPERIMENT_COMMIT=<40-character-commit> experiments/exp435_repeat_capacity/sky.variant_analyze.yaml
```

Durable output:
`s3://oa-bolinas/experiments/exp435/retrieval/dna-exp435-repeat-variant-deltas-r1/`.
