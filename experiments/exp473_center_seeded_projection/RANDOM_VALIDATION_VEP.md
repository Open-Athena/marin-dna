# Terminal VEP comparison for the random-validation control

This additive workflow compares the step-4,999 CDS full-window model trained with the row-random validation split against the reused issue #417 CDS full-window model trained with chromosome 18 held out.
It scores only the new random-validation checkpoint and reuses the existing terminal predictions and metrics for the issue #417 arm.

The evaluation reads only each pinned dataset repository's `train.parquet` file.
That development split contains odd autosomes and chromosome X.
Even-autosome and chromosome-Y labels, predictions, and aggregate metrics remain held out.

The reported CDS rows are Mendelian missense, splicing, and synonymous; Complex-trait missense; and SGE missense and splicing.
Complete mature-miRNA match groups are removed before any matched-data metric is computed.
Mendelian and Complex-trait AUPRC and Group SMD deltas use aligned match-group bootstrap draws.
SGE uses the official per-arm assay-macro AUPRC and bootstrap SE.

The new workflow owns outputs under `s3://oa-bolinas/snakemake/analysis/evals_v2/results/issue473/<snapshot-commit>/random_validation_vep/`.
The reused issue #417 artifacts remain at the immutable issue #473 evaluation snapshot under `results/issue473/ae90f6d9e4b23ebe8fb1bd2314baa66cb82b37c1/development_eval/`.
No established evals_v2 rule, output path, or baseline artifact is edited or recomputed.

Run the experiment-project tests and an isolated dry-run before launch:

```bash
cd experiments/exp473_center_seeded_projection
uv run --python /usr/bin/python3.12 --locked pytest

cd ../../snakemake/analysis/evals_v2
PYTHONPATH=../../../experiments/exp473_center_seeded_projection/src \
uv run --locked --group genome-s3 \
  python -m exp473_center_seeded_projection.random_validation_vep_config \
  --output /tmp/issue_473_random_validation_vep.yaml \
  --snapshot-commit <full-commit-sha>
uv run --locked --group genome-s3 snakemake --dry-run \
  --snakefile ../../../experiments/exp473_center_seeded_projection/workflow/RandomValidationVep.smk \
  --profile workflow/profiles/default \
  --configfile /tmp/issue_473_random_validation_vep.yaml \
  --cores 4 \
  --resources gpu=1 mem_mb=12000 \
  issue_473_random_validation_vep_all
```

Launch the three new score cells and comparison on one EC2 A10G:

```bash
sky launch \
  experiments/exp473_center_seeded_projection/sky/random_validation_vep.yaml \
  -c exp473-random-validation-vep \
  --env EXP473_VEP_COMMIT=<full-commit-sha> \
  --down
```
