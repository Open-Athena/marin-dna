# Issue #473 projection-policy trajectories

These figures compare full-window and center-1 projection models across all nine saved checkpoints from steps 1,000 through 4,999.
All four models used chromosome-18-held-out training validation splits.
The plotted VEP trajectories use only the canonical development `train` split containing odd-numbered autosomes and chromosome X.
No held-out labeled rows were evaluated.

The CDS full-window values come from the repaired canonical `evals_v2` model family `exp417-cds-combined-vertebrates-step-{step}`.
The CDS center-1 and both enhancer arms come from the issue-specific development artifacts under experiment commit `ae90f6d9e4b23ebe8fb1bd2314baa66cb82b37c1`.
The latter checkpoints expose the intended dual-schema RoPE metadata and do not have the legacy issue #417 loading mismatch.

CDS reporting is restricted to Mendelian missense, splicing, and synonymous; Complex-trait missense; and SGE missense and splicing.
Enhancer reporting is restricted to Mendelian and Complex-trait distal.
SGE is omitted for enhancer models because it is outside their training-region scope.

Lines show actual AUPRC values.
Vertical uncapped error bars show plus or minus one standard error from the producing analysis artifact.
The gray dashed line is positive prevalence.
Each y-axis is anchored at prevalence and extends slightly lower only when needed to display a point's uncertainty interval.
Mendelian and Complex-trait matched-set prevalence is one positive per ten rows.
SGE prevalence is the equal-weight mean of per-accession positive prevalence over the supported accessions used by the assay-macro endpoint.

The repaired CDS source tables are in the immutable development analysis bundle at `s3://oa-bolinas/snakemake/analysis/evals_v2/results/issue473/ae90f6d9e4b23ebe8fb1bd2314baa66cb82b37c1/damage_control/cb2fe372485d8287fa72a4e0faeae1d80b830178/`.
The enhancer source tables are in the immutable development analysis bundle at `s3://oa-bolinas/snakemake/analysis/evals_v2/results/issue473/ae90f6d9e4b23ebe8fb1bd2314baa66cb82b37c1/analysis/89bcd07baf27d0326bf6efbbf101c6204fb0a7db/`.

Run `uv run --locked python ../../.agents/artifacts/issue-473/projection-policy-trajectories/plot.py` from `experiments/exp473_center_seeded_projection` to regenerate both SVGs.
