# Issue #473 training-split control figures

These figures compare the step-4,999 CDS full-window models trained with chromosome-18-held-out and row-random-held-out validation splits.

`validation_loss.csv` records the complete available loss trajectory for each arm.
The random-held-out values are W&B `eval/loss` rows from [`dna-exp473-0p25b-cds-fullwindow-random-val-v1`](https://wandb.ai/gonzalobenegas/marin/runs/dna-exp473-0p25b-cds-fullwindow-random-val-v1).
The chr18-held-out W&B run contains no validation rows because guarded resumptions disabled W&B logging.
Its panel therefore uses the exact offline `evals_v2` replay on the original public 16,384-row validation shard from `s3://oa-bolinas/snakemake/analysis/evals_v2/results/issue473/410f442a31185dbddb97fab8b23b8842701fcfc6/issue417_validation_control/analysis/loss_points.parquet`.
The panels use independent y-axes because they evaluate different validation rows and use slightly different loss kernels.

`auprc.csv` records canonical `evals_v2` development-`train` endpoints for both terminal checkpoints.
Mendelian and Complex-trait matched-set prevalence is one positive per ten rows.
SGE prevalence is the equal-weight mean of per-accession positive prevalence over the same supported accessions used by the assay-macro AUPRC endpoint.
Error bars are one standard error from the producing metric artifact and have no terminal caps.

Run `uv run --locked python ../../.agents/artifacts/issue-473/split-control/plot.py` from `experiments/exp473_center_seeded_projection` to regenerate both SVGs.
