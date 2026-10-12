# exp586 CoreWeave Evaluation Operations

> This document governs final evaluation work managed with the `run-training-sweep-cw` skill.
> Read it at the start of every evaluation heartbeat before inspecting SQLite, W&B, or Iris.

## Invariants

Evaluate only permanent final checkpoints selected by the user.
Use H100s only, whole eight-GPU nodes, batch priority, and Iris user `eczech`.
Treat W&B as execution-state truth and use Iris only for exact-job liveness, placement, and concrete startup failures.
Check live capacity before every submission or placement change.
Keep at most one active dispatch for each logical evaluation trial.
Do not rerun historical comparison models.
Do not update MarinDNA issue 586 automatically.
Publish code and cumulative results in the PlantCAD2 `codex/exp586` pull request.

## First Model

The first model is `exp586-plantcad2-d768-scratch-lrm1-seed0-v1` at permanent update 412,290.
Its checkpoint is `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/checkpoints/exp586-plantcad2-d768-scratch-lrm1-seed0-v1/2026.10.08/checkpoints/step-412290`.
The checkpoint metadata SHA-256 is `e887cd81dc658f46aee76faebb3313943733d889210b47f4c562ac450418162e`.
The checkpoint records the eight-token DNA tokenizer and the `ring` training backend.
The evaluation uses dropless Sonic inference only after Sonic agrees with the recorded ring backend on bounded real genomic inputs.

## Protocol

Reuse the exact experiment 582 PlantCAD2 inputs, task formulas, held-out memberships, and historical artifacts.
Evaluate all 20 original tasks at 8,192 bases, the 19 centered tasks at 2,048 bases, and maize allele frequency at both contexts.
Report natural causal readouts, two-sided forward and reverse-complement products, frozen whole-window probes, and frozen target-position probes.
Validate exact checkpoint and tokenizer binding, zero routed drops, finite normalized probabilities, nonconstant outputs, strand alignment, and saved-probe round trips.
Run a bounded real-data pilot before the full split.

## State And Artifacts

The local control database is `scratch/exp586-final-eval/exp586_eval.sqlite`.
Immutable database backups belong under `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/operations/evaluation/`.
Evaluation inputs and wheels belong under `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/evaluations/final-v1/inputs/`.
Model outputs use the label `scratch-lrm1-seed0` under `results/`, `analysis/`, `probes/`, and `validation/`.
Every uploaded object is content-addressed or immutable and verified by size and SHA-256 after upload.

## Placement

Method validation, the pilot, each full-split worker, and postprocessing each use one eight-H100 node.
The full split keeps the proven 16-worker exp582 partition so source Parquet files are not duplicated across workers.
Dispatch workers only when one CoreWeave backend reports enough current batch-priority capacity for the planned gang.
W&B run IDs begin with `exp586-plantcad2-d768-scratch-lrm1-seed0-` and identify the phase and worker.

## Change Record

2026-10-12: The final scratch 1× checkpoint was selected as the only completed experiment 586 model.
2026-10-12: Checkpoint metadata showed that scratch training finished with ring EP, correcting the copied pooled-EP parity assumption before GPU evaluation.
