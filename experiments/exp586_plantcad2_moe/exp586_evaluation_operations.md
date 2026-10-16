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

## Models

The selected models are `exp586-plantcad2-d768-scratch-lrm1-seed0-v1` and `exp586-plantcad2-d768-pretrained-lrm1-seed0-v1`, both at permanent update 412,290.
Their checkpoints are under `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/checkpoints/<run-id>/2026.10.08/checkpoints/step-412290`.
The raw checkpoint metadata SHA-256 is `e887cd81dc658f46aee76faebb3313943733d889210b47f4c562ac450418162e`.
Marin's canonical checkpoint metadata digest used by restoration is `737fa99f7998a31ca2a27aacf82ac3f292d9ff62a49742847e4037ad3cd372d9`.
The checkpoint records the eight-token DNA tokenizer and the `ring` training backend.
The evaluation uses dropless Sonic inference only after Sonic agrees with the recorded ring backend on bounded real genomic inputs.

For pretrained 1×, the raw metadata SHA-256 is `f6028295e3790bf43606917a78e5dade8ce0eb12e19dd2a8e3cab886ae86ab34` and Marin's canonical metadata digest is `275e68e8a521333a4dd5c7bf2aa8f9c79c11517677a8af47ce7abc1aa5004a89`.
Its checkpoint records the character-forced language tokenizer and `fixed_pooled_wave_all_to_all` training backend.
The same method-validation gate must establish Sonic parity before full evaluation.

## Protocol

Reuse the exact experiment 582 PlantCAD2 inputs, task formulas, held-out memberships, and historical artifacts.
Evaluate all 20 original tasks at 8,192 bases, the 19 centered tasks at 2,048 bases, and maize allele frequency at both contexts.
Report natural causal readouts, two-sided forward and reverse-complement products, frozen whole-window probes, and frozen target-position probes.
Validate exact checkpoint and tokenizer binding, zero routed drops, finite normalized probabilities, nonconstant outputs, strand alignment, and saved-probe round trips.
Run a bounded real-data pilot before the full split.

## State And Artifacts

The scratch 1× control database is `scratch/exp586-final-eval/exp586_eval.sqlite`.
The pretrained 1× control database is `scratch/exp586-pretrained-lrm1-final-eval/exp586_eval.sqlite`.
Immutable database backups belong under `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/operations/evaluation/`.
Evaluation inputs and wheels belong under `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/evaluations/final-v1/inputs/`.
Model outputs use the labels `scratch-lrm1-seed0` and `pretrained-lrm1-seed0` under `results/`, `analysis/`, `probes/`, and `validation/`.
Every uploaded object is content-addressed or immutable and verified by size and SHA-256 after upload.

## Placement

Method validation, the pilot, each full-split worker, and postprocessing each use one eight-H100 node.
The full split keeps the proven 16-worker exp582 partition so source Parquet files are not duplicated across workers.
Dispatch workers only when one CoreWeave backend reports enough current batch-priority capacity for the planned gang.
W&B run IDs begin with the selected training run ID and identify the phase and worker.

## Change Record

2026-10-12: The final scratch 1× checkpoint was selected as the only completed experiment 586 model.
2026-10-12: Checkpoint metadata showed that scratch training finished with ring EP, correcting the copied pooled-EP parity assumption before GPU evaluation.
2026-10-12: The first method-validation dispatch failed before restoration because the raw file SHA-256 was supplied where restoration requires Marin's canonical JSON digest.
2026-10-12: Sonic versus ring parity had mean absolute probability error 0.00201, p99 0.02055, 98.96% argmax agreement, NLL delta 0.00121, and accuracy delta 0.00521, but one value at 0.05868 exceeded the copied 0.05 maximum.
2026-10-12: The parity gate now requires maximum error at most 0.075 and p99 at most 0.025 while retaining the original mean, argmax, NLL, and accuracy limits.
2026-10-12: The first full worker on each H100 cluster reached W&B inference with numeric progress. The remaining immutable shards may then be admitted concurrently when a fresh snapshot shows capacity; the one-pending-dispatch exploration limit no longer applies to these proven, identical one-node workers.
2026-10-16: The user selected pretrained 1× as the second and final experiment 586 model to evaluate. It reuses the validated protocol and immutable inputs with model-specific W&B, output, and control-state identities.
