# exp586 motif probe operations

The user authorized motif probes on 2026-11-05 for every checkpoint in both existing trajectories, followed by updating the existing results comment on PlantCAD2 PR #22.
This supersedes the earlier exclusion of motif probes for this additional study only.
Preserve all existing zero-shot, classification and AF results.
There are ten run/checkpoint positions and nine unique checkpoints because the two runs share the language-only starting model.

## Scientific contract

Use the original eight motif recovery tasks and source rows at 8K genomic windows.
The whole designated 2–3-base motif is absent from both model passes.
Concatenate final-layer last-token activations from the forward left flank and reverse-complement right flank, in that order.
Fit an L2 multiclass logistic head predicting the complete motif, and report whole-motif exact-match accuracy.
Select regularization using the existing six-value C grid and three-fold chromosome-grouped cross-validation on development chromosomes only.
Reuse the existing species-specific whole-chromosome test holdouts and duplicate-visible-input purge.
Exclude ambiguous target motifs consistently with the original motif accuracy calculation; retain ambiguous visible context unchanged.
Record class frequencies, a training-majority-motif baseline, shuffled-outcome controls and serialized-head prediction parity.
Keep one fixed directional feature definition across all tasks and checkpoints.

Reuse cached core/non-core features only after matching both complete visible flanks in their exact forward/RC order and verifying checkpoint, row and chunk identities.
Canonical strand equivalence is sufficient for split duplicate detection but insufficient for directional embedding reuse.
Infer only required training/test examples without verified matching cached embeddings.
Shared language-only checkpoint results are computed once and reused in both trajectories.

## Execution and durability

Use the existing CoreWeave skill helpers and direct, verified CoreWeave S3 access; stop if direct reads cannot be restored.
Every Iris root and child must resolve to batch priority under user eczech.
Use H100s only on cw-rno2a and cw-us-east-02a, in whole eight-GPU nodes, if inference is necessary.
Refresh utilization before admitting GPU work; retain the existing 576-GPU ceiling rather than reserving that capacity in advance.
CPU-only probe fits may use up to 32 CPUs and 128 GB RAM per root, concurrently when useful; release resources when each task finishes.
Use the operator's W&B credentials in eric-czech/marin, tag dna-exp586, and evaluation identities containing motif-probe/final-eval.
W&B is the source of progress; Iris is for exact-root liveness and concrete startup failures.
Use 30-minute heartbeats during established remote work, with one-hour reslice and three-hour restart thresholds; deadline remains 2026-11-18T23:59:59Z.

Native durable owner: s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/evaluations/frozen-classification-v1/motif-checkpoint-trajectory-v1/.
The immutable source manifest, prepared memberships, cache bindings, code wheel, fitting requests, results, predictions and fitted heads live beneath that owner.
Use scratch/motif-probes/evaluation.sqlite and the skill's transactional mutations for observations and exact job identities.
Publish immutable, size/SHA256/readback-verified ledger backups beneath operations/motif-checkpoint-evaluation before dispatch and after state changes.
Append research milestones to .agents/logbooks/exp586.md; do not post interim issue or PR updates.

## Reporting contract

Update only https://github.com/plantcad/plantcad2/pull/22#issuecomment-6647191065 after completing and validating the study.
Keep WIP as the first line, avoid the robot emoji and first-person wording, and retain the heatmap in collapsed detail.
Both token and W&B-run-progress dot/line figures have six metric pairs, with zero-shot above linear probe within each pair.
Top row: Composite, Conservation, Motifs.
Bottom row: Core/non-core, SV impact, Allele frequency.
Each figure has twelve panels: a two-row, three-column arrangement of paired tasks, physically four subplot rows by three columns.
Both composites average conservation, motifs, core/non-core and SV with equal category weights; AF stays separate.
Include motif probes in the existing methods table and explain their outcome, fixed split and exact-match metric briefly.
No separate motif-only plot is to be published.
