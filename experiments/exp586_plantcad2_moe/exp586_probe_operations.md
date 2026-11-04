# exp586 frozen classification probes

Read this document before every execution pass.
Use `run-training-sweep-cw` for capacity, dispatch, W&B progress, and durable state.

## Scope

The user authorized these additional probes on 2026-11-04 after reviewing the feature visibility rules.
Evaluate only the final language-pretrained 0.22T and 1T d768 checkpoints already used in the completed evaluation.
Run conservation (three tasks), core/non-core (eight tasks), and SV (one task).
Keep motif recovery and the existing allele-frequency results unchanged.
Return a simple comparison table in chat; do not revise the draft or post to GitHub.

## Scientific contract

Use the pinned original full task inputs.
For conservation and core/non-core, physically exclude the designated nucleotide or entire central motif before model inference.
Compute independent final forward-left and reverse-complement-right flank states.
Compare their average and concatenation using regularized logistic classifiers, without updating the DNA model.
For SV, retain the dataset's observed REF/ALT pairs and use real-token mean embeddings with the historical `[REF, ALT-REF]` construction, averaged across strands or concatenated by direction.
Validate input alignment, tokenization, repeat/batch invariance, zero expert drops, and counterfactual target exclusion before full extraction.
Recover genomic coordinates and construct common leakage-controlled training/test memberships before fitting or choosing readouts.
Choose feature scaling and classifier regularization using development rows only.
Compare cached zero-shot predictions on the same held-out membership; keep supervised and zero-shot scores distinct.
Do not invent a 20-task supervised composite from the twelve classification tasks.

## Execution policy

Use H100s only, whole eight-GPU nodes, on `cw-us-east-02a` and `cw-rno2a`, at most 256 GPUs concurrently.
Every root uses `--priority batch --user eczech`; children explicitly use or inherit batch priority.
Use W&B `eric-czech/marin`, group `exp586-plantcad2-moe-frozen-classification` and tag `dna-exp586`.
Use the skill utilization helper before dispatch and capacity changes.
Follow W&B progress; inspect Iris only for exact-job liveness or concrete startup failures.
Use 30-minute heartbeats after remote progress is proven, with one-hour reslicing and three-hour restart thresholds.
The hard deadline remains 2026-11-17T23:59:59Z.
Keep one writer per immutable chunk and one active dispatch per logical trial.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| cw-us-east-02a | H100 | 1 | 8 | eligible | Existing exp586 runtime and dropless backend validated here; new extraction needs a bounded gate |
| cw-rno2a | H100 | 1 | 8 | eligible | Existing exp586 runtime and dropless backend validated here; new extraction needs a bounded gate |

## Durable ownership

Use `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/evaluations/frozen-classification-v1/` for inputs, extraction receipts, features, validation, classifiers, and results.
Use `scratch/probes/evaluation.sqlite` for the local execution ledger and the skill's transactional persistence helper for mutations and consistent backups.
Publish and directly verify immutable ledger backups under `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/operations/frozen-classification/` before external actions and after state mutations.
If local state is lost, restore the newest integrity-checked immutable backup.
Keep execution code in this experiment project and dataset preparation/classifier fitting in the PlantCAD2 `marindna_exp586` project on `codex/exp586-probes`.
Keep research chronology in `.agents/logbooks/exp586.md`; do not duplicate fleet state there.

## Change record

2026-11-04: Both checkpoints passed six-case hidden-state validation against the independently restored FP32 ordered scatter backend, including ambiguous SV bases and the final source file of a multi-file task.
The first full 0.22T worker has W&B-confirmed extraction progress.
Admit up to sixteen identical one-node extraction workers per condition after its gate passes, overriding the skill's default pending-target limit of one for this bounded homogeneous evaluation workload.
Keep the total submitted allocation at or below 256 H100s; place 0.22T in East and 1T in Reno while capacity permits.
CPU-only classifier fitting may use batch roots with up to 32 CPUs and 128 GB RAM each after extraction, with identities and progress recorded as ledger events.
