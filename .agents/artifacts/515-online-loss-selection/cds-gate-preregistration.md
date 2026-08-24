## Protocol amendment: exp58 CDS gate before RefSeq

The RefSeq restart is queued but will not launch yet.
The next run is a shallow CDS gate using the original `exp58-animals` lineage.

### Checkpoints and tokenization

- Student initialization: `gs://marin-dna-us-central1/checkpoints/exp58-animals-r01-1e3682/hf/step-1000`.
- Frozen teacher: `gs://marin-dna-us-central1/checkpoints/exp58-animals-r01-1e3682/hf/step-16999`.
- Both exports contain the same Qwen3 configuration and six-token tokenizer.
- The tokenizer has no BOS or EOS; every 256-base row has 255 causal next-token targets.

### Training data and schedule

- Corpus: `marin-dna/genomes-v4-genome_set-animals-intervals-v5_256_128` at revision `04d374450a0f78f0ab5e17a8bc7b7c4baeb8295c`.
- This is the original animal-CDS corpus used by exp58.
- Lowercase repeat targets are excluded in every arm, while their bases remain visible as context.
- Run one shared 100-step uniform bridge with linear warmup from 1e-5 to 1e-3.
- Fork all seven arms from the complete bridge checkpoint and train every arm for exactly 100 additional steps at constant 1e-3.
- Every arm uses effective batch 2,048 and the same 204,800 post-bridge input rows.

### Seven arms

1. Uniform eligible-token CE.
2. Random 50% eligible-token CE.
3. Current-student lowest-loss 50% CE.
4. Current-student middle-loss 50% CE.
5. Current-student highest-loss 50% CE.
6. Pure full-distribution `KL(p_teacher || p_student)` at temperature 1 over every eligible target, with no hard-label CE term.
7. Hard-label CE on the lowest-loss 50% of eligible targets according to frozen final-teacher NLL.

The teacher is frozen and identical for arms 6 and 7.
Teacher-low thresholds are computed among nonrepeat eligible targets within each sequence, matching the online selectors' 50% density contract.

### Gate evaluation and stop

- Primary endpoint: pooled missense plus splicing AUPRC on `marin-dna/evals_mendelian_traits`, train split, revision `4aed58e50c5dea0b878a665007af2ef9e5108e9f`.
- The endpoint contains 8,990 variants in 899 matched groups, with one positive and nine negatives per group.
- Report missense and splicing AUPRC separately.
- Synonymous variants are excluded by user decision.
- Compare each of the six nonuniform arms with bridge using one-sided paired match-group permutation tests and Holm-adjust the six p-values.
- Stop after all seven 100-step arm checkpoints and evaluations; do not continue any arm until the user reviews the gate.

CSV metrics and per-variant predictions are authoritative.
W&B is optional.
The existing Lambda A100 remains subject to the original allocation clock, the $48 GPU stop, and the authorized $50 all-in cap.

Focused validation passed: 30 tests covering BOS/no-BOS alignment, selectors, pure teacher KL, teacher-low masking, checkpoint compatibility, retained-cluster launch behavior, exact resume metadata, and statistical tests.
