## Protocol amendment: continue uniform CE and pure teacher KL to step 200

This amendment is recorded before launching any additional optimizer step.

### Fixed lineage

- Parent run: `364abd024f3c-20260824t1320z` at implementation commit `480e6e17a9097d29f0881cbef5506d244276f89e`.
- Student initialization: `exp58-animals-r01-1e3682` Hugging Face step 1,000.
- Frozen teacher: the compatible final step-16,999 export from the same run.
- Training data: `marin-dna/genomes-v4-genome_set-animals-intervals-v5_256_128` revision `04d374450a0f78f0ab5e17a8bc7b7c4baeb8295c`.
- Evaluation data: pinned Mendelian train split, pooled `missense_variant` plus `splicing`, with 8,990 variants in 899 matched groups.
- Compute: retain the existing Lambda A100 at $1.99/hour under the $48 GPU stop and $50 all-in ceiling.

### Arms and continuation

- Continue only `uniform-100`, using hard CE on every eligible nonrepeat target.
- Continue only `teacher-kl-full`, using pure `KL(teacher || student)` at temperature 1 on every eligible nonrepeat target.
- Resume each arm's full step-100 Lightning checkpoint rather than restarting or forking weights.
- Restore the arm-specific model, AdamW moments, scheduler position, selector state, data position, and Python, NumPy, PyTorch, and CUDA RNG state.
- Train exactly 100 more arm-local steps, ending each arm at local step 200.
- Preserve the constant 1e-3 plateau already reached after the original 20-step arm warmup, with no new warmup.
- Consume absolute sequence-plan rows 409,600 through 614,400 at effective batch 2,048.
- Materialize a longer deterministic plan and require its first 409,600 sequence and species rows to hash identically to the completed parent plan.

### Evaluation and inference

- Evaluate both step-200 checkpoints on the identical pinned missense-plus-splicing frame.
- Report pooled, missense, and splicing AUPRC for both arms.
- Primary comparison: teacher KL step 200 versus uniform CE step 200 using a two-sided paired match-group swap test with 20,000 permutations.
- Exploratory trajectories: compare each arm at step 200 with its own step-100 predictions using two-sided paired match-group swap tests with 20,000 permutations.
- Adjust the two trajectory p-values with Holm familywise correction.
- Treat the primary test as descriptive evidence from this one paired run rather than a population-level training-seed claim.

### Stop rule and artifacts

- Stop after both arms reach step 200 and both evaluations and registered tests finish.
- Do not continue random-50 or any ranked-half arm in this extension.
- Do not launch the queued RefSeq restart as part of this extension.
- Store extension checkpoints, CSV predictions, JSON summaries, plan-lineage hashes, comparison statistics, runtimes, and final manifest under a new immutable issue-owned S3 snapshot.
- Leave W&B disabled and use CSV as the authoritative metric record.
