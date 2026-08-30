# Issue 535: 4B uniform five-region continuation

This locked experiment continues the permanent native checkpoint at global step 215573 for 160,000 added steps.
It uses an equal-weight mixture of CDS, upstream, downstream, ncRNA exon, and non-promoter cCRE caches.
The 20-step smoke and production run have separate artifact and W&B identities.

The smoke output is `gs://marin-us-east5/checkpoints/dna-exp535-4b-uniform-five-region-smoke-v5p8/2026.08.30`.
The production output is `gs://marin-us-east5/checkpoints/dna-exp535-4b-uniform-five-region/2026.08.30`.
Native recovery checkpoints are under each output's `checkpoints/` directory, and HF exports are under `hf/`.
The production checkpointer keeps rolling 10-minute recovery states, permanent state 343573 immediately before cooldown, and forced final state 375573.

Launch the smoke from this directory with a batch-priority Iris coordinator in `us-east5` and `--mode smoke --run`.
After confirming native restore, `v5p-8`, finite advancing loss, native checkpoint 215593, and HF export 215593, launch the same committed code with `--mode production --run`.
Do not switch accelerator family or storage backend without updating issue #535.
