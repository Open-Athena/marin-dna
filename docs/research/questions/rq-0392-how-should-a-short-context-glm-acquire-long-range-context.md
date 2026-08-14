# How should a short-context gLM acquire long-range context?

## Metadata

| Field | Value |
|---|---|
| Question ID | `RQ-0392` |
| Status | `active` |
| Overall confidence | `low` |
| Evidence considered through | `2026-08-13` |
| Predecessor issues | [#392](https://github.com/Open-Athena/marin-dna/issues/392) |

## Question and scope

How should we turn a genomic language model pretrained on short windows—typically sized to model a single functional element—into a model that can use long-range genomic context while retaining nucleotide-level resolution? At fixed compute and data, how do second-stage long-context language modeling, direct long-context downstream fine-tuning, and a hierarchical local-to-global architecture compare? Which strategy best preserves what the short-context model learned while adding useful dependencies across tens to hundreds of kilobases?

## Current answer

No MarinDNA experiment directly compares long-context strategies. Existing work favors reusing short-context representations through staged, hierarchical, or downstream adaptation, but the best choice depends on required output resolution and compute; confidence is low, and a matched comparison is the main gap.

No MarinDNA experiment directly compares ways to add genuinely long-range context to a short-window model. A 256-versus-512 bp pretraining comparison found no clear VEP difference, so it does not decide behavior at tens or hundreds of kilobases.

Three strategies remain viable. Continued long-context language modeling could produce a reusable base model but adds the most compute and risks diluting functional signal with abundant background sequence. Direct long-context downstream fine-tuning is the minimum-complexity baseline and aligns context learning with labels, but sparse labels may not teach reusable long-range structure. A local-to-global model reuses the short-context encoder and delegates cross-window interactions to a second model; full-resolution tiling preserves base-level outputs, while pooling is probably required at much longer contexts.

A short local encoder has improved a 2,114 bp supervised accessibility model in published work, which supports feasibility of local-to-global transfer. That setup freezes the local encoder, never lets it attend across chunk boundaries, and remains far below chromosome-scale context, so it does not establish long-context language understanding.

Confidence is low on a universal winner. Direct downstream extension should be the baseline; local-to-global modeling is the leading scalable option for nucleotide-resolution tasks; continued language modeling is justified only if gains transfer across several long-range tasks. Any comparison must match parameters, training tokens, and compute and must verify dependence on distant sequence.

## Confidence and limitations

No MarinDNA experiment directly compares long-context strategies. Existing work favors reusing short-context representations through staged, hierarchical, or downstream adaptation, but the best choice depends on required output resolution and compute; confidence is low, and a matched comparison is the main gap.

Confidence is low on a universal winner. Direct downstream extension should be the baseline; local-to-global modeling is the leading scalable option for nucleotide-resolution tasks; continued language modeling is justified only if gains transfer across several long-range tasks. Any comparison must match parameters, training tokens, and compute and must verify dependence on distant sequence.

## Operational consequence

No MarinDNA experiment directly compares ways to add genuinely long-range context to a short-window model. A 256-versus-512 bp pretraining comparison found no clear VEP difference, so it does not decide behavior at tens or hundreds of kilobases.

## Supporting evidence

- [ARSENAL](https://www.biorxiv.org/content/10.64898/2026.02.05.703637v3) pretrains a 350 bp masked DNA model on ENCODE cCREs, tiles frozen per-base embeddings across a 2,114 bp input, and feeds them to a ChromBPNet-style dilated CNN. Across five cell lines it improves accessibility count prediction and caQTL/dsQTL scoring over a matched one-hot model. The [implementation](https://github.com/amanpatel101/arsenal-chrombpnet/blob/dcfa42b1786713e131bb113f4c6d20acc046185d/chrombpnet/chrombpnet.py#L188-L223) freezes the encoder by default, and its [center-out chunking](https://github.com/amanpatel101/arsenal-chrombpnet/blob/dcfa42b1786713e131bb113f4c6d20acc046185d/chrombpnet/chrombpnet.py#L391-L456) retains one embedding per base. This is direct evidence for full-resolution local-to-global transfer. It does not test long-context pretraining, cross-chunk attention in the gLM, pooling, or contexts beyond 2,114 bp.
- [AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0) predicts thousands of functional tracks from 1 Mb sequence using a multiscale supervised architecture. It demonstrates that long context and nucleotide-scale outputs can coexist through hierarchical computation. It does not test initialization from a short-context self-supervised gLM, so it is an architecture precedent rather than evidence for transfer.
- [#396](https://github.com/Open-Athena/marin-dna/issues/396) asks whether gLM pretraining improves sequence-to-function models. Its controlled scratch-versus-pretrained benchmark is a natural first consumer of any local-to-global design, but accessibility context is much shorter than the full long-range question.
- [#395](https://github.com/Open-Athena/marin-dna/issues/395) asks how long-window background sequence should be selected, sampled, or weighted. This matters because uniformly weighted language modeling on long mammalian windows may devote most gradient mass to sequence outside the functional element of interest.

## Contradictory evidence

The predecessor issue did not maintain a separate contradictory-evidence section. Its caveats and negative results are preserved in Current answer and Supporting evidence.

## Related experiments

- [#37](https://github.com/Open-Athena/marin-dna/issues/37) compared 256 bp and 512 bp pretraining contexts and found no clear VEP difference. It supports short windows as a workable local baseline and proposed downstream extension or hierarchy, but the tested lengths are too short to distinguish the three long-context strategies.

## Open questions

- **What long-range capability do we want first?** Candidate tests should require distant context by construction—for example enhancer–promoter interactions, gene-level expression, long-range splicing regulation, or another task where masking distant sequence should measurably hurt.
- **What context lengths define the useful regime?** Compare a small ladder spanning the current local context through the expected biological scale, rather than testing only one large endpoint.
- **How should a second-stage LM sample and weight sequence?** Compare uniform long windows, functional-element-centered windows, and functional or conservation-aware loss weights. How sensitive are the conclusions to incomplete and lineage-specific annotations?
- **Does loss weighting damage likelihood interpretation?** If the long-context model is trained with nonuniform per-base weights, which language-modeling and zero-shot variant scores remain comparable to the short-context model?
- **How do we avoid forgetting local sequence grammar?** Test freezing versus fine-tuning the local model, mixing short and long examples, and progressive context curricula.
- **Should we preserve per-base embeddings or pool them?** ARSENAL-style concatenation supplies a no-pooling baseline. Compare it with mean/attention pooling, learned summary tokens, and multiple coarse tokens per chunk as context grows.
- **How should local windows be tiled?** ARSENAL uses center-aligned, mostly non-overlapping 350 bp chunks with special handling for the two sequence ends. Compare overlap, stride, and blending schemes, and measure prediction or representation discontinuities at internal chunk boundaries explicitly.
- **How does global information return to nucleotide resolution?** Compare broadcasting a global embedding, cross-attention from local positions to coarse tokens, and local decoding with skip connections.
- **Does end-to-end fine-tuning matter?** ARSENAL’s released adapter freezes the local encoder by default. A frozen local encoder is cheaper and gives a clean test of representation reuse, but joint training may be necessary for local features to expose information useful at the global level.
- **What is the fair compute-matched comparison?** Match parameter count where possible and report training tokens, FLOPs, peak memory, and wall time. A longer sequence at the same number of examples is not a compute-matched intervention.
- **Is the model using distant context?** Evaluate with distance-stratified ablations: crop the input, mask or shuffle distal sequence, perturb the candidate interacting element, and measure performance as a function of distance.
- **How should leakage controls change?** Longer overlapping windows increase the chance that homologous or shifted sequence crosses train/test boundaries; update the similarity and locus-holdout rules before interpreting small gains.
- **What is the smallest decisive first experiment?** One option is a downstream task with known long-range dependence and five matched arms: short-context baseline, direct full-resolution extension, frozen ARSENAL-style per-base tiling, frozen-local pooled hierarchy, and jointly trained pooled hierarchy. A second-stage LM arm should follow once that comparison establishes that the task and evaluation can detect useful long-range context.

## History

- 2026-08-14 — Migrated from the predecessor research-question issue [#392](https://github.com/Open-Athena/marin-dna/issues/392). The issue remains the historical source for its original body and comments.
