# How should genomic anchors be selected and projected across species?

> [!NOTE]
> **TL;DR:** Center-1-bp projection is the default for new multispecies training datasets because it is simpler than full-window projection, increased aggregate species–anchor recovery, and produced broadly similar one-seed development AUPRC trajectories for CDS and enhancer specialists; existing full-window artifacts remain valid historical controls.

## Question

How do human anchor-inclusion and cross-species alignment-projection choices change which genomic regions and species are represented in multispecies DNA language-model training data, and how do those choices affect downstream model quality?
We want to understand the trade-off across genomic-region classes among evolutionary breadth, confidence that an extracted target window is homologous to its human anchor, and usefulness for model training.
Projection yield and clade recovery are intermediate measurements; downstream model performance is the final criterion.

## Current answer

The full-window projection baseline is operational and well specified.
It starts from conservation-filtered 255 bp human anchors, projects the full interval, requires all compatible fragments to map to one target chromosome and strand without overlap, takes their outer span, accepts spans from 128 to 512 bp, and resizes around the span midpoint to 255 bp.
The final target window can therefore contain unaligned flanking sequence.

[Experiment #473](../experiments/473-center-seeded-projection.md) compared that contract with center-1 projection, which projects the central nucleotide, requires a unique target locus, and extracts a fixed 255 bp target-genome window around it.
Across five regions, center-1 recovered 82.28% of requested species–anchor pairs, compared with 79.60% for full-window projection.
The difference was region-specific: CDS recovery increased by 2.679 percentage points, while enhancer-centered cCRE recovery decreased by 0.892 points.
The sampled reverse-trace audit found no general increase in external flank.

At matched tokens, the corrected one-seed development AUPRC trajectories showed no consistent policy advantage across the region-relevant CDS and enhancer benchmarks.
All three terminal CDS Mendelian paired intervals included zero.
The terminal enhancer Mendelian result favored full-window projection by 0.056 with a paired interval excluding zero, but distal Complex-trait AUPRC was nearly identical and no additional seed tested replication.

Center-1 is the operational default for new projection datasets because its projection contract is simpler, aggregate recovery is higher, and the downstream trajectories are broadly similar.
This choice does not establish statistical equivalence.
Existing full-window rules and artifacts remain available for reproducibility and historical comparisons.
Projection yield alone remains insufficient for future policy decisions, especially in regions without matched downstream training.
Tiny fragments, duplicated loci, midpoint definition, span thresholds, multiple landmarks, and fragment-selection policies remain open choices for other region classes.

<details>
<summary>Related work</summary>

- [#121](https://github.com/Open-Athena/marin-dna/issues/121) developed the early Zoonomia HAL enhancer-projection path.
  It established operational feasibility for human-anchored mammalian data but was limited to an enhancer-oriented pipeline.
- [#149](https://github.com/Open-Athena/marin-dna/issues/149) generalized human-anchored projection into a genome-wide mammalian training program.
  It established the reusable data concept; projection semantics were inherited rather than experimentally optimized.
- [#153](https://github.com/Open-Athena/marin-dna/issues/153) compared MAF streaming with halLiftover using no-duplicate mappings and selected HAL for speed and operational simplicity.
  It also established the single-locus, 128–512 bp outer-span, midpoint-resize baseline.
  The comparison chose a backend, not the best biological projection policy.
- [#227](https://github.com/Open-Athena/marin-dna/issues/227) fixed the v4 base-pair and window-labeling contract used after projection.
  It separates annotation semantics from projection semantics but does not measure homolog quality.
- [#230](https://github.com/Open-Athena/marin-dna/issues/230) and [#233](https://github.com/Open-Athena/marin-dna/issues/233) added rank-based species subsetting and built the 19-order cohort.
  They make recovery-versus-species-density experiments possible; they do not choose anchor or projection policy.
- [#417](https://github.com/Open-Athena/marin-dna/issues/417) applies one projection contract across Zoonomia HAL and UCSC MultiZ sources and records backend-uniform rejection reasons.
  It improves comparability and accounting while leaving the contract itself unvalidated.
- [Ye et al., Predicting functional constraints across evolutionary timescales with phylogeny-informed genomic language models](https://pmc.ncbi.nlm.nih.gov/articles/PMC12458161/) shows that GPN-Star calibrated entropy can outperform classical phyloP or PhastCons for constraint and variant prioritization and that primate, mammal, and vertebrate timescales differ by endpoint.
  This motivates alternative anchor-inclusion scores.
  It does not determine the best projection semantics.
- The released [GPN-Star genome-wide scores](https://huggingface.co/datasets/songlab/gpn-star-scores) provide calibrated position-level entropy at primate, mammal, and vertebrate timescales.
  Count-matched thresholds can separate score choice from selected-genome fraction; within-window selected-base fraction remains a separate axis.

</details>

<details>
<summary>Related experiments</summary>

- [#120](https://github.com/Open-Athena/marin-dna/issues/120) compared cheap pairwise methods against lifted enhancer orthologs.
  It showed that local similarity search can recover useful cross-species cCRE candidates with a measurable recall frontier and motivated projection alternatives.
- [#136](https://github.com/Open-Athena/marin-dna/issues/136) compared enhancer-curation strategies through training and VEP.
  Human-anchored projection outperformed per-species segmentation in that setting, supporting the general projection idea without isolating fragment semantics.
- [#160](https://github.com/Open-Athena/marin-dna/issues/160) trained on the first whole-genome and TSS-proximal Zoonomia projections.
  The successful runs established end-to-end viability of the baseline dataset, not its optimality.
- [#166](https://github.com/Open-Athena/marin-dna/issues/166) scaled models on the whole-genome projected dataset.
  It tests whether the materialized corpus supports larger training, while anchor and projection choices remain fixed.
- [#177](https://github.com/Open-Athena/marin-dna/issues/177) audited N stretches and repeat masking in the projected corpus.
  It did not find major corruption from those sources and documented repeat handling; homolog correctness was outside scope.
- [#187](https://github.com/Open-Athena/marin-dna/issues/187) trained 1B region specialists on the v3 projected partitions.
  It showed that projected region-specific corpora carry differentiated training signal but used the inherited projection policy.
- [#213](https://github.com/Open-Athena/marin-dna/issues/213) measured how VEP positives intersect conservation-selected anchors.
  It found much higher Mendelian than complex-trait coverage, showing that anchor inclusion changes downstream scope before projection begins.
- [#221](https://github.com/Open-Athena/marin-dna/issues/221) compared region-label assignment policies and found that 34% of anchors changed label.
  It shows that annotation assignment is a major independent source of dataset variation.
- [#232](https://github.com/Open-Athena/marin-dna/issues/232) trained matched v4 region specialists.
  The home-domain diagonal supports the usefulness of the projected partitions and supplies downstream metrics for future policy comparisons.
- [#255](https://github.com/Open-Athena/marin-dna/issues/255) compared 108-family and 19-order projected cohorts at matched compute.
  Region-dependent differences show that species recovery and density can matter even when projection semantics are fixed.
- [#351](https://github.com/Open-Athena/marin-dna/issues/351) compared clean enhancer-centered anchors with uniform tiling.
  Centering was suggestively better for distal VEP but unequal epoch counts prevented attribution to anchor geometry.
- [#353](https://github.com/Open-Athena/marin-dna/issues/353) compared human-anchored nucleotide CDS projection with native per-species annotation at vertebrate and animal scales.
  Projection produced useful data but lost distant species and did not dominate every evaluation, directly exposing the recovery-versus-construction tradeoff.
- [Experiment #473](../experiments/473-center-seeded-projection.md) compared full-window and center-1 projection on fixed anchors with recovery, reverse-trace QC, matched-token training, paired Mendelian uncertainty, Complex traits, and SGE.
  Its corrected one-seed development trajectories are broadly similar, supporting center-1 as the simpler default while retaining full-window artifacts as historical controls.

</details>

<details>
<summary>Possible directions</summary>

- For regions beyond CDS and enhancer-centered cCREs, which projection semantics best balance species recovery with confidence that the extracted target window is homologous to the human anchor: full-window projection, center-seeded projection, multiple projected landmarks, or fragment/locus-based alternatives?
- Within full-window projection, should tiny fragments be removed before locus checks, should one fragment be selected as canonical, or should all compatible fragments define the target span?
  How should same-chromosome/strand requirements, span-length cutoffs, midpoint choice, and permitted unaligned flank vary?
- How do phyloP and GPN-Star inclusion criteria, the evolutionary timescale of the score, the base-level cutoff, and the required within-window selected fraction change genome-wide anchor composition and the percentage of primates, mammals, and vertebrates recovered?
- How heterogeneous are these effects across genomic annotations and sequence contexts?
  CDS-centered and cCRE-enhancer-centered windows are useful contrasting initial probes, rather than the scope of the research question itself.
- Which projection diagnostics are needed beyond overall yield, including unique versus multiple mappings, aligned coverage around center-seeded windows, and per-species, per-clade, and per-region recovery?
- How should controlled training comparisons separate sequence quality and evolutionary breadth from dataset quantity, and which independent coding, regulatory, and genome-wide evaluations should determine success?

</details>
