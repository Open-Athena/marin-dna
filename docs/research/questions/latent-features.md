# What latent biological features do gLMs learn?

## TL;DR

Sparse-autoencoder analyses identify reproducible splice, stop-codon, accessibility, and coding-context feature responses across layers and dictionaries, while several broader semantic claims failed to replicate. The program is paused indefinitely; confidence is moderate for the local causal features and low for a complete biological inventory, with reference-annotation and broader variant tiers still unfinished.

## Question

What biologically meaningful latent features do genomic language models learn, how are those features organized across layers, orientations, training stages, and model families, and what genomic computations do they support?

MarinDNA m5.1 is the first experimental system for this question, not its scope. We will use sparse autoencoders as the best reasonable feature-discovery instrument available today. The goal is not to establish that SAEs beat every raw-activation, probe, or sequence method. Comparisons and controls should be included only when they rule out a concrete artifact or are necessary for a stronger robustness or causal claim; the priority is rapid, FDR-controlled association discovery, biological interpretation, and targeted follow-up.

**Primary discovery principle:** prioritize how features change between matched reference and alternate sequences at real or designed variants. Paired Δ features are less studied than single-sequence annotation features, align directly with variant effect prediction, and provide a local counterfactual that removes much background-locus variation. Absolute reference/alternate activations, human-reference inventories, and sequence logos remain essential supporting views for interpreting a response, but they are not the main endpoint.

## Current answer

The program is paused indefinitely as of 2026-08-10, and all scoped experiment issues are closed. The strongest result is that m5.1 contains sparse features with reproducible local biological responses, but the inventory is selective and strongly dependent on layer, dictionary, orientation, and response definition.

Splice-acceptor, splice-donor, and stop-creation responses survive targeted perturbation, untouched-context replication, and transfer across independently trained dictionaries. A broad synonymous/codon-degeneracy interpretation did not replicate. A better-trained dictionary instead exposed a narrow leucine-codon-family response. These results support local causal sequence grammars with moderate confidence and argue against assigning semantics from decoder similarity or a single association.

Layer choice is task-specific. Middle layers performed best on the frozen broad-consequence endpoint, while final-layer features carried the strongest Mendelian-label, complex-trait-label, and accessibility-direction associations. Reconstruction quality improved with more SAE training but did not reliably predict biological yield. Feature counts also overstate biological multiplicity because correlated response families can contain many feature IDs.

The paired-variant protocol is part of the current answer: retain reference, alternate, signed change, and magnitude; report forward and reverse-complement orientations separately; predeclare any aggregate; test all eligible features with complete-family correction; and distinguish association, interpretation, and intervention. Unsigned AlphaGenome associations currently identify broad GC/CpG-conditioned accessibility or promoter-effect magnitude rather than tissue-specific semantics. Accessibility-direction discoveries are highly redundant and overlap broad consequence response, so accessibility causality remains untested.

Reference-sequence analyses show substantial repeat and annotation-state capacity. Final-layer repeat information remains after strict composition matching, while promoter-like sequence is easier to identify than enhancer-like sequence. Composition/repeat qualification and exact boundary localization are still needed before claiming genomic segmentation. The next useful biological tiers are UTR, promoter/TSS-proximal, and annotated ncRNA variants, followed later by enhancer/cCRE variants.

<details>
<summary>Related work</summary>

- [Inside a genomic language model](https://marindna-latent-feature-atlas.gsbenegas.chatgpt.site) is the current visual synthesis of the project’s methods, layer organization, positive findings, negative results, and claim boundaries. The [commit-pinned source](https://github.com/Open-Athena/marin-dna/blob/234f3073121d8de1f51a5e16e8637ac1986152e2/experiments/issue288_latent_feature_atlas/index.html) preserves the reviewed version. It is a summary artifact, not independent evidence.
- [Korsakova and Kelley, Learning monosemantic features in multitask DNA regulatory sequence models via sparse autoencoder decomposition](https://openreview.net/forum?id=AlLZnZX01x) trained TopK SAEs on early Borzoi layers and annotated features with repeats, motifs, and regulatory elements. Motifs specialized by depth, orientation, and flanking context, and larger dictionaries split concepts. This motivates layer panels, normalized activations, FWD/RC inspection, and context-aware interpretation. The remaining gap is transfer from a supervised regulatory model to paired variant effects in a causal gLM.
- [Evo 2 feature work](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1) trained a large BatchTopK SAE on a genomic foundation model and recovered biological sequence features at scale. It supports SAE feasibility but does not establish that individual features are monosemantic or causally relevant to variant effects.
- [Language Modeling Materializes a World Model of Protein Biology](https://www.biorxiv.org/content/10.1101/2024.12.18.629098v1) reports concept splitting, decoder neighborhoods, and feature combinations in protein models. It motivates analyzing feature families and co-activation systems alongside individual IDs. The open question is whether the same organizational principles hold for genomic sequence and across reverse-complement views.
- The fixed methodology treats biological labels and automated descriptions as discovery aids. A strong interpretation requires corrected association evidence, sequence localization, paired or designed perturbations, orientation analysis, and independent contexts when the claim warrants it. Causal intervention on the base model remains a higher bar than activation response alone.

</details>

<details>
<summary>Related experiments</summary>

- [#418](https://github.com/Open-Athena/marin-dna/issues/418) trained and validated the first production-shaped m5.1 block-10 BatchTopK SAE. It established workable reconstruction and sparsity and produced initial splice and nucleotide features, but did not by itself validate biological semantics.
- [#420](https://github.com/Open-Athena/marin-dna/issues/420) tested the first fixed feature panel on Mendelian variants. Pathogenic label was null overall and within subsets, while seven-way consequence/region prediction reached macro-AUPRC 0.2409 versus 0.1429 chance, showing encoded region information without pathogenicity separation.
- [#421](https://github.com/Open-Athena/marin-dna/issues/421) scanned unsigned SAE changes against AlphaGenome tracks. Leading block-10/19 signals were broad GC/CpG-conditioned accessibility or promoter-effect magnitude features; tissue-specific interpretations failed robustness checks, and one tail-sensitive lead instead connected to Mendelian label.
- [#422](https://github.com/Open-Athena/marin-dna/issues/422) ran the broad 35-consequence inventory across blocks 1, 10, and 19. Splice, stop, missense, miRNA, promoter-like, and 5′ UTR signals were selective rather than universal, and FWD/RC discoveries agreed in direction when shared despite incomplete ID overlap.
- [#424](https://github.com/Open-Athena/marin-dna/issues/424) froze the paired-variant and strand protocol. It retained ref, alt, signed and unsigned responses, required separate FWD/RC reporting, and selected signed mean as the then-best global reducer while documenting that same-ID strand invariance is uncommon.
- [#426](https://github.com/Open-Athena/marin-dna/issues/426) compared SAE layer and 5M versus 25M activation budgets. Block 10/5M led the frozen broad-consequence endpoint, later layers led some coding tasks, and better reconstruction at 25M did not consistently improve biological yield.
- [#428](https://github.com/Open-Athena/marin-dna/issues/428) replicated fixed coding-context features on a larger phase- and substitution-matched panel. Effects persisted but attenuated, and a local 31-bp sequence model was stronger, narrowing the interpretation to orientation-complementary local coding context.
- [#429](https://github.com/Open-Athena/marin-dna/issues/429) prospectively replicated causal acceptor, donor, stop-creation, and synonymous/codon-degeneracy perturbation effects on untouched contexts. This established the first local causal feature set for the program.
- [#431](https://github.com/Open-Athena/marin-dna/issues/431) transferred those semantic queries across independently trained dictionaries. Acceptor, donor, and stop creation replicated; synonymous/codon degeneracy did not, despite stable decoder-neighborhood geometry.
- [#432](https://github.com/Open-Athena/marin-dna/issues/432) repeated the causal protocol in the healthier 25M dictionary. Splice and stop semantics persisted; the broad synonymous interpretation remained null, while exhaustive search found a narrow leucine-codon-family response.
- [#434](https://github.com/Open-Athena/marin-dna/issues/434) scanned accessibility-QTL causality and direction. A final-layer signed-direction family appeared in the 559-positive dsQTL pilot, but it was highly redundant, overlapped broad consequence response, and did not test causality because negatives were absent.
- [#435](https://github.com/Open-Athena/marin-dna/issues/435) mapped repeat capacity across blocks 1, 10, and 19. Repeat hierarchy became more specific with depth; final-layer repeat information survived composition matching, and selected repeat grammars showed causal motif-loss responses, while many shallower signals were composition-linked.
- [#436](https://github.com/Open-Athena/marin-dna/issues/436) mapped Mendelian pathogenicity across layers. Individual-feature signal was weak in blocks 1/10 and stronger but distributed in block 19; focal associations do not yet bridge the full performance of official likelihood or whole-window embedding readouts.
- [#438](https://github.com/Open-Athena/marin-dna/issues/438) mapped complex-trait label signal across layers. Block 19 dominated the fixed panel, magnitude was more useful than signed change, and recurrent feature 1662 transferred to held-out data but currently supports local coding-impact/codon-context sensitivity rather than a broad causal mechanism.
- [#440](https://github.com/Open-Athena/marin-dna/issues/440) tested reference annotation states. The first pass found layer-specific gene-state inventories and strong promoter-like signals but little enhancer-specific signal; composition/repeat controls and transcript-boundary case studies remain unfinished.

</details>

## Possible directions

### 1. Paired-variant protocol: current answer and extensions

Reference and alternate sequences are a matched counterfactual pair, not two independent examples. This paired response is the default scientific unit for [#288](https://github.com/Open-Athena/marin-dna/issues/288); single-sequence enrichment is supporting interpretation. [#424](https://github.com/Open-Athena/marin-dna/issues/424) established the current protocol:

- retain reference activation, alternate activation, signed Δ, |Δ|, and inactive→active / active→inactive states for each orientation;
- use signed Δ as the primary variant-effect quantity because threshold crossings miss graded splice effects and magnitude-only views erase informative direction for dense features such as the stop-codon feature;
- use all eligible variants, outcome-independent feature-support requirements, complete-family BH correction, and effect-size reporting for association discovery; reserve genomic holdouts and block-bootstrap uncertainty for targeted robustness or causal follow-up;
- preserve absolute activations so a background-locus feature can be distinguished from a variant-induced feature change.

Still open:

- [#429](https://github.com/Open-Athena/marin-dna/issues/429) has now frozen and tested the first spatial extension: reverse RC positions into genomic coordinates; preserve focal scoring; choose focal, ±15 bp local maximum, or local sum on validation blocks only; and report held-out profiles. Local maximum decisively improves acceptor and stop-gain features, while donor-fifth-base and synonymous remain focal;
- how to condition or stratify on substitution type, local annotation, repeat family, and labels when needed without turning the program into a general baseline bake-off;
- the direct missense-versus-synonymous task is heavily phase-confounded: a discovery-fit codon-position score reaches held-out AUROC 0.927. [#428](https://github.com/Open-Athena/marin-dna/issues/428) now replicates the three frozen SAE scores after matching within codon position and transcript-oriented substitution, but at conditional AUROC 0.590–0.603 rather than the pilot's 0.825–0.853; a fixed 31-bp local-sequence model reaches 0.760. Treat these as stable but modest codon-context features. CDS and splice variants remain the first substantive biological tier as well as the protocol proving ground; broaden beyond missense-versus-synonymous and do not spend the whole program trying to force an amino-acid interpretation.

### 2. Bidirectionality and reverse complementation

FWD and RC remain primary, separately reported views. [#424](https://github.com/Open-Athena/marin-dna/issues/424) found that same-ID rowwise invariance is uncommon: on held-out variants only 3.86% of adequately supported features had `|r| ≥ 0.05`. Nevertheless, signed same-ID mean won the global validation comparison and transferred best to held-out blocks because useful features can cover complementary strand contexts.

Current decision:

- use `(ΔFWD + ΔRC) / 2` as the primary aggregate for broad consequence-family screens, retain both component views, and report `max(|ΔFWD|, |ΔRC|)` as a sensitivity analysis; [#426](https://github.com/Open-Athena/marin-dna/issues/426) shows that max absolute may become the better endpoint for a narrower binary mutation-response question, but this must be declared per analysis rather than selected post hoc;
- explicit coding-strand alignment is not a better reducer for the three selected coding IDs: aligned versus anti-aligned AUROC is 0.643/0.656 for f12658 and 0.632/0.629 for f13637, both below signed mean; f11064 coding-aligned absolute is 0.734 versus 0.806 for max absolute. These features are not currently strict transcript-orientation features;
- do not enforce same-ID RC consistency in the first SAE sweep;
- begin with RC-balanced SAE training examples under the ordinary objective;
- treat brute-force cross-ID matching as exploratory. Stable pairs exist, but 13–14 of 70 discovery winners became constant on validation/test, demonstrating winner's curse.

Still open:

- whether mutual-nearest, support-stable cross-ID pairs preserve biological semantics across datasets;
- whether set-level consistency is more appropriate than feature-index consistency;
- when strand-specific biology should intentionally remain unaggregated.

### 3. SAE quality and scaling

Experiment [#426](https://github.com/Open-Athena/marin-dna/issues/426) completed the first coarse budget × layer comparison. It rejects two convenient shortcuts: the final layer is not globally best, and longer training / better reconstruction do not imply greater biological feature yield. Blocks 10/16 are the best starting points for broad consequence discovery, while block 19 remains valuable for narrower coding contrasts. The 25M arm is not an upper bound: later runs may scale up to eight billion activations when a biological endpoint, feature stability, or model-relevant reconstruction—not MSE alone—justifies the cost and the run receives its own compute authorization.

[#431](https://github.com/Open-Athena/marin-dna/issues/431) adds a sharper failure mode: the [#426](https://github.com/Open-Athena/marin-dna/issues/426) and fresh 5M normalized exports both have negative FVE and JumpReLU L0 around 1,700, while the existing block-10/25M export reaches FVE 0.809 and L0 242.8. [#432](https://github.com/Open-Athena/marin-dna/issues/432) then showed that the healthier 25M checkpoint preserves the robust splice/stop features and exposes a narrow leucine codon-family response, but does not recover a broad or decoder-aligned synonymous feature. Reconstruction health appears useful for resolving specificity, yet still does not substitute for biological association and targeted causal evidence.

SAE improvements should be evaluated by rerunning the fixed first/middle/final biological panels; the layer hypothesis does not prescribe a particular SAE recipe.

1. Treat [#426](https://github.com/Open-Athena/marin-dna/issues/426)'s block-4/10/16/19 sweep as evidence that layer effects are task-specific, while using the fixed first/middle/final panel for general association mapping; do not infer a globally best layer from one outcome.
2. Vary dictionary expansion, K/L0 or threshold sparsity, BatchTopK versus related architectures, and random seed, comparing FDR-controlled biological association yield and feature stability rather than reconstruction alone.
3. Diagnose the 25M block-19 concentration (inactive fraction 0.568; top-1% activation share 0.748) before scaling that recipe; use larger activation budgets up to eight billion only when a measured bottleneck justifies them.
4. Test whether RC-balanced SAE training or an explicit orientation objective changes invariant-feature yield.
5. Retain reconstructed-model CE/KL degradation alongside FVE, cosine, L0, dead-feature, activation-concentration, and compute diagnostics; [#426](https://github.com/Open-Athena/marin-dna/issues/426) shows that even these health metrics are supporting evidence rather than the scientific endpoint.
6. Normalize feature activations against a fixed human-reference corpus before comparing activation magnitudes or ranking features with different dynamic ranges and prevalence.
7. Compare decoder-space neighborhoods and direction alignment across dictionaries, checkpoints, and seeds; measure biological yield for feature families as well as individual IDs.
8. After robust individual features emerge, test whether co-activation modules capture higher-order motif or region grammar.

The decisive research metrics are FDR-controlled biological association yield, feature and feature-family stability across seeds/checkpoints, motif/context coherence, FWD/RC behavior, and causal validation. Reuse fixed datasets and analysis definitions when comparing SAE recipes; preserve untouched contexts only for targeted robustness, transfer, or causal claims that require them.

### 4. Biological follow-up

The literature suggests the following prioritized **variant-response** inventory for short-context m5.1 analysis. Reference-sequence scans are useful both for naming and localizing Δ features and as a distinct annotation/segmentation application:

- **Sequence and composition responses:** how substitutions perturb nucleotide, GC/CpG, homopolymer, low-complexity, and local compositional features.
- **Repeats and mobile sequence:** variant responses stratified by repeat class, family, subfamily, and divergence/age—including endogenous retroviral/LTR, LINE/SINE, satellite, and simple-repeat sequence. Audit both Δ-responsive feature IDs and their share of nonzero Δ slots and total |Δ| mass; reference activation capacity is a secondary diagnostic.
- **Local regulatory variant grammar:** perturbations in 5′ and 3′ UTRs, TF and RBP motifs, motif orientation/flanking context, promoters/TSS, cCRE categories, and local motif combinations or spacing.
- **Gene architecture and RNA variants:** splice donors/acceptors, polypyrimidine tracts, exon/intron boundaries, polyadenylation signals, ORF/intergenic sequence, and annotated tRNA, rRNA, miRNA, snoRNA, lncRNA, and other ncRNA classes.
- **Coding variant effects:** codon phase/frame, start and stop signals, synonymous versus missense changes, frameshifts, premature stops, and amino-acid or protein-secondary-structure signals recoverable from coding DNA.
- **Feature organization:** FWD/RC feature pairs or sets, broad-to-specific feature hierarchies, context-specialized members of one motif family, decoder-space neighborhoods, and compositional feature programs.

The practical paired-variant ladder is: **CDS and splicing variants as the first biological step and substantive tier** → 5′/3′ UTR variants → promoter/TSS-proximal and annotated ncRNA variants → enhancer/cCRE variants. Enhancers are the hardest tier because their local sequence vocabulary is heterogeneous and cell-state dependent; claims about distal enhancer–promoter relationships remain out of scope for a 255 bp model.

Applied follow-ups:

- **Reference-sequence segmentation and gene annotation ([#440](https://github.com/Open-Athena/marin-dna/issues/440)):** Scan the human reference at base-pair resolution for features whose activation identifies local genomic states or sharply marks boundaries—for example exon versus intron, CDS versus 5′/3′ UTR, splice boundaries, promoter/TSS, intergenic sequence, and cCRE classes. Evaluate both pointwise class association and boundary localization, keeping FWD/RC separate initially. This is a standalone application of the learned feature inventory rather than a variant-effect endpoint; for the 255 bp m5.1 system it concerns local sequence annotation, not long-range regulatory interactions.
- **CDS and splicing variants:** This is the first biological step, because known splice and genetic-code grammar provides unusually strong falsification tests while remaining central to variant effect prediction. [#429](https://github.com/Open-Athena/marin-dna/issues/429) has held-out donor/acceptor/stop/synonymous leads, matched discovery-context perturbations, and a prospective hash-sampled test-context panel on which all five original-dictionary causal contrasts clear zero. [#431](https://github.com/Open-Athena/marin-dna/issues/431) then showed that acceptor, donor, and stop semantics survive two independent dictionaries even when IDs permute, while synonymous/codon degeneracy does not survive either 5M dictionary or either exhaustive search. [#432](https://github.com/Open-Athena/marin-dna/issues/432) confirmed the three robust semantics at 25M and found only a narrower leucine codon-family/local-degeneracy feature under exhaustive search, not general synonymous semantics. Broader missense, start/stop-loss, splice-subclass, and real frameshift/indel panels remain separate follow-ups. This tier remains scientifically important after the protocol is validated.
- **Local regulatory variant ladder:** Find Δ features for 5′/3′ UTR variants first, then strand-aware promoter/TSS-proximal and annotated ncRNA variants, before the harder enhancer/cCRE tier. Use FDR-controlled paired associations, spatial localization around the edit, motif/context inspection, and targeted mutagenesis at every stage; add independent-context replication when advancing a stronger mechanistic claim.
- **Repeat response capacity:** At each tier, quantify repeat class/family/subfamily associations and the share of Δ-responsive feature IDs, nonzero Δ slots, total |Δ| mass, and decoder neighborhoods devoted to repeat-overlapping variants. Reference-sequence capacity is supporting context.
- **Regulatory interpretation:** [#421](https://github.com/Open-Athena/marin-dna/issues/421) now characterizes block-19 feature 219 and block-10 feature 11137 as broad, strand-reproducible, GC/CpG-conditioned accessibility/promoter-effect magnitude features. Feature 11928's AlphaGenome association is tail-driven, but its Mendelian-label association is strand-consistent and should be compared with [#436](https://github.com/Open-Athena/marin-dna/issues/436)'s final-layer inventory. Next resolve track-composition robustness, then use targeted perturbation for mechanism and review the signed/default-scorer phase.
- **Curriculum and generalization:** Which features appeared or sharpened during the three-way to five-way continuation, and do they generalize across held-out human loci and homologous mammalian loci?
- **Causality:** For a small preregistered set of robust features, do in-distribution ablation or clamping change downstream predictions in the biologically expected direction?
- **Efficiency:** What is the cheapest correctness-equivalent extraction and SAE training recipe that preserves the scientific conclusions?

The authoritative forward-looking queue is maintained in **Current execution plan** above.

Session compute ledger: approximately **$35.2 / $50** for the autonomous work authorized in this session, including $1.17 for the shared [#420](https://github.com/Open-Athena/marin-dna/issues/420)/[#421](https://github.com/Open-Athena/marin-dna/issues/421) c7i.8xlarge CPU run, approximately $4.45 for [#436](https://github.com/Open-Athena/marin-dna/issues/436) block-1 training, paired extraction, and the focal statistical scan, and approximately $0.81 for [#438](https://github.com/Open-Athena/marin-dna/issues/438)'s feature-1662 saturation extraction and analysis, plus approximately $2.9 for [#435](https://github.com/Open-Athena/marin-dna/issues/435)'s repeat inventory, reference and variant panels, A10G extraction, sparse CPU associations, sensitivity passes, paired-delta analysis, motif/context pass, causal single-base saturation, and repeat-aware Mendelian-label analysis, under $1 for [#422](https://github.com/Open-Athena/marin-dna/issues/422)'s three-layer H100 extraction plus complete-family CPU analysis and audit, under $1 for [#434](https://github.com/Open-Athena/marin-dna/issues/434)'s positive-panel materialization plus H100 direction extraction and scan, and $0.53 for [#434](https://github.com/Open-Athena/marin-dna/issues/434)'s f1829 mutagenesis/intervention pass. This session may use at most one paid CPU instance and one paid GPU instance concurrently; the limit is per session, not global across other Codex sessions. No paid [#288](https://github.com/Open-Athena/marin-dna/issues/288) CPU or GPU instance is currently running in this session. This is not a lifetime budget for [#288](https://github.com/Open-Athena/marin-dna/issues/288); future sessions and experiments may receive separate compute authorizations.
