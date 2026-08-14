# exp232 follow-up: why doesn't distal (enhancer) VEP improve with training?

## Metadata

| Field | Value |
|---|---|
| Question ID | `RQ-0283` |
| Status | `closed` |
| Overall confidence | `unknown` |
| Evidence considered through | `2026-08-13` |
| Predecessor issues | [#283](https://github.com/Open-Athena/marin-dna/issues/283) |

## Question and scope

Why did the `ccre_non_promoter` specialist learn an enhancer constraint signal without improving distal variant-effect prediction, and which changes to training-set curation, window placement, model scale, training duration, or evaluation explain the failure?

## Current answer

The flat distal-enhancer VEP result was largely caused by coding contamination and mixed cCRE classes: a curated follow-up raised distal AUPRC from 0.127 to 0.299 and 0.272 while removing the splicing leak. Enhancer-centered windows later reached 0.366 versus 0.308 for tiled windows, but unequal epoch counts and 58 distal positives leave the anchoring effect unresolved. Confidence is high on the curation diagnosis and low on window-centering causality.

The original enhancer specialist learned a rising enhancer-versus-background likelihood gap while distal Mendelian AUPRC stayed near 0.10–0.13. Its off-domain splicing AUPRC reached 0.238, which suggested that the training population contained coding sequence and mixed regulatory classes.

The curation diagnosis is now supported with high confidence. Excluding exon-overlapping windows and restricting the non-promoter cCRE population raised distal AUPRC from 0.127 to 0.299 and 0.272 in two curated arms, reduced splicing AUPRC to about 0.095, and changed the enhancer-gap-to-distal-AUPRC correlation from −0.10 to 0.89 and 0.95. Coding contamination and target heterogeneity explain most of the original failure.

Enhancer-centered windows later reached distal AUPRC 0.366 versus 0.308 for clean tiled windows. Confidence in a centering effect is low because the centered arm saw about 9.7 epochs versus 3.7, neither learning curve had plateaued, and the distal benchmark contains 58 positives. The curation question is answered; window placement, convergence, scale, and evaluation variance remain open.

## Confidence and limitations

The flat distal-enhancer VEP result was largely caused by coding contamination and mixed cCRE classes: a curated follow-up raised distal AUPRC from 0.127 to 0.299 and 0.272 while removing the splicing leak. Enhancer-centered windows later reached 0.366 versus 0.308 for tiled windows, but unequal epoch counts and 58 distal positives leave the anchoring effect unresolved. Confidence is high on the curation diagnosis and low on window-centering causality.

The curation diagnosis is now supported with high confidence. Excluding exon-overlapping windows and restricting the non-promoter cCRE population raised distal AUPRC from 0.127 to 0.299 and 0.272 in two curated arms, reduced splicing AUPRC to about 0.095, and changed the enhancer-gap-to-distal-AUPRC correlation from −0.10 to 0.89 and 0.95. Coding contamination and target heterogeneity explain most of the original failure.

Enhancer-centered windows later reached distal AUPRC 0.366 versus 0.308 for clean tiled windows. Confidence in a centering effect is low because the centered arm saw about 9.7 epochs versus 3.7, neither learning curve had plateaued, and the distal benchmark contains 58 positives. The curation question is answered; window placement, convergence, scale, and evaluation variance remain open.

## Operational consequence

The original enhancer specialist learned a rising enhancer-versus-background likelihood gap while distal Mendelian AUPRC stayed near 0.10–0.13. Its off-domain splicing AUPRC reached 0.238, which suggested that the training population contained coding sequence and mixed regulatory classes.

## Supporting evidence

- [#227](https://github.com/Open-Athena/marin-dna/issues/227) and [#228](https://github.com/Open-Athena/marin-dna/issues/228) define and materialize the v4 region-labeling contract used by the baseline and curated follow-ups. The setup uses base-pair priority followed by window-majority labeling, a protein-coding TSS band, and zero cCRE flank. These records make the contamination diagnosis reproducible, but they do not compare curation policies through training.
- The v4 validation recipes provide the conserved-versus-background likelihood gaps used as a mechanistic readout. Their main implication is that a rising enhancer gap can coexist with flat distal VEP when the training population is contaminated; the remaining gap is whether the likelihood metric predicts VEP reliably after curation across other region classes.

## Contradictory evidence

The predecessor issue did not maintain a separate contradictory-evidence section. Its caveats and negative results are preserved in Current answer and Supporting evidence.

## Related experiments

- [#8](https://github.com/Open-Athena/marin-dna/issues/8) established the likelihood-gap framing used to compare functional and background sequence. The enhancer case supplies an important limitation: the gap rose while distal VEP stayed flat until the training population was cleaned.
- [#232](https://github.com/Open-Athena/marin-dna/issues/232) produced the original matched six-specialist result. Its non-promoter cCRE arm reached distal AUPRC 0.127, splicing AUPRC 0.238, and a −0.10 enhancer-gap-to-distal-AUPRC correlation, defining the anomaly this question explains.
- [#279](https://github.com/Open-Athena/marin-dna/issues/279) found that missense VEP could degrade with model scale. It weakens the generic “use a larger model” hypothesis but does not directly test enhancer scaling.
- [#326](https://github.com/Open-Athena/marin-dna/issues/326) tested exon-overlap removal and enhancer-only curation. Distal AUPRC rose to 0.299 and 0.272, splicing fell to about 0.095, and the gap-to-AUPRC correlation became strongly positive, supporting coding contamination and target heterogeneity as the main causes.
- [#351](https://github.com/Open-Athena/marin-dna/issues/351) compared clean tiled and enhancer-centered windows at fixed optimizer steps. Centering reached 0.366 versus 0.308 distal AUPRC, but unequal epoch counts, 58 positives, and non-plateaued trajectories leave the anchoring effect unresolved.

## Open questions

- Match tiled and centered arms by both optimizer steps and realized epochs, then repeat with enough seeds to estimate variance.
- Test whether the centering trend persists on a larger or independent distal regulatory benchmark.
- Measure whether longer training or larger models add signal after the curation defects are removed.
- Quantify how much of the remaining uncertainty comes from the 58-positive Mendelian distal set.

## History

- 2026-08-14 — Migrated from the predecessor research-question issue [#283](https://github.com/Open-Athena/marin-dna/issues/283). The issue remains the historical source for its original body and comments.
