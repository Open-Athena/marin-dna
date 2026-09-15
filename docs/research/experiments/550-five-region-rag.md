# Five-region RAG with order-level vertebrate representatives

> [!NOTE]
> **TL;DR:** The broader-context, longer-trained 46M RAG recipe improved Mendelian and SGE zero-shot point estimates and all three frozen probes over the earlier 46M prototype, but Complex Traits zero-shot AUPRC fell; human was the only primate, so human VEP had no non-human primate context.

## Findings

The five-region recipe extended the small fixed-ortholog RAG result beyond seven mammals, using available windows from a panel of 39 non-human vertebrate representatives plus human.
At the final checkpoint, Mendelian and SGE zero-shot macro AUPRC exceeded the earlier 46M prototype by 0.0453 and 0.0326, while Complex Traits was lower by 0.0262.
Frozen-probe point estimates increased on all three benchmarks, with the largest change on Mendelian, from 0.4088 to 0.5502.
These are comparisons between complete recipes, not estimates of the effect of adding species or deduplicating taxonomic orders.

More optimization within this lineage did not improve every readout uniformly.
Between 50,000 and 100,000 updates, Mendelian and SGE zero-shot point estimates rose, Complex Traits fell slightly, and all three frozen-probe estimates rose.
The strongest GPN-Star arm still led each zero-shot macro benchmark, including all six eligible Complex Traits subsets.
The result supports further study of small retrieval-conditioned readers while leaving the Complex Traits deficit unresolved.

## Evidence

Final aggregate validation language-modeling loss was 0.5454, with an equal-region macro loss of 0.5423.
Validation used 2,000 chr18 documents, 400 from each region; every chr18 human anchor was excluded from training.
Loss covered nonpadding next-token targets across all available species segments, including human, rather than human targets alone.
Different species availability, ordering, and missing-sequence conventions prevent treating this loss as a matched comparison with the earlier prototype.

One 46M causal model trained from scratch on 1,110,006 published documents with equal sampling weight for CDS, TSS/5′ UTR, 3′ UTR, ncRNA, and enhancers.
Training ran for 100,000 updates of 200 documents, or 204.8 billion allocated token positions including masked padding.
The earlier [fixed-ortholog prototype](402-fixed-ortholog-rag.md) trained for 30,000 updates and 62.9 billion token presentations.
The optimizer schedule was resolved for the new batch and training horizon.

Each new document contained available 255-base windows from 18 non-human mammalian and 21 non-mammalian vertebrate order representatives, plus human as the sole primate representative.
A reproducible permutation varied across documents and stayed fixed on repeated presentations; human participated in the training permutation and was appended last for VEP.
Unavailable windows were omitted, and documents were padded to 10,240 positions with padding targets masked.
The earlier prototype used seven fixed mammalian slots, missing-window `N` placeholders, and human in the final slot.
Both approaches obtained ortholog windows from precomputed genomic alignments.

Evaluation used the same 51,623 development variants as the historical comparison: 16,140 Mendelian, 11,630 Complex Traits, and 23,853 SGE rows on odd autosomes and chromosome X.
Reported macros exclude mature miRNA and benchmark-specific low-support cells; eligible support matched across checkpoints and the historical baseline.
Values are macro AUPRC ± bootstrap standard error; differences use unrounded estimates.

| Benchmark | Protocol | Earlier 46M, final | New 46M, 50k | New 46M, 100k |
| --- | --- | ---: | ---: | ---: |
| Mendelian | Zero-shot | 0.3955 ± 0.0157 | 0.4037 ± 0.0158 | 0.4407 ± 0.0148 |
| Complex Traits | Zero-shot | 0.1840 ± 0.0149 | 0.1637 ± 0.0126 | 0.1578 ± 0.0128 |
| SGE | Zero-shot | 0.4767 ± 0.0115 | 0.4758 ± 0.0114 | 0.5093 ± 0.0117 |
| Mendelian | Frozen probe | 0.4088 ± 0.0290 | 0.5202 ± 0.0236 | 0.5502 ± 0.0282 |
| Complex Traits | Frozen probe | 0.2976 ± 0.0235 | 0.2929 ± 0.0238 | 0.3131 ± 0.0197 |
| SGE | Frozen probe | 0.4185 ± 0.0095 | 0.4629 ± 0.0113 | 0.5026 ± 0.0110 |

Zero-shot scoring averaged forward and reverse-complement likelihood ratios, using signed scores for Mendelian and SGE and absolute scores for Complex Traits.
Frozen probes used human-token allele embeddings and the existing chromosome-held-out fitting protocol.
Matched-data zero-shot metrics pool variants within consequence subsets, while probes aggregate per-chromosome AUPRC with chromosome-cluster uncertainty; compare checkpoints within a protocol, not the difference between these two readouts.

Complex Traits zero-shot macro AUPRC fell from 0.1871 at 10k to 0.1558 at 20k, rose to 0.1637 at 50k, then fell to 0.1578 at 100k.
Its highest observed point estimate was at 10k.
At 100k, GPN-Star M remained higher on Mendelian and Complex Traits at 0.5380 and 0.2781, respectively.
The new model's SGE point estimate approached GPN-Star M's 0.5157, but GPN-Star V was higher at 0.5585.
These baseline comparisons used matching development dataset revisions and eligible support with each model's registered scoring protocol.

## Limitations

- Human was the sole primate in the order-level panel, so human predictions had no non-human primate orthologs in context.
  The experiment does not test whether close primate context improves human VEP, especially on Complex Traits.
- One training seed and no matched ablation separate species breadth, taxonomic deduplication, segment order, missing-window handling, training footprint, or optimization exposure.
  The 50k-to-100k comparison follows one scheduled training trajectory and is not a fixed-compute duration ablation.
- Reported differences are point estimates with marginal standard errors; no paired significance test or replicated scaling law was established.
- All downstream evidence concerns development-cohort SNVs, including probe fitting and checkpoint comparisons.
  Held-out labeled performance and indel effects were not evaluated.
- Retrieval was derived from existing alignments.
  The experiment does not establish learned or online retrieval quality, general unaligned-query accuracy, or genome-scale serving cost.

## Related questions

- [Can autoregressive RAG gLMs be accurate and practical?](../questions/retrieval-augmented-models.md)
- [Why do MarinDNA models lag on complex-trait VEP?](../questions/complex-trait-vep.md)

## Research record

- [Experiment issue #550](https://github.com/Open-Athena/marin-dna/issues/550)
