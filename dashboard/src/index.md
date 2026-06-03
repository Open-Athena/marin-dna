---
title: MarinDNA Observatory
toc: false
---

# MarinDNA Observatory

Public, version-controlled **benchmarks** and **interpretation** for genomic language models trained under [MarinDNA](https://github.com/Open-Athena/marin-dna). Two pillars: how well each model ranks variants, and what each model has learned.

## Benchmarks

Matched-pair variant-effect leaderboards:

- [**Mendelian traits**](./leaderboards/mendelian) — OMIM ∪ HGMD ∪ Smedley pathogenic SNVs (AF < 0.1%) vs gnomAD AF > 0.1%, 1:9 matched on consequence + chrom + continuous distance features. Sort axis: Macro Avg.
- [**Complex traits**](./leaderboards/complex) — UKBB fine-mapped variants (`max(PIP) > 0.9`) vs non-fine-mapped, 1:9 matched on consequence + chrom + distance + MAF. Sort axis: Global.
- [**caQTL**](./leaderboards/caqtl) / [**dsQTL**](./leaderboards/dsqtl) — DART-Eval Task-5 chromatin-accessibility / DNase-I-sensitivity QTLs (no matching; AUPRC / Pearson / Spearman).

A model family's AUPRC depends on which score you compute it from — the **protocol** pages compare scoring approaches head-to-head on the same models and dataset:

- [**Protocol: MarinDNA**](./protocols/marin_dna) — LLR vs NucDep
- [**Protocol: Evo 2**](./protocols/evo2) — LLR vs NucDep
- [**Protocol: GPN-Star**](./protocols/gpn-star) — calibrated (cLLR) vs uncalibrated LLR

## Interpretation

Visual analyses of what the trained models have internalized:

- [**Categorical Jacobian / nucleotide dependency**](./interpretation/nucleotide-dependency) — per-locus dependency maps: how substituting one position shifts the model's predicted nucleotide distribution elsewhere, revealing coupled functional elements (independently developed for protein and genomic LMs). See [#237](https://github.com/Open-Athena/marin-dna/issues/237).

## Reference

- [**Models**](./models) — every entry above, with family / training / source links.
- [**About**](./about) — methodology, the agent-readable data tier, and how to add a model or an interpretation type.
