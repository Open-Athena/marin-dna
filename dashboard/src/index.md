---
title: MarinDNA Leaderboard
toc: false
---

# MarinDNA Leaderboard

Public, version-controlled leaderboards for genomic language models trained under [MarinDNA](https://github.com/Open-Athena/marin-dna), evaluated on matched-pair variant-effect prediction.

## Leaderboards

- [**Mendelian traits**](./leaderboards/mendelian) — OMIM ∪ HGMD ∪ Smedley pathogenic SNVs (AF < 0.1%) vs gnomAD AF > 0.1%, 1:9 matched on consequence + chrom + continuous distance features. Sort axis: Macro Avg.
- [**Complex traits**](./leaderboards/complex) — UKBB fine-mapped variants (`max(PIP) > 0.9`) vs non-fine-mapped, 1:9 matched on consequence + chrom + distance + MAF. Sort axis: Global.

## Protocols / Scoring approaches

A model family's AUPRC depends on which score column you compute it from. These pages compare protocols head-to-head — same models, same dataset, different scoring approach.

- [**MarinDNA**](./protocols/marin_dna) — LLR vs NucDep
- [**Evo 2**](./protocols/evo2) — LLR vs NucDep
- [**GPN-Star**](./protocols/gpn-star) — calibrated (cLLR) vs uncalibrated LLR

## Reference

- [**Models**](./models) — every entry on the leaderboards, with family / training / source links.
- [**About**](./about) — methodology, agent-readable data tier, "how to add a model" workflow.
