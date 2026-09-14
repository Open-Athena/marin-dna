The final five-region RAG checkpoint remains below the strongest GPN-Star variant on all three development macro AUPRC point estimates.
It is close to GPN-Star M on SGE, while GPN-Star V leads that benchmark.
Mendelian synonymous and SGE splicing are the two displayed base subsets where RAG exceeds all three GPN-Star variants.
All six eligible Complex Traits subsets are below all three GPN-Star variants.

This is a zero-shot comparison on the same pinned development dataset revisions (odd autosomes and chromosome X).
Every displayed support count matches across models.
GPN-Star uses its registered calibrated LLR scores, signed for Mendelian/SGE and absolute for Complex Traits; RAG uses its registered strand-averaged LLR protocol.
V denotes vertebrate, M mammal, and P primate.
Values are AUPRC ± bootstrap standard error.

Mendelian traits

| Subset | RAG 46M, final 100k | GPN-Star V | GPN-Star M | GPN-Star P |
| --- | ---: | ---: | ---: | ---: |
| Macro average | 0.4407 ± 0.0148 | 0.5003 ± 0.0156 | 0.5380 ± 0.0149 | 0.4447 ± 0.0146 |
| Missense | 0.6011 ± 0.0193 | 0.6809 ± 0.0168 | 0.6308 ± 0.0181 | 0.5133 ± 0.0197 |
| Splicing | 0.5964 ± 0.0283 | 0.5289 ± 0.0258 | 0.6265 ± 0.0236 | 0.5332 ± 0.0251 |
| 5′ UTR | 0.4404 ± 0.0336 | 0.4601 ± 0.0338 | 0.4364 ± 0.0341 | 0.3543 ± 0.0312 |
| Promoter | 0.3079 ± 0.0278 | 0.4668 ± 0.0320 | 0.4957 ± 0.0312 | 0.4004 ± 0.0311 |
| ncRNA | 0.4428 ± 0.0482 | 0.5841 ± 0.0441 | 0.6574 ± 0.0401 | 0.5974 ± 0.0411 |
| 3′ UTR | 0.3887 ± 0.0505 | 0.4073 ± 0.0502 | 0.5011 ± 0.0527 | 0.3877 ± 0.0480 |
| Distal | 0.2470 ± 0.0419 | 0.5062 ± 0.0608 | 0.4837 ± 0.0576 | 0.4910 ± 0.0550 |
| Synonymous | 0.5015 ± 0.0651 | 0.3681 ± 0.0662 | 0.4724 ± 0.0593 | 0.2804 ± 0.0598 |

Complex traits

| Subset | RAG 46M, final 100k | GPN-Star V | GPN-Star M | GPN-Star P |
| --- | ---: | ---: | ---: | ---: |
| Macro average | 0.1578 ± 0.0128 | 0.2515 ± 0.0193 | 0.2781 ± 0.0204 | 0.2631 ± 0.0189 |
| Distal | 0.1199 ± 0.0055 | 0.1912 ± 0.0138 | 0.2135 ± 0.0144 | 0.2044 ± 0.0140 |
| Missense | 0.1910 ± 0.0157 | 0.2304 ± 0.0185 | 0.2667 ± 0.0218 | 0.2310 ± 0.0200 |
| Promoter | 0.1304 ± 0.0133 | 0.3102 ± 0.0371 | 0.3057 ± 0.0379 | 0.3082 ± 0.0386 |
| 3′ UTR | 0.1226 ± 0.0205 | 0.1925 ± 0.0450 | 0.2104 ± 0.0443 | 0.1769 ± 0.0342 |
| ncRNA | 0.1541 ± 0.0331 | 0.2722 ± 0.0637 | 0.3244 ± 0.0658 | 0.2914 ± 0.0701 |
| 5′ UTR | 0.2289 ± 0.0626 | 0.3123 ± 0.0735 | 0.3480 ± 0.0812 | 0.3664 ± 0.0686 |

SGE

| Subset | RAG 46M, final 100k | GPN-Star V | GPN-Star M | GPN-Star P |
| --- | ---: | ---: | ---: | ---: |
| Macro average | 0.5093 ± 0.0117 | 0.5585 ± 0.0114 | 0.5157 ± 0.0115 | 0.4106 ± 0.0113 |
| Missense | 0.4332 ± 0.0107 | 0.5262 ± 0.0104 | 0.4407 ± 0.0100 | 0.3416 ± 0.0093 |
| Splicing | 0.6211 ± 0.0225 | 0.5864 ± 0.0230 | 0.6162 ± 0.0229 | 0.5230 ± 0.0238 |
| Both (pooled) | 0.4338 ± 0.0099 | 0.5083 ± 0.0095 | 0.4273 ± 0.0091 | 0.3363 ± 0.0085 |

The largest Mendelian deficits are distal, ncRNA, and promoter variants.
Against GPN-Star M, the Mendelian macro difference is -0.0972, Complex Traits -0.1203, and SGE -0.0065.
Against the strongest SGE variant, GPN-Star V, the SGE macro difference is -0.0492.
These describe point estimates and do not establish paired statistical significance or equivalence.

Mature miRNA is excluded, and Complex Traits splicing/synonymous are omitted under the 30-positive support rule.
SGE Both is pooled within accessions and is not included in Macro.
GPN-Star frozen-probe results are not available in this registered baseline; RAG's supervised probe values are not used in this zero-shot comparison.

The archived [GPN-Star pipeline configuration](https://github.com/Open-Athena/marin-dna/blob/00a4e3530310a2c42b85702800bdc71428c829f6/snakemake/gpn_star_eval/config/config.yaml) pins all three dataset revisions, the train split, and score protocols.
The [metric rules](https://github.com/Open-Athena/marin-dna/blob/00a4e3530310a2c42b85702800bdc71428c829f6/snakemake/gpn_star_eval/workflow/rules/metrics.smk) use the same benchmark metric contracts.
Source S3 keys, local SHA-256 identities, exact values, support checks, and differences are in the companion JSON report.
