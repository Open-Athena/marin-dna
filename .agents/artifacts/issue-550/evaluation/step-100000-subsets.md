Final 100k checkpoint, development AUPRC ± bootstrap standard error.
The model covers broad mixed genomic regions; all benchmark rows come from odd autosomes and chromosome X.
Zero-shot scores use strand-averaged LLR with the benchmark-specific sign/absolute-value transform.
Frozen probes use the maintained leave-one-chromosome-out protocol.
Matched-data zero-shot and probe metrics use different aggregation: pooled subset AUPRC versus per-chromosome AUPRC, respectively.
Their columns should not be treated as a direct estimate of the benefit from probing.

Mendelian traits

| Subset | Zero-shot | Frozen probe |
| --- | ---: | ---: |
| Macro average | 0.4407 ± 0.0148 | 0.5502 ± 0.0282 |
| Missense | 0.6011 ± 0.0193 | 0.6242 ± 0.0181 |
| Splicing | 0.5964 ± 0.0283 | 0.7040 ± 0.0272 |
| 5′ UTR | 0.4404 ± 0.0336 | 0.4935 ± 0.0809 |
| Promoter | 0.3079 ± 0.0278 | 0.5102 ± 0.0696 |
| ncRNA | 0.4428 ± 0.0482 | 0.7720 ± 0.1086 |
| 3′ UTR | 0.3887 ± 0.0505 | 0.4268 ± 0.0720 |
| Distal | 0.2470 ± 0.0419 | 0.3344 ± 0.0607 |
| Synonymous | 0.5015 ± 0.0651 | 0.5366 ± 0.0596 |

Complex traits

| Subset | Zero-shot | Frozen probe |
| --- | ---: | ---: |
| Macro average | 0.1578 ± 0.0128 | 0.3131 ± 0.0197 |
| Distal | 0.1199 ± 0.0055 | 0.2823 ± 0.0122 |
| Missense | 0.1910 ± 0.0157 | 0.3006 ± 0.0252 |
| Promoter | 0.1304 ± 0.0133 | 0.3590 ± 0.0422 |
| 3′ UTR | 0.1226 ± 0.0205 | 0.3925 ± 0.0516 |
| ncRNA | 0.1541 ± 0.0331 | 0.3054 ± 0.0466 |
| 5′ UTR | 0.2289 ± 0.0626 | 0.2387 ± 0.0587 |

SGE

| Subset | Zero-shot | Frozen probe |
| --- | ---: | ---: |
| Macro average | 0.5093 ± 0.0117 | 0.5026 ± 0.0110 |
| Missense | 0.4332 ± 0.0107 | 0.3887 ± 0.0097 |
| Splicing | 0.6211 ± 0.0225 | 0.6811 ± 0.0215 |
| Both (pooled) | 0.4338 ± 0.0099 | — |

Mature miRNA is excluded from the reported subsets and macros.
Complex Traits splicing and synonymous are omitted because they have 19 and 17 positive match groups, below the 30-positive reporting threshold.
SGE averages within the maintained accession/subset contract; its pooled Both row is excluded from Macro.
The standard SGE probe metric omits Both because scores from separately trained subset probes are not pooled.
