The 50,000-step checkpoint completed development VEP and frozen probes at 6:26 p.m. NYC on September 14.
All 15 canonical S3 objects were content-verified, logs were recovered after the host reboot, and the worker is terminated.

Development zero-shot macro AUPRC, estimate ± bootstrap standard error:

| Benchmark | 50k | Final 100k |
| --- | ---: | ---: |
| Mendelian | 0.4037 ± 0.0158 | 0.4407 ± 0.0148 |
| Complex Traits | 0.1637 ± 0.0126 | 0.1578 ± 0.0128 |
| SGE | 0.4758 ± 0.0114 | 0.5093 ± 0.0117 |

Development frozen-probe macro AUPRC, estimate ± bootstrap standard error:

| Benchmark | 50k | Final 100k |
| --- | ---: | ---: |
| Mendelian | 0.5202 ± 0.0236 | 0.5502 ± 0.0282 |
| Complex Traits | 0.2929 ± 0.0238 | 0.3131 ± 0.0197 |
| SGE | 0.4629 ± 0.0113 | 0.5026 ± 0.0110 |

Both checkpoints use identical development cohorts, eligible support, scoring, and probing protocols.
The final zero-shot point estimates are higher by 0.0371 on Mendelian and 0.0335 on SGE; 50k is higher by 0.0059 on Complex Traits.
The final probe point estimates are higher on all three benchmarks.
No paired statistical significance test was performed.

Matched-data zero-shot and probe metrics have different aggregation contracts; compare checkpoints within each table.
These broad-model reports cover odd autosomes and chromosome X and omit mature miRNA and benchmark-specific low-support cells.
The known raw matched-metric filtering inconsistency in #574 does not affect these reported macros.
