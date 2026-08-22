# One-pass MNTP adaptation of MarinDNA m5.1

> [!NOTE]
> **TL;DR:** A 1,000-step BICO LoRA adaptation improved one-pass nucleotide prediction during training but remained worse than the causal source, while Mendelian VEP showed no statistically resolved change.

## Findings

The standard-rate rank-16 BICO LoRA run did not produce a useful one-pass bidirectional model.
Its nucleotide prediction improved during adaptation, then plateaued below the causal source despite access to both sequence directions.
The symmetric two-causal-pass control remained better than either model.

Mendelian macro AUPRC showed no statistically resolved improvement or degradation across the run.
Every paired checkpoint interval versus step 0 included zero.

The source model remained frozen, the adapter-disabled causal readout was bit-exact, and the latest run had finite losses and gradient norms without clipping.
The result does not support selecting a VEP checkpoint or extending the same recipe beyond 1,000 steps.

## Evidence

The fixed validation panel used 640 identical masked nucleotide targets: 128 sequences from each of CDS, downstream, enhancer, ncRNA, and upstream data.
The source used its causal next-token readout, while BICO used full attention with reflected future RoPE and excluded masked keys.

| Readout | Four-way cross-entropy | Accuracy |
|---|---:|---:|
| Causal source | 1.050770 | 50.63% |
| BICO LoRA step 0 | 1.387224 | 32.97% |
| BICO LoRA step 1,000 | 1.273889 | 41.09% |
| Symmetric two-pass control | 0.913447 | 62.50% |

Training used 94,000 sequences, 24.1 million model tokens, a physical batch of 94 without accumulation, and a `5e-5` peak learning rate.
The schedule warmed up for 100 steps, stayed constant through step 800, and decayed through step 1,000.

The Mendelian analysis used 16,100 odd-autosome/X development variants across eight consequence subsets.
Paired uncertainty used 2,000 seed-0 match-group bootstrap replicates inside each subset.

| Step | Mendelian macro AUPRC | Change from step 0 | Paired 95% interval |
|---:|---:|---:|---:|
| 0 | 0.104816 | 0 | reference |
| 100 | 0.111331 | +0.006515 | [-0.000809, +0.013839] |
| 400 | 0.108776 | +0.003960 | [-0.002557, +0.010476] |
| 1,000 | 0.107966 | +0.003150 | [-0.003377, +0.009677] |

## Limitations

The standard-rate run used one seed, one source checkpoint, one 1,000-step schedule, and LoRA on one attention formulation.
The nucleotide comparison intentionally favors BICO by giving it both sequence directions, so its deficit is a failed engineering gate rather than a matched-objective likelihood comparison.
The Mendelian result measures change within the BICO run and does not establish parity with source-CLM VEP.
Direct masked-site VEP may not measure useful internal representations or supervised sequence-to-function performance.
No held-out even-autosome or chromosome-Y labels or metrics were accessed.

## Related questions

- [Can causal gLM checkpoints be cheaply adapted into bidirectional representation models?](../questions/bidirectional-models.md)

## Research record

- [Experiment issue #479](https://github.com/Open-Athena/marin-dna/issues/479)
