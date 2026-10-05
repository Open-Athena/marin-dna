# Online loss selection and teacher distillation for CDS continuation

> [!NOTE]
> **TL;DR:** In one shallow exp58 animal-CDS continuation, current- or teacher-loss-ranked half-token objectives sharply reduced Mendelian missense-plus-splicing AUPRC, while pure final-checkpoint teacher KL exceeded uniform CE at step 200; one seed and privileged later-lineage supervision limit the inference.

## Findings

Hard selection by token loss did not improve this continuation.
After 100 arm-local steps, all three current-student loss-ranked halves and the frozen-teacher lowest-loss half were significantly worse than the shared bridge.
Randomly retaining half of eligible targets remained above the bridge but underperformed uniform CE, providing no compelling benefit from sparse gradients alone.

Pure full-distribution teacher KL and uniform CE were the strongest objectives at the first gate.
After both arms resumed to 200 arm-local steps, teacher KL reached 0.183331 pooled missense-plus-splicing AUPRC versus 0.170727 for uniform CE.
Their paired difference was 0.012604 AUPRC with a two-sided match-group permutation p-value of 0.019549.

Neither arm changed significantly from step 100 to step 200 after Holm correction.
The evidence therefore distinguishes the two objectives at the step-200 evaluation within this run, but does not establish a reliable upward or downward trajectory for either arm.

## Evidence

No held-out language-model validation loss was evaluated.
Only training loss was logged, so the accepted findings rest on downstream variant-effect prediction.

The experiment initialized a 256-base, no-BOS Qwen3 model from exp58-animals step 1,000 and used its pinned animal-CDS corpus.
Lowercase repeat targets were excluded, while repeat bases remained visible as context.
The shared 100-step uniform-CE bridge used fresh AdamW and a linear warmup from 1e-5 to 1e-3.
Seven independent 100-step objective forks each reset AdamW and scheduler state, used a 20-step warmup to 1e-3, and held the learning rate constant at 1e-3 thereafter.
The uniform and teacher-KL arms later resumed their complete step-100 states for another 100 steps without a second warmup.

The primary evaluation pooled 5,800 missense and 3,190 splicing records from the pinned Mendelian-traits train split.
The 8,990 records formed 899 matched groups with one positive and nine negatives per group.

| Checkpoint | Pooled | Missense | Splicing |
|---|---:|---:|---:|
| Shared bridge | 0.156796 | 0.126489 | 0.219649 |
| Uniform CE, step 100 | 0.175704 | 0.139221 | 0.249623 |
| Random-50 CE, step 100 | 0.161681 | 0.127705 | 0.234459 |
| Student-low-50 CE, step 100 | 0.110347 | 0.100109 | 0.133439 |
| Student-middle-50 CE, step 100 | 0.109192 | 0.100417 | 0.128960 |
| Student-high-50 CE, step 100 | 0.108288 | 0.105148 | 0.115249 |
| Teacher KL, step 100 | 0.177410 | 0.142064 | 0.246134 |
| Teacher-low-50 CE, step 100 | 0.129372 | 0.118574 | 0.149306 |
| Uniform CE, step 200 | 0.170727 | 0.132595 | 0.247808 |
| Teacher KL, step 200 | **0.183331** | **0.148211** | **0.254956** |

_AUPRC rounded to six decimals at each completed evaluation; pooled missense plus splicing is primary._

At step 100, one-sided paired match-group swap tests compared each nonuniform objective with the bridge using 20,000 permutations and Holm correction across six comparisons.
Random-50 and teacher KL each had adjusted p_worse = 1.0.
The four loss-ranked half-token objectives each had adjusted p_worse = 0.000300.

At step 200, the two-sided paired teacher-KL-versus-uniform test used the same match-group unit and 20,000 permutations.
Teacher KL exceeded uniform by 0.012604 AUPRC with p = 0.019549.
The step-200-minus-step-100 changes were -0.004977 for uniform and +0.005921 for teacher KL; their Holm-adjusted two-sided p-values were both 0.563772.

Every arm processed the same number and order of input rows through step 100.
The two resumed arms used byte-identical plan prefixes through step 200 and restored model, optimizer, scheduler, selector, data-position, and random-number-generator state.
Teacher KL required a frozen-teacher forward pass and averaged 28.88 seconds per added step, compared with 22.17 seconds for uniform CE.
The comparison therefore matches processed input rather than wall time or accelerator compute.

## Limitations

- One paired training seed was evaluated.
  The permutation p-values quantify uncertainty across matched evaluation records within this run, not variation across training seeds.
- The teacher is the final step-16,999 checkpoint from the student's own training lineage.
  Its soft targets provide privileged later-lineage supervision, so the KL result is not evidence for token selection alone.
- Every fork, including uniform CE, reset optimizer and scheduler state and repeated the same warmup.
  The result is specific to a high-learning-rate objective transition rather than uninterrupted exp58 training.
- The endpoint used the Mendelian-traits train split and excluded synonymous variants.
  The result may not generalize beyond missense and splicing prediction.
- The model, animal-CDS corpus, repeat exclusion, 256-base context, and shallow continuation are specific to exp58.
  No RefSeq-corpus replication was run.
- Matching processed input did not match training compute because teacher KL was slower per step.

## Related questions

- [Which genomic regions to train on, and how to find them?](../questions/training-regions.md)

## Research record

- [Experiment issue #515](https://github.com/Open-Athena/marin-dna/issues/515)
