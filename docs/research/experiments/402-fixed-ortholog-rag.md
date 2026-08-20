# Fixed-ortholog retrieval prototype

> [!NOTE]
> **TL;DR:** The fixed-ortholog 104M model reached VEP performance not seen in comparable small single-sequence models in MarinDNA's experiment history or Gonzalo Benegas's prior work; Gonzalo attributes the unusually strong performance to retrieval, while the experiment does not quantify the gain against a matched single-sequence arm; broader species coverage, longer training, and larger readers are the clearest next scale axes.

![Graphical abstract showing seven mammalian ortholog windows conditioning a 104M causal model and its development-cohort VEP comparisons](figures/402/marindna-rag-visual-abstract.svg)

_The fixed offline retrieval method and official development/`train` macro-AUPRC comparisons; error bars are ±1 SE and dataset facets use independent axes._
_The figure has no matched no-retrieval training arm, so it does not quantify the retrieval gain._

## Findings

The fixed-ortholog models used the aligned mammalian context, and the 104M model reached unusually strong VEP performance for its size.
Removing, shifting, or replacing the ortholog context worsened human-token validation loss.
The 104M model also exceeded the 1B single-sequence MarinDNA m5.1 reference on all three zero-shot development-cohort point estimates and on the Complex Traits and SGE frozen probes.

Gonzalo Benegas's assessment is that no single-sequence model in the 45M- to 104M-parameter range across MarinDNA's prior experiments, his earlier work, or the broader work known to him has achieved comparable VEP performance.
Together with the context perturbations, he interprets fixed ortholog retrieval as the source of the unusually high performance.
The experiment does not quantify the gain over an otherwise-matched single-sequence model.

The finding applies to an offline, alignment-derived context scheme.
It does not establish the accuracy or practicality of online retrieval, arbitrary unaligned queries, or indel-effect prediction.

## Evidence

Each training document contained seven fixed 255-base mammalian windows projected through the Zoonomia alignment, followed by the homologous 255-base human window.
The 46M- and 104M-parameter causal models each trained from scratch for 30,000 optimizer updates and 62.9 billion token presentations.
Each size had one seed, and both validation losses were finite and still falling at the final checkpoint.

On the official development/`train` cohorts, the 104M model had the following macro-AUPRC point estimates, reported as percentages.
Error terms are ±1 SE.

| Dataset | Protocol | Fixed-ortholog 104M | MarinDNA 1B m5.1 | GPN-Star M | phyloP 447m |
|---|---|---:|---:|---:|---:|
| Mendelian | Zero-shot | 42.57 ± 1.56 | 39.49 ± 1.55 | 53.80 ± 1.49 | 40.13 ± 1.51 |
| Mendelian | Frozen probe | 46.58 ± 2.27 | 48.94 ± 2.23 | N/A | N/A |
| Complex traits | Zero-shot | 18.20 ± 1.55 | 16.10 ± 1.48 | 27.81 ± 2.04 | 22.40 ± 1.81 |
| Complex traits | Frozen probe | 30.91 ± 2.42 | 22.45 ± 1.37 | N/A | N/A |
| SGE | Zero-shot | 48.20 ± 1.14 | 35.83 ± 1.11 | 51.57 ± 1.15 | 33.69 ± 1.01 |
| SGE | Frozen probe | 47.54 ± 1.00 | 38.31 ± 1.15 | N/A | N/A |

These are point-estimate comparisons, not significance claims.
GPN-Star M led all three zero-shot comparisons.
The fixed-ortholog model led the learned frozen-probe comparison on Complex Traits and SGE, while m5.1 led Mendelian.

Behavioral checks showed that the model used the prefixed context.
Removing, rolling, or replacing ortholog windows worsened human-token validation loss; changing the sequence-boundary tokens also changed outputs.
Available projected bases received more attention than missing-`N` controls, with attention concentrated near the expected aligned causal offset.
Across segment slots, loss decreased as identity to the best earlier available segment increased; the within-slot Spearman correlation was -0.471 for the 46M model and -0.470 for the 104M model across 13,169 available windows.

## Promising directions

The strongest near-term scale opportunity is broader species coverage because the current context covers only seven mammalian species and no non-mammalian vertebrates.
Longer optimization is another direct axis because both validation losses were still falling at 30,000 updates.
Larger readers are also promising because only 46M- and 104M-parameter models were tested and the 104M result was unusually strong.
One seed per size does not establish a scaling law, so the gain from model scale remains to be measured.

## Limitations

- No matched human-only training arm quantifies the retrieval gain under identical training conditions.
  The qualitative comparison with small single-sequence models relies on Gonzalo Benegas's knowledge of MarinDNA's experiment history and prior work rather than a catalogued baseline table.
- No wrong-species training arm identifies how much of the benefit requires genuine orthology.
- Species, phylogenetic group, fixed slot, and accumulated earlier context were confounded by one hard-coded order.
- Evaluation covered SNVs on development cohorts.
  No held-out labeled results or indel-effect evaluation were included.
- Retrieval was precomputed from a whole-genome alignment; online retrieval quality, latency, index size, and serving cost remain unmeasured.

## Related questions

- [Can autoregressive RAG gLMs be accurate and practical?](../questions/retrieval-augmented-models.md)

## Research record

- [Experiment issue #402](https://github.com/Open-Athena/marin-dna/issues/402)
