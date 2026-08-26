# Anchor-free clustering of mammalian genome windows

> [!NOTE]
> **TL;DR:** Symmetric clustering of independently tiled 255 or 511 bp mammalian genome windows did not provide a viable conservation selector: candidate generation missed many known homologs, exhaustive alignment exposed a separate short-window recovery ceiling, alternative seed graphs lost precision or recall, and monolithic Linclust crashed on the exact 298.5-million-window panel.

## Findings

Do not continue tuning or scaling the tested anchor-free short-window clustering recipe.
On projected human, mouse, and armadillo controls, the strongest scalable Linclust setting remained a high-precision under-clusterer.
It recovered 54.9% of known cross-species pairs and 46.1% of exact three-species anchors in a five-million-window genomic background while leaving 4.84 million clusters.
Four independent hash shifts raised recall by only 1.3 percentage points for approximately four times the clustering work, and denser one-pass seed sampling did not improve recall.

The missed pairs were not solely a Linclust prefilter problem.
An exhaustive no-prefilter graph recovered 71.9% of known pairs at 100% observed pair precision on the bounded 255 bp fixture, leaving 28.1% unrecovered even with only an E-value acceptance threshold.
On the 123 anchors shared between window-length fixtures, the same exhaustive control recovered 71.3% of pairs at 255 bp but only 37.7% at 511 bp.
Longer windows around a projected center therefore diluted rather than strengthened the homologous signal in this construction.

The alternative scalable recipes did not yield a useful precision-recall tradeoff.
A source-aware seed graph reached 70.1% recall in a one-million-window background, but strict precision fell to 50.9% and 45.6% of truth records shared components with genomic decoys.
Exhaustive alignment of its bounded truth-containing components removed all observed decoy pairs but reduced global recall to 52.3%, below Linclust.
DECIPHER Clusterize formed giant mixed components and contaminated every truth record in the one-million-window comparison.

The exact 20-order panel retained 298,524,220 of 469,611,559 candidate 255 bp windows, totaling 76,123,676,100 retained bases.
MMseqs2 18.8cc5c `kmermatcher` segfaulted on this monolithic database with both the measured 148-seed recipe and a 64-seed retry that planned one in-memory split.
The latter reached a sampled 325,828,228 KiB RSS while approximately 192 GiB of host memory remained available, so its failure was not a host out-of-memory event.
No full-panel assignments or direct phyloP metrics were produced.

These results do not rule out a materially different method that uses positional or syntenic evidence, an anchor genome, or another candidate representation.
They do show that more hash shifts, denser short-seed sampling, broader alignment of the same candidates, a longer centered window, or another monolithic Linclust run is not justified by this experiment.

## Evidence

The primary bounded fixture contained 128 projected human, mouse, and armadillo anchors with three sequences and three known cross-species pairs per anchor.
It was injected into balanced, independently sampled real-genome backgrounds so true-pair recovery and genomic-decoy contamination could be measured separately.

| Scalable approach | Background | True-pair recall | Strict precision | Exact anchors | Candidate time | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| Linclust, automatic k17, 148 seeds | 5,000,000 | 54.9% | 99.1% | 46.1% | 157.78 s | 12.6 GiB |
| Four Linclust hash shifts | 5,000,000 | 56.2% | 97.7% | 47.7% | 641.90 s | Not aggregated |
| Linclust, automatic k17, 239 seeds | 5,000,000 | 53.4% | 97.6% | 44.5% | 184.24 s | 19.4 GiB |
| Source-aware k19 seed graph | 1,000,000 | 70.1% | 50.9% | 33.6% | 362.39 s | 25.6 GiB |

The 148-seed Linclust arm produced 4,841,534 clusters in the five-million-window background.
This refuted the expectation that clustering would approach the number of sequences divided by the number of species: independently sampled genome tiles usually have no sampled orthologous partner and therefore remain singletons.

The exhaustive controls separated candidate loss from accepted-alignment loss.
Removing the k-mer prefilter raised recall from 35.4% to 46.6% at the original 0.50 identity and 0.80 coverage gate.
Relaxing coverage raised it further, but only the E-value-only graph reached 71.9% recall, still with 100% observed precision.
Changing minimum identity from 0.40 to 0.30 at 0.50 coverage changed no pairs, identifying coverage and short-window alignability rather than identity as the operative acceptance limits.

The 20-genome run used exactly one selected assembly from each of 20 mammalian orders and all retained tiles from each assembly.
Both paid full-panel attempts completed database creation before failing in `kmermatcher` seed-list generation.
The five on-demand panel attempts, including setup and workflow corrections, used an estimated $8.11 of EC2 compute and all workers were terminated.

## Limitations

- The truth fixture used projected human, mouse, and armadillo loci.
  It measures recovery of known homologous groups rather than direct classification of phyloP-defined conservation.
- The bounded fixture contained 128 anchors, and five anchors had to be replaced in the 511 bp construction because expansion introduced ambiguity or majority-masked sequence.
- The independently sampled genomic background is appropriate for measuring collisions and singleton behavior but rarely includes the exact cross-species counterpart of a background window.
- The full 20-genome clustering did not complete, so the experiment produced no whole-panel cluster statistics or held-out human phyloP evaluation.
- The scaling failure applies to MMseqs2 18.8cc5c and the tested monolithic nucleotide Linclust recipes.
  It does not establish that every sharded, distributed, or position-aware clustering algorithm must fail.
- No genomic language model was trained on a clustering-derived footprint.
  The experiment evaluates conservation-candidate recovery, not downstream training utility.
- The raw S3 inputs and outputs were deleted after the negative disposition to stop ongoing storage charges.
  Commit-pinned code, configurations, tests, manifests, metrics, run details, and the complete logbook remain public, but the deleted artifacts can no longer be inspected byte-for-byte at their published S3 locations.

## Related questions

- [Which genomic regions to train on, and how to find them?](../questions/training-regions.md)

## Research record

- [Experiment issue #521](https://github.com/Open-Athena/marin-dna/issues/521)
- [Complete standalone workflow at the final code snapshot](https://github.com/Open-Athena/marin-dna/tree/e02d1637dc41f886ffdc5dd071228314f2a58631/snakemake/analysis/linclust_conservation)
- [Final append-only experiment logbook](https://github.com/Open-Athena/marin-dna/blob/efdd8e2b7c96230851d315cdbeaf1dd51fbb2fb1/.agents/logbooks/linclust-conservation.md)
- [Commit-pinned code index and artifact disposition](https://github.com/Open-Athena/marin-dna/issues/521#issuecomment-5427008645)
