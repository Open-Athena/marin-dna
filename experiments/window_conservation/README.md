# Unary window conservation by species k-mer prevalence

Issue [577](https://github.com/Open-Athena/marin-dna/issues/577) tests one conservation proxy score per genomic window.
No pair list, nearest-neighbor output, or clustering is constructed.

## Method and computational cost

Scan each species once and count distinct-species support for canonical k-mers selected by a fixed 1/64 hash sample.
Keep one counter per selected distinct word, with the most recent species ID and within-species copy count so repeats never increase the species count.
The same word always receives the same sampling decision across genomes and strands.
A second pass scores every complete nonoverlapping 4,096 bp window from its sampled words, subtracting its own species from prevalence.
Words crossing window boundaries are assigned by their end position; terminal incomplete windows are omitted.
Report any-other-species support, at-least-two-other-species support, and mean other-species breadth, each with and without a maximum-four-copies-in-every-species filter.
The denominator includes all sampled distinct words even when the copy filter rejects their contribution.
Zero-seed windows have score zero and an explicit seed count.

For total input bases B and U distinct sampled words, hash counting and scoring take expected O(B) time and O(U) working space at fixed window length.
There is no species-by-species query loop and no array proportional to all windows allocated for each window.
Small per-window sorts deduplicate at most W words; with variable W this contributes O(B log W).
The serialized index uses 16 bytes per word; hash tables require additional RAM.
Input FASTA storage and output tables also scale linearly in input size.
This is an expected hash-table bound, not an adversarial worst-case guarantee.

## Biological test

Complete reference sequences from #568 are indexed without selecting regions by known homology or conservation annotations.
The primary output is a window score, and the endpoint is conserved-base enrichment at a fixed selected-base budget.
The existing vertebrate-projection phyloP_447m bigWig supplies evaluation labels; it never enters the scoring index.
Reuse the pipeline's phyloP >= 2.2162 definition, counting missing bigWig values as non-conserved over the whole-window denominator and reporting coverage separately.
The secondary positive-window definition requires at least 20% conserved bases, matching the pipeline's fraction criterion at our different window size.
The pinned S3 object and checksum identify the exact existing annotation; no new conservation-label source is introduced.
These labels are alignment-derived conservation annotations, not independent biological ground truth.
`config/protocol.json` fixes the method grid, chromosome split, controls, selection rule, and advancement criterion before metrics are inspected.
Select on chromosome 1, publish the selection, and inspect chromosome 2 only afterward.
Chromosome separation prevents overlapping-window leakage; homologous repeat families can still cross chromosomes.
Other species receive scores, but biological accuracy is validated only on human.
Three real species cannot establish sensitivity or phylogenetic calibration across 1,000 species.
Large-species synthetic tests evaluate computational scaling separately.

## Execution

Run all data work on the budget-capped EC2 worker.
Use one process at a time and one thread for each numeric runtime.

```bash
uv sync --locked --group dev
g++ -O3 -std=c++17 -Wall -Wextra -Werror src/window_conservation/prevalence.cpp -o prevalence
uv run --locked pytest
uv run --locked python -m window_conservation.prepare --root /data/issue577
uv run --locked python -m window_conservation.run --root /data/issue577
```

Per-stage command receipts, wall time, exit code, GNU time peak RSS, per-window score tables, annotation provenance, and selected-word index sizes are retained.
Durable ownership is `s3://oa-bolinas/issues/577/` under a producing-commit prefix.
The source and complete experiment record remain on the permanent research branch; accepted findings are delivered separately through a documentation PR.
