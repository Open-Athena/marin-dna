# Unary window conservation by species k-mer prevalence

Issue [577](https://github.com/Open-Athena/marin-dna/issues/577) tests one conservation proxy score per genomic window.
No pair list, nearest-neighbor output, or clustering is constructed.

## Method and computational cost

Seed the index with words from the query chromosomes, then scan each complete species once and count distinct-species support for canonical k-mers selected by a fixed 1/4 hash sample.
Keep one counter per selected distinct word, with the most recent species ID and within-species copy count so repeats never increase the species count.
The same word always receives the same sampling decision across genomes and strands.
A second pass scores every complete nonoverlapping 100 bp interval from its sampled words, subtracting its own species from prevalence.
Words crossing window boundaries are assigned by their end position; terminal incomplete windows are omitted.
Report any-other-species support, at-least-two-other-species support, and mean other-species breadth, each with and without a maximum-four-copies-in-every-species filter.
The denominator includes all sampled distinct words even when the copy filter rejects their contribution.
Zero-seed windows have score zero and an explicit seed count.

For total input bases B, query bases Q, and U distinct sampled query words, hash counting and scoring take expected O(B + Q) time and O(U) working space at fixed window length.
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
After inspecting development results, also freeze a secondary setting maximizing selected conserved-base fraction; this comparison does not replace the original matched-enrichment selection or gate.
Export its BED6+2 files with the `stretches-density-` prefix and report its uncertainty separately.
Chromosome separation prevents overlapping-window leakage; homologous repeat families can still cross chromosomes.
This biological pilot scores all complete 100 bp intervals on human chromosomes 1 and 2.
A query-restricted index retains only words present on those chromosomes, but counts their copies and distinct-species support across every complete input genome.
The restricted index gives identical scores to a global index for its query intervals; a test asserts this equivalence.
Adjacent selected intervals merge into BED stretches, without bridging an unselected or invalid interval.
The output is BED6+2: chromosome, start, end, candidate name, rounded mean score scaled to 0–1,000, unstranded marker, original mean score, and number of constituent intervals.
The 100 bp bins set boundary resolution; this pilot does not claim nucleotide-accurate boundaries.
The 4 kb development experiment was superseded following the user’s resolution clarification, before inspecting held-out labels.
Its results are preserved separately as exploratory history.
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
uv run --locked python -m window_conservation.local_run --root /data/issue577
uv run --locked python -m window_conservation.evaluate --root /data/issue577 --split dev
```

Per-stage command receipts, wall time, exit code, GNU time peak RSS, per-window score tables, annotation provenance, and selected-word index sizes are retained.
Durable ownership is `s3://oa-bolinas/issues/577/` under a producing-commit prefix.
The source and complete experiment record remain on the permanent research branch; accepted findings are delivered separately through a documentation PR.

## Selection and scaling boundaries

The ordinary fixed-budget selector uses partition-based selection and a linear pass to merge adjacent bins, avoiding an all-interval ranking sort.
Retaining M interval scores for this exact budget adds O(M) working space; the combined scoring-and-selection bound is O(U + M), with O(B + M) expected work at fixed interval length.
Bootstrap evaluation reuses a sorted score order to handle resampled multiplicities; that uncertainty calculation is separate from producing the stretches.
The three-species biological run retains only query-chromosome words, so its memory footprint must not be extrapolated as if it indexed every genome.
The synthetic scaling run constructs the global index and scores every synthetic species at the same 1/4 sampling density and 100 bp resolution.
It varies species count (125–1,000) and intervals per species (1,024–8,192) independently, with three timing repetitions per shape.
It measures construction and score-table generation; scientific evaluation, downloading, preparation, and fixed-budget selection are separate stages.
An in-memory global index still grows with the number of distinct sampled words and may require impractical memory at thousands of complete genomes.
Repeatedly rescanning all genomes for bounded query batches would add a batch-count factor; the pilot does not claim that strategy preserves total linear work when scoring every genome.
The original pilot did not include a disk-partitioned global implementation; the extension below tests one.

## Expanded panel and bounded-memory comparisons

`config/extension.json` preserves the completed chr2 pilot and declares chr1 development plus fresh chr3 validation for the extension.
The nested panels contain 3, 6, and 10 complete mammalian genomes; sampling compares fixed rates 1/4, 1/8, and 1/16 with the bottom 16 and 32 hashes of all words in each 100 bp query interval.
For fixed-size sketches, selected query words are counted against every word in each complete support genome, even if another interval would omit that word from its sketch.
The bottom-16 comparison uses a superset index seeded by bottom-32, and thinner rates similarly reuse the 1/4 index; separate query-profile runs measure each smaller index's own footprint.

`compact.cpp` uses contiguous 16-byte slots while preserving exact word identity and species/copy counts.
It retains one query index across incremental species panels and scoring, avoiding repeated index serialization and loading.
The original dense implementation remains the independent parity reference.

`partition.cpp` routes sampled word occurrences into disk partitions in one input pass, counts each partition exactly, and writes nonzero partial window scores into contiguous window chunks.
A final pass assembles scores using at most 262,144 interval accumulators.
There is no complete-genome rescan per partition.
Expected counting/scoring work is O(B + M) for fixed interval width, with O(U/P + C + P) resident state for P word partitions and C intervals per aggregation chunk, assuming roughly balanced hash partitions.
Temporary disk and I/O scale with sampled occurrences and partial score records.
The measured implementation accepts up to 512 partitions; excessive partition size or skew can still exceed RAM and is not handled by recursive repartitioning.

`stream_select.py` makes exact fixed-budget selections independently per species without retaining every window score in RAM.
It counts distinct score values for one species at a time, finds the threshold in eight weighted radix passes, resolves boundary ties with eight disk radix passes over 64-bit hashes, and merges adjacent selected intervals in a final score-stream pass.
For M intervals, T boundary ties, and D distinct scores summed across species, selection uses O(M + 8D + 8T) expected work, which is O(M) because D and T are at most M.
Resident selection state is O(maximum distinct scores in one species + species), and scratch disk is O(T).
For the any-species scores at fixed 100 bp width, the finite set of numerator/denominator fractions bounds the number of score values per species independently of genome length.
Species must appear contiguously in the score stream; within a species, intervals must be in genomic order for merging.
Output is a BED6+2 file per species plus the selection receipt.

`biology.py` describes fixed selections using the existing pipeline GTF/cCRE assets and checksum-pinned UCSC RepeatMasker annotations.
Annotation types may overlap; reported fractions are not an exclusive partition.
GTF coordinates convert from 1-based closed, and bare primary chromosome names explicitly map to the corresponding hg38 `chr` names at the boundary.
`spatial.py` tests planted tract length, substitution, indels, bin offset, duplicated targets, and shuffled negative controls without using them to retune the real-data selector.

Run the extension sequentially on the authorized worker:

```bash
uv run --locked python -m window_conservation.extension_prepare --root /data/issue577
uv run --locked python -m window_conservation.extension_run --root /data/issue577
uv run --locked python -m window_conservation.extension_evaluate --root /data/issue577 --split dev
# Commit and publish report/selection.json before the validation command.
uv run --locked python -m window_conservation.extension_evaluate --root /data/issue577 --split validation --freeze-sha COMMIT
uv run --locked python -m window_conservation.memory_scaling --root /data/issue577
uv run --locked python -m window_conservation.full_global --root /data/issue577
uv run --locked python -m window_conservation.query_profiles --root /data/issue577
uv run --locked python -m window_conservation.biology --root /data/issue577 --split pilot
uv run --locked python -m window_conservation.biology --root /data/issue577 --split extension
uv run --locked python -m window_conservation.spatial --root /data/issue577
uv run --locked python -m window_conservation.extension_report --root /data/issue577
```

The full-genome resource trial scores every complete 100 bp interval in the original three genomes, verifies every original human query score, and produces exact 5% per-genome selections with `any_copy4`.
The resource matrix repeats the original baseline, compact table, and disk partitions contemporaneously, with three repetitions per setting and complete baseline score parity at 1/4 sampling.
