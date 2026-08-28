# HAL-to-chain investigation for issue #517

## Question

Can the human-to-family-deduplicated-species coordinate mappings be materialized from the Zoonomia HAL once and then reused, without exporting an intermediate MAF or TAF?

## Conclusion

Yes.
HAL already supports direct coordinate queries, and the official released Cactus tool `cactus-hal2chains` can materialize pairwise UCSC chains directly from HAL without MAF or TAF.
Its default conversion is a streaming pipeline:

```text
halLiftover HAL query query_whole_genome.bed target stdout --outPSL
  | pslPosTarget stdin stdout
  | axtChain -psl ... stdin target.2bit query.2bit stdout
  | gzip
```

PSL is therefore used as a transient stream, but no multiple-alignment export is required.
The conversion does not avoid HAL traversal: it pays for one whole-genome `halLiftover` pipeline per genome pair and stores the result for reuse.

The human decision is to optimize for arbitrary future interval projection rather than only the current 128-bp grid.
The target artifact is therefore a reusable whole-genome human-to-species chain for each of the 107 family-deduplicated mammals.
A coordinate-only grid cache remains a possible specialized optimization, but it is not the desired deliverable.

## What HAL and chain represent

HAL is not a flat lookup table.
It is a hierarchical graph-based multiple alignment with ancestral genomes, parent/child segment links, and explicit paralogy relations.
`halLiftover` traverses that graph between any source and target and maps BED intervals base by base.

A UCSC chain is a pairwise set of scored, co-linear alignment blocks with gaps.
It is a materialized pairwise view of the HAL alignment, not a byte-for-byte extraction of an already stored chain.
The `axtChain` stage scores and groups PSL blocks, so a chain can make different choices from a direct interval query in duplicated or competing alignment regions.

HAL does contain an apparent `hal2chain` program, but its source warns that it was never finished or tested and directs users to `halLiftover --outPSL` instead.
It should not be used for this experiment.

## Official converter status

The maintained converter is present in Cactus 3.1.4, which the current projection workflow pins.
That release creates one Toil job and one full HAL copy per genome pair, and its source explicitly records that this design does not scale to huge HAL files.

Cactus 3.3.0 adds the needed batching interface.
It can copy the HAL once per batch and run several `halLiftover | axtChain` pipelines concurrently with `--batchCount`, `--batchCores`, and `--batchParallelHal2chains`.
This version should be evaluated in an isolated pilot rather than silently replacing the Cactus 3.1.4 dependency of the completed strict controls.

The converter's own resource model is important for the 1,262,706,573,453-byte HAL.
For non-`--inMemory` operation it estimates that `halLiftover` may touch roughly one tenth of the HAL through the page cache, in addition to the `axtChain` working set.
Its documented 900-GiB example needed 165 GiB after an 83-GiB batch was memory-killed.
The current 96-GiB `c6id.12xlarge` projection node is therefore not an assumed-safe chain-generation node, even though it was adequate for our sparse center-1 queries.
The pilot must measure peak RSS and page-cache pressure on appropriately sized EC2 memory and local NVMe rather than reuse the old instance choice by default.

## Direction is easy to reverse accidentally

The Cactus output is named `target_vs_query.chain.gz`.
Its `axtChain` invocation puts the Cactus `target` genome on the UCSC chain `tName` or reference side and the Cactus `query` genome on the `qName` side.
UCSC `liftOver` consumes coordinates from the chain's `tName` side.

Consequently, a chain used for human-to-species projection must have `Homo_sapiens` on the Cactus `--targetGenomes` side and the destination species on `--queryGenomes`.
Every pilot chain header must be checked before use; the expected header has a human sequence as `tName` and the destination-species sequence as `qName`.

## Strict-control semantic caveat

The current mammal projector calls `halLiftover --noDupes`.
HAL defines this flag as refusing to follow duplication edges in the graph.

`cactus-hal2chains` does not expose or pass `--noDupes` in its default pipeline.
The official default chain is therefore not automatically a strict replacement for the current projector.
It may contain alternative paralogous chains, and downstream `liftOver` without `-multiple` may resolve overlapping candidates differently from direct `halLiftover --noDupes`.

The strict pilot should compare two chain recipes:

1. The unmodified Cactus 3.3.0 chain, to measure the supported default behavior.
2. A minimally modified pipeline whose `halLiftover --outPSL` stage also uses `--noDupes`, with direction normalized for human-to-species use.

Neither variant should replace the projector until exact center-coordinate, strand, no-mapping, and multiplicity parity has been measured under the existing acceptance contract.

## Options considered for issue #523

| Option | MAF or TAF | Up-front work | Reuse scope | Main caveat |
| --- | --- | --- | --- | --- |
| Current batched direct HAL | None | 107 sparse `halLiftover --noDupes` calls per experiment | None | Re-reads the 1.26-TB HAL for every selector experiment |
| **Pairwise chain set, selected** | None; streamed PSL only | 107 whole-genome HAL-to-chain pipelines once | Any later intervals | Expensive generation and default duplication semantics differ |
| Fixed-grid coordinate cache | None | One full 22,948,560-center direct projection per species | Any selector or arm on this uniform grid | Not a general-purpose genome-to-genome liftover artifact |
| Custom libHAL mapper | None | New maintained C++ implementation | Potentially arbitrary | HAL's multi-target column iterator exists but is documented as not the most efficient traversal, so this is higher engineering risk |

The current direct implementation is already efficient for a one-off filtered experiment because all centers are placed in one BED per species and the 107 single-threaded calls run concurrently.
It made exactly 107 HAL invocations for both completed selector builds.
The 1,627,410-window GPN run took 41 minutes 30 seconds, while the smaller 1,136,854-window strict-phyloP run took 46 minutes 37 seconds.
That non-monotonic observation shows that the earlier 14.10-fold linear extrapolation to all windows is a planning bound, not a validated scaling law.

An all-grid coordinate cache would contain 22,948,560 × 107 = 2,455,495,920 human-center/species requests before adding the human reference.
Although it could be compact and exact for the present experiment, it would not support a later change in tiling, window length, anchor position, or arbitrary annotations without returning to HAL.
That loss of flexibility is why it is not the selected artifact.

## Recommended bounded pilot

Do not generate all 107 chains first.
Use one near primate, one rodent, and one distant mammal, for example `Papio_anubis`, `Mus_musculus`, and `Loxodonta_africana`.

For each species:

1. Generate the released Cactus 3.3.0 default chain and a `--noDupes` chain candidate from the same immutable HAL.
2. Record chain-generation wall time, CPU time, peak RSS, peak page-cache or cgroup memory, scratch disk, and compressed chain size.
3. Check chain direction from the header before invoking `liftOver`.
4. Project the same current center-1 BED with direct `halLiftover --noDupes` and both chains.
5. Compare exact mapped or unmapped state, target sequence name, zero-based half-open coordinate, strand, and mapping multiplicity for every query.
6. Run the all-22,948,560-center BED through the validated chain and time only the `liftOver` stage.
7. Calculate the reuse break-even point from chain-generation time divided by the per-run time saved relative to direct HAL.

If strict parity is high, generate, validate, and pin all 107 whole-genome chains.
The generation cost and reuse break-even remain useful planning measurements, but the whole-genome flexibility is the primary reason to build them.
If generation is too expensive on the first resource profile, optimize batching, memory, and staging before considering a narrower artifact.
If parity differs materially, retain direct `halLiftover --noDupes` as the scientific source of truth and treat chain results only as an exploratory backend.

No cloud job was launched for this investigation.

## Sources

- [HAL README at commit `a94735bf`](https://github.com/ComparativeGenomicsToolkit/hal/blob/a94735bf24bc18d9f989563b5d38496214633156/README.md#hierarchical-alignment-hal-format-api-v23) describes HAL as an indexed graph-based multiple alignment and documents direct base-by-base `halLiftover` queries.
- [HAL `halLiftover` CLI source](https://github.com/ComparativeGenomicsToolkit/hal/blob/a94735bf24bc18d9f989563b5d38496214633156/liftover/impl/halLiftoverMain.cpp) defines `--noDupes` as not mapping between duplications and supports direct PSL output.
- [HAL native iterator API](https://github.com/ComparativeGenomicsToolkit/hal/blob/a94735bf24bc18d9f989563b5d38496214633156/api/inc/halColumnIterator.h) accepts a target-genome set but warns that reference-column iteration is not HAL's most efficient traversal.
- [Unsupported `hal2chain` source](https://github.com/ComparativeGenomicsToolkit/hal/blob/a94735bf24bc18d9f989563b5d38496214633156/blockViz/impl/hal2chain.cpp) contains the unfinished and untested warning.
- [Cactus 3.3.0 chain-export documentation](https://github.com/ComparativeGenomicsToolkit/cactus/blob/v3.3.0/doc/progressive.md#chains-export) documents `cactus-hal2chains` and its batch interface.
- [Cactus 3.3.0 implementation](https://github.com/ComparativeGenomicsToolkit/cactus/blob/v3.3.0/src/cactus/maf/cactus_hal2chains.py#L483-L523) shows the direct `halLiftover | pslPosTarget | axtChain | gzip` pipeline.
- [UCSC chain-format documentation](https://genome.ucsc.edu/goldenPath/help/chain.html) defines the target/reference and query sides and zero-based half-open coordinates.
- [Issue #523](https://github.com/Open-Athena/marin-dna/issues/523) owns the separate full-unfiltered projection and format benchmark.
