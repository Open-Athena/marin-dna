# Local k-mer conservation retrieval

This permanent experiment branch owns the bounded study in [issue #568](https://github.com/Open-Athena/marin-dna/issues/568).
It is an independent Python project; the one-off benchmark is not intended for `main`.

## Frozen protocol

Regenerate the original 128-anchor, human/mouse/armadillo 255 bp fixture using the function and configuration at `e02d1637dc41f886ffdc5dd071228314f2a58631`.
Keep those controls and add 128 deterministically sampled anchors from the same projection sources.
Extract 4,096 bp genomic contexts using the original assemblies and projected centers; reject out-of-bounds extractions but retain ambiguous and repeat-rich flanks.
Merge overlapping contexts into locus components, including overlaps in any species, before development/held-out assignment.
Keep all homologs in the same split.
These homology-linked components serve only to isolate splits.
Candidate budgets and queries use separate physical loci, formed only by overlapping intervals on the same chromosome in the same species.
One query can have multiple known target loci; score each known locus pair, spend the candidate budget once per query, and charge query work once.
The fixed target universe also contains 1,000 nonoverlapping genomic background contexts per species, composition/complexity-matched genomic challenges, and injected shuffled controls.
Real-genome background hits are unresolved, including repeat/paralog challenges; shuffled controls are measured separately as injected decoys.

The initial screen crosses `W = 64, 128, 255, 511, 1024` with `k = 9, 13, 17, 21` at `floor(W/2)` stride.
Tile the same covered contexts with deterministic, independent species/contig offsets.
Include boundary windows to cover context ends and report their count.
Canonicalize reverse complements, skip k-mers spanning non-ACGT bases, and record masking as an explicit ablation rather than silently dropping repeat windows.
Score complete k-mer sets with Jaccard similarity.
For a query locus and target species, aggregate by the maximum window-pair score and collapse each target locus before applying candidate budgets `C = 1, 10, 100`.
Ties use a deterministic identifier hash unrelated to truth; zero-similarity pairs are not retrieved.
Report score-positive window pairs, unique candidates, index bytes, timings, and peak memory so smaller strides cannot obtain free candidate work.
The exact sparse k-mer inverted index preserves every shared feature and therefore also implements a full-set scoring baseline; compare it with brute-force set scoring on a bounded audit subset.

Before inspecting development results, set the exploratory advancement gate to at least 80% development locus-pair recall at `C=10`, with at most 10% injected-decoy candidates in the returned top ten.
This threshold is a study decision, not a field standard.
Compare the best settings at full, half, and quarter strides, repeat masking, local subwindows, and a small scale union under the same unique-candidate budget.
Use synthetic planted tracts with known boundaries to isolate span, divergence, offsets, and indels.
Choose settings on development loci only, then freeze the choice before held-out evaluation.
For surviving settings compare full sets, independent-permutation MinHash scans, compatible banded LSH, and matched Linclust, separating alignment verification cost and recall loss.
Advance to conservation enrichment only after an indexed method improves the measured recall–cost frontier and the gain survives held-out evaluation.
No whole-genome production run or model training is part of this study.

## Resources and ownership

The user authorized at most $30 EC2 costs.
The initial worker is one `c7i.4xlarge` in `us-east-2` (16 vCPUs, 32 GiB RAM), with a 60 GiB encrypted gp3 root disk that is deleted on termination.
Its OS shutdown deadline is ten hours after bootstrap, and instance-initiated shutdown is configured to terminate.
The planning estimate is $0.714/hour compute, approximately $7.14 at the deadline, plus less than $0.20 for the short-lived root disk and public IPv4.
Reserve the remaining budget for necessary bounded reruns; record launches and actual lifetime estimates before any additional instance.
Expected peak analysis memory is below 16 GiB; stop a parameter arm on excessive repetitive postings or projected runtime beyond the allocation.
All full-data work, dependency installation, and tests run on EC2.
The shared 2-vCPU VM only orchestrates commands and handles small text artifacts.

Issue-owned data and results use versioned paths below `s3://oa-bolinas/issues/568/`.
Do not duplicate the existing genome source assets into the result namespace.
Preserve input ETags/sizes, fixture checksums, the producing code snapshot, per-locus predictions, and resource records.
Small summary artifacts live on this permanent branch; reviewed interpretations are extracted into a separate documentation pull request.

## Reproduction

From this directory on an appropriately sized worker:

```bash
uv sync --locked --group dev
uv run --locked pytest
uv run --locked kmer-fixture --root /data/issue568 --prior /data/issue568/prior
uv run --locked kmer-screen --root /data/issue568 --split dev
```

The `prior` path must contain the pinned #521 source tree extracted from the commit above.
See the logbook for exact producing commits and commands.

The completed fixture audit identified disjoint genomic intervals within six homology-linked split groups.
The initial metadata and figures from the first screen are superseded.
`python -m kmer_conservation.relabel_fixture --root /data/issue568 --output /data/issue568/v2` corrects locus identities while asserting that every sequence, genomic interval, identifier, and development/held-out assignment stays identical.
Final comparisons use this version-two fixture; see its manifest and audit.
Synthetic duplicate controls use `python -m kmer_conservation.synthetic --root /data/issue568/v2 --duplicates 10`, with ten independently mutated tract copies per target species and locus.
These are planted paralog-like competitors, not annotated biological paralogs.
