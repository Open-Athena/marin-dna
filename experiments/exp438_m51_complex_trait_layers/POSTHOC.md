# Feature 1662 post-hoc interpretation

This follow-up starts from the completed r2 extraction and treats block-19/25M
feature 1662 as a post-hoc candidate. It does not alter the preregistered
experiment or add a new confirmatory claim.

The workflow is intentionally staged:

1. `interpret_candidate.py` restores explicit zero rows for feature 1662 and
   audits FWD/RC activation states, response tails, allele changes, and loci.
2. `inspect_decoder.py` measures the feature's decoder-space neighborhood.
3. `annotate_missense_vep_parallel.py` obtains picked-transcript SIFT,
   PolyPhen, BLOSUM62, codon, gene, and clinical annotations for all 2,500
   missense variants from the official Ensembl VEP REST API. Each 50-variant
   response is checkpointed, and four concurrent workers remain far below the
   documented API rate limit.
4. `analyze_candidate_mechanism.py` tests associations with the VEP scores and
   summarizes label enrichment, ref→alt strata, codon position, and response
   tails.
5. `condition_candidate_mechanism.py` residualizes each response against
   ref→alt class plus codon position, then additionally against SIFT, PolyPhen,
   and BLOSUM62. Welch and Mann–Whitney families receive BH correction.
6. `plot_candidate_mechanism.py` produces the response-decile SVG/PNG.

All outputs are descriptive and post-hoc. The durable artifact prefix is:

```text
s3://oa-bolinas/experiments/exp438/retrieval/dna-exp438-feature1662-posthoc-r1/
```

The original extraction and SAE weights remain at their existing verified S3
locations; this compact archive does not duplicate them.
