# Feature 1662 saturation perturbation

This post-hoc mechanistic experiment asks whether the held-out, label-associated
block-19 SAE feature 1662 is specifically sensitive to coding structure. The
frozen design is recorded in issue #438 before annotation or activation
extraction.

## Frozen design

- Start from the 2,040 missense rows in the pinned official complex-traits test
  split.
- Annotate with the official Ensembl REST VEP endpoint using `pick=1` and
  `canonical=1`.
- Keep picked protein-coding, single-codon SNVs with an unambiguous non-stop
  reference codon and verified allele orientation.
- Select 128 contexts at each focal codon position by the frozen SHA-256 key.
  Selection cannot read labels, SAE responses, prediction scores, or clinical
  annotations.
- In each 255-bp reference context, make all three possible substitutions at
  every genomic offset from -15 through +15.
- Measure the focal-token activation of block-19 SAE feature 1662 separately in
  forward and reverse-complement sequence orientation.

The planned tests are generic focal sensitivity, codon position 2 versus 1 and
3, and within-context nonsynonymous versus synonymous sensitivity. Full signed
and unsigned profiles are descriptive. The coding-consequence test requires at
least 30 contexts containing both synonymous and nonsynonymous center edits;
otherwise it is reported as underpowered and fails its strict criterion.

## Reproduce

The design step needs the exact pinned `test.parquet` and a 40-character
`EXPERIMENT_COMMIT`:

```bash
uv run --project experiments/exp438_m51_complex_trait_layers --no-dev python \
  experiments/exp438_m51_complex_trait_layers/prepare_feature1662_saturation.py \
  --panel /path/to/test.parquet \
  --output-dir /path/to/dna-exp438-feature1662-saturation-r1-design
```

After uploading the validated design directory to its S3 prefix, launch
`sky.saturation.yaml` with the same pinned commit in `EXPERIMENT_COMMIT`.

## Durable storage

- Design: `s3://oa-bolinas/experiments/exp438/retrieval/dna-exp438-feature1662-saturation-r1-design/`
- Extraction, analysis, plots, and archive manifest:
  `s3://oa-bolinas/experiments/exp438/retrieval/dna-exp438-feature1662-saturation-r1/`

Local and Sky disks are staging areas only. Both S3 directories carry
per-artifact SHA-256 hashes, and the final run additionally carries a recursive
archive manifest.
