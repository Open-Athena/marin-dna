# exp421: SAE variant effects and AlphaGenome tracks

This bounded analysis asks whether the magnitude of an m5.1 SAE feature response to a Mendelian variant is associated with AlphaGenome's precomputed, unsigned per-track L2 effect score. It pools all variants in the primary analysis; `label` and `subset` are not used for stratification.

The analysis uses chromosome-held-out discovery, validation, and test splits. Pearson correlation screens all eligible feature–track pairs on discovery data. The top validation-consistent candidates are frozen before the test split is read. Held-out Pearson correlation is the primary metric; Spearman correlation is a predeclared secondary confirmation of monotonic associations. Match-group-centered Pearson and Spearman correlations are sensitivities rather than the primary estimands.

The current AlphaGenome metadata endpoint is saved with the results. Its assay counts match the historical score export exactly, but the original May 2026 metadata snapshot was not preserved, so named-track interpretation carries that provenance caveat.

## Canonical biosample mapping

Before the grouped-L2 analysis, `biosample_mapping.py` materializes a metadata-only mapping whose canonical key is `biosample_type|ontology_curie`. This merges donors, replicates, and same-ontology label synonyms while keeping tissues, primary cells, in-vitro differentiated cells, organoids, and cell lines distinct. It writes the full track mapping, canonical-unit and assay-by-unit catalogs, an explicit review queue, hashes, and a human-readable audit without reading AlphaGenome scores or SAE outputs.

Run the unit tests from this directory:

```bash
uv run --with numpy --with scipy --with polars --with matplotlib --with pytest pytest -q test_association.py
```

The remote task is defined in `sky.yaml`. Its output includes the discovery and held-out tables, top test variants, a Markdown summary, an overview figure, and a hash manifest.
