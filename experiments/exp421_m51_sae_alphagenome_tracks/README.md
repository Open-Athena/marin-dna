# exp421: SAE variant effects and AlphaGenome tracks

This bounded analysis asks whether the magnitude of an m5.1 SAE feature response to a Mendelian variant is associated with AlphaGenome's precomputed, unsigned per-track L2 effect score. It pools all variants in the primary analysis; `label` and `subset` are not used for stratification.

The original analysis uses chromosome-held-out discovery, validation, and test splits. Pearson correlation screens all eligible feature–track pairs on discovery data. The top validation-consistent candidates are frozen before the test split is read. Held-out Pearson correlation is the primary metric; Spearman correlation is a predeclared secondary confirmation of monotonic associations. Match-group-centered Pearson and Spearman correlations are sensitivities rather than the primary estimands.

`grouped_l2_association.py` is the registered complete-family revisit. It uses all variants and every globally supported feature from the block-1, block-10, and block-19 25M SAEs; keeps FWD and RC separate; and tests maxima at six resolutions: overall, assay, high-level tissue, major cell lineage, assay-by-tissue, and assay-by-lineage. Pearson on raw `abs(delta)` is primary, Pearson after `log1p` is the scale sensitivity, and Spearman is shared by both monotone feature scales. BH is applied to every feature-by-outcome pair within each layer, resolution, orientation, and statistic family.

The current AlphaGenome metadata endpoint is saved with the results. Its assay counts match the historical score export exactly, but the original May 2026 metadata snapshot was not preserved, so named-track interpretation carries that provenance caveat.

## Canonical biosample mapping

Before the grouped-L2 analysis, `biosample_mapping.py` materializes a metadata-only exact-ontology map whose canonical key is `biosample_type|ontology_curie`. This merges donors, replicates, and same-ontology label synonyms while preserving the source concepts for audit. `biosample_taxonomy.py` then maps those concepts onto two independently approved axes: 17 broad tissue/organ groups and 12 major cell lineages. A track may receive one group on each axis; unresolved concepts are excluded from the corresponding family rather than pooled into a heterogeneous `other` group. The taxonomy materialization copies its metadata inputs, writes track- and unit-level mappings and combination catalogs, and emits hashes plus a human-readable audit without reading variant labels, AlphaGenome scores, or SAE outputs.

Run the unit tests from this directory:

```bash
uv run --with numpy --with scipy --with polars --with matplotlib --with pytest pytest -q test_association.py
uv run --with numpy --with scipy --with polars --with pytest pytest -q test_grouped_l2_association.py
uv run --with polars --with pytest pytest -q test_biosample_mapping.py test_biosample_taxonomy.py
```

The remote task is defined in `sky.yaml`. Its output includes the discovery and held-out tables, top test variants, a Markdown summary, an overview figure, and a hash manifest.
