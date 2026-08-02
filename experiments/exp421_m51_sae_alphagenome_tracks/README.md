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

The historical split-based pass is defined in `sky.yaml`. The six-resolution complete-family revisit is defined in `sky.grouped_l2.yaml`; it downloads the frozen panel, 25M SAE activations, high-level taxonomy, and AlphaGenome L2 table directly from S3, then uploads a hash-manifested result archive.

Create compact result tables and both PNG and SVG synthesis figures with:

```bash
uv run --with polars --with numpy --with pandas --with matplotlib --with seaborn python experiments/exp421_m51_sae_alphagenome_tracks/summarize_grouped_l2.py --result-root <retrieved-result-directory> --output-dir <summary-directory>
```

## Leading-feature characterization

`characterize_grouped_l2_features.py` is a post-hoc follow-up for the three leading grouped-L2 features. It keeps FWD and RC separate, measures top-1%-trimmed robustness, joins the outcome-blind repeat panel, tests sequence/consequence/repeat annotations, and exports oriented top-response contexts. These analyses characterize already-selected features and are not a second confirmatory screen.

Run its focused tests with:

```bash
uv run --project experiments/exp421_422_statistical_revisits pytest -q experiments/exp421_m51_sae_alphagenome_tracks/test_characterize_grouped_l2_features.py
```

The bounded CPU runner is `sky.characterize_grouped_l2.yaml`. It retrieves the frozen panel, selected 25M SAE activations, high-level taxonomy, AlphaGenome L2 table, audited repeat annotations, and exact indexed GRCh38 FASTA, then uploads a hash-manifested result archive.
