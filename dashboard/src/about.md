---
title: About
---

# About this observatory

This site is the public face of `marin-dna`, in two pillars: **Benchmarks** — how well each gLM / conservation track / external baseline ranks pathogenic / causal variants against matched negative controls — and **Interpretation**, which visualizes what the trained models have learned.

The benchmark tables replace the hand-curated ones on [#161](https://github.com/Open-Athena/marin-dna/issues/161) (Mendelian) and [#162](https://github.com/Open-Athena/marin-dna/issues/162) (Complex).

## Benchmarks methodology

**AUPRC** (area under the precision–recall curve) on the full ranked list of variants within a subset. Each positive is matched 1:9 against nearest-neighbor negatives sharing consequence + chromosome + (continuous) TSS/exon-distance features, so the positive rate is **10%** by design — a random ranker scores 0.10, a perfect ranker scores 1.00.

**SE** is the cluster bootstrap over `match_group`s (1000 resamples). Bootstrapping at the group level preserves the matched-pair clustering that gives the metric meaning. Implemented in [`snakemake/analysis/evals_v2/src/marin_dna_evals/metrics.py`](https://github.com/Open-Athena/marin-dna/blob/main/snakemake/analysis/evals_v2/src/marin_dna_evals/metrics.py).

Each method × dataset emits two aggregate rows alongside the per-subset cells:

- **Global** — AUPRC across **all** match groups, regardless of per-subset size.
- **Macro Avg** — unweighted mean of per-subset AUPRCs across the K subsets meeting the n_positives ≥ 30 threshold (see *Subset threshold* below). SE is `√(Σ SE²) / K`.

**Sort axis.** Mendelian sorts by Macro Avg (the variant composition is dominated by missense — a ClinVar annotator-history artifact, not pathogenicity reality — so Global AUPRC over-weights protein-coding-specialist methods). Complex traits sorts by Global.

**Subset threshold.** A subset is shown as a per-subset column (and contributes to Macro Avg) only if it has at least **30 positives** — i.e. `n_positives ≥ 30`, which on the headers (where `n` is total variants, exactly 10× positives at 1:9) corresponds to `n ≥ 300`. Subsets below the threshold still contribute to Global.

**Train split only.** Test is held out for the final-eval pass. All numbers here reflect train development.

## Interpretation methodology

**Categorical Jacobian / nucleotide dependency maps** measure how substituting the base at one position shifts the model's predicted nucleotide distribution at every other position, over a locus-sized window — an `L×L` map whose off-diagonal structure flags coupled positions (splice sites, structured elements). The method was discovered independently for protein language models (categorical Jacobian; Zhang et al., *PNAS* 2024) and genomic LMs (nucleotide dependency; Tomaz da Silva et al., *Nat. Genet.* 2025). Our models are causal, so each map stitches a forward and a reverse-complement pass and symmetrizes (`mean`). Computed by the evals_v2 pipeline; see [#237](https://github.com/Open-Athena/marin-dna/issues/237) for the method and the autoregressive correctness argument. Each map links out to the region in the UCSC Genome Browser.

## Agent-readable data

The dashboard is a presentation layer over plain-text source files. To consume the data programmatically, fetch one of:

- **`dashboard/models.yaml`** in the repo — canonical metadata for every method. `gh api repos/Open-Athena/marin-dna/contents/dashboard/models.yaml` or `git show main:dashboard/models.yaml`.
- **`/data/models.json`** under this site — models.yaml normalized to JSON. Same fields as the YAML.
- **`/data/leaderboard.parquet`** under this site — long-form `(method × dataset × subset)` AUPRC + SE + `n` (total variants in the subset, or K on the macro row) + `n_positives` (positives in the subset, used for the ≥30 display gate). Readable from Python (`pl.read_parquet(URL)`) or DuckDB (`SELECT * FROM read_parquet('URL')`).
- **`/data/datasets.json`** under this site — per-dataset metadata (HF commit, score type, etc.).
- **`/data/nuc_dep.zip`** under this site — interpretation artifacts: the nucleotide-dependency heatmap SVGs (`{combine}/{locus}/{model}.svg`) plus a `manifest.json` of per-locus metadata + UCSC links, built from the evals_v2 `nuc_dep` outputs on S3.

Every field shown in a table or tooltip is present in those files; the rendered HTML never hides information behind a click.

## Adding a new method

1. Append a YAML block to [`dashboard/models.yaml`](https://github.com/Open-Athena/marin-dna/blob/main/dashboard/models.yaml) (registry order; tag the appropriate `datasets`).
2. For `family: marin_dna`, also add the model to [`snakemake/analysis/evals_v2/config/config.yaml`](https://github.com/Open-Athena/marin-dna/blob/main/snakemake/analysis/evals_v2/config/config.yaml).
3. Run the evals_v2 pipeline → parquet written to S3.
4. Open a PR; CI rebuilds this site and the new row appears.

The schema is documented at the top of models.yaml.

## Adding an interpretation type

The Interpretation pillar is a convention, not a framework — each analysis is one nav entry + one page + one data loader:

1. Produce the artifacts in a pipeline (e.g. evals_v2) and sync them to S3.
2. Add a data loader under [`dashboard/src/data/`](https://github.com/Open-Athena/marin-dna/tree/main/dashboard/src/data) that fetches them — derive keys from config rather than enumerating S3 (the CI role has `GetObject`, not `ListBucket`), and skip not-yet-materialized artifacts. Put non-trivial Python in the owning project package so it is testable (e.g. [`interpretation_catalog.py`](https://github.com/Open-Athena/marin-dna/blob/main/snakemake/analysis/evals_v2/src/marin_dna_evals/interpretation_catalog.py)).
3. Add a page under `dashboard/src/interpretation/` and a nav entry under the **Interpretation** section in [`observablehq.config.js`](https://github.com/Open-Athena/marin-dna/blob/main/dashboard/observablehq.config.js).
4. Open a PR; CI rebuilds the site.
