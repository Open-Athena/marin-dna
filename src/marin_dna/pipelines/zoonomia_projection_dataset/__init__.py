"""Pipeline-specific helpers for ``snakemake/zoonomia_projection_dataset/``.

- ``validation``: Ensembl-flavored region builders + case-encoded validation
  parquet construction for the seven per-recipe validation HF datasets
  (val_cds / val_utr5 / val_utr3 / val_ncrna / val_promoter / val_enhancer
  / val_tss_pc).
- ``region_labels``: per-anchor region-type annotation + per-subset HF
  dataset cards for the six v3 partition datasets (v3_cds / v3_utr3 /
  v3_ncrna_exon / v3_tss_region_and_utr5 / v3_ccre_non_promoter / v3_bg).
"""

# GitHub coordinates for commit-pinned permalinks in per-repo HF dataset
# cards. Used by ``validation.write_hf_readme`` and
# ``region_labels.write_subset_hf_readme``.
_GITHUB_PIPELINE_PATH = "snakemake/zoonomia_projection_dataset"
_GITHUB_REPO = "Open-Athena/marin-dna"
