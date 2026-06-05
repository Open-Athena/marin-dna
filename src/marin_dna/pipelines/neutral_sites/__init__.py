"""Build the GPN-Star neutral-site set on GRCh38 for mutation-rate calibration.

Model-independent genome engineering: it produces a reusable parquet of
high-confidence neutral ``(chrom, pos, ref)`` sites that the calibration step in
``snakemake/analysis/evals_v2`` then scores per checkpoint. See
``snakemake/neutral_sites/README.md`` and issue #267.
"""
