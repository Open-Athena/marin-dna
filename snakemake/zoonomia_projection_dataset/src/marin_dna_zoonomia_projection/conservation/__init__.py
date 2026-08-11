"""Conservation-track utilities used by ``snakemake/zoonomia_projection_dataset/``.

- ``histogram``: per-base value histograms over defined regions of a phyloP bigWig.
- ``calibration``: pick a threshold for one track that matches the genome-wide
  passing nucleotide count of another track at a reference threshold.
- ``scoring``: parse ``bigWigAverageOverBed`` output into a typed Polars frame.
"""
