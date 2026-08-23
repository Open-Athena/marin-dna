"""Pipeline-specific helpers for ``snakemake/training_dataset/``.

- ``hf_readme``: per-repo HuggingFace dataset-card (``README.md``) generators
  for the ``genomes-v*`` training and validation datasets.
  Each uploaded dataset ships a commit-pinned, recipe-accurate card.
"""

# GitHub coordinates for commit-pinned permalinks embedded in the per-repo HF
# dataset cards. Used by ``hf_readme.build_training_readme`` /
# ``build_validation_readme``.
_GITHUB_PIPELINE_PATH = "snakemake/training_dataset"
_GITHUB_REPO = "Open-Athena/marin-dna"
