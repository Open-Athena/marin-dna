"""Common imports + helpers for alphagenome_eval rules.

Two independent paths share this pipeline (and its AlphaGenome API setup):

- **Matched-group baseline** (`datasets`: mendelian/complex) — per-track L2 → max →
  AUPRC ± cluster-bootstrap SE (cluster = `match_group`, the 1:k matched set). Rules:
  `compute_per_track_l2`, `aggregate_max`,
  `compute_metrics` (`score.smk` / `metrics.smk`).
- **DNase-LFC QTL predictions** (`dnase_lfc.datasets`: caqtl/dsqtl) — the single
  GM12878-DNase LFC scorer for the supervised official-metrics benchmark (#311). Rule:
  `score_dnase_lfc` (`dnase_lfc.smk`). The benchmark *metrics* are not here — they are
  the model-agnostic `scripts/qtl_benchmark.py` driver.
"""

import pandas as pd
from datasets import load_dataset

from marin_dna.pipelines.evals.alphagenome import score_variants_alphagenome
from marin_dna.pipelines.evals.conservation import REQUIRED_VARIANT_COLUMNS
from marin_dna.pipelines.evals.metrics import compute_auprc_metrics


def get_dataset_config(name):
    for d in config["datasets"]:
        if d["name"] == name:
            return d
    raise ValueError(f"dataset {name!r} not found in config")


# Matched-group datasets. Pinned in a `wildcard_constraints` on every matched-group
# rule so its `{dataset}` wildcard can't match the DNase-LFC `{ds}` paths.
DATASETS = [d["name"] for d in config["datasets"]]
MATCHED_CONSTRAINT = "|".join(DATASETS)

# DNase-LFC QTL prediction datasets (#311).
DNASE_LFC = config["dnase_lfc"]
DNASE_LFC_DATASETS = [d["name"] for d in DNASE_LFC["datasets"]]
DNASE_LFC_CONSTRAINT = "|".join(DNASE_LFC_DATASETS)


def get_dnase_lfc_revision(name):
    for d in DNASE_LFC["datasets"]:
        if d["name"] == name:
            return d["hf_revision"]
    raise ValueError(f"dnase_lfc dataset {name!r} not found in config")
