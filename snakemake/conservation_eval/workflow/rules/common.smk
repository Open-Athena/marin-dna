from pathlib import Path

import pandas as pd

from datasets import load_dataset

from bolinas.pipelines.evals.conservation import (
    CONSERVATION_TRACKS,
    REQUIRED_VARIANT_COLUMNS,
    aggregate_conservation_metrics,
    score_variants_at_positions,
)


SCORES = config["scores"]
SPLITS = config["splits"]
DATASETS = [d["name"] for d in config["datasets"]]
INPUT_HF_PREFIX = config["input_hf_prefix"]


def get_dataset_config(name: str) -> dict:
    # Per-dataset entry lookup (name, hf_revision). Mirrors the helper in
    # snakemake/analysis/evals_v2/workflow/rules/common.smk.
    for d in config["datasets"]:
        if d["name"] == name:
            return d
    raise KeyError(f"unknown dataset {name!r}")


# Sanity-check up-front: every score listed in config must be a known track.
_unknown = set(SCORES) - set(CONSERVATION_TRACKS)
assert not _unknown, (
    f"unknown scores in config: {_unknown}. " f"Known: {sorted(CONSERVATION_TRACKS)}"
)
