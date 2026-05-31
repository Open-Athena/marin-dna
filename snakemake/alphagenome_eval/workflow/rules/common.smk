"""Common imports + helpers for alphagenome_eval rules."""

import os

import pandas as pd
from datasets import load_dataset

from marin_dna.pipelines.evals.alphagenome import score_variants_alphagenome
from marin_dna.pipelines.evals.conservation import (
    QTL_VARIANT_COLUMNS,
    REQUIRED_VARIANT_COLUMNS,
)
from marin_dna.pipelines.evals.metrics import compute_auprc_metrics, compute_qtl_metrics

# Per-dataset eval protocol — see snakemake/analysis/evals_v2 common.smk.
# `matched_pair` (default) → per-subset AUPRC + cluster bootstrap.
# `qtl_global` → global AUPRC + positives-only effect_size correlation on the
# unmatched DART-Eval QTL datasets (caqtl/dsqtl).
EVAL_PROTOCOLS = ("matched_pair", "qtl_global")


def get_dataset_config(name):
    for d in config["datasets"]:
        if d["name"] == name:
            return d
    raise ValueError(f"dataset {name!r} not found in config")


def get_dataset_protocol(name):
    """Eval protocol for a dataset (default `matched_pair`)."""
    return get_dataset_config(name).get("eval_protocol", "matched_pair")


def get_dataset_variant_columns(name):
    """Required variant columns for a dataset, keyed by eval protocol."""
    return (
        QTL_VARIANT_COLUMNS
        if get_dataset_protocol(name) == "qtl_global"
        else REQUIRED_VARIANT_COLUMNS
    )


DATASETS = [d["name"] for d in config["datasets"]]

# Fail fast on an unknown eval_protocol (typo) at parse time.
for _d in config["datasets"]:
    _ep = _d.get("eval_protocol", "matched_pair")
    assert _ep in EVAL_PROTOCOLS, (
        f"dataset {_d['name']!r} `eval_protocol` must be one of "
        f"{sorted(EVAL_PROTOCOLS)}, got {_ep!r}"
    )
