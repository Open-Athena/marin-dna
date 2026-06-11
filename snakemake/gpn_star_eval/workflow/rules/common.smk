"""Common imports + helpers for gpn_star_eval rules."""

import pandas as pd
from datasets import load_dataset

from marin_dna.pipelines.evals.conservation import (
    REQUIRED_VARIANT_COLUMNS,
    SGE_VARIANT_COLUMNS,
)
from marin_dna.pipelines.evals.gpn_star import predictions_url, score_variants_gpn_star
from marin_dna.pipelines.evals.metrics import compute_auprc_metrics, compute_sge_metrics

# `datasets` is a list of {name, hf_revision, eval_protocol?} (revisions pinned to
# the HF commits evals_v2 targets). Names drive the wildcards; the revision map
# pins `load_dataset` so the prediction parquets row-align deterministically.
DATASETS = [d["name"] for d in config["datasets"]]
HF_REVISION = {d["name"]: d["hf_revision"] for d in config["datasets"]}
MODELS = config["models"]
SCORE_COLUMNS = config["score_columns"]

# Per-dataset eval protocol — mirrors snakemake/conservation_eval common.smk.
# `matched_pair` (default) → per-subset AUPRC + cluster bootstrap on `match_group`
# (`compute_auprc_metrics`). `sge` (#301) → per-accession × consequence-subset
# AUPRC on the binary `label` of evals_sge (`compute_sge_metrics`). No `qtl_global`
# here — GPN-Star has no current-revision QTL prediction upload.
EVAL_PROTOCOLS = ("matched_pair", "sge")


def get_dataset_config(name: str) -> dict:
    """Per-dataset config entry (name, hf_revision, eval_protocol)."""
    for d in config["datasets"]:
        if d["name"] == name:
            return d
    raise KeyError(f"unknown dataset {name!r}")


def get_dataset_protocol(name: str) -> str:
    """Eval protocol for a dataset (default `matched_pair`)."""
    return get_dataset_config(name).get("eval_protocol", "matched_pair")


def get_dataset_variant_columns(name: str) -> tuple[str, ...]:
    """Required variant columns for a dataset, keyed by eval protocol. These are
    carried verbatim from the HF dataset into the scores parquet so the metric
    step (`compute_auprc_metrics` / `compute_sge_metrics`) has its grouping +
    label columns."""
    if get_dataset_protocol(name) == "sge":
        return SGE_VARIANT_COLUMNS
    return REQUIRED_VARIANT_COLUMNS


# Fail fast on an unknown eval_protocol (typo) at parse time.
for _d in config["datasets"]:
    _ep = _d.get("eval_protocol", "matched_pair")
    assert _ep in EVAL_PROTOCOLS, (
        f"dataset {_d['name']!r} `eval_protocol` must be one of "
        f"{sorted(EVAL_PROTOCOLS)}, got {_ep!r}"
    )
