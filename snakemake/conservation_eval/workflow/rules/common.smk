from pathlib import Path

import pandas as pd

from datasets import load_dataset

from marin_dna.pipelines.evals.conservation import (
    CONSERVATION_TRACKS,
    QTL_VARIANT_COLUMNS,
    REQUIRED_VARIANT_COLUMNS,
    SGE_VARIANT_COLUMNS,
    aggregate_conservation_metrics,
    aggregate_conservation_qtl_metrics,
    aggregate_conservation_sge_metrics,
    score_variants_at_positions,
)


SCORES = config["scores"]
SPLITS = config["splits"]
DATASETS = [d["name"] for d in config["datasets"]]
INPUT_HF_PREFIX = config["input_hf_prefix"]

# Per-dataset eval protocol — see snakemake/analysis/evals_v2 common.smk.
# `matched_pair` (default) → per-subset AUPRC + cluster bootstrap.
# `qtl_global` → global AUPRC + positives-only effect_size correlation on the
# unmatched DART-Eval QTL datasets (caqtl/dsqtl).
# `sge` (issue #301) → per-accession × consequence-subset Spearman + AUPRC on
# evals_sge, via the same shared `compute_sge_metrics` as evals_v2.
EVAL_PROTOCOLS = ("matched_pair", "qtl_global", "sge")


def get_dataset_config(name: str) -> dict:
    # Per-dataset entry lookup (name, hf_revision). Mirrors the helper in
    # snakemake/analysis/evals_v2/workflow/rules/common.smk.
    for d in config["datasets"]:
        if d["name"] == name:
            return d
    raise KeyError(f"unknown dataset {name!r}")


def get_dataset_protocol(name: str) -> str:
    """Eval protocol for a dataset (default `matched_pair`)."""
    return get_dataset_config(name).get("eval_protocol", "matched_pair")


def get_dataset_variant_columns(name: str) -> tuple[str, ...]:
    """Required variant columns for a dataset, keyed by eval protocol."""
    protocol = get_dataset_protocol(name)
    if protocol == "qtl_global":
        return QTL_VARIANT_COLUMNS
    if protocol == "sge":
        return SGE_VARIANT_COLUMNS
    return REQUIRED_VARIANT_COLUMNS


# Fail fast on an unknown eval_protocol (typo) at parse time.
for _d in config["datasets"]:
    _ep = _d.get("eval_protocol", "matched_pair")
    assert _ep in EVAL_PROTOCOLS, (
        f"dataset {_d['name']!r} `eval_protocol` must be one of "
        f"{sorted(EVAL_PROTOCOLS)}, got {_ep!r}"
    )


# Sanity-check up-front: every score listed in config must be a known track.
_unknown = set(SCORES) - set(CONSERVATION_TRACKS)
assert not _unknown, (
    f"unknown scores in config: {_unknown}. " f"Known: {sorted(CONSERVATION_TRACKS)}"
)
