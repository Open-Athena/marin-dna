"""Common imports + helpers for evals_v2 rules."""

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from marin_dna.pipelines.evals.conservation import (
    QTL_VARIANT_COLUMNS,
    REQUIRED_VARIANT_COLUMNS,
)
from marin_dna.pipelines.evals.inference import compute_variant_scores
from marin_dna.pipelines.evals.metrics import (
    SCORE_PROTOCOLS,
    compute_auprc_metrics,
    compute_qtl_metrics,
)

# Per-dataset eval protocol. `matched_pair` (default) → per-subset AUPRC +
# cluster bootstrap over `match_group` (mendelian/complex). `qtl_global` →
# global AUPRC + positives-only effect_size correlation on the unmatched
# DART-Eval QTL datasets (caqtl/dsqtl), which carry no subset/match_group.
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


def get_model_config(name):
    for m in config["models"]:
        if m["name"] == name:
            return m
    raise ValueError(f"model {name!r} not found in config")


# Each model entry must declare exactly one source — fail loud here so a
# typo in config doesn't surface as a confusing rule error later.
for _m in config["models"]:
    _has_gcs = "gcs_path" in _m
    _has_hf = "hf_repo" in _m
    assert (
        _has_gcs ^ _has_hf
    ), f"model {_m['name']!r} must have exactly one of `gcs_path` or `hf_repo`"

# Same fail-fast for per-dataset score_protocol — a typo would surface
# late as a KeyError inside the metrics rule's `SCORE_PROTOCOLS[protocol]`.
for _d in config["datasets"]:
    assert _d["score_protocol"] in SCORE_PROTOCOLS, (
        f"dataset {_d['name']!r} `score_protocol` must be one of "
        f"{sorted(SCORE_PROTOCOLS)}, got {_d['score_protocol']!r}"
    )
    _ep = _d.get("eval_protocol", "matched_pair")
    assert _ep in EVAL_PROTOCOLS, (
        f"dataset {_d['name']!r} `eval_protocol` must be one of "
        f"{sorted(EVAL_PROTOCOLS)}, got {_ep!r}"
    )


# Wildcard alternations used across rules.
DATASETS = [d["name"] for d in config["datasets"]]
MODELS = [m["name"] for m in config["models"]]


def get_model_datasets(model_name):
    """Datasets a given model is evaluated on.

    Defaults to all configured datasets; a model entry may set
    ``datasets: [name, …]`` to restrict evaluation to a subset.
    """
    cfg = get_model_config(model_name)
    if "datasets" not in cfg:
        return DATASETS
    bad = [d for d in cfg["datasets"] if d not in DATASETS]
    assert not bad, (
        f"model {model_name!r} `datasets` references unknown names: {bad} "
        f"(known: {DATASETS})"
    )
    return cfg["datasets"]


def get_model_batch_size(model_name):
    """Per-model ``batch_size`` if set, else the global ``inference.batch_size``."""
    bs = get_model_config(model_name).get(
        "batch_size", config["inference"]["batch_size"]
    )
    assert (
        isinstance(bs, int) and bs > 0
    ), f"model {model_name!r} `batch_size` must be a positive int, got {bs!r}"
    return bs


# --- Nucleotide dependency maps (interpretation, issue #237) ----------------
# Optional `nuc_dep:` config section. Targets are kept off `rule all`; see
# rules/interpretation.smk.
NUC_DEP_CFG = config.get("nuc_dep", {})
NUC_DEP_LOCI = list(NUC_DEP_CFG.get("loci", {}))
NUC_DEP_MODELS = NUC_DEP_CFG.get("models", [])


def get_nuc_dep_ord():
    """Vector-norm order for collapsing each 4x4 Jacobian block. ``inf`` (the
    default, as in GPN-Star) → ``np.inf``; otherwise a float."""
    raw = NUC_DEP_CFG.get("ord", "inf")
    if isinstance(raw, str) and raw.lower() in ("inf", "infinity"):
        return np.inf
    return float(raw)


# Fail fast: every nuc_dep model is a known checkpoint, and every locus fits the
# context window of every model it would run on (the categorical Jacobian needs
# the whole locus inside one window).
for _m in NUC_DEP_MODELS:
    assert _m in MODELS, f"nuc_dep model {_m!r} not found in `models`"
for _loc, _c in NUC_DEP_CFG.get("loci", {}).items():
    assert _c["end"] > _c["start"], (
        f"nuc_dep locus {_loc!r}: end {_c['end']} must exceed start {_c['start']}"
    )
    for _m in NUC_DEP_MODELS:
        _ws = get_model_config(_m)["window_size"]
        assert _c["end"] - _c["start"] <= _ws, (
            f"nuc_dep locus {_loc!r} span {_c['end'] - _c['start']} bp exceeds "
            f"model {_m!r} window_size {_ws}"
        )
