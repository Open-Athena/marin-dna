"""Common imports + helpers for evals_v2 rules."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset

from marin_dna_evals.conservation import (
    QTL_VARIANT_COLUMNS,
    REQUIRED_VARIANT_COLUMNS,
    SGE_VARIANT_COLUMNS,
)
from marin_dna_evals.inference import compute_variant_scores
from marin_dna_evals.metrics import (
    SCORE_PROTOCOLS,
    compute_auprc_metrics,
    compute_qtl_metrics,
    compute_sge_metrics,
    compute_sge_probe_metrics,
    per_chrom_ap_table,
)
from marin_dna_evals.variant_probe import PAIR_COMBOS, run_subset_probes

# Per-dataset eval protocol. `matched_pair` (default) → per-subset AUPRC +
# cluster bootstrap over `match_group` (mendelian/complex). `qtl_global` →
# global AUPRC + positives-only effect_size correlation on the unmatched
# DART-Eval QTL datasets (caqtl/dsqtl), which carry no subset/match_group.
# `sge` (issue #301) → the saturation-genome-editing dataset (evals_sge):
# per-accession (mavedb_urn) × consequence-subset AUPRC on the binary `label`
# (impactful = calibrated abnormal), macro-averaged over subsets and accessions
# (compute_sge_metrics).
EVAL_PROTOCOLS = ("matched_pair", "qtl_global", "sge")


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
    protocol = get_dataset_protocol(name)
    if protocol == "qtl_global":
        return QTL_VARIANT_COLUMNS
    if protocol == "sge":
        return SGE_VARIANT_COLUMNS
    return REQUIRED_VARIANT_COLUMNS


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
    if _has_hf:
        assert _m.get("hf_revision"), f"HF model {_m['name']!r} must pin `hf_revision`"

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

# The pooled embedding (#318) is the FWD+RC average, so it needs both strands —
# fail at config load rather than store a silently fwd-only vector mislabeled as
# the average.
assert (
    not config["inference"].get("return_embeddings", False) or config["inference"]["rc"]
), "inference.return_embeddings=true requires inference.rc=true"


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
NUC_DEP_COMBINES = NUC_DEP_CFG.get("combines", ["mean"])


def get_nuc_dep_window(model):
    """nuc_dep context window for a model: a fixed ``nuc_dep.window_size`` (so all
    models see the same input, #240) if set, else the model's native window."""
    return NUC_DEP_CFG.get("window_size") or get_model_config(model)["window_size"]


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
for _c in NUC_DEP_COMBINES:
    assert _c in ("mean", "max"), f"nuc_dep combine {_c!r} must be 'mean' or 'max'"
for _m in NUC_DEP_MODELS:
    assert _m in MODELS, f"nuc_dep model {_m!r} not found in `models`"
for _loc, _c in NUC_DEP_CFG.get("loci", {}).items():
    assert (
        _c["end"] > _c["start"]
    ), f"nuc_dep locus {_loc!r}: end {_c['end']} must exceed start {_c['start']}"
    assert _c["strand"] in (
        "+",
        "-",
    ), f"nuc_dep locus {_loc!r}: strand must be '+' or '-', got {_c['strand']!r}"
    for _m in NUC_DEP_MODELS:
        _ws = get_nuc_dep_window(_m)
        assert _c["end"] - _c["start"] <= _ws, (
            f"nuc_dep locus {_loc!r} span {_c['end'] - _c['start']} bp exceeds "
            f"nuc_dep window_size {_ws}"
        )


# --- Embedding UMAP (interpretation, issue #246) ----------------------------
# Optional `umap_embeddings:` config section; targets kept off `rule all`
# (see rules/embedding_umap.smk). Reuses the `models:` registry.
UMAP_CFG = config.get("umap_embeddings", {})
UMAP_MODELS = UMAP_CFG.get("models", [])

# Fail fast: every umap model is a known checkpoint whose context window can
# hold the center-pooled region (window_size >= n_center_bp).
for _m in UMAP_MODELS:
    assert _m in MODELS, f"umap_embeddings model {_m!r} not found in `models`"
    _ws = get_model_config(_m)["window_size"]
    _nc = UMAP_CFG.get("n_center_bp", 100)
    assert (
        _ws >= _nc
    ), f"umap_embeddings model {_m!r} window_size {_ws} < n_center_bp {_nc}"


# --- LL gap (functional vs non-functional log-likelihood, issue #274) --------
# Optional `ll_gap:` config section; targets kept off `rule all` (see
# rules/ll_gap.smk). Reuses the `models:` registry. Its `datasets` are mixed-case
# validation-interval HF datasets (a `seq` column, uppercase = phyloP-functional)
# — distinct from the variant `datasets:` list above.
LL_GAP_CFG = config.get("ll_gap", {})
LL_GAP_MODELS = LL_GAP_CFG.get("models", [])
LL_GAP_DATASETS = [d["name"] for d in LL_GAP_CFG.get("datasets", [])]


def get_ll_gap_dataset_config(name):
    for d in LL_GAP_CFG.get("datasets", []):
        if d["name"] == name:
            return d
    raise ValueError(f"ll_gap dataset {name!r} not found in config")


# Fail fast: every ll_gap model is a known checkpoint, and every ll_gap dataset
# pins an HF repo + revision (so a data bump triggers re-execution).
for _m in LL_GAP_MODELS:
    assert _m in MODELS, f"ll_gap model {_m!r} not found in `models`"
for _d in LL_GAP_CFG.get("datasets", []):
    assert (
        "hf_repo" in _d and "hf_revision" in _d
    ), f"ll_gap dataset {_d.get('name')!r} needs `hf_repo` + `hf_revision`"
# --- Conservation × repeat predictability (issue #478) ----------------------
PREDICTABILITY_478_CFG = config.get("predictability_478", {})
PREDICTABILITY_478_MODELS = PREDICTABILITY_478_CFG.get("models", [])
PREDICTABILITY_478_DATASETS = [
    dataset["name"] for dataset in PREDICTABILITY_478_CFG.get("datasets", [])
]
PREDICTABILITY_478_VERSION = PREDICTABILITY_478_CFG.get("artifact_version", "v1")
PREDICTABILITY_478_PILOT_MODEL = (
    PREDICTABILITY_478_MODELS[0] if PREDICTABILITY_478_MODELS else None
)


def get_predictability_478_dataset_config(name):
    for dataset in PREDICTABILITY_478_CFG.get("datasets", []):
        if dataset["name"] == name:
            return dataset
    raise ValueError(f"predictability_478 dataset {name!r} not found")


def get_predictability_478_batch_size(model):
    batch_sizes = PREDICTABILITY_478_CFG.get("batch_sizes", {})
    value = batch_sizes.get(model, get_model_batch_size(model))
    assert (
        isinstance(value, int) and value > 0
    ), f"predictability_478 batch size for {model!r} must be positive"
    return value


if PREDICTABILITY_478_CFG:
    assert PREDICTABILITY_478_VERSION == "v1", (
        "unknown predictability_478 artifact schema version "
        f"{PREDICTABILITY_478_VERSION!r}"
    )
    assert PREDICTABILITY_478_DATASETS == ["cds", "upstream", "downstream"]
    assert PREDICTABILITY_478_CFG["window_size"] == 255
    assert (
        PREDICTABILITY_478_CFG["primary_start"],
        PREDICTABILITY_478_CFG["primary_end_exclusive"],
    ) == (32, 223)
    assert PREDICTABILITY_478_CFG["assembly"] == "GCF_000001405.40"
    assert PREDICTABILITY_478_CFG["repeat_twobit"].endswith("/GCF_000001405.40.2bit")
    assert PREDICTABILITY_478_CFG["cds_gtf"].endswith("/GCF_000001405.40.gtf.gz")
    for _model in PREDICTABILITY_478_MODELS:
        assert (
            _model in MODELS
        ), f"predictability_478 model {_model!r} not found in `models`"
        assert get_model_config(_model)["window_size"] == 255
        get_predictability_478_batch_size(_model)
    for _dataset in PREDICTABILITY_478_CFG["datasets"]:
        assert {"name", "hf_repo", "hf_revision"} <= set(_dataset)


# --- Linear probe (frozen-embedding VEP, issue #320) ------------------------
# Optional `probe:` config section; targets kept OFF `rule all` (see
# rules/probe.smk). Reuses the `models:` registry. Each probe entry is
# `{name, datasets: [...]}` — datasets are listed explicitly (not derived) because
# probing a cell requires its scores parquet to carry emb_ref/emb_alt
# (inference.return_embeddings), which is per-cell, not per-model.
PROBE_CFG = config.get("probe", {})
PROBE_MODELS = PROBE_CFG.get("models", [])

# Probe pair-feature per dataset: directional datasets (minus_llr) take the signed
# `concat_ref_delta = [ref, alt−ref]`; swap-invariant ones (abs_llr) take the
# symmetric `sum_absdiff = [ref+alt, |alt−ref|]` (issue #314 feature rule).
PROBE_FEATURE_BY_PROTOCOL = {
    "minus_llr": "concat_ref_delta",
    "abs_llr": "sum_absdiff",
}

# Fail fast: the feature map must cover every score protocol a dataset can declare
# (config load already asserts each dataset's score_protocol ∈ SCORE_PROTOCOLS), so
# get_probe_feature never raises a bare KeyError late at rule-eval — mirroring the
# guard the metrics path already has on its own SCORE_PROTOCOLS lookup.
_unmapped_protocols = sorted(set(SCORE_PROTOCOLS) - set(PROBE_FEATURE_BY_PROTOCOL))
assert not _unmapped_protocols, (
    f"PROBE_FEATURE_BY_PROTOCOL is missing entries for score protocols "
    f"{_unmapped_protocols}; add them or get_probe_feature will KeyError"
)


def get_probe_feature(name):
    """Probe pair-feature combo for a dataset, from its `score_protocol`; an
    explicit `probe_feature` on the dataset entry overrides."""
    cfg = get_dataset_config(name)
    if "probe_feature" in cfg:
        return cfg["probe_feature"]
    return PROBE_FEATURE_BY_PROTOCOL[cfg["score_protocol"]]


# Fail fast: every probe entry names a known checkpoint and lists datasets that
# model is actually evaluated on; any explicit `probe_feature` override is a known
# combo. A typo would otherwise surface late inside the rule.
for _pm in PROBE_MODELS:
    assert isinstance(_pm, dict) and {"name", "datasets"} <= _pm.keys(), (
        f"probe `models` entry must be a mapping with `name` + `datasets` keys "
        f"(unlike the bare-string `models:` of umap/ll_gap), got {_pm!r}"
    )
    assert _pm["name"] in MODELS, f"probe model {_pm['name']!r} not found in `models`"
    _model_datasets = get_model_datasets(_pm["name"])
    for _d in _pm["datasets"]:
        assert _d in _model_datasets, (
            f"probe model {_pm['name']!r} dataset {_d!r} not in its evaluated "
            f"datasets {_model_datasets}"
        )
for _d in config["datasets"]:
    _pf = _d.get("probe_feature")
    assert _pf is None or _pf in PAIR_COMBOS, (
        f"dataset {_d['name']!r} `probe_feature` must be one of {PAIR_COMBOS}, "
        f"got {_pf!r}"
    )
