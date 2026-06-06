"""Common imports + helpers for gpn_star_eval rules."""

import pandas as pd
from datasets import load_dataset

from marin_dna.pipelines.evals.conservation import REQUIRED_VARIANT_COLUMNS
from marin_dna.pipelines.evals.gpn_star import predictions_url, score_variants_gpn_star
from marin_dna.pipelines.evals.metrics import compute_auprc_metrics

# `datasets` is a list of {name, hf_revision} (revisions pinned to the k=9 /
# AUPRC HF commits). Names drive the wildcards; the revision map pins
# `load_dataset` so the prediction parquets row-align deterministically.
DATASETS = [d["name"] for d in config["datasets"]]
HF_REVISION = {d["name"]: d["hf_revision"] for d in config["datasets"]}
MODELS = config["models"]
SCORE_COLUMNS = config["score_columns"]
