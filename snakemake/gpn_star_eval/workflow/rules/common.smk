"""Common imports + helpers for gpn_star_eval rules."""

import pandas as pd
from datasets import load_dataset

from marin_dna.pipelines.evals.conservation import REQUIRED_VARIANT_COLUMNS
from marin_dna.pipelines.evals.gpn_star import predictions_url, score_variants_gpn_star
from marin_dna.pipelines.evals.metrics import compute_pairwise_metrics

DATASETS = config["datasets"]
MODELS = config["models"]
SCORE_COLUMNS = config["score_columns"]
