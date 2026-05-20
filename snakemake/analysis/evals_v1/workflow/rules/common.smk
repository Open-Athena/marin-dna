"""Common imports, constants, and helper functions for all rules."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score

from marin_dna.pipelines.evals.inference import compute_variant_scores
from marin_dna.pipelines.evals.metrics import aggregate_metrics, compute_metrics
from marin_dna.pipelines.evals.plotting import plot_metrics_vs_step, plot_models_comparison

COORDINATES = ["chrom", "pos", "ref", "alt"]


def get_dataset_config(dataset_name):
    for dataset in config["datasets"]:
        if dataset["name"] == dataset_name:
            return dataset
    raise ValueError(f"Dataset {dataset_name} not found in config")


def get_model_config(model_name):
    for model in config["models"]:
        if model["name"] == model_name:
            return model
    raise ValueError(f"Model {model_name} not found in config")


def get_original_dataset_path(dataset_name):
    return config["baselines"]["dataset_mapping"][dataset_name]


def load_baseline_data(
    plot_config: dict,
) -> dict[tuple[str, str], dict[str, float]]:
    """Load baseline metrics and structure them for plotting.

    Args:
        plot_config: Custom plot configuration containing dataset_subsets.

    Returns:
        Dictionary mapping (dataset, subset) to {baseline_display_name: metric_value}.
    """
    baselines_config = config.get("baselines", {})
    dataset_mapping = baselines_config.get("dataset_mapping", {})
    all_baselines = baselines_config.get("models", [])

    result: dict[tuple[str, str], dict[str, float]] = {}

    for ds in plot_config.get("dataset_subsets", []):
        include_baselines = ds.get("include_baselines", False)
        if not include_baselines:
            continue

        dataset = ds["dataset"]
        subset = ds["subset"]

        if dataset not in dataset_mapping:
            continue

        if isinstance(include_baselines, list):
            baselines_to_include = include_baselines
        else:
            baselines_to_include = all_baselines

        key = (dataset, subset)
        result[key] = {}

        for baseline in baselines_to_include:
            model_id = get_baseline_model_id(baseline)
            display_name = get_baseline_display_name(baseline)
            filepath = f"results/baselines/metrics/{dataset}/{model_id}.parquet"
            try:
                df = pd.read_parquet(filepath)
                subset_data = df[df["subset"] == subset]
                if not subset_data.empty:
                    result[key][display_name] = subset_data["value"].iloc[0]
            except FileNotFoundError:
                pass

    return result
