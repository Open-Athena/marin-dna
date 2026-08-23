"""Metrics and transformations for variant effect prediction."""

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

# ============================================================================
# TRANSFORM FUNCTIONS
# ============================================================================


def transform_minus(scores: torch.Tensor) -> torch.Tensor:
    """Negate scores (higher LLR = more likely → higher score = more pathogenic).

    Args:
        scores: Raw LLR scores from model

    Returns:
        Negated scores
    """
    return -1 * scores


def transform_identity(scores: torch.Tensor) -> torch.Tensor:
    """No transformation (use raw scores).

    Args:
        scores: Raw LLR scores from model

    Returns:
        Unchanged scores
    """
    return scores


def transform_abs(scores: torch.Tensor) -> torch.Tensor:
    """Absolute value (magnitude of effect).

    Args:
        scores: Raw LLR scores from model

    Returns:
        Absolute value of scores
    """
    return torch.abs(scores)


# Registry: string name -> function
TRANSFORMS = {
    "minus": transform_minus,
    "identity": transform_identity,
    "abs": transform_abs,
}


def get_transform(name: str):
    """Get transform function by name.

    Args:
        name: Transform name (e.g., "minus", "identity", "abs")

    Returns:
        Transform function

    Raises:
        ValueError: If transform name not found
    """
    if name not in TRANSFORMS:
        available = ", ".join(sorted(TRANSFORMS.keys()))
        raise ValueError(f"Unknown transform: '{name}'. Available: {available}")
    return TRANSFORMS[name]


# ============================================================================
# METRIC FUNCTIONS
# ============================================================================


def metric_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUPRC (Average Precision).

    Args:
        labels: True binary labels
        scores: Predicted scores

    Returns:
        AUPRC value
    """
    return average_precision_score(labels, scores)


def metric_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC (ROC AUC).

    Args:
        labels: True binary labels
        scores: Predicted scores

    Returns:
        AUROC value
    """
    return roc_auc_score(labels, scores)


def metric_spearman(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Spearman correlation.

    Args:
        labels: True continuous values
        scores: Predicted scores

    Returns:
        Spearman correlation coefficient
    """
    corr, _ = spearmanr(scores, labels)
    return corr


def metric_pearson(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Pearson correlation.

    Args:
        labels: True continuous values
        scores: Predicted scores

    Returns:
        Pearson correlation coefficient
    """
    corr, _ = pearsonr(scores, labels)
    return corr


# Registry: string name -> function
METRICS = {
    "auprc": metric_auprc,
    "auroc": metric_auroc,
    "spearman": metric_spearman,
    "pearson": metric_pearson,
}


def get_metric(name: str):
    """Get metric function by name.

    Args:
        name: Metric name (e.g., "auprc", "auroc", "spearman", "pearson")

    Returns:
        Metric function

    Raises:
        ValueError: If metric name not found
    """
    if name not in METRICS:
        available = ", ".join(sorted(METRICS.keys()))
        raise ValueError(f"Unknown metric: '{name}'. Available: {available}")
    return METRICS[name]
