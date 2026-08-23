"""Tests for metrics and transformations."""

import numpy as np
import pytest
import torch

from glm_experiments.utils.metrics import (
    METRICS,
    TRANSFORMS,
    get_metric,
    get_transform,
    metric_auprc,
    metric_auroc,
    metric_pearson,
    metric_spearman,
    transform_abs,
    transform_identity,
    transform_minus,
)


class TestTransforms:
    """Test transform functions."""

    def test_transform_minus(self):
        """Test negation transform."""
        scores = torch.tensor([1.0, -2.0, 3.0])
        result = transform_minus(scores)
        expected = torch.tensor([-1.0, 2.0, -3.0])
        assert torch.allclose(result, expected)

    def test_transform_identity(self):
        """Test identity transform."""
        scores = torch.tensor([1.0, -2.0, 3.0])
        result = transform_identity(scores)
        assert torch.allclose(result, scores)

    def test_transform_abs(self):
        """Test absolute value transform."""
        scores = torch.tensor([1.0, -2.0, 3.0])
        result = transform_abs(scores)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(result, expected)

    def test_get_transform_valid(self):
        """Test getting valid transform functions."""
        assert get_transform("minus") == transform_minus
        assert get_transform("identity") == transform_identity
        assert get_transform("abs") == transform_abs

    def test_get_transform_invalid(self):
        """Test getting invalid transform raises ValueError."""
        with pytest.raises(ValueError, match="Unknown transform: 'invalid'"):
            get_transform("invalid")

    def test_transforms_registry_complete(self):
        """Test that TRANSFORMS registry contains all expected transforms."""
        expected_transforms = {"minus", "identity", "abs"}
        assert set(TRANSFORMS.keys()) == expected_transforms


class TestMetrics:
    """Test metric functions."""

    def test_metric_auprc(self):
        """Test AUPRC metric."""
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        result = metric_auprc(labels, scores)
        assert 0.0 <= result <= 1.0
        # Perfect separation should give high AUPRC
        perfect_scores = np.array([0.0, 0.0, 1.0, 1.0])
        perfect_result = metric_auprc(labels, perfect_scores)
        assert perfect_result > 0.9

    def test_metric_auroc(self):
        """Test AUROC metric."""
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        result = metric_auroc(labels, scores)
        assert 0.0 <= result <= 1.0
        # Perfect separation should give AUROC = 1.0
        perfect_scores = np.array([0.0, 0.0, 1.0, 1.0])
        perfect_result = metric_auroc(labels, perfect_scores)
        assert perfect_result == 1.0

    def test_metric_spearman(self):
        """Test Spearman correlation metric."""
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scores = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        result = metric_spearman(labels, scores)
        assert -1.0 <= result <= 1.0
        # Perfect monotonic relationship should give correlation ≈ 1.0
        perfect_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        perfect_result = metric_spearman(labels, perfect_scores)
        assert np.isclose(perfect_result, 1.0)

    def test_metric_pearson(self):
        """Test Pearson correlation metric."""
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scores = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        result = metric_pearson(labels, scores)
        assert -1.0 <= result <= 1.0
        # Perfect linear relationship should give correlation = 1.0
        perfect_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        perfect_result = metric_pearson(labels, perfect_scores)
        assert np.isclose(perfect_result, 1.0)

    def test_get_metric_valid(self):
        """Test getting valid metric functions."""
        assert get_metric("auprc") == metric_auprc
        assert get_metric("auroc") == metric_auroc
        assert get_metric("spearman") == metric_spearman
        assert get_metric("pearson") == metric_pearson

    def test_get_metric_invalid(self):
        """Test getting invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric: 'invalid'"):
            get_metric("invalid")

    def test_metrics_registry_complete(self):
        """Test that METRICS registry contains all expected metrics."""
        expected_metrics = {"auprc", "auroc", "spearman", "pearson"}
        assert set(METRICS.keys()) == expected_metrics
