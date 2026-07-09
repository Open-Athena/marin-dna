"""Fold orchestration: dev-fold multi-seed runs and full nested-LOCO (#369).

A fresh model is (re)built per seed / per fold via ``build_fn`` — ``train_fold`` mutates
the weights, so seeds and folds must not share a model instance.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from marin_dna.pipelines.evals.metrics import per_chrom_weighted_ap
from marin_dna.pipelines.finetune.data import (
    VariantWindows,
    chrom_fold_masks,
    nested_loco_folds,
)
from marin_dna.pipelines.finetune.model import SiameseLoRAClassifier
from marin_dna.pipelines.finetune.train import FoldResult, TrainConfig, train_fold

BuildFn = Callable[[], SiameseLoRAClassifier]


def _free(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_dev_fold(
    build_fn: BuildFn,
    windows: VariantWindows,
    cfg: TrainConfig,
    device: torch.device,
    *,
    test_chrom: str = "1",
    val_chrom: str = "3",
    seeds: tuple[int, ...] = (0, 1, 2),
) -> list[FoldResult]:
    """Leave-out-``test_chrom`` dev fold, one run per seed (val_chrom drives early stop)."""
    masks = chrom_fold_masks(windows.chrom, test_chrom, val_chrom)
    results: list[FoldResult] = []
    for seed in seeds:
        clf = build_fn().to(device)
        results.append(
            train_fold(
                clf, windows, masks, cfg, device, seed=seed,
                test_chrom=test_chrom, val_chrom=val_chrom,
            )
        )
        del clf
        _free(device)
    return results


def run_nested_loco(
    build_fn: BuildFn,
    windows: VariantWindows,
    cfg: TrainConfig,
    device: torch.device,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, list[FoldResult], float]:
    """Full nested leave-one-chromosome-out; returns (OOF logits, fold results, AUPRC).

    Each fold's val-selected test scores are written into the OOF vector; the single
    per-chromosome-weighted AUPRC over the assembled OOF is the honest headline number.
    """
    oof = np.full(len(windows), np.nan, dtype=float)
    fold_results: list[FoldResult] = []
    for test_chrom, val_chrom, masks in nested_loco_folds(windows.chrom):
        clf = build_fn().to(device)
        res = train_fold(
            clf, windows, masks, cfg, device, seed=seed,
            test_chrom=test_chrom, val_chrom=val_chrom,
        )
        assert res.best_test_scores is not None and res.test_idx is not None
        oof[res.test_idx] = res.best_test_scores
        fold_results.append(res)
        del clf
        _free(device)
    assert not np.isnan(oof).any(), "some variant was never in a test fold"
    overall = per_chrom_weighted_ap(windows.label, oof, windows.chrom)
    return oof, fold_results, overall
