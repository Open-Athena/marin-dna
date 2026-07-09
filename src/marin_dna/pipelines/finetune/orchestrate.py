"""Fold orchestration: dev-fold multi-seed runs and full nested-LOCO (#369).

A fresh model is (re)built per seed / per fold via ``build_fn`` — ``run_fold`` mutates the
weights, so seeds and folds must not share a model instance. ``make_logger(name)`` returns
a fresh per-fold wandb logger (or ``None``); the run is finished after each fold.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from marin_dna.pipelines.evals.metrics import per_chrom_weighted_ap
from marin_dna.pipelines.finetune.data import (
    VariantWindows,
    chrom_fold_masks,
    nested_loco_folds,
)
from marin_dna.pipelines.finetune.model import SiameseLoRAClassifier
from marin_dna.pipelines.finetune.train import FoldResult, TrainConfig, run_fold

BuildFn = Callable[[], SiameseLoRAClassifier]
MakeLogger = Callable[[str], object] | None


def _finish(logger) -> None:
    if logger is not None:
        import wandb

        wandb.finish()


def run_dev_fold(
    build_fn: BuildFn,
    windows: VariantWindows,
    cfg: TrainConfig,
    *,
    test_chrom: str = "1",
    val_chrom: str = "3",
    seeds: tuple[int, ...] = (0, 1, 2),
    make_logger: MakeLogger = None,
) -> list[FoldResult]:
    """Leave-out-``test_chrom`` dev fold, one run (+ wandb run) per seed."""
    masks = chrom_fold_masks(windows.chrom, test_chrom, val_chrom)
    results: list[FoldResult] = []
    for seed in seeds:
        logger = make_logger(f"test{test_chrom}-s{seed}") if make_logger else None
        results.append(
            run_fold(build_fn(), windows, masks, cfg, seed=seed,
                     test_chrom=test_chrom, val_chrom=val_chrom, wandb_logger=logger)
        )
        _finish(logger)
    return results


def run_nested_loco(
    build_fn: BuildFn,
    windows: VariantWindows,
    cfg: TrainConfig,
    *,
    seed: int = 0,
    make_logger: MakeLogger = None,
) -> tuple[np.ndarray, list[FoldResult], float]:
    """Full nested leave-one-chromosome-out; returns (OOF logits, fold results, AUPRC)."""
    oof = np.full(len(windows), np.nan, dtype=float)
    fold_results: list[FoldResult] = []
    for test_chrom, val_chrom, masks in nested_loco_folds(windows.chrom):
        logger = make_logger(f"test{test_chrom}-s{seed}") if make_logger else None
        res = run_fold(build_fn(), windows, masks, cfg, seed=seed,
                       test_chrom=test_chrom, val_chrom=val_chrom, wandb_logger=logger)
        _finish(logger)
        assert res.best_test_scores is not None and res.test_idx is not None
        oof[res.test_idx] = res.best_test_scores
        fold_results.append(res)
    assert not np.isnan(oof).any(), "some variant was never in a test fold"
    overall = per_chrom_weighted_ap(windows.label, oof, windows.chrom)
    return oof, fold_results, overall
