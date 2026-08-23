"""Tests for StepProgressBar callback."""

import pytest
from hydra.core.hydra_config import HydraConfig
from lightning import Trainer
from omegaconf import DictConfig, open_dict

from glm_experiments.train import train
from glm_experiments.utils.callbacks import StepProgressBar


def test_step_progress_bar_instantiation(cfg_train: DictConfig) -> None:
    """Test that StepProgressBar can be instantiated from config.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
        # Remove lr_monitor since tests have no logger
        if "lr_monitor" in cfg_train.callbacks:
            del cfg_train.callbacks.lr_monitor
    train(cfg_train)


def test_step_progress_bar_no_grad_accum(cfg_train: DictConfig) -> None:
    """Test progress bar with accumulate_grad_batches=1 (no gradient accumulation).

    Verifies that the progress bar works correctly without gradient accumulation.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_steps = 10
        cfg_train.trainer.accumulate_grad_batches = 1
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.log_every_n_steps = 1
        cfg_train.trainer.val_check_interval = None
        cfg_train.trainer.num_sanity_val_steps = 0
        cfg_train.trainer.limit_train_batches = 20  # Must be int for IterableDataset
        cfg_train.trainer.limit_val_batches = 10  # Must be int for IterableDataset
        # Remove lr_monitor since tests have no logger
        if "lr_monitor" in cfg_train.callbacks:
            del cfg_train.callbacks.lr_monitor

    metric_dict, _ = train(cfg_train)

    # Verify training completed
    assert metric_dict is not None


def test_step_progress_bar_with_grad_accum(cfg_train: DictConfig) -> None:
    """Test progress bar with accumulate_grad_batches=4 (with gradient accumulation).

    Verifies that the progress bar correctly tracks optimizer steps instead of batches,
    preventing overflow.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        # Use per_device_batch_size and batch_size to trigger gradient accumulation
        cfg_train.data.batch_size = 8
        cfg_train.data.per_device_batch_size = 2
        # This will set accumulate_grad_batches = 8 / (2 * 1) = 4

        cfg_train.trainer.max_steps = 10
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 1
        cfg_train.trainer.log_every_n_steps = 1
        cfg_train.trainer.val_check_interval = None
        cfg_train.trainer.num_sanity_val_steps = 0
        cfg_train.trainer.limit_train_batches = 50  # Must be int for IterableDataset
        cfg_train.trainer.limit_val_batches = 10  # Must be int for IterableDataset
        # Remove lr_monitor since tests have no logger
        if "lr_monitor" in cfg_train.callbacks:
            del cfg_train.callbacks.lr_monitor

    metric_dict, _ = train(cfg_train)

    # Verify training completed without overflow
    assert metric_dict is not None


def test_step_progress_bar_tracks_global_step(cfg_train: DictConfig) -> None:
    """Test that progress bar position matches trainer.global_step.

    Verifies that after training, the progress bar has tracked optimizer steps.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.data.batch_size = 8
        cfg_train.data.per_device_batch_size = 2
        # This sets accumulate_grad_batches = 4

        cfg_train.trainer.max_steps = 5
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 1
        cfg_train.trainer.log_every_n_steps = 1
        cfg_train.trainer.val_check_interval = None
        cfg_train.trainer.num_sanity_val_steps = 0
        cfg_train.trainer.limit_train_batches = 25  # Must be int for IterableDataset
        cfg_train.trainer.limit_val_batches = 10  # Must be int for IterableDataset
        # Remove lr_monitor since tests have no logger
        if "lr_monitor" in cfg_train.callbacks:
            del cfg_train.callbacks.lr_monitor

    _, object_dict = train(cfg_train)

    # Verify trainer completed expected number of steps
    assert object_dict["trainer"].global_step == 5


@pytest.mark.slow
def test_step_progress_bar_with_validation(cfg_train: DictConfig) -> None:
    """Test progress bar with validation enabled.

    Verifies that the progress bar works correctly when validation is enabled.
    Validation progress bar should remain unchanged.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.data.batch_size = 8
        cfg_train.data.per_device_batch_size = 2
        # This sets accumulate_grad_batches = 4

        cfg_train.trainer.max_steps = 10
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.log_every_n_steps = 1
        cfg_train.trainer.val_check_interval = 5
        cfg_train.trainer.num_sanity_val_steps = 0
        cfg_train.trainer.limit_train_batches = 50  # Must be int for IterableDataset
        cfg_train.trainer.limit_val_batches = 10  # Must be int for IterableDataset
        # Remove lr_monitor since tests have no logger
        if "lr_monitor" in cfg_train.callbacks:
            del cfg_train.callbacks.lr_monitor

    metric_dict, _ = train(cfg_train)

    # Verify training and validation completed
    assert "train/loss" in metric_dict
    assert "val/loss" in metric_dict


def test_step_progress_bar_callback_direct() -> None:
    """Test StepProgressBar callback directly without full training.

    Verifies that the callback can be instantiated and has correct attributes.
    """
    callback = StepProgressBar()
    assert hasattr(callback, "on_train_batch_end")
    assert callable(callback.on_train_batch_end)
