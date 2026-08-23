"""Custom progress bar that tracks optimizer steps instead of batches.

This progress bar callback correctly tracks optimizer steps when using gradient
accumulation, preventing the progress bar from showing batch_idx / total_steps
which causes overflow (e.g., "40000/10000" when accumulate_grad_batches=4).

Instead, it shows step_idx / total_steps consistently (e.g., "10000/10000").
"""

from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import TQDMProgressBar


class StepProgressBar(TQDMProgressBar):
    """Progress bar that tracks optimizer steps instead of batches for training.

    When gradient accumulation is used, the standard progress bar shows batch indices
    which can exceed the total steps, causing overflow. This callback ensures the
    progress bar correctly tracks optimizer steps.

    Only modifies training progress bar. Validation, testing, and prediction progress
    bars remain unchanged.
    """

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update progress bar after training batch.

        Updates progress bar position to match optimizer steps (trainer.global_step)
        instead of batch index. Only updates after optimizer steps when using gradient
        accumulation.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Current LightningModule.
            outputs: Training step outputs.
            batch: Current batch.
            batch_idx: Current batch index.
        """
        # Call parent to handle metrics update
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        # Manually set progress bar position to global_step
        # This ensures the bar shows step_idx / total_steps instead of batch_idx / total_steps
        if self.train_progress_bar is not None:
            self.train_progress_bar.n = trainer.global_step
            self.train_progress_bar.refresh()
