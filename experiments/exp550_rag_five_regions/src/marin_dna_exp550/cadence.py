"""Align this experiment's saves and LM evaluations with completed updates.

Upstream metrics retain their zero-based convention. Only the three milestone
callbacks receive the completed-update index; the resumable state is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from haliax import Axis
from levanter.adaptor import AdaptorConfig, AdaptorExportConfig, NoAdaptorConfig
from levanter.data.dataset import AsyncDataset
from levanter.data.loader import DataLoader
from levanter.trainer import StepInfo, TrainerHooks

MILESTONE_CALLBACKS = {
    ("levanter.trainer", "Trainer._add_default_hooks.<locals>.checkpoint_hook"),
    ("levanter.compat.hf_checkpoints", "save_hf_checkpoint_callback.<locals>.cb"),
    ("levanter.eval", "cb_tagged_lm_evaluate.<locals>.eval_callback"),
}


def bound_training_loader(trainer: Any) -> None:
    """Bound lookahead without changing the original loader's batch or shuffle."""
    original = trainer.data_loader

    def create(
        dataset: AsyncDataset[Any], batch: Axis | int | None = None
    ) -> DataLoader[Any]:
        loader = original(dataset, batch)
        loader.max_buffered_batches = 8
        loader.fetch_batch_size = 8
        return loader

    trainer.data_loader = create


def callback_identity(callback: Any) -> tuple[str, str]:
    function = getattr(callback, "fn", callback)
    return getattr(function, "__module__", ""), getattr(function, "__qualname__", "")


class CompletedStepInfo(StepInfo):
    @property
    def step(self) -> int:
        return self.next_step


class CompletedUpdateHooks(TrainerHooks):
    @classmethod
    def from_existing(cls, existing: TrainerHooks) -> CompletedUpdateHooks:
        result = cls()
        identities = [callback_identity(hook.fn) for hook in existing.hooks]
        if any(identities.count(identity) != 1 for identity in MILESTONE_CALLBACKS):
            raise ValueError(
                "pinned native, HF, and LM-evaluation callbacks changed or are missing"
            )
        result.hooks = existing.hooks
        result.jit_hooks = existing.jit_hooks
        return result

    def run_hooks(self, info: StepInfo, force: bool = False) -> None:
        completed = info.next_step
        milestone_info = CompletedStepInfo(
            info.state, info.loss, info.step_duration, info._event_handler
        )
        for hook in self.hooks:
            identity = callback_identity(hook.fn)
            if identity in MILESTONE_CALLBACKS:
                if completed == 0 and identity[0] != "levanter.eval":
                    continue
                if force or (completed > 0 and completed % hook.every == 0):
                    hook.fn.on_step(milestone_info, force=force)
            elif force or (info.step > 1 and info.step % hook.every == 0):
                hook.fn.on_step(info, force=force)


@AdaptorConfig.register_subclass("exp550_completed_updates")
@dataclass(frozen=True)
class CompletedUpdateAdaptor(NoAdaptorConfig):
    def install_export_hooks(
        self,
        *,
        trainer: Any,
        converter: Any,
        tokenizer: Any,
        export: AdaptorExportConfig,
    ) -> None:
        super().install_export_hooks(
            trainer=trainer, converter=converter, tokenizer=tokenizer, export=export
        )
        trainer.hooks = CompletedUpdateHooks.from_existing(trainer.hooks)
        # Upstream constructs the training loader after installing export hooks.
        bound_training_loader(trainer)
