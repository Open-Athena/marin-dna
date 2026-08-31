"""Exact-step Hugging Face exports for the issue #535 continuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from levanter.adaptor import AdaptorConfig, AdaptorExportConfig, NoAdaptorConfig
from levanter.compat.hf_checkpoints import HFCheckpointConverter, save_hf_checkpoint_callback
from rigging.filesystem.storage_path import prefix_join


@AdaptorConfig.register_subclass("exact-hf")
@dataclass(frozen=True)
class ExactHfExportConfig(NoAdaptorConfig):
    """Export HF checkpoints at exact completed-step labels after a resume."""

    steps: tuple[int, ...] = ()

    def install_export_hooks(
        self,
        *,
        trainer: Any,
        converter: HFCheckpointConverter | None,
        tokenizer: Any,
        export: AdaptorExportConfig,
    ) -> None:
        del tokenizer
        if export.hf_save_path is None:
            raise ValueError("exact HF exports require hf_save_path")
        if converter is None:
            raise ValueError("exact HF exports require an HF-compatible model")
        base_path = export.hf_save_path
        if trainer.config.checkpointer.append_run_id_to_base_path:
            base_path = prefix_join(base_path, trainer.run_id)
        callback = save_hf_checkpoint_callback(
            base_path,
            converter,
            upload_to_hf=False,
            save_dtype=None,
            generation_config=export.generation_config,
        )
        targets = frozenset(self.steps)
        saved: set[int] = set()

        def save_exact_step(info: Any) -> None:
            step = int(info.step)
            if step in targets and step not in saved:
                callback(info)
                saved.add(step)

        trainer.add_hook(save_exact_step, every=1)
