"""Lightning DataModule backed by immutable exp479 sequence plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from exp479_mntp.data import Objective, SequenceCollator, SequencePlanDataset


class ExperimentDataModule(L.LightningDataModule):
    """Serve one training plan and two deterministic validation corruptions."""

    def __init__(
        self,
        *,
        train_plan: Path,
        validation_plan: Path,
        tokenizer: PreTrainedTokenizerBase,
        objective: Objective,
        canonical_token_ids: tuple[int, ...],
        mask_token_id: int | None,
        batch_size: int,
        seed: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.train_plan_path = train_plan
        self.validation_plan_path = validation_plan
        self.tokenizer = tokenizer
        self.objective = objective
        self.canonical_token_ids = canonical_token_ids
        self.mask_token_id = mask_token_id
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.train_dataset: SequencePlanDataset | None = None
        self.validation_dataset: SequencePlanDataset | None = None
        self._restored_state: dict[str, Any] | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        self.train_dataset = SequencePlanDataset(self.train_plan_path)
        self.validation_dataset = SequencePlanDataset(self.validation_plan_path)
        if self._restored_state is not None:
            if self._restored_state["train_plan_sha256"] != self.train_dataset.sha256:
                raise RuntimeError("training plan changed since the checkpoint was written")
            if self._restored_state["validation_plan_sha256"] != self.validation_dataset.sha256:
                raise RuntimeError("validation plan changed since the checkpoint was written")

    def _loader(
        self,
        dataset: SequencePlanDataset,
        *,
        validation_mode: str,
        drop_last: bool,
    ) -> StatefulDataLoader[dict[str, Any]]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        return StatefulDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            generator=generator,
            collate_fn=SequenceCollator(
                tokenizer=self.tokenizer,
                objective=self.objective,
                canonical_token_ids=self.canonical_token_ids,
                mask_token_id=self.mask_token_id,
                seed=self.seed,
                validation_mode=validation_mode,  # type: ignore[arg-type]
            ),
        )

    def train_dataloader(self) -> StatefulDataLoader[dict[str, Any]]:
        if self.train_dataset is None:
            raise RuntimeError("setup() must run before train_dataloader()")
        return self._loader(self.train_dataset, validation_mode="diffusion", drop_last=True)

    def val_dataloader(self) -> list[StatefulDataLoader[dict[str, Any]]]:
        if self.validation_dataset is None:
            raise RuntimeError("setup() must run before val_dataloader()")
        if self.objective == "clm":
            return [
                self._loader(self.validation_dataset, validation_mode="diffusion", drop_last=False)
            ]
        return [
            self._loader(self.validation_dataset, validation_mode="diffusion", drop_last=False),
            self._loader(self.validation_dataset, validation_mode="single", drop_last=False),
        ]

    def state_dict(self) -> dict[str, Any]:
        if self.train_dataset is None or self.validation_dataset is None:
            return {}
        return {
            "train_plan_sha256": self.train_dataset.sha256,
            "validation_plan_sha256": self.validation_dataset.sha256,
            "next_sequence_plan_batch": int(self.trainer.global_step) if self.trainer else 0,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._restored_state = dict(state_dict)
