from __future__ import annotations

import copy
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import Qwen3Config, Qwen3ForCausalLM

from exp479_mntp.module import AdaptationModule


class BatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.rows: list[dict[str, torch.Tensor]] = []
        for sample_id in range(8):
            input_ids = torch.tensor([2, 3, 4, 7, 5, 6], dtype=torch.long)
            input_ids[-1] = 3 + sample_id % 4
            labels = torch.full_like(input_ids, -100)
            labels[2] = 5
            weights = torch.zeros_like(input_ids, dtype=torch.float32)
            weights[2] = 1.0
            self.rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": labels,
                    "loss_weights": weights,
                    "sample_ids": torch.tensor(sample_id),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


def _model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=16,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config)


def _trainer(root: Path, max_steps: int) -> L.Trainer:
    return L.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_steps=max_steps,
        max_epochs=-1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        limit_val_batches=0,
        default_root_dir=root,
    )


def _module(model: Qwen3ForCausalLM) -> AdaptationModule:
    return AdaptationModule(
        model=model,
        arm="scratch_mntp",
        batch_size=2,
        train_steps=4,
        warmup_steps=1,
        cooldown_start_step=3,
    )


def _state(module: AdaptationModule) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def test_interrupted_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    torch.manual_seed(479)
    initial = _model()
    uninterrupted = _module(copy.deepcopy(initial))
    loader = StatefulDataLoader(BatchDataset(), batch_size=2, shuffle=False)
    _trainer(tmp_path / "uninterrupted", max_steps=4).fit(uninterrupted, train_dataloaders=loader)

    interrupted = _module(copy.deepcopy(initial))
    first_trainer = _trainer(tmp_path / "interrupted", max_steps=2)
    first_trainer.fit(interrupted, train_dataloaders=loader)
    checkpoint = tmp_path / "step-2.ckpt"
    first_trainer.save_checkpoint(checkpoint)

    resumed = _module(copy.deepcopy(initial))
    _trainer(tmp_path / "resumed", max_steps=4).fit(
        resumed,
        train_dataloaders=loader,
        ckpt_path=checkpoint,
    )

    for name, expected in _state(uninterrupted).items():
        torch.testing.assert_close(_state(resumed)[name], expected, rtol=0, atol=0)
