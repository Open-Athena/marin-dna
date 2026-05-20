"""AlphaGenome convolutional-probe enhancer classifier."""

from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from alphagenome_pytorch.model import AlphaGenome, SequenceEncoder
from transformers import get_cosine_schedule_with_warmup

ENCODER_OUTPUT_DIM = 1536


def load_pretrained_encoder(weights_path: str | Path) -> SequenceEncoder:
    """Load encoder weights from a full AlphaGenome checkpoint.

    Creates a full AlphaGenome model, extracts the encoder state_dict,
    loads it into a fresh SequenceEncoder, then discards the rest.
    """
    full_model = AlphaGenome.from_pretrained(weights_path, device="cpu")
    encoder_state = full_model.encoder.state_dict()
    del full_model

    encoder = SequenceEncoder()
    encoder.load_state_dict(encoder_state)
    return encoder


class EnhancerClassifier(L.LightningModule):
    """Binary enhancer classifier using AlphaGenome's CNN encoder trunk.

    Architecture: SequenceEncoder → AdaptiveAvgPool1d(1) → LayerNorm → Linear(1536, 1)
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.1,
        freeze_backbone: bool = True,
        warmup_fraction: float = 0.1,
        num_training_steps: int | None = None,
        mlp_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["weights_path"])

        if weights_path is not None:
            self.encoder = load_pretrained_encoder(weights_path)
        else:
            self.encoder = SequenceEncoder()

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if mlp_hidden_dim > 0:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.LayerNorm(ENCODER_OUTPUT_DIM),
                nn.Linear(ENCODER_OUTPUT_DIM, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden_dim, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.LayerNorm(ENCODER_OUTPUT_DIM),
                nn.Linear(ENCODER_OUTPUT_DIM, 1),
            )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.val_auroc = torchmetrics.AUROC(task="binary")
        self.val_auprc = torchmetrics.AveragePrecision(task="binary")
        self.val_logits = torchmetrics.CatMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk, _intermediates = self.encoder(x)
        logits = self.head(trunk).squeeze(-1)
        return logits

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        optimizer = self.optimizers()
        assert not isinstance(optimizer, list)
        self.log("lr", optimizer.param_groups[0]["lr"], prog_bar=True)
        return loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        self.log("grad_norm", grad_norm, on_step=True, prog_bar=False)

    def on_validation_epoch_start(self) -> None:
        self.val_logits.reset()

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        self.val_auroc.update(preds, y.int())
        self.val_auprc.update(preds, y.int())
        self.val_logits.update(logits)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True, sync_dist=True)  # type: ignore[func-returns-value]
        self.log("val_auprc", self.val_auprc.compute(), prog_bar=True, sync_dist=True)  # type: ignore[func-returns-value]
        self.val_auroc.reset()
        self.val_auprc.reset()

    def configure_optimizers(self) -> dict:  # type: ignore[override]
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
        num_steps = self.hparams["num_training_steps"]
        if num_steps is None:
            raise ValueError("num_training_steps must be set for LR scheduling")
        num_warmup = int(self.hparams["warmup_fraction"] * num_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=num_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
