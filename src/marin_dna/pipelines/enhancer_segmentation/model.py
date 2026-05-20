"""AlphaGenome per-bin enhancer segmenter."""

import logging
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from alphagenome_pytorch.model import AlphaGenome, SequenceEncoder, TransformerTower
from transformers import get_cosine_schedule_with_warmup

from marin_dna.pipelines.enhancer_classification.model import (
    ENCODER_OUTPUT_DIM,
    load_pretrained_encoder,
)

log = logging.getLogger(__name__)


def _load_pretrained_encoder_tower_org(
    weights_path: str | Path, n_transformer_layers: int
) -> tuple[SequenceEncoder, TransformerTower, torch.Tensor]:
    """Load encoder + tower + human-row organism embedding from an AlphaGenome
    checkpoint. Returns ``(encoder, tower, organism_embed_row)`` where tower's
    blocks are truncated to the first ``n_transformer_layers`` and the
    organism row is the pretrained human (index 0) embedding.
    """
    full_model = AlphaGenome.from_pretrained(weights_path, device="cpu")

    encoder = SequenceEncoder()
    encoder.load_state_dict(full_model.encoder.state_dict())

    tower = TransformerTower(d_model=ENCODER_OUTPUT_DIM)
    tower.load_state_dict(full_model.tower.state_dict())
    if n_transformer_layers < len(tower.blocks):
        tower.blocks = tower.blocks[:n_transformer_layers]

    # Row 0 = human in the pretrained 2-row embedding. We deliberately drop
    # the mouse row: the segmenter has a single learnable "species" vector
    # hard-coded to human, which still fine-tunes during training but has no
    # branching on genome at inference time (future species default to human).
    organism_row = full_model.organism_embed.weight[0].detach().clone()

    del full_model
    return encoder, tower, organism_row


class _SingleSpeciesTower(nn.Module):
    """Truncated transformer tower fronted by a single learnable 1536-vector
    species embedding, hard-coded to human (index 0 of the pretrained model).

    See issue #115 discussion: the AlphaGenome pretrained tower was trained
    with an organism embedding added to the trunk before attention; feeding
    it raw encoder output without that embedding is out-of-distribution. We
    preserve the pretrained regime for human but intentionally collapse the
    2-row organism embedding to 1 learnable row so the final model has no
    species-specific bias at inference — new genomes just use index 0 and
    the embedding fine-tunes during training.
    """

    def __init__(
        self,
        tower: TransformerTower,
        init_embedding: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.tower = tower
        self.species_embed = nn.Parameter(
            init_embedding.clone()
            if init_embedding is not None
            else torch.zeros(ENCODER_OUTPUT_DIM)
        )

    def forward(self, trunk_ncl: torch.Tensor) -> torch.Tensor:
        # Encoder output is (B, C, S); tower expects NLC (B, S, C).
        x = trunk_ncl.transpose(1, 2) + self.species_embed
        x, _pair = self.tower(x)
        return x.transpose(1, 2)  # back to (B, C, S) for the Conv1d head


class EnhancerSegmenter(L.LightningModule):
    """Per-bin enhancer segmenter using AlphaGenome's CNN encoder trunk.

    Architecture: SequenceEncoder -> Conv1d(1536, 1, kernel_size=1) -> per-bin logits.
    The encoder downsamples 128x, so an input of shape (B, L, 4) yields
    (B, L/128) logits.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.1,
        freeze_backbone: bool = False,
        warmup_fraction: float = 0.1,
        num_training_steps: int | None = None,
        pos_weight: float = 1.0,
        genomes: list[str] | None = None,
        n_transformer_layers: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["weights_path"])
        self.genomes: list[str] = list(genomes) if genomes else []

        if weights_path is not None and n_transformer_layers > 0:
            encoder, tower, org_row = _load_pretrained_encoder_tower_org(
                weights_path, n_transformer_layers
            )
            self.encoder = encoder
            self.tower = _SingleSpeciesTower(tower, init_embedding=org_row)
            log.info(
                "Loaded pretrained AlphaGenome encoder + tower (first %d blocks)"
                " with single learnable species embedding (init: human) from %s",
                n_transformer_layers,
                weights_path,
            )
        elif weights_path is not None:
            self.encoder = load_pretrained_encoder(weights_path)
            self.tower = None
            log.info("Loaded pretrained AlphaGenome encoder from %s", weights_path)
        else:
            self.encoder = SequenceEncoder()
            if n_transformer_layers > 0:
                fresh_tower = TransformerTower(d_model=ENCODER_OUTPUT_DIM)
                fresh_tower.blocks = fresh_tower.blocks[:n_transformer_layers]
                self.tower = _SingleSpeciesTower(fresh_tower, init_embedding=None)
            else:
                self.tower = None
            log.info("No weights_path provided — encoder initialized from scratch")

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            if self.tower is not None:
                for param in self.tower.parameters():
                    param.requires_grad = False

        # GroupNorm over channels (equivalent to LayerNorm on each position's
        # 1536-vector) before the final projection. Matches the classifier's
        # LayerNorm-before-Linear pattern and stabilizes gradient norms
        # (pre-norm an earlier run sat at mean=35 / max=90 with clip=1.0).
        self.head = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=ENCODER_OUTPUT_DIM),
            nn.Conv1d(ENCODER_OUTPUT_DIM, 1, kernel_size=1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
        )
        # Overall AUPRC plus a per-species AUPRC metric per genome.
        self.val_auprc = torchmetrics.AveragePrecision(task="binary")
        self.val_auprc_per_species = nn.ModuleDict(
            {g: torchmetrics.AveragePrecision(task="binary") for g in self.genomes}
        )
        # Accumulate flat logits so train.py can write val_predictions.parquet.
        self.val_logits = torchmetrics.CatMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk, _intermediates = self.encoder(x)
        if self.tower is not None:
            trunk = self.tower(trunk)
        logits = self.head(trunk).squeeze(1)
        return logits

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y, _g = batch
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
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        x, y, g = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        flat_logits = logits.reshape(-1)
        flat_labels = y.reshape(-1).int()
        preds = torch.sigmoid(flat_logits)
        self.val_auprc.update(preds, flat_labels)
        self.val_logits.update(flat_logits)

        # Per-species AUPRC: each row of y belongs to genome g[row]; expand g
        # across bin positions to match the flattened logits/labels shape.
        num_bins = y.shape[1]
        g_flat = g.repeat_interleave(num_bins)
        for i, name in enumerate(self.genomes):
            mask = g_flat == i
            if mask.any():
                self.val_auprc_per_species[name].update(preds[mask], flat_labels[mask])  # type: ignore[operator]

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_auprc", self.val_auprc.compute(), prog_bar=True, sync_dist=True)  # type: ignore[func-returns-value]
        self.val_auprc.reset()
        for name, metric in self.val_auprc_per_species.items():
            # AveragePrecision.compute() raises if no samples were seen
            # (e.g. sparse genomes under limit_val_batches < 1.0).
            if metric.update_count == 0:  # type: ignore[operator]
                continue
            self.log(
                f"val_auprc/{name}",
                metric.compute(),  # type: ignore[operator]
                prog_bar=False,
                sync_dist=True,
            )
            metric.reset()  # type: ignore[operator]

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
