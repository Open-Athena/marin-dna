"""CLI training script for the enhancer classifier.

Usage::

    uv run python -m marin_dna.pipelines.enhancer_classification.train \
        --train-parquet data/train.parquet \
        --val-parquet data/validation.parquet \
        --weights-path alphagenome.pth \
        --output-ckpt results/model/default/v3/best.ckpt \
        --output-metrics results/model/default/v3/metrics.json
"""

import argparse
import json
import logging
from pathlib import Path

import lightning as L
import numpy as np
import polars as pl
import torch
import wandb
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from marin_dna.pipelines.enhancer_classification.dataset import EnhancerDataset
from marin_dna.pipelines.enhancer_classification.model import EnhancerClassifier

log = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")

WANDB_PROJECT = "bolinas-enhancer-classification"
WANDB_ENTITY = "gonzalobenegas"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train enhancer classifier")
    parser.add_argument("--train-parquet", type=str, required=True)
    parser.add_argument("--val-parquet", type=str, required=True)
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--output-ckpt", type=str, required=True)
    parser.add_argument("--output-metrics", type=str, required=True)
    parser.add_argument("--output-val-predictions", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument(
        "--freeze-backbone", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--mlp-hidden-dim", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-run", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_ckpt = Path(args.output_ckpt)
    output_metrics = Path(args.output_metrics)
    output_dir = output_ckpt.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(args.seed, workers=True)

    train_ds = EnhancerDataset(args.train_parquet)
    val_ds = EnhancerDataset(args.val_parquet)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    num_training_steps = len(train_loader)

    model = EnhancerClassifier(
        weights_path=args.weights_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        freeze_backbone=args.freeze_backbone,
        warmup_fraction=args.warmup_fraction,
        num_training_steps=num_training_steps,
        mlp_hidden_dim=args.mlp_hidden_dim,
    )

    model = torch.compile(model)  # type: ignore[assignment]

    loggers = [
        CSVLogger(output_dir, name="logs"),
        WandbLogger(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=args.wandb_run,
            save_dir=output_dir,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=1,
        precision="bf16-mixed",
        accelerator="gpu",
        devices=1,  # single-GPU only; multi-GPU not supported
        gradient_clip_val=args.gradient_clip_val,
        logger=loggers,
        default_root_dir=output_dir,
    )

    # Capture W&B run ID before training finalizes it
    wandb_logger = next((lg for lg in loggers if isinstance(lg, WandbLogger)), None)
    wandb_run_id: str | None = None
    if wandb_logger is not None:
        wandb_run_id = wandb_logger.experiment.id

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save checkpoint
    trainer.save_checkpoint(output_ckpt)

    # Collect metrics
    metrics = {
        "val_auroc": float(trainer.callback_metrics.get("val_auroc", 0.0)),
        "val_auprc": float(trainer.callback_metrics.get("val_auprc", 0.0)),
    }

    # Per-sample validation predictions and per-species metrics
    logits_array = model.val_logits.compute().cpu().numpy()
    val_meta = pl.read_parquet(
        args.val_parquet,
        columns=["genome", "chrom", "start", "end", "strand", "label"],
    )
    assert len(logits_array) == len(val_meta), (
        f"Logit count ({len(logits_array)}) != validation rows ({len(val_meta)})"
    )
    val_meta = val_meta.with_columns(pl.Series("logit", logits_array))

    if args.output_val_predictions:
        Path(args.output_val_predictions).parent.mkdir(parents=True, exist_ok=True)
        val_meta.write_parquet(args.output_val_predictions)

    # Per-species metrics
    per_species: dict[str, float] = {}
    for genome in val_meta["genome"].unique().sort().to_list():
        subset = val_meta.filter(pl.col("genome") == genome)
        labels = subset["label"].to_numpy()
        probs = expit(subset["logit"].to_numpy())
        if len(np.unique(labels)) == 2:
            per_species[f"val_auroc/{genome}"] = float(roc_auc_score(labels, probs))
            per_species[f"val_auprc/{genome}"] = float(
                average_precision_score(labels, probs)
            )
    metrics.update(per_species)

    # Log per-species metrics to W&B (run was finalized by Lightning,
    # so reopen it with resume="must" using the captured run ID)
    if per_species and wandb_run_id is not None:
        try:
            run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                id=wandb_run_id,
                resume="must",
            )
            run.log(per_species)
            run.finish()
        except Exception:
            log.warning("Failed to log per-species metrics to W&B", exc_info=True)

    output_metrics.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
