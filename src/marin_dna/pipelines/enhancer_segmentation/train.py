"""CLI training script for the per-bin enhancer segmenter.

Usage::

    uv run python -m marin_dna.pipelines.enhancer_segmentation.train \\
        --train-parquet results/dataset/segmentation/seg_v1/train.parquet \\
        --val-parquet results/dataset/segmentation/seg_v1/validation.parquet \\
        --weights-path results/weights/model_all_folds.safetensors \\
        --output-ckpt results/model/segmentation/default/seg_v1/best.ckpt \\
        --output-metrics results/model/segmentation/default/seg_v1/metrics.json \\
        --output-val-predictions results/model/segmentation/default/seg_v1/val_predictions.parquet \\
        --wandb-run default-seg_v1
"""

import argparse
import json
import logging
import math
from pathlib import Path

import lightning as L
import numpy as np
import polars as pl
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from marin_dna.pipelines.enhancer_segmentation.dataset import SegmentationDataset
from marin_dna.pipelines.enhancer_segmentation.model import EnhancerSegmenter
from marin_dna.pipelines.enhancer_segmentation.predictions import build_bin_predictions

log = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")

WANDB_PROJECT = "bolinas-enhancer-segmentation"
WANDB_ENTITY = "gonzalobenegas"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-bin enhancer segmenter")
    parser.add_argument("--train-parquet", type=str, required=True)
    parser.add_argument("--val-parquet", type=str, required=True)
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--output-ckpt", type=str, required=True)
    parser.add_argument("--output-metrics", type=str, required=True)
    parser.add_argument("--output-val-predictions", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument(
        "--freeze-backbone", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument(
        "--n-transformer-layers",
        type=int,
        default=0,
        help="Number of AlphaGenome transformer blocks to stack after the "
        "CNN encoder (0 = encoder-only).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Lightning Trainer.max_steps (-1 = unlimited). "
        "Takes precedence over --limit-train-batches for the LR schedule.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=1.0,
        help="Fraction (<=1.0) of training batches per epoch, or an int number "
        "of batches. Mirrors Lightning Trainer.limit_train_batches.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=1.0,
        help="Fraction (<=1.0) or int number of validation batches. Mirrors "
        "Lightning Trainer.limit_val_batches.",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate per optimizer step. "
        "Use to keep the effective batch size constant when reducing "
        "--batch-size for memory; total nucleotides per optimizer step = "
        "batch_size * accumulate_grad_batches * window_size.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-run", type=str, required=True)
    return parser.parse_args()


def _as_batches_arg(v: float) -> int | float:
    """Interpret a CLI float as Lightning's int|float batches argument.

    Lightning semantics: int = number of batches, float in [0.0, 1.0] =
    fraction. Values in (0, 1] stay as a fraction (1.0 = all batches);
    values > 1 with no fractional part become an explicit batch count.
    """
    if v > 1 and float(v).is_integer():
        return int(v)
    return float(v)


def compute_pos_weight(train_parquet: str) -> float:
    """Compute BCE pos_weight = (# negative bins) / (# positive bins) from the
    training parquet. Derives the value from the exact dataset the model sees
    so it cannot drift from the data (subsampling, splits, RC augmentation).
    """
    labels = pl.read_parquet(train_parquet, columns=["labels"])["labels"]
    arr = np.asarray(labels.to_list(), dtype=np.uint8)
    n_pos = int(arr.sum())
    n_neg = int(arr.size - n_pos)
    if n_pos == 0:
        raise ValueError("Training set has no positive bins; cannot set pos_weight")
    return n_neg / n_pos


def main() -> None:
    args = parse_args()
    output_ckpt = Path(args.output_ckpt)
    output_metrics = Path(args.output_metrics)
    output_dir = output_ckpt.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(args.seed, workers=True)

    pos_weight = compute_pos_weight(args.train_parquet)
    log.info("Computed pos_weight=%.4f from training set", pos_weight)

    # Build a single genome_to_idx shared by train + val so per-species AUPRC
    # metrics reference a stable integer mapping regardless of split contents.
    train_genomes = (
        pl.read_parquet(args.train_parquet, columns=["genome"])["genome"]
        .unique()
        .to_list()
    )
    val_genomes = (
        pl.read_parquet(args.val_parquet, columns=["genome"])["genome"]
        .unique()
        .to_list()
    )
    genome_to_idx = {
        g: i for i, g in enumerate(sorted(set(train_genomes) | set(val_genomes)))
    }

    train_ds = SegmentationDataset(
        args.train_parquet, augment_rc=True, genome_to_idx=genome_to_idx
    )
    val_ds = SegmentationDataset(
        args.val_parquet, augment_rc=False, genome_to_idx=genome_to_idx
    )

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

    # Schedule over optimizer steps: Lightning counts post-accumulation steps,
    # so divide by accumulate_grad_batches.
    full_micro_batches = len(train_loader)
    acc = max(1, args.accumulate_grad_batches)
    if args.max_steps > 0:
        num_training_steps = args.max_steps
    else:
        if args.limit_train_batches <= 1.0:
            micro = max(1, int(full_micro_batches * args.limit_train_batches))
        else:
            micro = min(full_micro_batches, int(args.limit_train_batches))
        num_training_steps = max(1, math.ceil(micro / acc))

    model = EnhancerSegmenter(
        weights_path=args.weights_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        freeze_backbone=args.freeze_backbone,
        warmup_fraction=args.warmup_fraction,
        num_training_steps=num_training_steps,
        pos_weight=pos_weight,
        genomes=train_ds.genomes,
        n_transformer_layers=args.n_transformer_layers,
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
        max_steps=args.max_steps,
        limit_train_batches=_as_batches_arg(args.limit_train_batches),
        limit_val_batches=_as_batches_arg(args.limit_val_batches),
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision="bf16-mixed",
        accelerator="gpu",
        devices=1,  # single-GPU only; multi-GPU not supported
        gradient_clip_val=args.gradient_clip_val,
        logger=loggers,
        default_root_dir=output_dir,
    )

    # Record pos_weight and weights_path to the W&B run while it's still open
    # (trainer.fit finalizes it). The classification pipeline's post-fit
    # wandb.init(resume='must') pattern fails silently — per-species AUPRC is
    # now logged live by the Lightning module's on_validation_epoch_end instead.
    wandb_logger = next((lg for lg in loggers if isinstance(lg, WandbLogger)), None)
    if wandb_logger is not None:
        wandb_logger.experiment.config.update(
            {
                "pos_weight": pos_weight,
                "weights_path": str(args.weights_path) if args.weights_path else None,
            },
            allow_val_change=True,
        )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Force a final validation pass on the trained weights. Lightning only
    # runs val at the end of an epoch by default; with `max_steps` cutting
    # off mid-epoch the final val otherwise doesn't fire, so val_logits
    # would hold only the sanity-check tensors from step 0.
    trainer.validate(model, dataloaders=val_loader, verbose=False)

    trainer.save_checkpoint(output_ckpt)

    # Collect every val_* metric Lightning logged (includes per-species AUPRC).
    metrics: dict[str, float] = {
        k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("val_")
    }
    metrics["pos_weight"] = pos_weight

    logits_flat = model.val_logits.compute().cpu().numpy()
    val_meta = pl.read_parquet(
        args.val_parquet,
        columns=["genome", "chrom", "start", "end", "strand", "labels"],
    )
    # With limit_val_batches < 1.0 Lightning iterates a prefix of the
    # (unshuffled) val loader; the val parquet is shuffled at build time so
    # the prefix is a random sample. Truncate val_meta to what was seen.
    num_bins = len(val_meta["labels"][0])
    n_windows_seen = len(logits_flat) // num_bins
    if n_windows_seen < len(val_meta):
        val_meta = val_meta.head(n_windows_seen)

    if args.output_val_predictions and len(val_meta) > 0:
        predictions = build_bin_predictions(val_meta, logits_flat)
        Path(args.output_val_predictions).parent.mkdir(parents=True, exist_ok=True)
        predictions.write_parquet(args.output_val_predictions)

    output_metrics.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
