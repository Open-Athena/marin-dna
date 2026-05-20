"""Whole-genome enhancer prediction with sliding-window inference.

Scans genomic regions with a trained EnhancerClassifier, producing
per-window logits in parquet format. Designed for genome-wide enhancer
annotation across many species.

Usage::

    python -m marin_dna.pipelines.enhancer_classification.predict_genome \
        --genome genome.2bit \
        --checkpoint best.ckpt \
        --windows windows.parquet \
        --output predictions.parquet
"""

import argparse
import logging
import time
from pathlib import Path

import lightning as L
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from marin_dna.pipelines.enhancer_classification.genome_window_dataset import (
    GenomeWindowDataset,
)
from marin_dna.pipelines.enhancer_classification.model import EnhancerClassifier

logger = logging.getLogger(__name__)


def predict_genome(
    genome_path: Path,
    checkpoint_path: Path,
    windows: pl.DataFrame,
    batch_size: int = 512,
    num_workers: int = 4,
) -> pl.DataFrame:
    """Run enhancer prediction on pre-computed windows.

    Args:
        genome_path: Path to genome file (.2bit or .fa/.fa.gz).
        checkpoint_path: Path to trained EnhancerClassifier checkpoint.
        windows: DataFrame with ``chrom``, ``start``, ``end`` columns.
        batch_size: Inference batch size.
        num_workers: Number of DataLoader workers.

    Returns:
        The input DataFrame with an added ``logit`` column (float32).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    model = EnhancerClassifier.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    model.to(device)
    model.eval()
    model = torch.compile(model)  # type: ignore[assignment]

    dataset = GenomeWindowDataset(genome_path, windows)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    trainer = L.Trainer(
        accelerator="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=False,
        enable_checkpointing=False,
    )

    t0 = time.perf_counter()
    predictions = trainer.predict(model, dataloader)
    elapsed = time.perf_counter() - t0

    assert predictions is not None
    all_logits = np.concatenate([p.float().numpy() for p in predictions])  # type: ignore[union-attr]

    logger.info(
        "Predicted %d windows in %.1fs (%.0f windows/sec)",
        len(all_logits),
        elapsed,
        len(all_logits) / elapsed,
    )

    return windows.with_columns(pl.Series("logit", all_logits, dtype=pl.Float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict enhancer logits across genomic regions."
    )
    parser.add_argument(
        "--genome", type=Path, required=True, help="Genome file (.2bit or .fa.gz)"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Lightning checkpoint"
    )
    parser.add_argument(
        "--windows", type=Path, required=True, help="Window coordinates parquet"
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Inference batch size"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    logger.info("Loading windows from %s", args.windows)
    windows = pl.read_parquet(args.windows)
    logger.info(
        "  %d windows across %d chromosomes", len(windows), windows["chrom"].n_unique()
    )

    result = predict_genome(
        genome_path=args.genome,
        checkpoint_path=args.checkpoint,
        windows=windows,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(args.output)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
