"""Whole-genome enhancer prediction with per-bin segmentation model.

Tiles genomic regions with non-overlapping windows and runs the
EnhancerSegmenter to produce per-bin logits at 128bp resolution.

Usage::

    python -m marin_dna.pipelines.enhancer_segmentation.predict_genome \
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
from marin_dna.pipelines.enhancer_segmentation.model import EnhancerSegmenter

logger = logging.getLogger(__name__)


def tile_chromosomes(
    chrom_sizes: dict[str, int], window_size: int
) -> list[tuple[str, int, int]]:
    """Emit non-overlapping full-context windows for each chromosome.

    Only windows that fit entirely within a chromosome are emitted —
    the model was trained on unpadded sequence and cannot be trusted
    on N-padded input. Contigs shorter than ``window_size`` and the
    last ``size % window_size`` bases of each chromosome are therefore
    uncovered. See discussion on issue #118.
    """
    windows: list[tuple[str, int, int]] = []
    for chrom, size in chrom_sizes.items():
        n_windows = size // window_size
        for i in range(n_windows):
            start = i * window_size
            windows.append((chrom, start, start + window_size))
    return windows


def predict_genome(
    genome_path: Path,
    checkpoint_path: Path,
    windows: pl.DataFrame,
    bin_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
) -> pl.DataFrame:
    """Run segmentation prediction on pre-computed windows.

    Args:
        genome_path: Path to genome file (.2bit or .fa/.fa.gz).
        checkpoint_path: Path to trained EnhancerSegmenter checkpoint.
        windows: DataFrame with ``chrom``, ``start``, ``end`` columns.
        bin_size: Size of each output bin in bp.
        batch_size: Inference batch size.
        num_workers: Number of DataLoader workers.

    Returns:
        DataFrame with columns ``chrom``, ``bin_start``, ``bin_end``, ``logit``
        — one row per bin across all windows.
    """
    window_size = int(windows["end"][0] - windows["start"][0])
    if window_size % bin_size != 0:
        raise ValueError(
            f"window_size ({window_size}) must be a multiple of bin_size ({bin_size})"
        )
    expected_num_bins = window_size // bin_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    model = EnhancerSegmenter.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    if torch.cuda.is_available():
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

    if predictions is None:
        raise RuntimeError("trainer.predict returned None")
    all_logits = np.concatenate([p.float().numpy() for p in predictions])  # type: ignore[union-attr]
    num_bins = all_logits.shape[1]
    if num_bins != expected_num_bins:
        raise ValueError(
            f"Model produced {num_bins} bins per window but window_size "
            f"({window_size}) / bin_size ({bin_size}) = {expected_num_bins}. "
            f"Check that --bin-size matches the model's native downsampling."
        )
    total_bins = len(windows) * num_bins

    logger.info(
        "Predicted %d windows (%d bins) in %.1fs (%.0f bins/sec)",
        len(windows),
        total_bins,
        elapsed,
        total_bins / elapsed,
    )

    chroms = np.repeat(windows["chrom"].to_numpy(), num_bins)
    window_starts = np.repeat(windows["start"].to_numpy(), num_bins)
    bin_indices = np.tile(np.arange(num_bins, dtype=np.int32), len(windows))
    bin_starts = window_starts + bin_indices * bin_size
    bin_ends = bin_starts + bin_size

    return pl.DataFrame(
        {
            "chrom": chroms,
            "bin_start": bin_starts,
            "bin_end": bin_ends,
            "logit": all_logits.reshape(-1).astype(np.float32),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict per-bin enhancer logits across genomic regions."
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
    parser.add_argument("--bin-size", type=int, default=128, help="Bin size in bp")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Inference batch size"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Truncate windows to first N for smoke tests (0 = no limit)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output parquet")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    logger.info("Loading windows from %s", args.windows)
    windows = pl.read_parquet(args.windows)
    if args.max_windows > 0:
        windows = windows.head(args.max_windows)
        logger.info("  Truncated to first %d windows for smoke test", args.max_windows)
    logger.info(
        "  %d windows across %d chromosomes", len(windows), windows["chrom"].n_unique()
    )

    result = predict_genome(
        genome_path=args.genome,
        checkpoint_path=args.checkpoint,
        windows=windows,
        bin_size=args.bin_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(args.output)
    logger.info("Wrote %s (%d bins)", args.output, len(result))


if __name__ == "__main__":
    main()
