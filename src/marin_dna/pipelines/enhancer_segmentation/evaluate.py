"""Evaluate a trained EnhancerSegmenter checkpoint on a validation parquet.

Runs a single forward pass over the entire validation parquet (no training,
no gradient) and writes a per-(window, bin) val_predictions parquet compatible
with the AUPRC / precision-recall tooling. Used to re-evaluate existing
checkpoints against a freshly built full-coverage validation set.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from marin_dna.pipelines.enhancer_segmentation.dataset import SegmentationDataset
from marin_dna.pipelines.enhancer_segmentation.model import EnhancerSegmenter
from marin_dna.pipelines.enhancer_segmentation.predictions import build_bin_predictions

log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("medium")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a segmenter checkpoint on a val parquet"
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--val-parquet", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EnhancerSegmenter.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    genome_to_idx = (
        {g: i for i, g in enumerate(model.genomes)} if model.genomes else None
    )
    val_ds = SegmentationDataset(
        args.val_parquet, augment_rc=False, genome_to_idx=genome_to_idx
    )
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    all_logits: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            x, _y, _g = batch
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
            all_logits.append(logits.float().cpu().numpy())

    logits_arr = np.concatenate(all_logits, axis=0)  # (N, num_bins)
    num_bins = logits_arr.shape[1]
    log.info("Ran inference on %d windows x %d bins", logits_arr.shape[0], num_bins)

    val_meta = pl.read_parquet(
        args.val_parquet,
        columns=["genome", "chrom", "start", "end", "strand", "labels"],
    )
    predictions = build_bin_predictions(val_meta, logits_arr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(args.output)
    log.info("Wrote %d prediction rows to %s", predictions.height, args.output)


if __name__ == "__main__":
    main()
