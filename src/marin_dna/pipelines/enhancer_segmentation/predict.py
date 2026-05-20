"""Generate per-bin enhancer probability BedGraph tracks.

Tiles a genomic region with non-overlapping ``window_size`` windows, runs the
segmenter, and writes one score per ``bin_size`` bin. The caller is expected
to supply a region whose length is already a multiple of ``window_size``
(compute this once offline per region; see the ``segmentation_visualization``
block in the pipeline config).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from alphagenome_pytorch.utils.sequence import sequence_to_onehot
from marin_dna.data.genome import Genome

from marin_dna.pipelines.enhancer_segmentation.model import EnhancerSegmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict per-bin enhancer probabilities over a genomic region."
    )
    parser.add_argument("--genome", type=Path, required=True, help="Genome .fa.gz")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Lightning checkpoint"
    )
    parser.add_argument("--chrom", type=str, required=True)
    parser.add_argument(
        "--start", type=int, required=True, help="Region start (0-based)"
    )
    parser.add_argument("--end", type=int, required=True, help="Region end (exclusive)")
    parser.add_argument("--window-size", type=int, default=16384)
    parser.add_argument("--bin-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", type=str, default="enhancer_prob_seg")
    return parser.parse_args()


def tile_region(start: int, end: int, window_size: int) -> list[tuple[int, int]]:
    """Non-overlapping ``window_size`` tiles covering ``[start, end)``.

    The region length must be an exact multiple of ``window_size``;
    pre-compute enlarged coords offline and pass them in.
    """
    length = end - start
    if length <= 0 or length % window_size != 0:
        raise ValueError(
            f"Region length {length} must be a positive multiple of {window_size}"
        )
    return [(s, s + window_size) for s in range(start, end, window_size)]


def predict_tiles(
    genome: Genome,
    model: EnhancerSegmenter,
    chrom: str,
    tiles: list[tuple[int, int]],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return ``(n_tiles, num_bins)`` array of per-bin probabilities."""
    all_probs: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i : i + batch_size]
            seqs = [genome(chrom, s, e) for s, e in batch]
            onehot = np.stack([sequence_to_onehot(s).astype(np.float32) for s in seqs])
            x = torch.from_numpy(onehot).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
            probs = torch.sigmoid(logits).float().cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def write_bedgraph(
    path: Path,
    chrom: str,
    tiles: list[tuple[int, int]],
    probs: np.ndarray,
    bin_size: int,
    track_name: str,
) -> None:
    ucsc_chrom = f"chr{chrom}"
    with open(path, "w") as f:
        f.write(
            f'track type=bedGraph name="{track_name}" '
            f"visibility=full autoScale=on windowingFunction=mean\n"
        )
        for tile_idx, (tile_start, _tile_end) in enumerate(tiles):
            for bin_idx in range(probs.shape[1]):
                bin_start = tile_start + bin_idx * bin_size
                bin_end = bin_start + bin_size
                f.write(
                    f"{ucsc_chrom}\t{bin_start}\t{bin_end}\t"
                    f"{probs[tile_idx, bin_idx]:.6f}\n"
                )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    genome = Genome(args.genome, subset_chroms={args.chrom})

    model = EnhancerSegmenter.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    tiles = tile_region(args.start, args.end, args.window_size)
    probs = predict_tiles(genome, model, args.chrom, tiles, args.batch_size, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_bedgraph(args.output, args.chrom, tiles, probs, args.bin_size, args.name)


if __name__ == "__main__":
    main()
