"""Generate enhancer probability BedGraph tracks for genomic regions.

Runs sliding-window inference with the AlphaGenome CNN probe enhancer
classifier and writes per-window probabilities in BedGraph format,
suitable for display in the UCSC Genome Browser.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from alphagenome_pytorch.utils.sequence import sequence_to_onehot
from marin_dna.data.genome import Genome

from marin_dna.pipelines.enhancer_classification.model import EnhancerClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict enhancer probabilities over a genomic region."
    )
    parser.add_argument("--genome", type=Path, required=True, help="Genome .fa.gz")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Lightning checkpoint"
    )
    parser.add_argument(
        "--chrom", type=str, required=True, help="Ensembl chromosome name (e.g. '7')"
    )
    parser.add_argument(
        "--start", type=int, required=True, help="Region start (0-based)"
    )
    parser.add_argument("--end", type=int, required=True, help="Region end (exclusive)")
    parser.add_argument("--window-size", type=int, default=255, help="Window size")
    parser.add_argument(
        "--step-size", type=int, required=True, help="Sliding window step size"
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Inference batch size"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output .bedgraph file"
    )
    parser.add_argument(
        "--name", type=str, default="enhancer_prob", help="Track name for BedGraph"
    )
    return parser.parse_args()


def sliding_windows(
    start: int, end: int, window_size: int, step_size: int
) -> list[tuple[int, int]]:
    """Generate sliding window coordinates over a region."""
    windows = []
    for w_start in range(start, end - window_size + 1, step_size):
        windows.append((w_start, w_start + window_size))
    return windows


def predict_region(
    genome: Genome,
    model: EnhancerClassifier,
    chrom: str,
    windows: list[tuple[int, int]],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run batch inference on sliding windows, returning enhancer probabilities."""
    all_probs: list[np.ndarray] = []

    for i in range(0, len(windows), batch_size):
        batch_windows = windows[i : i + batch_size]
        sequences = [genome(chrom, s, e) for s, e in batch_windows]
        onehot_batch = np.array(
            [sequence_to_onehot(seq).astype(np.float32) for seq in sequences]
        )
        x = torch.from_numpy(onehot_batch).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs)


def write_bedgraph(
    path: Path,
    chrom: str,
    windows: list[tuple[int, int]],
    scores: np.ndarray,
    track_name: str,
) -> None:
    """Write scores to BedGraph format with UCSC-compatible chromosome names."""
    ucsc_chrom = f"chr{chrom}"
    with open(path, "w") as f:
        f.write(
            f'track type=bedGraph name="{track_name}" visibility=full autoScale=on windowingFunction=mean\n'
        )
        for (start, end), score in zip(windows, scores):
            f.write(f"{ucsc_chrom}\t{start}\t{end}\t{score:.6f}\n")


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    genome = Genome(args.genome, subset_chroms={args.chrom})

    model = EnhancerClassifier.load_from_checkpoint(
        args.checkpoint, map_location=device
    )
    model.to(device)
    model.eval()

    windows = sliding_windows(args.start, args.end, args.window_size, args.step_size)

    scores = predict_region(genome, model, args.chrom, windows, args.batch_size, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_bedgraph(args.output, args.chrom, windows, scores, args.name)


if __name__ == "__main__":
    main()
