"""Inspect reproducible trajectory-group samples in local and UCSC context."""

from __future__ import annotations

import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("/tmp/ld489-preview.MRb9Kt")
BIOLOGY_ROOT = Path(__file__).resolve().parent / "biology"
METADATA_ROOT = ROOT / "metadata"
UCSC_API = "https://api.genome.ucsc.edu/getData/track"
CONTEXT_RADIUS = 30


def normalized_kmer_entropy(sequence: str, k: int = 3) -> float:
    """Return observed k-mer entropy divided by the four-base maximum."""
    kmers = [sequence[index : index + k] for index in range(len(sequence) - k + 1)]
    counts = Counter(kmers)
    probabilities = np.array(list(counts.values()), dtype=float) / len(kmers)
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return entropy / math.log2(4**k)


def longest_homopolymer(sequence: str) -> int:
    """Return the longest same-base run."""
    longest = 1
    current = 1
    for previous, value in zip(sequence, sequence[1:], strict=False):
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
    return longest


def best_short_period_match(sequence: str) -> tuple[int, float]:
    """Return the short period with the largest lagged base agreement."""
    best_period = 1
    best_match = 0.0
    for period in range(1, 13):
        match = float(np.mean(np.fromiter(
            (left == right for left, right in zip(sequence[period:], sequence[:-period])),
            dtype=bool,
        )))
        if match > best_match:
            best_period = period
            best_match = match
    return best_period, best_match


def local_context(samples: pd.DataFrame) -> pd.DataFrame:
    """Join the exact pinned validation sequences and local complexity metrics."""
    rows: list[dict[str, object]] = []
    for region, region_samples in samples.groupby("region", sort=False):
        metadata = pq.read_table(
            METADATA_ROOT / f"{region}.parquet",
            columns=["row_index", "sequence_upper", "is_repeat"],
        ).to_pandas()
        metadata = metadata.set_index("row_index")
        for sample in region_samples.to_dict(orient="records"):
            record = metadata.loc[int(sample["row_index"])]
            sequence = str(record["sequence_upper"])
            target = int(sample["target_pos"])
            start = target - CONTEXT_RADIUS
            end = target + CONTEXT_RADIUS + 1
            context = sequence[start:end]
            assert len(context) == 2 * CONTEXT_RADIUS + 1
            assert context[CONTEXT_RADIUS] == sample["base"]
            repeat_mask = np.asarray(record["is_repeat"], dtype=bool)[start:end]
            period, period_match = best_short_period_match(context)
            sample.update(
                {
                    "context_61": context,
                    "context_61_marked": (
                        context[:CONTEXT_RADIUS]
                        + "["
                        + context[CONTEXT_RADIUS]
                        + "]"
                        + context[CONTEXT_RADIUS + 1 :]
                    ),
                    "context_61_repeat_fraction": float(repeat_mask.mean()),
                    "context_61_3mer_entropy_normalized": normalized_kmer_entropy(context),
                    "context_61_longest_homopolymer": longest_homopolymer(context),
                    "context_61_best_period": period,
                    "context_61_best_period_match": period_match,
                }
            )
            rows.append(sample)
    return pd.DataFrame(rows)


def ucsc_track(track: str, chrom: str, start: int, end: int) -> dict[str, object]:
    """Query one 0-based half-open interval from the UCSC hg38 REST API."""
    query = urlencode(
        {
            "genome": "hg38",
            "track": track,
            "chrom": chrom,
            "start": max(0, start),
            "end": end,
        }
    ).replace("&", ";")
    with urlopen(f"{UCSC_API}?{query}", timeout=30) as response:
        payload = json.load(response)
    if "error" in payload:
        raise RuntimeError(str(payload))
    return payload


def interval_distance(start: int, end: int, feature_start: int, feature_end: int) -> int:
    if feature_end <= start:
        return start - feature_end
    if feature_start >= end:
        return feature_start - end
    return 0


def remote_context(sample: dict[str, object]) -> dict[str, object]:
    """Add exploratory UCSC repeat, duplication, and RefSeq context."""
    code = int(sample["chromosome_code"])
    chrom = f"chr{code if code <= 22 else {23: 'X', 24: 'Y', 25: 'M'}[code]}"
    position = int(sample["genomic_pos"])
    flank_start = position - 500
    flank_end = position + 501
    rmsk = ucsc_track("rmsk", chrom, flank_start, flank_end).get("rmsk", [])
    simple = ucsc_track("simpleRepeat", chrom, flank_start, flank_end).get(
        "simpleRepeat", []
    )
    superdups = ucsc_track(
        "genomicSuperDups", chrom, position, position + 1
    ).get("genomicSuperDups", [])
    genes = ucsc_track("ncbiRefSeq", chrom, position, position + 1).get(
        "ncbiRefSeq", []
    )
    rmsk_distances = [
        interval_distance(position, position + 1, int(item["genoStart"]), int(item["genoEnd"]))
        for item in rmsk
    ]
    simple_distances = [
        interval_distance(
            position, position + 1, int(item["chromStart"]), int(item["chromEnd"])
        )
        for item in simple
    ]
    sample.update(
        {
            "ucsc_chrom": chrom,
            "ucsc_rmsk_overlap": any(distance == 0 for distance in rmsk_distances),
            "ucsc_rmsk_nearest_bp": min(rmsk_distances, default=None),
            "ucsc_rmsk_nearby_names": ";".join(
                sorted({str(item["repName"]) for item in rmsk})
            ),
            "ucsc_simple_repeat_overlap": any(
                distance == 0 for distance in simple_distances
            ),
            "ucsc_simple_repeat_nearest_bp": min(simple_distances, default=None),
            "ucsc_segmental_duplication_overlap": bool(superdups),
            "ucsc_refseq_genes": ";".join(
                sorted({str(item.get("name2", item.get("name", ""))) for item in genes})
            ),
        }
    )
    return sample


def main() -> None:
    samples = pd.read_parquet(BIOLOGY_ROOT / "trajectory_group_samples.parquet")
    samples = local_context(samples)
    with ThreadPoolExecutor(max_workers=4) as executor:
        inspected = list(executor.map(remote_context, samples.to_dict(orient="records")))
    output = pd.DataFrame(inspected)
    output.to_parquet(BIOLOGY_ROOT / "trajectory_group_samples_inspected.parquet", index=False)
    output.to_csv(BIOLOGY_ROOT / "trajectory_group_samples_inspected.csv", index=False)
    print(
        output[
            [
                "region",
                "group",
                "ucsc_chrom",
                "genomic_pos",
                "is_conserved",
                "context_61_3mer_entropy_normalized",
                "context_61_longest_homopolymer",
                "ucsc_rmsk_overlap",
                "ucsc_simple_repeat_overlap",
                "ucsc_segmental_duplication_overlap",
                "ucsc_refseq_genes",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
