"""Build the preregistered high-purity human-reference panel for issue #440."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any

import polars as pl
from twobitreader import TwoBitFile

ISSUE = 440
WINDOW_BP = 255
FOCAL_INDEX = 127
SAMPLES_PER_CLASS = 2_048
SOURCE_LABELS_BYTES = 11_628_557
SOURCE_LABELS_SHA256 = (
    "a41b6a582b04ce13d47416b1d8c2a1e5d15cfb4d19b1ea5e05eec6fc018ccdbd"
)
SOURCE_PHYLOP_BYTES = 171_320_322
SOURCE_PHYLOP_SHA256 = (
    "b634d83111ce790d39cf84b752e7aa61655bacaf4d5425e3ea335915abbd929d"
)
SOURCE_TWOBIT_BYTES = 812_795_740
SOURCE_TWOBIT_SHA256 = (
    "c47af4db6aadd72efdafa926ddd0c5e185ba2109c816727b7c52fd956a928e27"
)
STANDARD_CHROMS = tuple(str(index) for index in range(1, 23)) + ("X", "Y")
NUCLEOTIDES = frozenset("ACGTN")

FUNCTIONAL_CLASSES = {
    "cds": "cds_frac",
    "utr3": "utr3_frac",
    "tss_region_and_utr5": "tss_region_and_utr5_frac",
    "ncrna_exon": "ncrna_exon_frac",
    "ccre_non_promoter": "ccre_non_promoter_frac",
}
REFERENCE_CLASSES = (*FUNCTIONAL_CLASSES, "intron", "intergenic")
FRACTION_COLUMNS = (
    "functional_frac",
    "cds_frac",
    "utr3_frac",
    "ncrna_exon_frac",
    "tss_region_and_utr5_frac",
    "ccre_non_promoter_frac",
    "gene_body_frac",
    "intron_frac",
    "intergenic_frac",
)

assert WINDOW_BP == 2 * FOCAL_INDEX + 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def assert_commit(value: str) -> None:
    assert len(value) == 40
    assert all(character in "0123456789abcdef" for character in value)


def stable_hash(reference_class: str, name: str) -> str:
    value = f"{reference_class}|{name}".encode()
    return hashlib.blake2b(value, digest_size=16).hexdigest()


def pure_class_candidates(labels: pl.DataFrame) -> pl.DataFrame:
    required = {
        "name",
        "chrom",
        "start",
        "end",
        "label",
        *FRACTION_COLUMNS,
    }
    assert required <= set(labels.columns), required - set(labels.columns)
    assert labels["name"].n_unique() == labels.height
    assert labels.filter(~pl.col("chrom").is_in(STANDARD_CHROMS)).is_empty()
    assert labels.filter(pl.col("end") - pl.col("start") != WINDOW_BP).is_empty()
    assert (
        labels.select(pl.col(FRACTION_COLUMNS).is_null().sum()).sum_horizontal().item()
        == 0
    )

    frames: list[pl.DataFrame] = []
    for reference_class, fraction_column in FUNCTIONAL_CLASSES.items():
        frames.append(
            labels.filter(
                (pl.col("label") == reference_class) & (pl.col(fraction_column) == 1.0)
            ).with_columns(pl.lit(reference_class).alias("reference_class"))
        )
    frames.extend(
        [
            labels.filter(
                (pl.col("label") == "background") & (pl.col("intron_frac") == 1.0)
            ).with_columns(pl.lit("intron").alias("reference_class")),
            labels.filter(
                (pl.col("label") == "background") & (pl.col("intergenic_frac") == 1.0)
            ).with_columns(pl.lit("intergenic").alias("reference_class")),
        ]
    )
    candidates = pl.concat(frames).sort("reference_class", "name")
    assert set(candidates["reference_class"].unique()) == set(REFERENCE_CLASSES)
    assert candidates["name"].n_unique() == candidates.height
    assert (
        candidates.select(
            pl.max_horizontal(pl.col(FRACTION_COLUMNS))
            .min()
            .alias("minimum_max_fraction")
        ).item()
        == 1.0
    )
    return candidates


def deterministic_balanced_sample(
    candidates: pl.DataFrame, *, samples_per_class: int = SAMPLES_PER_CLASS
) -> pl.DataFrame:
    assert samples_per_class > 0
    frames: list[pl.DataFrame] = []
    for reference_class in REFERENCE_CLASSES:
        class_candidates = candidates.filter(
            pl.col("reference_class") == reference_class
        )
        assert class_candidates.height >= samples_per_class
        frames.append(
            class_candidates.with_columns(
                pl.col("name")
                .map_elements(
                    partial(stable_hash, reference_class),
                    return_dtype=pl.String,
                )
                .alias("selection_hash")
            )
            .sort("selection_hash", "name")
            .head(samples_per_class)
        )
    sample = (
        pl.concat(frames)
        .sort("selection_hash", "reference_class", "name")
        .with_row_index("panel_row")
    )
    assert sample.height == samples_per_class * len(REFERENCE_CLASSES)
    assert sample["name"].n_unique() == sample.height
    assert sample["selection_hash"].n_unique() == sample.height
    assert (
        sample.select(pl.struct("chrom", "start", "end").n_unique()).item()
        == sample.height
    )
    assert dict(sample.group_by("reference_class").len().iter_rows()) == {
        reference_class: samples_per_class for reference_class in REFERENCE_CLASSES
    }
    return sample


def extract_sequence(
    genome: Mapping[str, Sequence[str]], *, chrom: str, start: int, end: int
) -> str:
    assert chrom in STANDARD_CHROMS
    assert 0 <= start < end and end - start == WINDOW_BP
    sequence = str(genome[chrom][start:end]).upper()
    assert len(sequence) == WINDOW_BP
    assert set(sequence) <= NUCLEOTIDES
    return sequence


def sequence_metrics(sequence: str) -> dict[str, float | int]:
    assert len(sequence) == WINDOW_BP and set(sequence) <= NUCLEOTIDES
    canonical = [base for base in sequence if base in "ACGT"]
    assert canonical
    counts = {base: canonical.count(base) for base in "ACGT"}
    entropy = -sum(
        (count / len(canonical)) * math.log2(count / len(canonical))
        for count in counts.values()
        if count
    )
    longest_run = 1
    current_run = 1
    for previous, current in pairwise(sequence):
        if previous == current:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
    return {
        "gc_fraction": (counts["G"] + counts["C"]) / len(canonical),
        "cpg_count": sequence.count("CG"),
        "sequence_entropy": entropy,
        "maximum_homopolymer": longest_run,
        "n_fraction": sequence.count("N") / WINDOW_BP,
    }


def build_panel(
    *,
    labels_path: Path,
    phylop_path: Path,
    genome_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    run_id = os.environ.get("RUN_ID", "")
    assert run_id
    for path, expected_bytes, expected_sha256 in (
        (labels_path, SOURCE_LABELS_BYTES, SOURCE_LABELS_SHA256),
        (phylop_path, SOURCE_PHYLOP_BYTES, SOURCE_PHYLOP_SHA256),
        (genome_path, SOURCE_TWOBIT_BYTES, SOURCE_TWOBIT_SHA256),
    ):
        assert path.is_file()
        assert path.stat().st_size == expected_bytes
        assert sha256_file(path) == expected_sha256

    labels = pl.read_parquet(labels_path)
    candidates = pure_class_candidates(labels)
    candidate_counts = dict(
        candidates.group_by("reference_class").len().sort("reference_class").iter_rows()
    )
    panel = deterministic_balanced_sample(candidates)
    phylop = pl.read_parquet(phylop_path).select(
        "name",
        "conserved_bases",
        "proportion_conserved",
        "mean_phylop",
        "n_valid_bases",
    )
    assert phylop["name"].n_unique() == phylop.height
    panel = panel.join(phylop, on="name", how="left", validate="1:1")
    assert (
        panel.select(
            pl.col(
                "conserved_bases",
                "proportion_conserved",
                "mean_phylop",
                "n_valid_bases",
            ).null_count()
        )
        .sum_horizontal()
        .item()
        == 0
    )
    assert panel.filter(pl.col("proportion_conserved") < 0.20).is_empty()

    genome = TwoBitFile(str(genome_path))
    sequences: list[str] = []
    metrics: list[dict[str, float | int]] = []
    for chrom, start, end in panel.select("chrom", "start", "end").iter_rows():
        sequence = extract_sequence(genome, chrom=chrom, start=start, end=end)
        sequences.append(sequence)
        metrics.append(sequence_metrics(sequence))
    panel = panel.with_columns(pl.Series("sequence", sequences)).hstack(
        pl.DataFrame(metrics)
    )
    assert panel.filter(pl.col("sequence").str.len_chars() != WINDOW_BP).is_empty()
    assert panel.filter(pl.col("n_fraction") > 0).is_empty()
    assert panel["panel_row"].to_list() == list(range(panel.height))

    output_dir.mkdir(parents=True)
    panel_path = output_dir / "panel.parquet"
    panel.write_parquet(panel_path, compression="zstd")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": run_id,
        "experiment_commit": experiment_commit,
        "scope": "balanced-100pct-pure-reference-state-focal-panel",
        "coordinate_system": "0-based half-open",
        "window_bp": WINDOW_BP,
        "focal_index": FOCAL_INDEX,
        "rows": panel.height,
        "samples_per_class": SAMPLES_PER_CLASS,
        "classes": list(REFERENCE_CLASSES),
        "candidate_counts": candidate_counts,
        "class_counts": dict(
            panel.group_by("reference_class").len().sort("reference_class").iter_rows()
        ),
        "sampling": "smallest BLAKE2b-128 hash of 'class|name'",
        "purity": "class-specific disjoint fraction exactly 1.0 across all 255 bp",
        "inputs": {
            "labels": {
                "uri": "s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results/human/intervals/region_labels/v4/min0.20.parquet",
                "bytes": SOURCE_LABELS_BYTES,
                "sha256": SOURCE_LABELS_SHA256,
            },
            "phylop": {
                "uri": "s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results/human/intervals/scored/phyloP_447m_windows.parquet",
                "bytes": SOURCE_PHYLOP_BYTES,
                "sha256": SOURCE_PHYLOP_SHA256,
            },
            "genome": {
                "uri": "s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results/human/genome.2bit",
                "bytes": SOURCE_TWOBIT_BYTES,
                "sha256": SOURCE_TWOBIT_SHA256,
                "assembly": "Ensembl release 115 GRCh38 soft-masked primary assembly",
            },
        },
        "panel": {
            "path": str(panel_path),
            "bytes": panel_path.stat().st_size,
            "sha256": sha256_file(panel_path),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--phylop", type=Path, required=True)
    parser.add_argument("--genome", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_panel(
        labels_path=args.labels,
        phylop_path=args.phylop,
        genome_path=args.genome,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
