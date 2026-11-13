"""Freeze an evaluation-only conservation panel outside LM training source regions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pysam

from exp593_dna.audit import file_sha256
from exp593_dna.build_annotations import contig_aliases, normalized_name
from exp593_dna.inputs import INPUTS_MANIFEST_SHA256
from exp593_dna.pipeline import DATASET_REVISION, reverse_complement

SOURCE_REPOSITORY = "kuleshov-group/cross-species-single-nucleotide-annotation"
SOURCE_REVISION = "ce38574d8a2aa8737167c6d5416a37a82d15eba4"
SOURCE_SHA256 = "67b986a54fcdef1a759da49164d1b210e216523c912e63513af972d4c869cb7c"
EXCLUSION_SHA256 = "57c06cab418804fac34d7b5a31e1ea88064f465ef659de466d288bf3c78d9413"
ALLOWED_CHROMOSOMES = ("1", "3", "5", "7", "9")


class Intervals:
    """Inclusive contact exclusion for conservatively expanded source intervals."""

    def __init__(self, values: list[tuple[int, int]]) -> None:
        merged: list[tuple[int, int]] = []
        for start, end in sorted(values):
            if start < 0 or end <= start:
                raise ValueError("Invalid genomic interval")
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        self.values = merged
        self.starts = [v[0] for v in merged]

    def touches(self, start: int, end: int) -> bool:
        index = bisect_right(self.starts, end) - 1
        return index >= 0 and self.values[index][1] >= start


def canonical_sequence_hash(sequence: str) -> str:
    dna = sequence.upper()
    return hashlib.sha256(min(dna, reverse_complement(dna)).encode()).hexdigest()


def training_intervals(
    root: Path,
    manifest_path: Path,
    *,
    expected_rows: int = 2_638_656,
    expected_shards: int = 44,
    expected_manifest_sha256: str = INPUTS_MANIFEST_SHA256,
) -> tuple[dict[str, Intervals], dict[str, int]]:
    """Verify every original training metadata shard before selecting Sorghum rows."""
    if file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Original training input manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset_revision"] != DATASET_REVISION:
        raise ValueError("Original training dataset revision changed")
    entries = {
        n: r
        for n, r in manifest["files"].items()
        if n.startswith("metadata/train/") and n.endswith(".parquet")
    }
    if len(entries) != expected_shards:
        raise ValueError("Original training metadata inventory changed")
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    checked = sorghum = 0
    for name, record in sorted(entries.items()):
        path = root / Path(name).name
        if (
            path.stat().st_size != record["bytes"]
            or file_sha256(path) != record["sha256"]
        ):
            raise ValueError("Original training metadata bytes changed")
        for batch in pq.ParquetFile(path).iter_batches(
            columns=["assembly", "chrom", "start", "end", "split"]
        ):
            for r in batch.to_pylist():
                checked += 1
                if r["split"] != "train":
                    raise ValueError("Training metadata contains a different split")
                if r["assembly"] == "Sbicolor":
                    sorghum += 1
                    # The source fields describe parent regions. Expanding both
                    # ends also covers their possible one-base convention shift.
                    intervals[normalized_name(r["chrom"])].append(
                        (max(0, r["start"] - 1), r["end"] + 1)
                    )
    if checked != expected_rows or not sorghum:
        raise ValueError("Training source-row coverage changed")
    return {k: Intervals(v) for k, v in intervals.items()}, {
        "all_training_rows_checked": checked,
        "sorghum_training_rows": sorghum,
    }


@dataclass(frozen=True)
class ConservationSite:
    """One published alignment label, never admitted to language-model training."""

    source_row: int
    chrom: str
    position_1based: int
    start: int
    end: int
    sequence: str
    label: int
    dna_sha256: str
    selection_rank: str
    target_index: int = 255
    use: str = "evaluation_only"

    def __post_init__(self) -> None:
        if (
            self.chrom not in ALLOWED_CHROMOSOMES
            or self.start < 0
            or len(self.sequence) != 512
            or self.start != self.position_1based - 256
            or self.end != self.start + 512
            or self.target_index != 255
            or self.sequence[self.target_index] not in "ACGT"
            or self.label not in (0, 1)
            or self.use != "evaluation_only"
            or self.dna_sha256 != canonical_sequence_hash(self.sequence)
        ):
            raise ValueError("Invalid conservation development site")


def conservation_candidates(
    source: Path,
    exclusions: Path,
    reference: Path,
    forbidden: dict[str, Intervals],
    *,
    seed: int = 593,
    expected_source_sha256: str = SOURCE_SHA256,
    expected_exclusion_sha256: str = EXCLUSION_SHA256,
) -> tuple[list[ConservationSite], dict[str, int], dict[str, Any]]:
    """Require exact full-window reference identity and exclude benchmark DNA/RC."""
    if (
        file_sha256(source) != expected_source_sha256
        or file_sha256(exclusions) != expected_exclusion_sha256
    ):
        raise ValueError("Pinned conservation source or exclusion bytes changed")
    with exclusions.open() as stream:
        excluded_dna = {
            canonical_sequence_hash(r["sequences"])
            for r in csv.DictReader(stream, delimiter="\t")
        }
    provenance = json.loads(reference.read_text())
    if (
        provenance["assembly_code"] != "Sbicolor"
        or provenance["accession"] != "GCF_000003195.3"
    ):
        raise ValueError("Conservation reference accession changed")
    genome = provenance["assets"]["genome_gzip"]
    if file_sha256(Path(genome["path"])) != genome["sha256"]:
        raise ValueError("Conservation reference bytes changed")
    counts: Counter[str] = Counter()
    unique: dict[str, ConservationSite] = {}
    conflicting = set()
    with pysam.FastaFile(provenance["fasta"]) as fasta, source.open() as stream:
        aliases = contig_aliases(
            fasta.references, Path(provenance["assets"]["assembly_report"]["path"])
        )
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["chrom", "pos", "label", "sequences"]:
            raise ValueError("Conservation source schema changed")
        for row, r in enumerate(reader):
            counts["source_rows"] += 1
            chrom = normalized_name(r["chrom"])
            if chrom not in ALLOWED_CHROMOSOMES:
                continue
            counts["odd_chromosome_rows"] += 1
            sequence = r["sequences"].upper()
            if len(sequence) != 512 or int(r["label"]) not in (0, 1):
                raise ValueError("Conservation source length or label changed")
            position = int(r["pos"])
            start, end = position - 256, position + 256
            if start < 0 or sequence[255] not in "ACGT":
                counts["invalid_target"] += 1
                continue
            if chrom not in forbidden:
                raise ValueError("Missing Sorghum training source chromosome")
            if forbidden[chrom].touches(start, end):
                counts["training_source_region_overlap"] += 1
                continue
            if fasta.fetch(aliases[chrom], start, end).upper() != sequence:
                counts["reference_sequence_mismatch"] += 1
                continue
            digest = canonical_sequence_hash(sequence)
            if digest in excluded_dna:
                counts["final_benchmark_dna_overlap"] += 1
                continue
            candidate = ConservationSite(
                row,
                chrom,
                position,
                start,
                end,
                sequence,
                int(r["label"]),
                digest,
                hashlib.sha256(
                    f"{seed}:{expected_source_sha256}:{row}".encode()
                ).hexdigest(),
            )
            if digest in conflicting:
                continue
            if previous := unique.get(digest):
                counts["duplicate_dna_or_rc"] += 1
                if previous.label != candidate.label:
                    counts["conflicting_dna_labels"] += 1
                    conflicting.add(digest)
                    del unique[digest]
                elif candidate.selection_rank < previous.selection_rank:
                    unique[digest] = candidate
            else:
                unique[digest] = candidate
    counts["eligible_unique_sites"] = len(unique)
    return (
        list(unique.values()),
        dict(counts),
        {
            "accession": provenance["accession"],
            "genome_gzip_sha256": genome["sha256"],
            "genome_url": genome["url"],
        },
    )


def select_matched_sites(
    candidates: list[ConservationSite], *, per_class: int = 256, seed: int = 593
) -> list[ConservationSite]:
    """Match chromosome and target base, with no overlapping selected contexts."""
    if per_class < 1:
        raise ValueError("Positive development sample count required")
    pools: dict[tuple[str, str, int], list[ConservationSite]] = defaultdict(list)
    for item in candidates:
        pools[item.chrom, item.sequence[item.target_index], item.label].append(item)
    for pool in pools.values():
        pool.sort(key=lambda r: r.selection_rank, reverse=True)
    cells = [(chrom, base) for chrom in ALLOWED_CHROMOSOMES for base in "ACGT"]
    np.random.default_rng(seed).shuffle(cells)
    occupied: dict[str, list[tuple[int, int]]] = defaultdict(list)
    selected: list[ConservationSite] = []

    def take(
        chrom: str, base: str, label: int, extra: list[tuple[int, int]]
    ) -> ConservationSite | None:
        while pool := pools[chrom, base, label]:
            candidate = pool.pop()
            if not any(
                candidate.start <= end and candidate.end >= start
                for start, end in occupied[chrom] + extra
            ):
                return candidate
        return None

    while len(selected) < 2 * per_class:
        before = len(selected)
        for chrom, base in cells:
            positive = take(chrom, base, 1, [])
            if positive is None:
                continue
            negative = take(chrom, base, 0, [(positive.start, positive.end)])
            if negative is None:
                continue
            selected.extend((positive, negative))
            occupied[chrom].extend((s.start, s.end) for s in (positive, negative))
            if len(selected) == 2 * per_class:
                break
        if len(selected) == before:
            raise ValueError("Insufficient matched, nonoverlapping conservation sites")
    if len({s.dna_sha256 for s in selected}) != len(selected):
        raise ValueError("Duplicate development DNA")
    return sorted(selected, key=lambda r: (r.chrom, r.start))


def prepare_conservation(
    source_root: Path,
    prep_root: Path,
    inputs_manifest: Path,
    output: Path,
    *,
    per_class: int = 256,
) -> dict[str, Any]:
    forbidden, coverage = training_intervals(
        prep_root / "metadata/train", inputs_manifest
    )
    candidates, counts, reference = conservation_candidates(
        source_root / "conservation_development_source.tsv",
        source_root / "valid.tsv",
        prep_root / "references/Sbicolor/reference.json",
        forbidden,
    )
    selected = select_matched_sites(candidates, per_class=per_class)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "conservation_development.jsonl"
    destination.write_text(
        "".join(json.dumps(asdict(s), sort_keys=True) + "\n" for s in selected)
    )
    report = {
        "purpose": "evaluation_only; never training or annotation guidance",
        "source_dataset": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_file": "Evolutionary_constraint/train.tsv",
        "source_sha256": SOURCE_SHA256,
        "exclusion_file": "Evolutionary_constraint/valid.tsv",
        "exclusion_sha256": EXCLUSION_SHA256,
        "exclusion_reason": "The upstream validation DNA is reused by PlantCAD2's final Andropogoneae benchmark.",
        "reference": reference,
        "lm_inputs_manifest_sha256": INPUTS_MANIFEST_SHA256,
        "coverage": coverage,
        "candidate_counts": counts,
        "selected_sites": len(selected),
        "label_counts": dict(Counter(s.label for s in selected)),
        "chromosome_target_base_matching": True,
        "selected_contexts_disjoint": True,
        "coordinate_convention": "Zero-based half-open start=pos-256, end=pos+256; published pos is one-based; target index 255.",
        "selection_seed": 593,
        "panel_sha256": file_sha256(destination),
        "limitation": "Source-coordinate and exact benchmark-DNA separation do not establish absence of homology across the 65 training genomes.",
        "scoring_protocol": "Before any prediction, freeze bare left/right-RC true-base log-probability average as the primary conservation score, plus minimal-FIM true-base log-probability. Report AUROC and average precision; no supervised fitting or model selection.",
    }
    (output / "conservation_development_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prep-root", type=Path, required=True)
    parser.add_argument("--inputs-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(prepare_conservation(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
