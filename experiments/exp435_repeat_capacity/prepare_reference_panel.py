"""Materialize the frozen, outcome-blind reference repeat panel for issue 435."""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pysam

from common import (
    ISSUE,
    PRIMARY_CHROMS,
    SEED,
    assert_commit,
    sha256_file,
    write_json,
)
from panel_common import (
    CATEGORY_CONTEXTS,
    EXPECTED_CLASS_COUNT,
    EXPECTED_FAMILY_COUNT,
    EXPECTED_SUBFAMILY_COUNT,
    FOCAL_INDEX,
    GLOBAL_SUBFAMILIES,
    MIN_CATEGORY_RAW_BP,
    MIN_CATEGORY_RECORDS,
    PANEL_RUN_ID,
    SUBFAMILIES_PER_FAMILY,
    UNIFORM_CONTROL_CONTEXTS,
    UNIFORM_REPEAT_CONTEXTS,
    WINDOW_BP,
    selection_hash,
    stable_seed,
)
from prepare_repeat_inventory import read_fai

INVENTORY_RUN_ID = "dna-exp435-repeat-inventory-r1"
INVENTORY_MANIFEST_SHA256 = (
    "47bf4dc4262a460476cb0c1e8a5de9695d8b389db3ad2aee7f7ce9af97d8efa6"
)
INVENTORY_S3 = (
    "s3://oa-bolinas/experiments/exp435/retrieval/dna-exp435-repeat-inventory-r1/"
)
FASTA_S3 = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)
CHROM_RANK = {chrom: index for index, chrom in enumerate(PRIMARY_CHROMS)}
NUCLEOTIDES = frozenset("ACGT")


def validate_inventory(
    inventory_dir: Path,
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    manifest_path = inventory_dir / "manifest.json"
    archive_manifest_path = inventory_dir / "archive_manifest.json"
    assert manifest_path.is_file() and archive_manifest_path.is_file()
    assert sha256_file(archive_manifest_path) == INVENTORY_MANIFEST_SHA256
    archive = json.loads(archive_manifest_path.read_text())
    assert archive["run_id"] == INVENTORY_RUN_ID
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == ISSUE and manifest["run_id"] == INVENTORY_RUN_ID
    assert manifest["analysis_status"] == "outcome_blind_annotation_inventory"
    for name, metadata in manifest["artifacts"].items():
        path = inventory_dir / name
        assert path.is_file() and path.stat().st_size == metadata["bytes"]
        assert sha256_file(path) == metadata["sha256"]
    annotations = pl.read_parquet(inventory_dir / "annotations.parquet")
    categories = pl.read_parquet(inventory_dir / "category_inventory.parquet")
    assert annotations.height == manifest["primary_records"] == 5_317_286
    assert set(annotations["chrom"].unique()) == set(PRIMARY_CHROMS)
    return manifest, annotations, categories


def merge_intervals(frame: pl.DataFrame) -> pl.DataFrame:
    assert frame.height > 0
    rows: list[dict[str, int | str]] = []
    by_chrom = {
        current["chrom"][0]: current.sort("start0", "end0")
        for current in frame.select("chrom", "start0", "end0").partition_by(
            "chrom", maintain_order=True
        )
    }
    for chrom in PRIMARY_CHROMS:
        current = by_chrom.get(chrom)
        if current is None:
            continue
        starts = current["start0"].to_numpy()
        ends = current["end0"].to_numpy()
        cumulative_end = np.maximum.accumulate(ends)
        is_new = np.concatenate([np.array([True]), starts[1:] > cumulative_end[:-1]])
        first_indices = np.flatnonzero(is_new)
        last_indices = np.concatenate([first_indices[1:] - 1, [starts.size - 1]])
        for start0, end0 in zip(
            starts[first_indices], cumulative_end[last_indices], strict=True
        ):
            assert 0 <= start0 < end0
            rows.append({"chrom": chrom, "start0": int(start0), "end0": int(end0)})
    result = pl.DataFrame(rows)
    assert result.height > 0
    return result


def sample_unique_offsets(total: int, size: int, *, namespace: str) -> list[int]:
    assert total >= size > 0
    rng = np.random.default_rng(stable_seed(namespace))
    selected: list[int] = []
    seen: set[int] = set()
    while len(selected) < size:
        remaining = size - len(selected)
        draws = rng.integers(0, total, size=max(32, remaining * 2), endpoint=False)
        for value in draws:
            integer = int(value)
            if integer not in seen:
                seen.add(integer)
                selected.append(integer)
                if len(selected) == size:
                    break
    return selected


def sample_interval_positions(
    intervals: pl.DataFrame, size: int, *, namespace: str
) -> list[tuple[str, int]]:
    assert intervals.height > 0 and size > 0
    lengths = (intervals["end0"] - intervals["start0"]).to_numpy()
    assert np.all(lengths > 0)
    cumulative = np.cumsum(lengths, dtype=np.int64)
    total = int(cumulative[-1])
    offsets = np.array(
        sample_unique_offsets(total, min(size, total), namespace=namespace),
        dtype=np.int64,
    )
    indices = np.searchsorted(cumulative, offsets, side="right")
    previous = np.where(indices == 0, 0, cumulative[indices - 1])
    positions = intervals["start0"].to_numpy()[indices] + offsets - previous
    chroms = intervals["chrom"].to_numpy()[indices]
    result = [
        (str(chrom), int(pos0)) for chrom, pos0 in zip(chroms, positions, strict=True)
    ]
    assert len(result) == len(set(result))
    return result


def repeat_free_focal_intervals(
    repeat_union: pl.DataFrame, chrom_lengths: Mapping[str, int]
) -> pl.DataFrame:
    rows: list[dict[str, int | str]] = []
    by_chrom = {
        current["chrom"][0]: current
        for current in repeat_union.partition_by("chrom", maintain_order=True)
    }
    for chrom in PRIMARY_CHROMS:
        previous_end = 0
        current = by_chrom.get(chrom)
        if current is not None:
            for start0, end0 in current.select("start0", "end0").iter_rows():
                focal_start = previous_end + FOCAL_INDEX
                focal_end = int(start0) - FOCAL_INDEX
                if focal_start < focal_end:
                    rows.append(
                        {"chrom": chrom, "start0": focal_start, "end0": focal_end}
                    )
                previous_end = max(previous_end, int(end0))
        focal_start = previous_end + FOCAL_INDEX
        focal_end = int(chrom_lengths[chrom]) - FOCAL_INDEX
        if focal_start < focal_end:
            rows.append({"chrom": chrom, "start0": focal_start, "end0": focal_end})
    result = pl.DataFrame(rows)
    assert result.height > 0
    return result


def selected_categories(categories: pl.DataFrame) -> dict[str, list[str]]:
    eligible = categories.filter(
        (pl.col("record_count") >= MIN_CATEGORY_RECORDS)
        & (pl.col("raw_annotated_bp") >= MIN_CATEGORY_RAW_BP)
    )
    classes = (
        eligible.filter(pl.col("level") == "class").sort("label")["label"].to_list()
    )
    families = (
        eligible.filter(pl.col("level") == "family").sort("label")["label"].to_list()
    )
    subfamilies = eligible.filter(pl.col("level") == "subfamily").sort(
        "raw_annotated_bp", "label", descending=[True, False]
    )
    rows = subfamilies.to_dicts()
    chosen = {row["label"] for row in rows[:GLOBAL_SUBFAMILIES]}
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        parts = str(row["label"]).split("|", maxsplit=2)
        assert len(parts) == 3
        by_family["|".join(parts[:2])].append(row)
    for family in families:
        chosen.update(
            row["label"] for row in by_family[family][:SUBFAMILIES_PER_FAMILY]
        )
    order = {
        row["label"]: (-int(row["raw_annotated_bp"]), row["label"]) for row in rows
    }
    subfamily_labels = sorted(chosen, key=order.__getitem__)
    assert len(classes) == EXPECTED_CLASS_COUNT
    assert len(families) == EXPECTED_FAMILY_COUNT
    assert len(subfamily_labels) == EXPECTED_SUBFAMILY_COUNT
    return {"class": classes, "family": families, "subfamily": subfamily_labels}


def add_hierarchy_labels(annotations: pl.DataFrame) -> pl.DataFrame:
    return annotations.with_columns(
        pl.concat_str("repeat_class", "repeat_family", separator="|").alias(
            "family_label"
        ),
        pl.concat_str(
            "repeat_class", "repeat_family", "repeat_name", separator="|"
        ).alias("subfamily_label"),
    )


def annotation_index(annotations: pl.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for current in annotations.partition_by("chrom", maintain_order=True):
        chrom = str(current["chrom"][0])
        ordered = current.sort("start0", "end0", "annotation_id")
        result[chrom] = {name: ordered[name].to_numpy() for name in ordered.columns}
    assert result and set(result) <= set(PRIMARY_CHROMS)
    return result


def annotate_points(
    points: Iterable[tuple[str, int]], annotations: pl.DataFrame
) -> dict[tuple[str, int], dict[str, Any] | None]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for chrom, pos0 in set(points):
        grouped[chrom].append(pos0)
    index = annotation_index(annotations)
    result: dict[tuple[str, int], dict[str, Any] | None] = {}
    for chrom, positions in grouped.items():
        data = index[chrom]
        starts = data["start0"]
        ends = data["end0"]
        active: set[int] = set()
        end_heap: list[tuple[int, int]] = []
        cursor = 0
        for pos0 in sorted(positions):
            while cursor < starts.size and int(starts[cursor]) <= pos0:
                active.add(cursor)
                heapq.heappush(end_heap, (int(ends[cursor]), cursor))
                cursor += 1
            while end_heap and end_heap[0][0] <= pos0:
                _, index_value = heapq.heappop(end_heap)
                active.discard(index_value)
            overlapping = [idx for idx in active if int(ends[idx]) > pos0]
            if not overlapping:
                result[(chrom, pos0)] = None
                continue
            overlapping.sort(key=lambda idx: int(data["annotation_id"][idx]))
            primary = min(
                overlapping,
                key=lambda idx: (
                    -int(data["sw_score"][idx]),
                    int(data["milli_div"][idx]),
                    -(int(ends[idx]) - int(starts[idx])),
                    int(data["annotation_id"][idx]),
                ),
            )
            result[(chrom, pos0)] = {
                "annotation_id": int(data["annotation_id"][primary]),
                "start0": int(starts[primary]),
                "end0": int(ends[primary]),
                "sw_score": int(data["sw_score"][primary]),
                "milli_div": int(data["milli_div"][primary]),
                "repeat_strand": str(data["strand"][primary]),
                "repeat_name": str(data["repeat_name"][primary]),
                "repeat_class": str(data["repeat_class"][primary]),
                "repeat_family": str(data["repeat_family"][primary]),
                "family_label": str(data["family_label"][primary]),
                "subfamily_label": str(data["subfamily_label"][primary]),
                "boundary_distance": min(
                    pos0 - int(starts[primary]), int(ends[primary]) - 1 - pos0
                ),
                "overlap_count": len(overlapping),
                "overlap_annotation_ids": [
                    int(data["annotation_id"][idx]) for idx in overlapping
                ],
                "overlap_subfamilies": [
                    str(data["subfamily_label"][idx]) for idx in overlapping
                ],
            }
    return result


def sequence_metrics(sequence: str) -> dict[str, int | float]:
    assert len(sequence) == WINDOW_BP and set(sequence) <= NUCLEOTIDES
    counts = np.array([sequence.count(base) for base in "ACGT"], dtype=np.float64)
    probabilities = counts[counts > 0] / WINDOW_BP
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    max_homopolymer = 1
    run = 1
    for previous, current in itertools.pairwise(sequence):
        run = run + 1 if current == previous else 1
        max_homopolymer = max(max_homopolymer, run)
    gc_count = int(counts[1] + counts[2])
    return {
        "gc_count": gc_count,
        "gc_fraction": gc_count / WINDOW_BP,
        "cpg_count": sequence.count("CG"),
        "shannon_entropy": entropy,
        "max_homopolymer": max_homopolymer,
    }


def repeat_covered_bp(
    chrom: str,
    start0: int,
    end0: int,
    union_index: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> int:
    starts, ends = union_index[chrom]
    cursor = int(np.searchsorted(ends, start0, side="right"))
    covered = 0
    while cursor < starts.size and int(starts[cursor]) < end0:
        covered += max(
            0, min(end0, int(ends[cursor])) - max(start0, int(starts[cursor]))
        )
        cursor += 1
    assert 0 <= covered <= end0 - start0
    return covered


def build_candidate_contexts(
    points: Iterable[tuple[str, int]],
    *,
    point_annotations: Mapping[tuple[str, int], dict[str, Any] | None],
    fasta: pysam.FastaFile,
    chrom_lengths: Mapping[str, int],
    repeat_union: pl.DataFrame,
) -> dict[tuple[str, int], dict[str, Any]]:
    union_index = {
        str(current["chrom"][0]): (
            current["start0"].to_numpy(),
            current["end0"].to_numpy(),
        )
        for current in repeat_union.partition_by("chrom", maintain_order=True)
    }
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for chrom, pos0 in sorted(
        set(points), key=lambda item: (CHROM_RANK[item[0]], item[1])
    ):
        start0, end0 = pos0 - FOCAL_INDEX, pos0 + FOCAL_INDEX + 1
        if start0 < 0 or end0 > chrom_lengths[chrom]:
            continue
        sequence = fasta.fetch(chrom, start0, end0).upper()
        if len(sequence) != WINDOW_BP or not set(sequence) <= NUCLEOTIDES:
            continue
        annotation = point_annotations.get((chrom, pos0))
        covered = repeat_covered_bp(chrom, start0, end0, union_index)
        row: dict[str, Any] = {
            "chrom": chrom,
            "pos0": pos0,
            "start0": start0,
            "end0": end0,
            "sequence": sequence,
            "is_repeat": annotation is not None,
            "repeat_fraction": covered / WINDOW_BP,
            **sequence_metrics(sequence),
        }
        if annotation is None:
            row.update(
                {
                    "annotation_id": None,
                    "primary_start0": None,
                    "primary_end0": None,
                    "sw_score": None,
                    "milli_div": None,
                    "repeat_strand": None,
                    "repeat_name": None,
                    "repeat_class": None,
                    "repeat_family": None,
                    "family_label": None,
                    "subfamily_label": None,
                    "boundary_distance": None,
                    "overlap_count": 0,
                    "overlap_annotation_ids": [],
                    "overlap_subfamilies": [],
                }
            )
        else:
            row.update(annotation)
            row["primary_start0"] = row.pop("start0")
            row["primary_end0"] = row.pop("end0")
            row["start0"] = start0
            row["end0"] = end0
        rows[(chrom, pos0)] = row
    return rows


def gc_edges(rows: Iterable[dict[str, Any]]) -> list[int]:
    values = np.array([int(row["gc_count"]) for row in rows])
    assert values.size == UNIFORM_REPEAT_CONTEXTS
    edges = np.quantile(values, np.linspace(0, 1, 11), method="nearest").astype(int)
    assert np.all(edges[:-1] <= edges[1:])
    return edges.tolist()


def gc_bin(gc_count: int, edges: list[int]) -> int:
    assert len(edges) == 11 and 0 <= gc_count <= WINDOW_BP
    return int(np.searchsorted(np.array(edges[1:-1]), gc_count, side="right"))


def match_rows(
    positives: list[dict[str, Any]],
    candidates: Iterable[dict[str, Any]],
    *,
    namespace: str,
    different_level: str | None = None,
    different_label: str | None = None,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    positive_coords = {(row["chrom"], row["pos0"]) for row in positives}
    for row in candidates:
        coord = (row["chrom"], row["pos0"])
        if coord in positive_coords:
            continue
        if different_level is not None:
            assert different_label is not None
            column = {
                "class": "repeat_class",
                "family": "family_label",
                "subfamily": "subfamily_label",
            }[different_level]
            if row[column] == different_label:
                continue
        buckets[(row["chrom"], int(row["gc_bin"]))].append(row)
    for key, values in buckets.items():
        values.sort(
            key=lambda row: selection_hash(namespace, row["chrom"], row["pos0"])
        )
    matched: list[dict[str, Any]] = []
    offsets: dict[tuple[str, int], int] = defaultdict(int)
    for positive in positives:
        key = (positive["chrom"], int(positive["gc_bin"]))
        offset = offsets[key]
        assert offset < len(buckets[key]), (namespace, key, offset, len(buckets[key]))
        matched.append(buckets[key][offset])
        offsets[key] += 1
    assert len(matched) == len(positives)
    assert len({(row["chrom"], row["pos0"]) for row in matched}) == len(matched)
    return matched


def category_label(row: Mapping[str, Any], level: str) -> str | None:
    return {
        "class": row["repeat_class"],
        "family": row["family_label"],
        "subfamily": row["subfamily_label"],
    }[level]


def materialize(
    *, inventory_dir: Path, fasta_path: Path, output_dir: Path
) -> dict[str, Any]:
    assert not output_dir.exists()
    commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(commit)
    assert os.environ.get("RUN_ID") == PANEL_RUN_ID
    inventory_manifest, annotations, category_inventory = validate_inventory(
        inventory_dir
    )
    chrom_lengths = read_fai(Path(f"{fasta_path}.fai"))
    assert Path(f"{fasta_path}.gzi").is_file()
    annotations = add_hierarchy_labels(annotations)
    repeat_union = merge_intervals(annotations)
    assert (
        int((repeat_union["end0"] - repeat_union["start0"]).sum())
        == inventory_manifest["repeat_union_bp"]
    )
    control_intervals = repeat_free_focal_intervals(repeat_union, chrom_lengths)
    chosen = selected_categories(category_inventory)

    uniform_repeat_candidates = sample_interval_positions(
        repeat_union,
        UNIFORM_REPEAT_CONTEXTS * 2,
        namespace=f"{PANEL_RUN_ID}|uniform-repeat",
    )
    control_candidates: list[tuple[str, int]] = []
    repeat_counts = defaultdict(int)
    for chrom, _ in uniform_repeat_candidates:
        repeat_counts[chrom] += 1
    for chrom in PRIMARY_CHROMS:
        current = control_intervals.filter(pl.col("chrom") == chrom)
        if repeat_counts[chrom] == 0:
            continue
        control_candidates.extend(
            sample_interval_positions(
                current,
                repeat_counts[chrom] * 8,
                namespace=f"{PANEL_RUN_ID}|uniform-control|{chrom}",
            )
        )

    category_candidates: dict[tuple[str, str], list[tuple[str, int]]] = {}
    label_columns = {
        "class": "repeat_class",
        "family": "family_label",
        "subfamily": "subfamily_label",
    }
    for level, labels in chosen.items():
        column = label_columns[level]
        selected_annotations = annotations.filter(pl.col(column).is_in(labels))
        by_label = {
            str(current[column][0]): current
            for current in selected_annotations.partition_by(
                column, maintain_order=True
            )
        }
        assert set(by_label) == set(labels)
        for label in labels:
            intervals = merge_intervals(by_label[label])
            category_candidates[(level, label)] = sample_interval_positions(
                intervals,
                CATEGORY_CONTEXTS * 2,
                namespace=f"{PANEL_RUN_ID}|category|{level}|{label}",
            )

    repeat_candidate_points = set(uniform_repeat_candidates)
    for values in category_candidates.values():
        repeat_candidate_points.update(values)
    point_annotations = annotate_points(repeat_candidate_points, annotations)
    for point in control_candidates:
        point_annotations[point] = None
    fasta = pysam.FastaFile(str(fasta_path))
    assert set(PRIMARY_CHROMS) <= set(fasta.references)
    candidate_rows = build_candidate_contexts(
        repeat_candidate_points | set(control_candidates),
        point_annotations=point_annotations,
        fasta=fasta,
        chrom_lengths=chrom_lengths,
        repeat_union=repeat_union,
    )
    fasta.close()

    uniform_repeat = [
        candidate_rows[point]
        for point in uniform_repeat_candidates
        if point in candidate_rows and candidate_rows[point]["is_repeat"]
    ][:UNIFORM_REPEAT_CONTEXTS]
    assert len(uniform_repeat) == UNIFORM_REPEAT_CONTEXTS
    edges = gc_edges(uniform_repeat)
    for row in candidate_rows.values():
        row["gc_bin"] = gc_bin(int(row["gc_count"]), edges)
    valid_controls = [
        candidate_rows[point]
        for point in control_candidates
        if point in candidate_rows
        and not candidate_rows[point]["is_repeat"]
        and candidate_rows[point]["repeat_fraction"] == 0.0
    ]
    uniform_controls = match_rows(
        uniform_repeat,
        valid_controls,
        namespace=f"{PANEL_RUN_ID}|uniform-control-match",
    )
    assert len(uniform_controls) == UNIFORM_CONTROL_CONTEXTS

    category_positives: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, points in category_candidates.items():
        level, label = key
        positives = [
            candidate_rows[point]
            for point in points
            if point in candidate_rows
            and candidate_rows[point]["is_repeat"]
            and category_label(candidate_rows[point], level) == label
        ][:CATEGORY_CONTEXTS]
        assert len(positives) == CATEGORY_CONTEXTS, (key, len(positives))
        category_positives[key] = positives

    repeat_pool_by_coord = {
        (row["chrom"], row["pos0"]): row
        for row in uniform_repeat
        + [row for values in category_positives.values() for row in values]
    }
    repeat_pool = list(repeat_pool_by_coord.values())
    category_negatives = {
        key: match_rows(
            positives,
            repeat_pool,
            namespace=f"{PANEL_RUN_ID}|negative|{key[0]}|{key[1]}",
            different_level=key[0],
            different_label=key[1],
        )
        for key, positives in category_positives.items()
    }

    selected_rows_by_coord = {
        (row["chrom"], row["pos0"]): row
        for row in uniform_repeat
        + uniform_controls
        + [row for values in category_positives.values() for row in values]
        + [row for values in category_negatives.values() for row in values]
    }
    selected_rows = sorted(
        selected_rows_by_coord.values(),
        key=lambda row: (CHROM_RANK[row["chrom"]], row["pos0"]),
    )
    for context_id, row in enumerate(selected_rows):
        row["context_id"] = context_id
    context_ids = {
        (row["chrom"], row["pos0"]): row["context_id"] for row in selected_rows
    }
    contexts = pl.DataFrame(selected_rows).select(
        "context_id",
        "chrom",
        "pos0",
        "start0",
        "end0",
        "sequence",
        "is_repeat",
        "annotation_id",
        "primary_start0",
        "primary_end0",
        "sw_score",
        "milli_div",
        "repeat_strand",
        "repeat_name",
        "repeat_class",
        "repeat_family",
        "family_label",
        "subfamily_label",
        "boundary_distance",
        "overlap_count",
        "overlap_annotation_ids",
        "overlap_subfamilies",
        "gc_count",
        "gc_fraction",
        "gc_bin",
        "cpg_count",
        "shannon_entropy",
        "max_homopolymer",
        "repeat_fraction",
    )
    uniform_pairs = pl.DataFrame(
        [
            {
                "pair_id": pair_id,
                "repeat_context_id": context_ids[(repeat["chrom"], repeat["pos0"])],
                "control_context_id": context_ids[(control["chrom"], control["pos0"])],
                "chrom": repeat["chrom"],
                "gc_bin": repeat["gc_bin"],
            }
            for pair_id, (repeat, control) in enumerate(
                zip(uniform_repeat, uniform_controls, strict=True)
            )
        ]
    )
    comparison_rows: list[dict[str, Any]] = []
    for level in ("class", "family", "subfamily"):
        for label in chosen[level]:
            key = (level, label)
            for pair_id, (positive, negative) in enumerate(
                zip(category_positives[key], category_negatives[key], strict=True)
            ):
                assert positive["chrom"] == negative["chrom"]
                assert positive["gc_bin"] == negative["gc_bin"]
                comparison_rows.append(
                    {
                        "level": level,
                        "label": label,
                        "pair_id": pair_id,
                        "positive_context_id": context_ids[
                            (positive["chrom"], positive["pos0"])
                        ],
                        "negative_context_id": context_ids[
                            (negative["chrom"], negative["pos0"])
                        ],
                        "chrom": positive["chrom"],
                        "gc_bin": positive["gc_bin"],
                    }
                )
    comparisons = pl.DataFrame(comparison_rows)
    assert uniform_pairs.height == UNIFORM_REPEAT_CONTEXTS
    assert (
        comparisons.height
        == (EXPECTED_CLASS_COUNT + EXPECTED_FAMILY_COUNT + EXPECTED_SUBFAMILY_COUNT)
        * CATEGORY_CONTEXTS
    )
    assert contexts["context_id"].to_list() == list(range(contexts.height))
    assert contexts["sequence"].str.len_chars().unique().to_list() == [WINDOW_BP]
    assert contexts.filter(
        ~pl.col("is_repeat") & (pl.col("repeat_fraction") != 0)
    ).is_empty()

    output_dir.mkdir(parents=True)
    context_path = output_dir / "contexts.parquet"
    uniform_path = output_dir / "uniform_pairs.parquet"
    comparison_path = output_dir / "category_comparisons.parquet"
    category_path = output_dir / "selected_categories.parquet"
    contexts.write_parquet(context_path, compression="zstd")
    uniform_pairs.write_parquet(uniform_path, compression="zstd")
    comparisons.write_parquet(comparison_path, compression="zstd")
    selected_category_rows = [
        {"level": level, "label": label, "contexts": CATEGORY_CONTEXTS}
        for level in ("class", "family", "subfamily")
        for label in chosen[level]
    ]
    pl.DataFrame(selected_category_rows).write_parquet(
        category_path, compression="zstd"
    )
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": PANEL_RUN_ID,
        "experiment_commit": commit,
        "analysis_status": "outcome_blind_reference_panel",
        "seed": SEED,
        "coordinate_system": "0-based half-open",
        "window_bp": WINDOW_BP,
        "focal_index": FOCAL_INDEX,
        "inventory": {
            "s3": INVENTORY_S3,
            "archive_manifest_sha256": INVENTORY_MANIFEST_SHA256,
            "manifest_sha256": sha256_file(inventory_dir / "manifest.json"),
        },
        "fasta": {
            "s3": FASTA_S3,
            "bytes": fasta_path.stat().st_size,
            "sha256": sha256_file(fasta_path),
            "fai_sha256": sha256_file(Path(f"{fasta_path}.fai")),
            "gzi_sha256": sha256_file(Path(f"{fasta_path}.gzi")),
        },
        "gc_edges": edges,
        "contexts": contexts.height,
        "repeat_contexts": contexts.filter(pl.col("is_repeat")).height,
        "repeat_free_contexts": contexts.filter(~pl.col("is_repeat")).height,
        "uniform_pairs": uniform_pairs.height,
        "category_comparisons": comparisons.height,
        "category_counts": {level: len(labels) for level, labels in chosen.items()},
        "category_support": {
            "minimum_records": MIN_CATEGORY_RECORDS,
            "minimum_raw_annotated_bp": MIN_CATEGORY_RAW_BP,
            "contexts_per_category": CATEGORY_CONTEXTS,
            "global_subfamilies": GLOBAL_SUBFAMILIES,
            "subfamilies_per_family": SUBFAMILIES_PER_FAMILY,
        },
        "overlap_policy": (
            "highest sw_score, then lower milli_div, longer interval, stable annotation_id"
        ),
    }
    result_path = output_dir / "results.json"
    write_json(result_path, result)
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (
            context_path,
            uniform_path,
            comparison_path,
            category_path,
            result_path,
        )
    }
    manifest = {**result, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    materialize(
        inventory_dir=args.inventory_dir,
        fasta_path=args.fasta,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
