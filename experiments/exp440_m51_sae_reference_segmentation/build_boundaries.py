"""Build the compact outcome-independent boundary panel for issue #440."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from marin_dna.data.utils import load_annotation
from marin_dna.pipelines.zoonomia_projection_dataset.validation import (
    filter_to_canonical_transcripts,
)
from twobitreader import TwoBitFile

from build_panel import WINDOW_BP, sha256_file, write_json
from extract_focal import ISSUE, assert_commit

BOUNDARY_PANEL_RUN_ID = "dna-exp440-boundary-panel-r1"
GTF_BYTES = 104_395_529
GTF_SHA256 = "2f8e31578c3aa2f35646927c4a3b3b0dcf0321e57c0ebd3ecc81afcbc836d1a8"
CCRE_BYTES = 10_361_229
CCRE_SHA256 = "35c322243a347ddcbcfac478c825ca9bb1af1cdfbbd876b64e788b5105a7afd5"
GENOME_BYTES = 812_795_740
GENOME_SHA256 = "c47af4db6aadd72efdafa926ddd0c5e185ba2109c816727b7c52fd956a928e27"
FLANK_BP = 32
TRANSCRIPT_CASES_PER_TYPE = 5
CCRE_CASES_PER_SUBTYPE = 3
PLS_CASES = 6
BOUNDARY_TYPES = (
    "cds_to_intron",
    "intron_to_cds",
    "utr5_to_cds",
    "cds_to_utr3",
    "els_edge",
    "pls_edge",
)


def _attribute(attribute: str, name: str) -> str | None:
    marker = f'{name} "'
    start = attribute.find(marker)
    if start < 0:
        return None
    value_start = start + len(marker)
    value_end = attribute.find('"', value_start)
    assert value_end >= 0
    return attribute[value_start:value_end]


def _contains(intervals: list[tuple[int, int]], start: int, end: int) -> bool:
    assert start < end
    return any(left <= start and end <= right for left, right in intervals)


def _aligned_segments(
    boundary_position0: int, direction: int, flank: int = FLANK_BP
) -> tuple[tuple[int, int], tuple[int, int]]:
    assert direction in (-1, 1) and flank > 0
    if direction == 1:
        return (
            (boundary_position0 - flank, boundary_position0),
            (boundary_position0, boundary_position0 + flank),
        )
    return (
        (boundary_position0, boundary_position0 + flank),
        (boundary_position0 - flank, boundary_position0),
    )


def gene_boundary_candidates(annotation: pl.DataFrame) -> list[dict[str, Any]]:
    """Derive exact canonical protein-coding transitions from 0-based GTF rows."""
    required = {"chrom", "feature", "start", "end", "strand", "attribute"}
    assert required <= set(annotation.columns)
    assert annotation.filter(~pl.col("strand").is_in(["+", "-"])).is_empty()

    transcript_info: dict[str, dict[str, str]] = {}
    exons: dict[str, list[tuple[int, int]]] = defaultdict(list)
    cds: dict[str, list[tuple[int, int]]] = defaultdict(list)
    utr5: dict[str, list[tuple[int, int]]] = defaultdict(list)
    utr3: dict[str, list[tuple[int, int]]] = defaultdict(list)
    feature_map = {
        "exon": exons,
        "CDS": cds,
        "five_prime_utr": utr5,
        "three_prime_utr": utr3,
    }
    for row in annotation.select(required).iter_rows(named=True):
        attribute = row["attribute"]
        transcript_id = _attribute(attribute, "transcript_id")
        if not transcript_id:
            continue
        transcript_biotype = _attribute(attribute, "transcript_biotype")
        if row["feature"] == "transcript":
            if transcript_biotype == "protein_coding":
                transcript_info[transcript_id] = {
                    "chrom": row["chrom"],
                    "strand": row["strand"],
                    "gene_id": _attribute(attribute, "gene_id") or "",
                    "gene_name": _attribute(attribute, "gene_name") or "",
                }
            continue
        if transcript_biotype != "protein_coding" or row["feature"] not in feature_map:
            continue
        assert row["start"] < row["end"]
        feature_map[row["feature"]][transcript_id].append(
            (int(row["start"]), int(row["end"]))
        )

    candidates: list[dict[str, Any]] = []

    def add(
        *,
        boundary_type: str,
        transcript_id: str,
        position: int,
        direction: int,
        state_before: str,
        state_after: str,
    ) -> None:
        info = transcript_info[transcript_id]
        candidates.append(
            {
                "boundary_type": boundary_type,
                "chrom": info["chrom"],
                "boundary_position0": position,
                "direction": direction,
                "strand": info["strand"],
                "gene_id": info["gene_id"],
                "gene_name": info["gene_name"],
                "transcript_id": transcript_id,
                "source_id": info["gene_id"],
                "ccre_subtype": None,
                "edge_side": None,
                "state_before": state_before,
                "state_after": state_after,
            }
        )

    for transcript_id, info in transcript_info.items():
        transcript_exons = sorted(set(exons.get(transcript_id, [])))
        transcript_cds = sorted(set(cds.get(transcript_id, [])))
        transcript_utr5 = sorted(set(utr5.get(transcript_id, [])))
        transcript_utr3 = sorted(set(utr3.get(transcript_id, [])))
        if not transcript_exons or not transcript_cds:
            continue
        direction = 1 if info["strand"] == "+" else -1
        ordered_exons = transcript_exons if direction == 1 else transcript_exons[::-1]
        for current, following in itertools.pairwise(ordered_exons):
            if direction == 1:
                donor = current[1]
                acceptor = following[0]
            else:
                donor = current[0]
                acceptor = following[1]
            before, after = _aligned_segments(donor, direction)
            if _contains(transcript_cds, *before):
                add(
                    boundary_type="cds_to_intron",
                    transcript_id=transcript_id,
                    position=donor,
                    direction=direction,
                    state_before="cds",
                    state_after="intron",
                )
            before, after = _aligned_segments(acceptor, direction)
            if _contains(transcript_cds, *after):
                add(
                    boundary_type="intron_to_cds",
                    transcript_id=transcript_id,
                    position=acceptor,
                    direction=direction,
                    state_before="intron",
                    state_after="cds",
                )

        cds_start = min(start for start, _ in transcript_cds)
        cds_end = max(end for _, end in transcript_cds)
        utr5_boundary = cds_start if direction == 1 else cds_end
        before, after = _aligned_segments(utr5_boundary, direction)
        if _contains(transcript_utr5, *before) and _contains(transcript_cds, *after):
            add(
                boundary_type="utr5_to_cds",
                transcript_id=transcript_id,
                position=utr5_boundary,
                direction=direction,
                state_before="utr5",
                state_after="cds",
            )
        utr3_boundary = cds_end if direction == 1 else cds_start
        before, after = _aligned_segments(utr3_boundary, direction)
        if _contains(transcript_cds, *before) and _contains(transcript_utr3, *after):
            add(
                boundary_type="cds_to_utr3",
                transcript_id=transcript_id,
                position=utr3_boundary,
                direction=direction,
                state_before="cds",
                state_after="utr3",
            )

    assert candidates
    assert {record["boundary_type"] for record in candidates} == set(BOUNDARY_TYPES[:4])
    return candidates


def ccre_edge_candidates(ccre: pl.DataFrame) -> list[dict[str, Any]]:
    required = {"chrom", "start", "end", "cre_class"}
    assert set(ccre.columns) == required
    result: list[dict[str, Any]] = []
    for chrom, start, end, subtype in ccre.filter(
        pl.col("cre_class").is_in(["pELS", "dELS", "PLS"])
    ).iter_rows():
        source_id = f"{chrom}:{start}-{end}:{subtype}"
        edge_hash = hashlib.blake2b(source_id.encode(), digest_size=8).digest()
        use_start = int.from_bytes(edge_hash, "big") % 2 == 0
        result.append(
            {
                "boundary_type": "pls_edge" if subtype == "PLS" else "els_edge",
                "chrom": chrom,
                "boundary_position0": start if use_start else end,
                "direction": 1 if use_start else -1,
                "strand": None,
                "gene_id": None,
                "gene_name": None,
                "transcript_id": None,
                "source_id": source_id,
                "ccre_subtype": subtype,
                "edge_side": "start" if use_start else "end",
                "state_before": "outside_ccre",
                "state_after": "els" if subtype in ("pELS", "dELS") else "pls",
            }
        )
    assert result
    return result


def stable_record_hash(record: Mapping[str, Any]) -> str:
    value = "|".join(
        str(record[key])
        for key in (
            "boundary_type",
            "source_id",
            "chrom",
            "boundary_position0",
            "direction",
        )
    )
    return hashlib.blake2b(value.encode(), digest_size=16).hexdigest()


def extract_context(
    genome: Mapping[str, Sequence[str]], *, chrom: str, boundary_position0: int
) -> tuple[int, int, str] | None:
    start = boundary_position0 - 127
    end = boundary_position0 + 128
    if chrom not in genome or start < 0 or end > len(genome[chrom]):
        return None
    sequence = str(genome[chrom][start:end]).upper()
    if len(sequence) != WINDOW_BP or not set(sequence) <= set("ACGT"):
        return None
    return start, end, sequence


def select_cases(
    candidates: list[dict[str, Any]],
    *,
    genome: Mapping[str, Sequence[str]],
    count: int,
    used_intervals: dict[str, list[tuple[int, int]]],
    used_sources: set[str],
) -> list[dict[str, Any]]:
    assert count > 0
    selected: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates, key=lambda item: (stable_record_hash(item), item["source_id"])
    ):
        source_id = candidate["source_id"]
        if not source_id or source_id in used_sources:
            continue
        context = extract_context(
            genome,
            chrom=candidate["chrom"],
            boundary_position0=candidate["boundary_position0"],
        )
        if context is None:
            continue
        start, end, sequence = context
        if any(
            start < old_end and old_start < end
            for old_start, old_end in used_intervals[candidate["chrom"]]
        ):
            continue
        record = dict(candidate)
        record.update(
            {
                "window_start": start,
                "window_end": end,
                "sequence": sequence,
                "selection_hash": stable_record_hash(candidate),
            }
        )
        selected.append(record)
        used_sources.add(source_id)
        used_intervals[candidate["chrom"]].append((start, end))
        if len(selected) == count:
            break
    assert len(selected) == count, (len(selected), count)
    return selected


def build_boundary_panel(
    *, gtf_path: Path, ccre_path: Path, genome_path: Path, output_dir: Path
) -> dict[str, Any]:
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    assert os.environ.get("RUN_ID") == BOUNDARY_PANEL_RUN_ID
    for path, expected_bytes, expected_sha in (
        (gtf_path, GTF_BYTES, GTF_SHA256),
        (ccre_path, CCRE_BYTES, CCRE_SHA256),
        (genome_path, GENOME_BYTES, GENOME_SHA256),
    ):
        assert path.is_file() and path.stat().st_size == expected_bytes
        assert sha256_file(path) == expected_sha

    annotation = filter_to_canonical_transcripts(load_annotation(str(gtf_path)))
    gene_candidates = gene_boundary_candidates(annotation)
    ccre = pl.read_parquet(ccre_path)
    ccre_candidates = ccre_edge_candidates(ccre)
    candidate_counts = dict(
        pl.DataFrame(
            gene_candidates + ccre_candidates,
            infer_schema_length=None,
        )
        .group_by("boundary_type")
        .len()
        .sort("boundary_type")
        .iter_rows()
    )
    genome = TwoBitFile(str(genome_path))
    used_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    used_sources: set[str] = set()
    selected: list[dict[str, Any]] = []
    for boundary_type in BOUNDARY_TYPES[:4]:
        selected.extend(
            select_cases(
                [
                    record
                    for record in gene_candidates
                    if record["boundary_type"] == boundary_type
                ],
                genome=genome,
                count=TRANSCRIPT_CASES_PER_TYPE,
                used_intervals=used_intervals,
                used_sources=used_sources,
            )
        )
    for subtype in ("pELS", "dELS"):
        selected.extend(
            select_cases(
                [
                    record
                    for record in ccre_candidates
                    if record["ccre_subtype"] == subtype
                ],
                genome=genome,
                count=CCRE_CASES_PER_SUBTYPE,
                used_intervals=used_intervals,
                used_sources=used_sources,
            )
        )
    selected.extend(
        select_cases(
            [record for record in ccre_candidates if record["ccre_subtype"] == "PLS"],
            genome=genome,
            count=PLS_CASES,
            used_intervals=used_intervals,
            used_sources=used_sources,
        )
    )
    panel = (
        pl.DataFrame(selected)
        .sort("boundary_type", "ccre_subtype", "selection_hash", nulls_last=True)
        .with_row_index("panel_row")
    )
    expected_counts = {
        **{
            boundary_type: TRANSCRIPT_CASES_PER_TYPE
            for boundary_type in BOUNDARY_TYPES[:4]
        },
        "els_edge": 2 * CCRE_CASES_PER_SUBTYPE,
        "pls_edge": PLS_CASES,
    }
    class_counts = dict(
        panel.group_by("boundary_type").len().sort("boundary_type").iter_rows()
    )
    assert class_counts == expected_counts
    assert panel.height == 32 and panel["source_id"].n_unique() == panel.height
    assert panel.filter(
        pl.col("window_end") - pl.col("window_start") != WINDOW_BP
    ).is_empty()
    assert panel.filter(pl.col("sequence").str.len_chars() != WINDOW_BP).is_empty()
    assert panel.filter(pl.col("sequence").str.contains("[^ACGT]")).is_empty()
    assert dict(
        panel.filter(pl.col("boundary_type") == "els_edge")
        .group_by("ccre_subtype")
        .len()
        .iter_rows()
    ) == {"pELS": 3, "dELS": 3}

    output_dir.mkdir(parents=True)
    panel_path = output_dir / "panel.parquet"
    panel.write_parquet(panel_path, compression="zstd")
    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": BOUNDARY_PANEL_RUN_ID,
        "analysis_status": "outcome_independent_compact_boundary_panel",
        "experiment_commit": experiment_commit,
        "coordinate_system": "0-based half-open; boundary is between bases",
        "window_bp": WINDOW_BP,
        "central_positions": [64, 191],
        "rows": panel.height,
        "class_counts": class_counts,
        "candidate_counts": candidate_counts,
        "sampling": "smallest BLAKE2b-128 hash per boundary type/source; distinct sources and non-overlapping contexts",
        "inputs": {
            "gtf": {"bytes": GTF_BYTES, "sha256": GTF_SHA256},
            "ccre": {"bytes": CCRE_BYTES, "sha256": CCRE_SHA256},
            "genome": {"bytes": GENOME_BYTES, "sha256": GENOME_SHA256},
        },
        "panel": {
            "bytes": panel_path.stat().st_size,
            "sha256": sha256_file(panel_path),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--ccre", type=Path, required=True)
    parser.add_argument("--genome", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_boundary_panel(
        gtf_path=args.gtf,
        ccre_path=args.ccre,
        genome_path=args.genome,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
