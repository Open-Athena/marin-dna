"""Measure sampled unaligned edge flanks from issue #473 named PSL traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_SIZE = 255
POLICIES = ("full_window", "center_1")
METRICS = ("left_flank", "right_flank", "external_flank", "internal_unaligned")


@dataclass(frozen=True)
class PslRecord:
    trace_id: str
    strand: str
    q_name: str
    q_size: int
    t_name: str
    t_size: int
    blocks: tuple[tuple[int, int, int], ...]


def _comma_ints(value: str, expected: int) -> tuple[int, ...]:
    result = tuple(int(part) for part in value.rstrip(",").split(",") if part)
    assert len(result) == expected
    return result


def parse_psl(path: Path) -> list[PslRecord]:
    """Parse the retained headerless 22-column ``--outPSLWithName`` output."""
    records: list[PslRecord] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            continue
        fields = line.split("\t")
        assert len(fields) == 22, f"{path}:{number}: expected 22 fields"
        count = int(fields[18])
        sizes = _comma_ints(fields[19], count)
        q_starts = _comma_ints(fields[20], count)
        t_starts = _comma_ints(fields[21], count)
        strand = fields[9]
        assert count > 0 and len(strand) in {1, 2}
        records.append(
            PslRecord(
                trace_id=fields[0],
                strand=strand,
                q_name=fields[10],
                q_size=int(fields[11]),
                t_name=fields[14],
                t_size=int(fields[15]),
                blocks=tuple(zip(q_starts, t_starts, sizes, strict=True)),
            )
        )
    return records


def _normalize(raw: int, size: int, genome_size: int, strand: str) -> int:
    start = raw if strand == "+" else genome_size - (raw + size)
    assert 0 <= start < start + size <= genome_size
    return start


def _clip_parameters(
    block_start: int, size: int, strand: str, start: int, end: int
) -> tuple[int, int] | None:
    left = max(block_start, start)
    right = min(block_start + size, end)
    if left >= right:
        return None
    if strand == "+":
        return left - block_start, right - block_start
    return block_start + size - right, block_start + size - left


def _from_parameters(
    block_start: int, size: int, strand: str, left: int, right: int
) -> tuple[int, int]:
    if strand == "+":
        return block_start + left, block_start + right
    return block_start + size - right, block_start + size - left


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = merged[-1][0], max(merged[-1][1], end)
    return merged


def _oriented_offsets(
    intervals: list[tuple[int, int]], start: int, end: int, strand: str
) -> list[tuple[int, int]]:
    offsets = []
    for left, right in intervals:
        assert start <= left < right <= end
        offsets.append(
            (left - start, right - start)
            if strand == "+"
            else (end - right, end - left)
        )
    return _merge(offsets)


def _measure(sample: pd.Series, records: list[PslRecord]) -> dict[str, object]:
    q_start, q_end = int(sample.t_start), int(sample.t_end)
    t_start, t_end = int(sample.source_start), int(sample.source_end)
    assert q_end - q_start == t_end - t_start == WINDOW_SIZE
    intervals: list[tuple[int, int]] = []
    matching = 0
    off_locus = 0
    for record in records:
        if record.q_name != sample.t_chrom or record.t_name != sample.source_chrom:
            off_locus += 1
            continue
        matching += 1
        q_strand = record.strand[0]
        t_strand = record.strand[1] if len(record.strand) == 2 else "+"
        for raw_q, raw_t, size in record.blocks:
            q_block = _normalize(raw_q, size, record.q_size, q_strand)
            t_block = _normalize(raw_t, size, record.t_size, t_strand)
            q_clip = _clip_parameters(q_block, size, q_strand, q_start, q_end)
            t_clip = _clip_parameters(t_block, size, t_strand, t_start, t_end)
            if q_clip is None or t_clip is None:
                continue
            left, right = max(q_clip[0], t_clip[0]), min(q_clip[1], t_clip[1])
            if left < right:
                intervals.append(
                    _from_parameters(q_block, size, q_strand, left, right)
                )
    offsets = _oriented_offsets(intervals, q_start, q_end, str(sample.t_strand))
    status = (
        "no_reverse_mapping"
        if not records
        else "off_expected_locus"
        if not matching
        else "no_anchor_overlap"
        if not offsets
        else "measured_exact_named_psl"
    )
    common = {
        key: sample[key]
        for key in (
            "trace_id",
            "projection_policy",
            "query_name",
            "region_label",
            "species",
            "alignment_name",
            "clade",
            "t_strand",
        )
    }
    if not offsets:
        return {
            **common,
            "matching_psl_rows": matching,
            "off_expected_locus_rows": off_locus,
            "aligned_to_anchor_bases": 0,
            "left_flank": np.nan,
            "right_flank": np.nan,
            "external_flank": np.nan,
            "internal_unaligned": np.nan,
            "center_base_aligned": pd.NA,
            "measurement_status": status,
        }
    aligned = sum(right - left for left, right in offsets)
    left_flank = offsets[0][0]
    right_flank = WINDOW_SIZE - offsets[-1][1]
    internal = WINDOW_SIZE - aligned - left_flank - right_flank
    assert min(aligned, left_flank, right_flank, internal) >= 0
    return {
        **common,
        "matching_psl_rows": matching,
        "off_expected_locus_rows": off_locus,
        "aligned_to_anchor_bases": aligned,
        "left_flank": left_flank,
        "right_flank": right_flank,
        "external_flank": left_flank + right_flank,
        "internal_unaligned": internal,
        "center_base_aligned": any(left <= 127 < right for left, right in offsets),
        "measurement_status": status,
    }


def measure_flanks(sample_path: Path, raw_root: Path) -> pd.DataFrame:
    sample = pd.read_csv(sample_path, sep="\t")
    assert set(sample.projection_policy) == set(POLICIES)
    assert sample.trace_id.is_unique
    rows: list[dict[str, object]] = []
    for (policy, alignment), group in sample.groupby(
        ["projection_policy", "alignment_name"], sort=True
    ):
        path = raw_root / str(policy) / f"{alignment}.psl"
        assert path.is_file(), f"missing retained trace {path}"
        by_trace: dict[str, list[PslRecord]] = defaultdict(list)
        for record in parse_psl(path):
            by_trace[record.trace_id].append(record)
        rows.extend(
            _measure(row, by_trace[str(row.trace_id)]) for _, row in group.iterrows()
        )
    result = pd.DataFrame(rows).sort_values("trace_id").reset_index(drop=True)
    result["center_base_aligned"] = result.center_base_aligned.astype("boolean")
    assert len(result) == len(sample) and result.trace_id.is_unique
    return result


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    measured = metrics[metrics.measurement_status == "measured_exact_named_psl"]
    per_anchor = measured.groupby(
        ["projection_policy", "region_label", "query_name"], as_index=False
    ).agg(
        left_flank=("left_flank", "mean"),
        right_flank=("right_flank", "mean"),
        external_flank=("external_flank", "mean"),
        internal_unaligned=("internal_unaligned", "mean"),
        center_aligned=("center_base_aligned", "mean"),
    )
    result = per_anchor.groupby(
        ["projection_policy", "region_label"], as_index=False
    ).agg(
        n_anchors=("query_name", "nunique"),
        mean_left_flank=("left_flank", "mean"),
        mean_right_flank=("right_flank", "mean"),
        mean_external_flank=("external_flank", "mean"),
        mean_internal_unaligned=("internal_unaligned", "mean"),
        center_base_aligned_fraction=("center_aligned", "mean"),
    )
    counts = measured.groupby(
        ["projection_policy", "region_label"], as_index=False
    ).size()
    return result.merge(counts.rename(columns={"size": "measured_rows"}))


def paired_deltas(
    metrics: pd.DataFrame, *, n_bootstrap: int, seed: int
) -> pd.DataFrame:
    measured = metrics[metrics.measurement_status == "measured_exact_named_psl"]
    keys = ["region_label", "query_name", "alignment_name"]
    columns = keys + list(METRICS)
    full = measured[measured.projection_policy == "full_window"][columns]
    center = measured[measured.projection_policy == "center_1"][columns]
    pairs = full.merge(center, on=keys, suffixes=("_full", "_center"))
    assert not pairs.empty and n_bootstrap > 0
    rows = []
    for region_index, (region, cell) in enumerate(
        pairs.groupby("region_label", sort=True)
    ):
        for metric_index, metric in enumerate(METRICS):
            delta = cell[f"{metric}_center"] - cell[f"{metric}_full"]
            values = (
                pd.DataFrame({"query_name": cell.query_name, "delta": delta})
                .groupby("query_name")
                .delta.mean()
                .to_numpy()
            )
            cell_seed = seed + region_index * 100 + metric_index
            rng = np.random.default_rng(cell_seed)
            draws = values[
                rng.integers(0, len(values), size=(n_bootstrap, len(values)))
            ].mean(axis=1)
            rows.append(
                {
                    "region_label": region,
                    "metric": metric,
                    "delta_center_minus_full": values.mean(),
                    "ci_low": np.quantile(draws, 0.025),
                    "ci_high": np.quantile(draws, 0.975),
                    "probability_center_lower": np.mean(draws < 0),
                    "n_pairs": len(cell),
                    "n_anchors": len(values),
                    "n_bootstrap": n_bootstrap,
                    "bootstrap_unit": "human_anchor",
                    "seed": cell_seed,
                }
            )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_analysis(
    sample_path: Path,
    raw_root: Path,
    output_dir: Path,
    *,
    analysis_commit: str,
    n_bootstrap: int,
    seed: int,
) -> None:
    assert len(analysis_commit) == 40 and set(analysis_commit) <= set(
        "0123456789abcdef"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = measure_flanks(sample_path, raw_root)
    summary = summarize(metrics)
    deltas = paired_deltas(metrics, n_bootstrap=n_bootstrap, seed=seed)
    outputs = {
        "metrics.parquet": metrics,
        "summary.parquet": summary,
        "paired_deltas.parquet": deltas,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output_dir / name, index=False)
    report = output_dir / "report.md"
    report.write_text(
        "# Issue #473 sampled unaligned flanks\n\n"
        "Flanks are human-oriented contiguous unaligned bases at each emitted "
        "window edge. Paired deltas are center 1 minus full window.\n\n"
        + summary.to_markdown(index=False)
        + "\n\n"
        + deltas.to_markdown(index=False)
        + "\n"
    )
    manifest = {
        "analysis_commit": analysis_commit,
        "coordinate_system": "0-based half-open",
        "source_kind": "deterministic sampled HAL named PSL trace",
        "raw_psl_files": len(list(raw_root.rglob("*.psl"))),
        "rows": len(metrics),
        "n_bootstrap": n_bootstrap,
        "bootstrap_unit": "human_anchor",
        "seed": seed,
        "outputs": {
            name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for name in (*outputs, "report.md")
            if (path := output_dir / name)
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=473)
    args = parser.parse_args()
    run_analysis(
        args.sample,
        args.raw_root,
        args.output_dir,
        analysis_commit=args.analysis_commit,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
