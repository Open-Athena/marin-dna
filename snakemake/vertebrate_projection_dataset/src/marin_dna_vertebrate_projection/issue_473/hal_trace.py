"""Reproducible sampled HAL traces for emitted-window alignment coverage."""

from __future__ import annotations

import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import polars as pl

TRACE_POLICIES = ("full_window", "center_1")
TRACE_METRIC_SCHEMA = pl.Schema(
    {
        "trace_id": pl.String,
        "projection_policy": pl.String,
        "query_name": pl.String,
        "source_chrom": pl.String,
        "source_start": pl.Int64,
        "source_end": pl.Int64,
        "region_label": pl.String,
        "species": pl.String,
        "alignment_name": pl.String,
        "clade": pl.String,
        "t_chrom": pl.String,
        "t_start": pl.Int64,
        "t_end": pl.Int64,
        "t_strand": pl.String,
        "fragment_count": pl.Int64,
        "psl_alignment_rows": pl.Int64,
        "psl_blocks": pl.Int64,
        "off_expected_locus_rows": pl.Int64,
        "emitted_window_aligned_bases": pl.Int64,
        "emitted_window_aligned_fraction": pl.Float64,
        "emitted_window_to_anchor_aligned_bases": pl.Int64,
        "emitted_window_to_anchor_aligned_fraction": pl.Float64,
        "human_anchor_aligned_bases": pl.Int64,
        "human_anchor_aligned_fraction": pl.Float64,
        "measurement_status": pl.String,
    }
)


@dataclass(frozen=True)
class NamedPslRecord:
    """One ``halLiftover --outPSLWithName`` row."""

    trace_id: str
    strand: str
    q_name: str
    q_size: int
    t_name: str
    t_size: int
    blocks: tuple[tuple[int, int, int], ...]


def _policy_sample_rows(
    path: str | Path,
    policy: str,
    *,
    seed: int,
    sample_modulus: int,
    core_threshold: int,
) -> pl.DataFrame:
    assert policy in TRACE_POLICIES
    rows = pl.scan_parquet(path).filter(pl.col("alignment_source") == "zoonomia_cactus")
    identity_hash = pl.concat_str(
        [pl.col("query_name"), pl.col("alignment_name")], separator="\t"
    ).hash(seed=seed)
    hashed = rows.with_columns(identity_hash.alias("_trace_hash"))
    is_zrs = (
        (pl.col("source_chrom") == "chr7")
        & (pl.col("source_start") < 156_793_500)
        & (pl.col("source_end") > 156_791_000)
    )
    is_core_sample = (pl.col("_trace_hash") % sample_modulus) < core_threshold
    is_fragment_sample = (pl.col("fragment_count") > 1) & (
        (pl.col("_trace_hash") % 100_000) < max(core_threshold, 1)
    )
    return (
        hashed.filter(is_core_sample | is_fragment_sample | is_zrs)
        .select(
            pl.lit(policy).alias("projection_policy"),
            "query_name",
            "source_chrom",
            "source_start",
            "source_end",
            "region_label",
            "species",
            "alignment_name",
            "clade",
            "t_chrom",
            "t_start",
            "t_end",
            "t_strand",
            "fragment_count",
        )
        .collect(engine="streaming")
    )


def write_hal_trace_sample(
    full_window_path: str | Path,
    center_1_path: str | Path,
    sample_path: str | Path,
    summary_path: str | Path,
    *,
    seed: int = 473,
    sample_modulus: int = 1_000_000,
    core_threshold: int = 8,
) -> None:
    """Select a bounded, stable sample without a genome-wide sort.

    The same hash key is used for both policies, so accepted intersections are
    sampled as pairs. All accepted ZRS rows are retained, while a separate
    sparse hash sample ensures fragmented HAL mappings are represented.
    """
    assert sample_modulus > 0
    assert 0 < core_threshold <= sample_modulus
    sample = pl.concat(
        [
            _policy_sample_rows(
                full_window_path,
                "full_window",
                seed=seed,
                sample_modulus=sample_modulus,
                core_threshold=core_threshold,
            ),
            _policy_sample_rows(
                center_1_path,
                "center_1",
                seed=seed,
                sample_modulus=sample_modulus,
                core_threshold=core_threshold,
            ),
        ],
        how="vertical",
    ).unique(subset=["projection_policy", "query_name", "alignment_name"])
    assert sample.height > 0, "deterministic HAL trace sample is empty"
    assert (sample["source_end"] - sample["source_start"] == 255).all()
    assert (sample["t_end"] - sample["t_start"] == 255).all()
    assert sample["t_strand"].is_in(["+", "-"]).all()
    sample = sample.sort(
        "projection_policy",
        "region_label",
        "query_name",
        "alignment_name",
    ).with_columns(
        pl.Series(
            "trace_id",
            [f"trace_{index:09d}" for index in range(sample.height)],
            dtype=pl.String,
        )
    )
    sample = sample.select("trace_id", pl.all().exclude("trace_id"))

    sample_output = Path(sample_path)
    summary_output = Path(summary_path)
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    sample.write_csv(sample_output, separator="\t")
    counts = (
        sample.group_by("projection_policy", "region_label")
        .len(name="rows")
        .sort("projection_policy", "region_label")
        .to_dicts()
    )
    summary_output.write_text(
        json.dumps(
            {
                "seed": seed,
                "sample_modulus": sample_modulus,
                "core_threshold": core_threshold,
                "selection_method": (
                    "stable pair hash plus sparse fragmented sample plus all ZRS"
                ),
                "full_window_source": str(full_window_path),
                "center_1_source": str(center_1_path),
                "rows": sample.height,
                "unique_anchors": sample["query_name"].n_unique(),
                "counts": counts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_hal_trace_bed(
    sample_path: str | Path,
    output_path: str | Path,
    *,
    policy: str,
    alignment_name: str,
) -> None:
    """Write sampled emitted target windows as 0-based BED6."""
    assert policy in TRACE_POLICIES
    sample = pl.read_csv(sample_path, separator="\t")
    rows = sample.filter(
        (pl.col("projection_policy") == policy)
        & (pl.col("alignment_name") == alignment_name)
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.select(
        pl.col("t_chrom"),
        pl.col("t_start"),
        pl.col("t_end"),
        pl.col("trace_id"),
        pl.lit(0).alias("score"),
        pl.col("t_strand"),
    ).write_csv(output, separator="\t", include_header=False)


def run_named_psl_trace(
    hal_path: str | Path,
    source_species: str,
    source_bed: str | Path,
    output_path: str | Path,
    *,
    target_species: str = "Homo_sapiens",
) -> None:
    """Map target-species emitted windows back to human as named PSL."""
    bed = Path(source_bed)
    output = Path(output_path)
    assert bed.is_file()
    output.parent.mkdir(parents=True, exist_ok=True)
    if bed.stat().st_size == 0:
        output.write_text("")
        return
    subprocess.run(
        [
            "halLiftover",
            "--noDupes",
            "--outPSLWithName",
            str(hal_path),
            source_species,
            str(bed),
            target_species,
            str(output),
        ],
        check=True,
    )


def _comma_ints(value: str, expected: int) -> tuple[int, ...]:
    parsed = tuple(int(part) for part in value.rstrip(",").split(",") if part)
    assert len(parsed) == expected
    return parsed


def parse_named_psl(path: str | Path) -> list[NamedPslRecord]:
    """Parse headerless 22-column named PSL output from halLiftover."""
    records: list[NamedPslRecord] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line:
            continue
        fields = line.split("\t")
        assert len(fields) == 22, (
            f"named PSL line {line_number} has {len(fields)} columns, expected 22"
        )
        block_count = int(fields[18])
        block_sizes = _comma_ints(fields[19], block_count)
        q_starts = _comma_ints(fields[20], block_count)
        t_starts = _comma_ints(fields[21], block_count)
        assert block_count > 0 and all(size > 0 for size in block_sizes)
        strand = fields[9]
        assert len(strand) in {1, 2} and set(strand) <= {"+", "-"}
        records.append(
            NamedPslRecord(
                trace_id=fields[0],
                strand=strand,
                q_name=fields[10],
                q_size=int(fields[11]),
                t_name=fields[14],
                t_size=int(fields[15]),
                blocks=tuple(zip(q_starts, t_starts, block_sizes, strict=True)),
            )
        )
    return records


def _normalize_psl_start(
    raw_start: int, size: int, genome_size: int, strand: str
) -> int:
    assert strand in {"+", "-"}
    start = raw_start if strand == "+" else genome_size - (raw_start + size)
    assert 0 <= start < start + size <= genome_size
    return start


def _interval_union_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    total = 0
    sorted_intervals = sorted(intervals)
    current_start, current_end = sorted_intervals[0]
    for start, end in sorted_intervals[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + current_end - current_start


def _parameter_clip(
    block_start: int,
    block_size: int,
    strand: str,
    clip_start: int,
    clip_end: int,
) -> tuple[int, int] | None:
    overlap_start = max(block_start, clip_start)
    overlap_end = min(block_start + block_size, clip_end)
    if overlap_start >= overlap_end:
        return None
    if strand == "+":
        return overlap_start - block_start, overlap_end - block_start
    return (
        block_start + block_size - overlap_end,
        block_start + block_size - overlap_start,
    )


def _parameter_to_interval(
    block_start: int,
    block_size: int,
    strand: str,
    parameter_start: int,
    parameter_end: int,
) -> tuple[int, int]:
    if strand == "+":
        return block_start + parameter_start, block_start + parameter_end
    return (
        block_start + block_size - parameter_end,
        block_start + block_size - parameter_start,
    )


def _measurement(
    sample: dict[str, object], records: list[NamedPslRecord]
) -> dict[str, object]:
    emitted_start = int(sample["t_start"])
    emitted_end = int(sample["t_end"])
    anchor_start = int(sample["source_start"])
    anchor_end = int(sample["source_end"])
    assert emitted_end - emitted_start == anchor_end - anchor_start == 255
    emitted_intervals: list[tuple[int, int]] = []
    against_anchor_intervals: list[tuple[int, int]] = []
    human_intervals: list[tuple[int, int]] = []
    matched_rows = 0
    block_count = 0
    off_expected = 0
    for record in records:
        if (
            record.q_name != sample["t_chrom"]
            or record.t_name != sample["source_chrom"]
        ):
            off_expected += 1
            continue
        matched_rows += 1
        q_strand = record.strand[0]
        t_strand = record.strand[1] if len(record.strand) == 2 else "+"
        for raw_q_start, raw_t_start, size in record.blocks:
            block_count += 1
            q_start = _normalize_psl_start(raw_q_start, size, record.q_size, q_strand)
            t_start = _normalize_psl_start(raw_t_start, size, record.t_size, t_strand)
            q_clip = _parameter_clip(
                q_start, size, q_strand, emitted_start, emitted_end
            )
            if q_clip is None:
                continue
            emitted_intervals.append(
                _parameter_to_interval(q_start, size, q_strand, *q_clip)
            )
            t_clip = _parameter_clip(t_start, size, t_strand, anchor_start, anchor_end)
            if t_clip is None:
                continue
            overlap_start = max(q_clip[0], t_clip[0])
            overlap_end = min(q_clip[1], t_clip[1])
            if overlap_start < overlap_end:
                against_anchor_intervals.append(
                    _parameter_to_interval(
                        q_start,
                        size,
                        q_strand,
                        overlap_start,
                        overlap_end,
                    )
                )
                human_intervals.append(
                    _parameter_to_interval(
                        t_start,
                        size,
                        t_strand,
                        overlap_start,
                        overlap_end,
                    )
                )
    emitted_bases = _interval_union_length(emitted_intervals)
    against_anchor_bases = _interval_union_length(against_anchor_intervals)
    human_bases = _interval_union_length(human_intervals)
    assert 0 <= against_anchor_bases <= emitted_bases <= 255
    assert 0 <= human_bases <= 255
    status = (
        "no_reverse_mapping"
        if not records
        else "off_expected_locus"
        if not matched_rows
        else "no_emitted_window_overlap"
        if not emitted_bases
        else "measured_exact_named_psl"
    )
    return {
        **sample,
        "psl_alignment_rows": matched_rows,
        "psl_blocks": block_count,
        "off_expected_locus_rows": off_expected,
        "emitted_window_aligned_bases": emitted_bases,
        "emitted_window_aligned_fraction": emitted_bases / 255.0,
        "emitted_window_to_anchor_aligned_bases": against_anchor_bases,
        "emitted_window_to_anchor_aligned_fraction": against_anchor_bases / 255.0,
        "human_anchor_aligned_bases": human_bases,
        "human_anchor_aligned_fraction": human_bases / 255.0,
        "measurement_status": status,
    }


def write_hal_trace_metrics(
    sample_path: str | Path,
    psl_path: str | Path,
    output_path: str | Path,
    *,
    policy: str,
    alignment_name: str,
) -> None:
    """Write one exact-coverage record per sampled accepted policy row."""
    assert policy in TRACE_POLICIES
    sample = pl.read_csv(sample_path, separator="\t").filter(
        (pl.col("projection_policy") == policy)
        & (pl.col("alignment_name") == alignment_name)
    )
    by_trace: dict[str, list[NamedPslRecord]] = defaultdict(list)
    expected = set(sample["trace_id"].to_list())
    for record in parse_named_psl(psl_path):
        assert record.trace_id in expected, f"unexpected trace id {record.trace_id}"
        by_trace[record.trace_id].append(record)
    rows = [
        _measurement(row, by_trace[str(row["trace_id"])]) for row in sample.to_dicts()
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = (
        pl.DataFrame(rows).select(TRACE_METRIC_SCHEMA.names()).cast(TRACE_METRIC_SCHEMA)
        if rows
        else pl.DataFrame(schema=TRACE_METRIC_SCHEMA)
    )
    frame.write_parquet(output)


def _normal_interval(
    mean: float, standard_deviation: float, count: int
) -> tuple[float, float]:
    standard_error = standard_deviation / math.sqrt(count) if count else 0.0
    return mean - 1.96 * standard_error, mean + 1.96 * standard_error


def write_hal_trace_summary(
    metric_paths: list[str],
    metrics_path: str | Path,
    summary_path: str | Path,
    uncertainty_path: str | Path,
    report_path: str | Path,
) -> None:
    """Combine sampled traces and report anchor-clustered uncertainty."""
    assert metric_paths
    frame = pl.concat([pl.read_parquet(path) for path in metric_paths], how="vertical")
    assert frame.height > 0, "HAL trace produced no sampled metric rows"
    assert frame["trace_id"].n_unique() == frame.height
    frame = frame.sort("projection_policy", "region_label", "query_name", "species")
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(metrics_path)

    measured = pl.col("measurement_status") == "measured_exact_named_psl"
    summary = (
        frame.group_by("projection_policy", "region_label", "clade")
        .agg(
            pl.len().cast(pl.Int64).alias("sampled_rows"),
            measured.sum().cast(pl.Int64).alias("measured_rows"),
            pl.col("emitted_window_aligned_fraction")
            .mean()
            .alias("mean_emitted_window_aligned_fraction"),
            pl.col("emitted_window_to_anchor_aligned_fraction")
            .mean()
            .alias("mean_emitted_window_to_anchor_aligned_fraction"),
            pl.col("emitted_window_to_anchor_aligned_fraction")
            .median()
            .alias("median_emitted_window_to_anchor_aligned_fraction"),
            pl.col("emitted_window_to_anchor_aligned_fraction")
            .quantile(0.1)
            .alias("q10_emitted_window_to_anchor_aligned_fraction"),
            pl.col("emitted_window_to_anchor_aligned_fraction")
            .quantile(0.9)
            .alias("q90_emitted_window_to_anchor_aligned_fraction"),
        )
        .sort("projection_policy", "region_label", "clade")
    )
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(summary_path)

    per_anchor = frame.group_by("projection_policy", "region_label", "query_name").agg(
        pl.col("emitted_window_to_anchor_aligned_fraction").mean().alias("anchor_mean")
    )
    uncertainty_rows: list[dict[str, object]] = []
    for group in per_anchor.partition_by(
        ["projection_policy", "region_label"], maintain_order=True
    ):
        policy = str(group["projection_policy"][0])
        region = str(group["region_label"][0])
        count = group.height
        mean = float(group["anchor_mean"].mean())
        sd_value = group["anchor_mean"].std()
        sd = 0.0 if sd_value is None else float(sd_value)
        low, high = _normal_interval(mean, sd, count)
        uncertainty_rows.append(
            {
                "comparison": policy,
                "region_label": region,
                "n_anchors": count,
                "mean": mean,
                "sd": sd,
                "se": sd / math.sqrt(count),
                "normal_95ci_low": low,
                "normal_95ci_high": high,
                "metric": "emitted_window_to_anchor_aligned_fraction",
            }
        )

    keys = ["query_name", "species", "region_label"]
    full = frame.filter(pl.col("projection_policy") == "full_window").select(
        *keys,
        pl.col("emitted_window_to_anchor_aligned_fraction").alias("full_window"),
    )
    center = frame.filter(pl.col("projection_policy") == "center_1").select(
        *keys,
        pl.col("emitted_window_to_anchor_aligned_fraction").alias("center_1"),
    )
    deltas = full.join(center, on=keys, how="inner").with_columns(
        (pl.col("center_1") - pl.col("full_window")).alias("delta")
    )
    per_anchor_delta = deltas.group_by("region_label", "query_name").agg(
        pl.col("delta").mean().alias("anchor_mean")
    )
    for group in per_anchor_delta.partition_by("region_label", maintain_order=True):
        region = str(group["region_label"][0])
        count = group.height
        mean = float(group["anchor_mean"].mean())
        sd_value = group["anchor_mean"].std()
        sd = 0.0 if sd_value is None else float(sd_value)
        low, high = _normal_interval(mean, sd, count)
        uncertainty_rows.append(
            {
                "comparison": "center_1_minus_full_window",
                "region_label": region,
                "n_anchors": count,
                "mean": mean,
                "sd": sd,
                "se": sd / math.sqrt(count),
                "normal_95ci_low": low,
                "normal_95ci_high": high,
                "metric": "paired_emitted_window_to_anchor_aligned_fraction",
            }
        )
    uncertainty = pl.DataFrame(uncertainty_rows).sort("comparison", "region_label")
    Path(uncertainty_path).parent.mkdir(parents=True, exist_ok=True)
    uncertainty.write_parquet(uncertainty_path)

    lines = [
        "# Issue #473 sampled HAL alignment trace",
        "",
        (
            "Exact base coverage was reproduced with `halLiftover "
            "--outPSLWithName` from each emitted target window back to the "
            "original human anchor. Coordinates are 0-based and half-open."
        ),
        "",
        (
            "Genome-wide paired diagnostics remain explicitly unavailable; these "
            "statistics apply only to the deterministic trace sample."
        ),
        "",
        "| policy | region | clade | sampled | measured | mean emitted aligned | mean emitted-to-anchor aligned |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            "| {projection_policy} | {region_label} | {clade} | {sampled_rows} | "
            "{measured_rows} | {mean_emitted_window_aligned_fraction:.4f} | "
            "{mean_emitted_window_to_anchor_aligned_fraction:.4f} |".format(**row)
        )
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines) + "\n")
