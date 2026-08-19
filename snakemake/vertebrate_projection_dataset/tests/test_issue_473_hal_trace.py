from __future__ import annotations

from pathlib import Path

import polars as pl
from marin_dna_vertebrate_projection.issue_473.hal_trace import (
    parse_named_psl,
    write_hal_trace_bed,
    write_hal_trace_metrics,
    write_hal_trace_sample,
)


def _accepted(policy: str, *, t_start: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_name": ["anchor"],
            "source_chrom": ["chr7"],
            "source_start": [1000],
            "source_end": [1255],
            "region_label": ["cds"],
            "species": ["Mus musculus"],
            "alignment_name": ["Mus_musculus"],
            "clade": ["mammals"],
            "alignment_source": ["zoonomia_cactus"],
            "t_chrom": ["chr5"],
            "t_start": [t_start],
            "t_end": [t_start + 255],
            "t_strand": ["+"],
            "fragment_count": [1 if policy == "center_1" else 2],
        }
    )


def test_sample_and_bed_are_policy_paired_and_half_open(tmp_path: Path) -> None:
    full_path = tmp_path / "full.parquet"
    center_path = tmp_path / "center.parquet"
    _accepted("full_window", t_start=100).write_parquet(full_path)
    _accepted("center_1", t_start=110).write_parquet(center_path)
    sample_path = tmp_path / "sample.tsv"
    summary_path = tmp_path / "sample.json"
    write_hal_trace_sample(
        full_path,
        center_path,
        sample_path,
        summary_path,
        sample_modulus=1,
        core_threshold=1,
    )

    sample = pl.read_csv(sample_path, separator="\t")
    assert sample.height == 2
    assert sample["query_name"].n_unique() == 1
    assert sample["trace_id"].n_unique() == 2
    bed_path = tmp_path / "center.bed"
    write_hal_trace_bed(
        sample_path,
        bed_path,
        policy="center_1",
        alignment_name="Mus_musculus",
    )
    fields = bed_path.read_text().rstrip().split("\t")
    assert fields[:3] == ["chr5", "110", "365"]
    assert fields[4:] == ["0", "+"]


def test_named_psl_metrics_clip_to_emitted_window_and_human_anchor(
    tmp_path: Path,
) -> None:
    sample = pl.DataFrame(
        {
            "trace_id": ["trace_000000000"],
            "projection_policy": ["center_1"],
            "query_name": ["anchor"],
            "source_chrom": ["chr7"],
            "source_start": [1000],
            "source_end": [1255],
            "region_label": ["cds"],
            "species": ["Mus musculus"],
            "alignment_name": ["Mus_musculus"],
            "clade": ["mammals"],
            "t_chrom": ["chr5"],
            "t_start": [100],
            "t_end": [355],
            "t_strand": ["+"],
            "fragment_count": [1],
        }
    )
    sample_path = tmp_path / "sample.tsv"
    sample.write_csv(sample_path, separator="\t")
    psl_path = tmp_path / "trace.psl"
    psl_path.write_text(
        "trace_000000000\t220\t0\t0\t0\t0\t0\t0\t0\t++\t"
        "chr5\t1000\t110\t340\tchr7\t2000\t990\t1260\t2\t"
        "100,120,\t110,220,\t990,1140,\n"
    )
    output = tmp_path / "metrics.parquet"
    write_hal_trace_metrics(
        sample_path,
        psl_path,
        output,
        policy="center_1",
        alignment_name="Mus_musculus",
    )

    metric = pl.read_parquet(output).row(0, named=True)
    assert metric["psl_alignment_rows"] == 1
    assert metric["psl_blocks"] == 2
    assert metric["emitted_window_aligned_bases"] == 220
    assert metric["emitted_window_to_anchor_aligned_bases"] == 205
    assert metric["human_anchor_aligned_bases"] == 205
    assert metric["measurement_status"] == "measured_exact_named_psl"


def test_negative_query_psl_coordinates_are_normalized(tmp_path: Path) -> None:
    psl = tmp_path / "negative.psl"
    psl.write_text(
        "trace_negative\t100\t0\t0\t0\t0\t0\t0\t0\t-+\t"
        "chr5\t1000\t790\t890\tchr7\t2000\t1000\t1100\t1\t"
        "100,\t790,\t1000,\n"
    )
    record = parse_named_psl(psl)[0]
    assert record.strand == "-+"
    assert record.blocks == ((790, 1000, 100),)
