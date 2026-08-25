from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.gpn_star_anchors import (
    GPN_ARMS,
    GPN_ASSIGNMENT_RECIPE,
    assign_gpn_six_arms,
    read_gpn_entropy_manifest,
    score_gpn_entropy_windows,
    write_gpn_anchor_catalog,
    write_gpn_selection_outputs,
)


def _write_windows(path: Path, rows: list[tuple[str, int, int, str]]) -> None:
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write("\t".join(map(str, row)) + "\n")


def test_read_gpn_entropy_manifest_validates_and_indexes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "chrom\tpath\trows\tsize_bytes\tsha256\tmin_pos\tmax_pos\n"
        "chr1\tdata/gpn-star-hg38-p243-200m/entropy/entropy_chr1.parquet"
        f"\t10\t100\t{'a' * 64}\t1\t10\n"
    )
    shards = read_gpn_entropy_manifest(manifest)
    assert list(shards) == ["chr1"]
    assert shards["chr1"].rows == 10
    assert shards["chr1"].min_pos == 1


def test_score_gpn_entropy_windows_converts_one_based_positions_and_is_strict(
    tmp_path: Path,
) -> None:
    windows = tmp_path / "windows.bed.gz"
    entropy = tmp_path / "entropy.parquet"
    scored = tmp_path / "scored.parquet"
    stats = tmp_path / "stats.json"
    _write_windows(
        windows,
        [
            ("chr1", 0, 255, "win0"),
            ("chr1", 128, 383, "win1"),
            ("chr1", 256, 511, "win2"),
        ],
    )
    pl.DataFrame(
        {
            "chrom": ["1"] * 6,
            "pos": [1, 128, 129, 255, 256, 257],
            "ref": ["A"] * 6,
            "entropy_calibrated": [0.01, 0.01, 0.01, 0.01, 0.10, 0.01],
        }
    ).write_parquet(entropy)

    observed = score_gpn_entropy_windows(
        windows,
        entropy,
        scored,
        stats,
        chrom="chr1",
        entropy_cutoff=0.10,
        expected_rows=6,
        expected_min_pos=1,
        expected_max_pos=257,
        batch_size=2,
    )

    frame = pl.read_parquet(scored)
    assert frame["gpn_selected_bases"].to_list() == [4, 3, 1]
    assert observed["selected_source_positions"] == 5
    assert json.loads(stats.read_text()) == observed


def test_write_gpn_selection_outputs_reports_10_and_20_percent(tmp_path: Path) -> None:
    scored = tmp_path / "scored.parquet"
    stats = tmp_path / "source-stats.json"
    selected = tmp_path / "selected.parquet"
    bed = tmp_path / "selected.bed.gz"
    summary = tmp_path / "summary.json"
    pl.DataFrame(
        {
            "chrom": ["chr1"] * 4,
            "start": [0, 128, 256, 384],
            "end": [255, 383, 511, 639],
            "name": ["w0", "w1", "w2", "w3"],
            "gpn_selected_bases": [25, 26, 50, 51],
            "proportion_gpn_selected": [25 / 255, 26 / 255, 50 / 255, 51 / 255],
            "gpn_entropy_cutoff": [0.081001] * 4,
        }
    ).write_parquet(scored)
    stats.write_text(json.dumps({"chrom": "chr1", "selected_source_positions": 7}))

    observed = write_gpn_selection_outputs(
        [str(scored)],
        [str(stats)],
        selected,
        bed,
        summary,
        min_selected_bases=51,
        expected_uniform_windows=4,
        expected_selected_source_positions=7,
        expected_windows_ge_10pct=3,
        expected_windows_ge_20pct=1,
    )

    assert pl.read_parquet(selected)["name"].to_list() == ["w3"]
    with gzip.open(bed, "rt") as handle:
        assert handle.read() == "1\t384\t639\tw3\n"
    assert observed["totals"]["windows_ge_10pct"] == 3


def _assignment_labels() -> pl.DataFrame:
    labels = [
        "cds",
        "utr3",
        "tss_region_and_utr5",
        "ncrna_exon",
        "ccre_non_promoter",
        "ccre_non_promoter",
        "background",
    ]
    fractions = {
        "cds_frac": [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
        "utr3_frac": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "tss_region_and_utr5_frac": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "ncrna_exon_frac": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "ccre_non_promoter_frac": [0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0],
    }
    return pl.DataFrame(
        {
            "name": [f"w{index}" for index in range(len(labels))],
            "chrom": ["1"] * len(labels),
            "start": [index * 128 for index in range(len(labels))],
            "end": [index * 128 + 255 for index in range(len(labels))],
            "label": labels,
            "functional_frac": [1.0] * 6 + [0.0],
            **fractions,
            "gene_body_frac": [0.0] * len(labels),
            "intron_frac": [0.0] * len(labels),
            "intergenic_frac": [1.0] * len(labels),
        }
    )


def test_assign_gpn_six_arms_is_exhaustive_and_uses_arm_a() -> None:
    assigned = assign_gpn_six_arms(_assignment_labels())
    assert assigned["arm"].to_list() == [
        "cds",
        "utr3",
        "tss_region_and_utr5",
        "ncrna_exon",
        "enhancer",
        "background",
        "background",
    ]
    assert assigned["assignment_reason"].to_list()[-2:] == [
        "ccre_rejected_by_arm_a_to_remainder",
        "v4_background_to_remainder",
    ]
    assert assigned["assignment_recipe"].unique().to_list() == [
        GPN_ASSIGNMENT_RECIPE
    ]


def test_write_gpn_anchor_catalog_writes_versioned_assignments(tmp_path: Path) -> None:
    labels = _assignment_labels()
    labels_path = tmp_path / "labels.parquet"
    selected_path = tmp_path / "selected.parquet"
    catalog_path = tmp_path / "catalog.parquet"
    assignments_path = tmp_path / "assignments.parquet"
    summary_path = tmp_path / "summary.json"
    labels.write_parquet(labels_path)
    selected = labels.select("name", "start", "end").with_columns(
        pl.lit("chr1").alias("chrom"),
        pl.lit(51).alias("gpn_selected_bases"),
        pl.lit(0.2).alias("proportion_gpn_selected"),
        pl.lit(0.081001).alias("gpn_entropy_cutoff"),
    )
    selected.write_parquet(selected_path)

    observed = write_gpn_anchor_catalog(
        labels_path,
        selected_path,
        catalog_path,
        assignments_path,
        summary_path,
        score_set="gpn-star-hg38-p243-200m",
        dataset_revision="a" * 40,
        min_selected_bases=51,
        expected_full_count=labels.height,
        required_arms=list(GPN_ARMS),
    )

    catalog = pl.read_parquet(catalog_path)
    assignments = pl.read_parquet(assignments_path)
    assert catalog.height == assignments.height == labels.height
    assert set(catalog["region_label"].to_list()) == set(GPN_ARMS)
    assert assignments["assignment_recipe"].unique().to_list() == [
        GPN_ASSIGNMENT_RECIPE
    ]
    assert assignments["gpn_min_selected_bases"].unique().to_list() == [51]
    assert sum(row["len"] for row in observed["by_arm"]) == labels.height
    assert observed["assignment_arm_count_sum"] == labels.height
