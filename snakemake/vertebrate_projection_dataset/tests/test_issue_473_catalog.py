from __future__ import annotations

from pathlib import Path

import polars as pl
from marin_dna_vertebrate_projection.issue_473.catalog import (
    ENHANCER_REGION,
    FIXED_REGIONS,
    build_fixed_scored_anchor_catalog,
    read_exp351_enhancer_anchors,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, list[Path], Path, Path]:
    labels = pl.DataFrame(
        {
            "name": ["cds-1", "utr3-1", "rna-1", "tss-1", "old-ccre"],
            "chrom": ["1", "2", "3", "4", "5"],
            "start": [0, 100, 200, 300, 400],
            "end": [255, 355, 455, 555, 655],
            "label": [
                "cds",
                "utr3",
                "ncrna_exon",
                "tss_region_and_utr5",
                "ccre_non_promoter",
            ],
        }
    )
    score = pl.DataFrame(
        {
            "name": labels["name"],
            "proportion_conserved": [0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    enhancer_score = pl.DataFrame(
        {
            "name": ["enh-1", "enh-2"],
            "proportion_conserved": [0.2, 0.9],
        }
    )
    labels_path = tmp_path / "labels.parquet"
    score_path = tmp_path / "chr1.parquet"
    enhancer_bed = tmp_path / "noexon.bed"
    enhancer_score_path = tmp_path / "enhancer-scored.parquet"
    labels.write_parquet(labels_path)
    score.write_parquet(score_path)
    enhancer_bed.write_text("7\t1000\t1255\tenh-1\nX\t2000\t2255\tenh-2\n")
    enhancer_score.write_parquet(enhancer_score_path)
    return labels_path, [score_path], enhancer_bed, enhancer_score_path


def test_read_exp351_enhancer_anchors_keeps_exact_rows(tmp_path: Path) -> None:
    _, _, enhancer_bed, enhancer_score = _write_inputs(tmp_path)
    result = read_exp351_enhancer_anchors(enhancer_bed, enhancer_score)

    assert result.height == 2
    assert set(result["source_chrom"]) == {"chr7", "chrX"}
    assert set(result["region_label"]) == {ENHANCER_REGION}
    assert (result["source_end"] - result["source_start"] == 255).all()


def test_fixed_catalog_replaces_old_ccre_with_exp351_population(
    tmp_path: Path,
) -> None:
    labels, scores, enhancer_bed, enhancer_score = _write_inputs(tmp_path)
    result = build_fixed_scored_anchor_catalog(
        labels,
        scores,
        enhancer_bed,
        enhancer_score,
        expected_enhancer_anchors=2,
    )

    assert result.height == 6
    assert set(result["region_label"]) == set(FIXED_REGIONS)
    assert "old-ccre" not in set(result["query_name"])
    assert {"enh-1", "enh-2"} <= set(result["query_name"])
    assert result["query_name"].n_unique() == result.height
