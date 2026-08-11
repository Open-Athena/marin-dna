"""Tests for ``marin_dna_zoonomia_projection.projection.tss``."""

from pathlib import Path

import polars as pl
import pytest

from marin_dna.data.utils import load_annotation
from marin_dna_zoonomia_projection.projection.tss import (
    get_ensembl_protein_coding_exons,
    write_mrna_tss_band_bed,
)


def _write_gtf(path: Path, rows: list[str]) -> None:
    """Write a tiny Ensembl-format GTF (1-based, tab-separated)."""
    path.write_text("\n".join(rows) + "\n")


def _gtf_row(
    chrom: str,
    feature: str,
    start_1based: int,
    end_1based: int,
    strand: str,
    attribute: str,
) -> str:
    return (
        f"{chrom}\tensembl\t{feature}\t{start_1based}\t{end_1based}"
        f"\t.\t{strand}\t.\t{attribute}"
    )


def test_get_ensembl_protein_coding_exons_filters_biotype(tmp_path: Path) -> None:
    """Only ``transcript_biotype == 'protein_coding'`` exons survive."""
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            # protein_coding mRNA: keep
            _gtf_row(
                "1",
                "exon",
                1001,
                1100,
                "+",
                'transcript_id "ENST_pc"; transcript_biotype "protein_coding";',
            ),
            # lncRNA: drop
            _gtf_row(
                "1",
                "exon",
                2001,
                2100,
                "+",
                'transcript_id "ENST_lnc"; transcript_biotype "lncRNA";',
            ),
            # processed_pseudogene: drop
            _gtf_row(
                "1",
                "exon",
                3001,
                3100,
                "+",
                'transcript_id "ENST_psd"; transcript_biotype "processed_pseudogene";',
            ),
            # NCBI-style gbkey "mRNA" without transcript_biotype "protein_coding": drop
            _gtf_row(
                "1",
                "exon",
                4001,
                4100,
                "+",
                'transcript_id "NM_xyz"; gbkey "mRNA";',
            ),
            # protein_coding on minus strand: keep
            _gtf_row(
                "2",
                "exon",
                5001,
                5050,
                "-",
                'transcript_id "ENST_pc_minus"; transcript_biotype "protein_coding";',
            ),
            # transcript-level (not exon): drop
            _gtf_row(
                "1",
                "transcript",
                1001,
                1100,
                "+",
                'transcript_id "ENST_pc"; transcript_biotype "protein_coding";',
            ),
        ],
    )
    ann = load_annotation(str(gtf))
    exons = get_ensembl_protein_coding_exons(ann)

    assert set(exons["transcript_id"]) == {"ENST_pc", "ENST_pc_minus"}
    assert exons.columns == ["chrom", "start", "end", "strand", "transcript_id"]
    # Coordinates should be 0-based (load_annotation does the conversion).
    keep_pos = exons.filter(pl.col("transcript_id") == "ENST_pc").row(0, named=True)
    assert keep_pos["start"] == 1000  # 1001 (1-based) → 1000 (0-based)
    assert keep_pos["end"] == 1100
    assert keep_pos["strand"] == "+"


def test_write_mrna_tss_band_bed_symmetric_flank(tmp_path: Path) -> None:
    """+ strand: band = [start - flank, start + flank]; − strand: [end - flank, end + flank]."""
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            # + strand: TSS at 5000 (1-based start) → 4999 (0-based)
            _gtf_row(
                "1",
                "exon",
                5000,
                5500,
                "+",
                'transcript_id "T_plus"; transcript_biotype "protein_coding";',
            ),
            # − strand: TSS is the rightmost exon end (genomic).
            # 1-based [3000, 4000] inclusive → 0-based [2999, 4000) → TSS at 4000.
            _gtf_row(
                "2",
                "exon",
                3000,
                4000,
                "-",
                'transcript_id "T_minus"; transcript_biotype "protein_coding";',
            ),
        ],
    )
    out_bed = tmp_path / "band.bed"
    n = write_mrna_tss_band_bed(gtf, flank=256, out_bed=out_bed)
    assert n == 2

    bed = pl.read_csv(
        str(out_bed),
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end"],
        schema_overrides={"chrom": pl.Utf8},
    )
    by_chrom = {r["chrom"]: r for r in bed.iter_rows(named=True)}

    # + strand: TSS=4999 → [4999 - 256, 4999 + 256] = [4743, 5255]
    assert by_chrom["1"]["start"] == 4743
    assert by_chrom["1"]["end"] == 5255
    assert by_chrom["1"]["end"] - by_chrom["1"]["start"] == 512

    # − strand: TSS=4000 → [4000 - 256, 4000 + 256] = [3744, 4256]
    assert by_chrom["2"]["start"] == 3744
    assert by_chrom["2"]["end"] == 4256
    assert by_chrom["2"]["end"] - by_chrom["2"]["start"] == 512


def test_write_mrna_tss_band_bed_merges_overlapping(tmp_path: Path) -> None:
    """Two transcripts with overlapping ± flank bands collapse to one row."""
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            # TSS at 1000 (0-based): band [744, 1256]
            _gtf_row(
                "1",
                "exon",
                1001,
                1500,
                "+",
                'transcript_id "T1"; transcript_biotype "protein_coding";',
            ),
            # TSS at 1100 (0-based): band [844, 1356] — overlaps T1's band
            _gtf_row(
                "1",
                "exon",
                1101,
                1600,
                "+",
                'transcript_id "T2"; transcript_biotype "protein_coding";',
            ),
        ],
    )
    out_bed = tmp_path / "band.bed"
    n = write_mrna_tss_band_bed(gtf, flank=256, out_bed=out_bed)
    assert n == 1, "overlapping bands should merge"

    bed = pl.read_csv(
        str(out_bed),
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end"],
        schema_overrides={"chrom": pl.Utf8},
    )
    row = bed.row(0, named=True)
    # Merged span = union of [744, 1256] and [844, 1356] = [744, 1356]
    assert row["chrom"] == "1"
    assert row["start"] == 744
    assert row["end"] == 1356


def test_write_mrna_tss_band_bed_empty_protein_coding_raises(tmp_path: Path) -> None:
    """Annotation with no protein_coding exons fires the loud assertion."""
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            _gtf_row(
                "1",
                "exon",
                1001,
                1100,
                "+",
                'transcript_id "ENST_lnc"; transcript_biotype "lncRNA";',
            ),
        ],
    )
    out_bed = tmp_path / "band.bed"
    with pytest.raises(AssertionError, match="no protein_coding exons"):
        write_mrna_tss_band_bed(gtf, flank=256, out_bed=out_bed)


def test_write_mrna_tss_band_bed_negative_flank_raises(tmp_path: Path) -> None:
    """Negative or zero flank fires the loud assertion."""
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            _gtf_row(
                "1",
                "exon",
                1001,
                1100,
                "+",
                'transcript_id "T1"; transcript_biotype "protein_coding";',
            ),
        ],
    )
    out_bed = tmp_path / "band.bed"
    with pytest.raises(AssertionError, match="flank must be positive"):
        write_mrna_tss_band_bed(gtf, flank=0, out_bed=out_bed)
    with pytest.raises(AssertionError, match="flank must be positive"):
        write_mrna_tss_band_bed(gtf, flank=-1, out_bed=out_bed)
