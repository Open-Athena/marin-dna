"""Tests for ``marin_dna_zoonomia_projection.validation``."""

from pathlib import Path

import pandas as pd
import polars as pl
import pyBigWig
import pytest

from marin_dna.data.intervals import GenomicSet
from marin_dna.data.utils import load_annotation
from marin_dna_zoonomia_projection.validation import (
    build_annotation_region,
    build_cre_region,
    build_tss_band_region,
    case_encode_sequences,
    filter_to_canonical_transcripts,
    get_ensembl_3_prime_utr,
    get_ensembl_5_prime_utr,
    get_ensembl_ncrna_exons,
    subsample_deterministic,
    write_hf_readme,
)

# ----------------------------------------------------------------------------
# Synthetic GTF helpers (mirror tests/projection/test_tss.py)
# ----------------------------------------------------------------------------


def _write_gtf(path: Path, rows: list[str]) -> None:
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


# ----------------------------------------------------------------------------
# filter_to_canonical_transcripts
# ----------------------------------------------------------------------------


def test_filter_to_canonical_keeps_descendant_rows(tmp_path: Path) -> None:
    """Keep all rows whose transcript_id is canonical; drop the rest."""
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            # Canonical transcript T1 + its exon + CDS
            _gtf_row(
                "1",
                "transcript",
                1001,
                2000,
                "+",
                'transcript_id "T1"; tag "Ensembl_canonical"; transcript_biotype "protein_coding";',
            ),
            _gtf_row(
                "1",
                "exon",
                1001,
                1500,
                "+",
                'transcript_id "T1"; transcript_biotype "protein_coding";',
            ),
            _gtf_row(
                "1",
                "CDS",
                1100,
                1400,
                "+",
                'transcript_id "T1"; transcript_biotype "protein_coding";',
            ),
            # Non-canonical transcript T2 + its exon: dropped
            _gtf_row(
                "1",
                "transcript",
                3001,
                4000,
                "+",
                'transcript_id "T2"; transcript_biotype "protein_coding";',
            ),
            _gtf_row(
                "1",
                "exon",
                3001,
                3500,
                "+",
                'transcript_id "T2"; transcript_biotype "protein_coding";',
            ),
        ],
    )
    ann = load_annotation(str(gtf))
    canonical = filter_to_canonical_transcripts(ann)

    transcript_ids = canonical["attribute"].str.extract(r'transcript_id "(.*?)"')
    keep = set(transcript_ids.drop_nulls().unique().to_list())
    assert keep == {"T1"}
    # transcript + exon + CDS rows for T1 = 3 rows
    assert len(canonical) == 3
    feats = set(canonical["feature"].to_list())
    assert feats == {"transcript", "exon", "CDS"}


def test_filter_to_canonical_no_canonical_raises(tmp_path: Path) -> None:
    """GTF with zero canonical-tagged transcripts fires the loud assertion."""
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            _gtf_row(
                "1",
                "transcript",
                1001,
                2000,
                "+",
                'transcript_id "T1"; transcript_biotype "protein_coding";',
            ),
        ],
    )
    ann = load_annotation(str(gtf))
    with pytest.raises(AssertionError, match="no transcripts tagged"):
        filter_to_canonical_transcripts(ann)


def test_filter_to_canonical_drops_gene_rows(tmp_path: Path) -> None:
    """Gene-feature rows have no transcript_id, so they're filtered out."""
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            _gtf_row("1", "gene", 1001, 2000, "+", 'gene_id "G1";'),
            _gtf_row(
                "1",
                "transcript",
                1001,
                2000,
                "+",
                'transcript_id "T1"; tag "Ensembl_canonical"; transcript_biotype "protein_coding";',
            ),
            _gtf_row(
                "1",
                "exon",
                1001,
                1500,
                "+",
                'transcript_id "T1"; transcript_biotype "protein_coding";',
            ),
        ],
    )
    ann = load_annotation(str(gtf))
    canonical = filter_to_canonical_transcripts(ann)
    feats = set(canonical["feature"].to_list())
    assert "gene" not in feats
    assert feats == {"transcript", "exon"}


# ----------------------------------------------------------------------------
# get_ensembl_5_prime_utr / get_ensembl_3_prime_utr
# ----------------------------------------------------------------------------


def _coding_transcript_rows(
    chrom: str,
    transcript_id: str,
    exon_start_1based: int,
    exon_end_1based: int,
    cds_start_1based: int,
    cds_end_1based: int,
    strand: str,
) -> list[str]:
    """Build a minimal protein_coding transcript: one exon + nested CDS."""
    attrs = f'transcript_id "{transcript_id}"; transcript_biotype "protein_coding";'
    return [
        _gtf_row(chrom, "exon", exon_start_1based, exon_end_1based, strand, attrs),
        _gtf_row(chrom, "CDS", cds_start_1based, cds_end_1based, strand, attrs),
    ]


def test_get_ensembl_5_prime_utr_protein_coding_plus_strand(tmp_path: Path) -> None:
    """+ strand: 5' UTR = exon segment from exon_start to cds_start."""
    gtf = tmp_path / "ann.gtf"
    rows = _coding_transcript_rows(
        chrom="1",
        transcript_id="T1",
        exon_start_1based=1001,
        exon_end_1based=2000,
        cds_start_1based=1101,
        cds_end_1based=1900,
        strand="+",
    )
    _write_gtf(gtf, rows)
    ann = load_annotation(str(gtf))
    utr = get_ensembl_5_prime_utr(ann)
    df = utr.to_pandas()
    assert len(df) == 1
    # 0-based: exon = [1000, 2000), CDS = [1100, 1900). 5' UTR = [1000, 1100).
    assert df.iloc[0]["start"] == 1000
    assert df.iloc[0]["end"] == 1100


def test_get_ensembl_5_prime_utr_handles_minus_strand(tmp_path: Path) -> None:
    """- strand: 5' UTR = exon segment from cds_end to exon_end (genomic right)."""
    gtf = tmp_path / "ann.gtf"
    rows = _coding_transcript_rows(
        chrom="1",
        transcript_id="T1_minus",
        exon_start_1based=1001,
        exon_end_1based=2000,
        cds_start_1based=1101,
        cds_end_1based=1900,
        strand="-",
    )
    _write_gtf(gtf, rows)
    ann = load_annotation(str(gtf))
    utr = get_ensembl_5_prime_utr(ann)
    df = utr.to_pandas()
    assert len(df) == 1
    # - strand: 5' UTR is genomically-rightmost exon segment past CDS end.
    # 0-based: exon = [1000, 2000), CDS = [1100, 1900). 5' UTR = [1900, 2000).
    assert df.iloc[0]["start"] == 1900
    assert df.iloc[0]["end"] == 2000


def test_get_ensembl_3_prime_utr_plus_strand(tmp_path: Path) -> None:
    """+ strand: 3' UTR = exon segment from cds_end to exon_end."""
    gtf = tmp_path / "ann.gtf"
    rows = _coding_transcript_rows(
        chrom="1",
        transcript_id="T1",
        exon_start_1based=1001,
        exon_end_1based=2000,
        cds_start_1based=1101,
        cds_end_1based=1900,
        strand="+",
    )
    _write_gtf(gtf, rows)
    ann = load_annotation(str(gtf))
    utr = get_ensembl_3_prime_utr(ann)
    df = utr.to_pandas()
    assert len(df) == 1
    # 0-based: 3' UTR on + = [1900, 2000).
    assert df.iloc[0]["start"] == 1900
    assert df.iloc[0]["end"] == 2000


def test_get_ensembl_5_prime_utr_lncrna_excluded(tmp_path: Path) -> None:
    """An lncRNA transcript with no CDS yields no 5' UTR."""
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            # lncRNA: has exon but no CDS — should not appear in 5' UTR set
            _gtf_row(
                "1",
                "exon",
                1001,
                2000,
                "+",
                'transcript_id "T_lnc"; transcript_biotype "lncRNA";',
            ),
            # Add a protein_coding transcript to make sure the function runs
            *_coding_transcript_rows(
                chrom="2",
                transcript_id="T_pc",
                exon_start_1based=1001,
                exon_end_1based=2000,
                cds_start_1based=1101,
                cds_end_1based=1900,
                strand="+",
            ),
        ],
    )
    ann = load_annotation(str(gtf))
    utr = get_ensembl_5_prime_utr(ann)
    df = utr.to_pandas()
    # Only the protein_coding transcript on chrom 2 contributes
    assert set(df["chrom"]) == {"2"}


# ----------------------------------------------------------------------------
# get_ensembl_ncrna_exons
# ----------------------------------------------------------------------------


def test_get_ensembl_ncrna_exons_keeps_lncrna(tmp_path: Path) -> None:
    """Ensembl lncRNA biotype is matched when present in the biotypes list."""
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
                'transcript_id "T_lnc"; gene_biotype "lncRNA"; transcript_biotype "lncRNA";',
            ),
            # mRNA exon should be excluded
            _gtf_row(
                "1",
                "exon",
                2001,
                2100,
                "+",
                'transcript_id "T_pc"; gene_biotype "protein_coding"; transcript_biotype "protein_coding";',
            ),
        ],
    )
    ann = load_annotation(str(gtf))
    nc = get_ensembl_ncrna_exons(ann, biotypes=["lncRNA", "miRNA"])
    df = nc.to_pandas()
    assert len(df) == 1
    assert df.iloc[0]["start"] == 1000


def test_get_ensembl_ncrna_exons_excludes_pseudogene(tmp_path: Path) -> None:
    """Pseudogene biotypes are filtered out via the gene_biotype guard."""
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
                'transcript_id "T_lnc"; gene_biotype "lncRNA"; transcript_biotype "lncRNA";',
            ),
            # processed_pseudogene gene_biotype: should be filtered
            _gtf_row(
                "1",
                "exon",
                2001,
                2100,
                "+",
                'transcript_id "T_psd"; gene_biotype "processed_pseudogene"; transcript_biotype "lncRNA";',
            ),
        ],
    )
    ann = load_annotation(str(gtf))
    nc = get_ensembl_ncrna_exons(ann, biotypes=["lncRNA"])
    df = nc.to_pandas()
    assert len(df) == 1
    assert df.iloc[0]["start"] == 1000


def test_get_ensembl_ncrna_exons_biotype_vocabulary_ensembl_only(
    tmp_path: Path,
) -> None:
    """Passing the Ensembl biotype list rejects the RefSeq ``"lnc_RNA"`` spelling."""
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            # RefSeq-style biotype: "lnc_RNA" (with underscore)
            _gtf_row(
                "1",
                "exon",
                1001,
                1100,
                "+",
                'transcript_id "T_refseq"; gene_biotype "lnc_RNA"; transcript_biotype "lnc_RNA";',
            ),
            # Ensembl-style biotype: "lncRNA"
            _gtf_row(
                "1",
                "exon",
                2001,
                2100,
                "+",
                'transcript_id "T_ensembl"; gene_biotype "lncRNA"; transcript_biotype "lncRNA";',
            ),
        ],
    )
    ann = load_annotation(str(gtf))
    nc = get_ensembl_ncrna_exons(ann, biotypes=["lncRNA"])
    df = nc.to_pandas()
    assert len(df) == 1
    # Only the Ensembl-style row survived
    assert df.iloc[0]["start"] == 2000


# ----------------------------------------------------------------------------
# build_annotation_region
# ----------------------------------------------------------------------------


def test_build_annotation_region_no_cross_recipe_subtraction(tmp_path: Path) -> None:
    """val_cds and val_utr5 retain their own bases — no cross-recipe subtraction.

    A canonical protein_coding transcript with adjacent 5' UTR and CDS:
    after add_flank(20), val_cds extends 20 bp upstream into the 5' UTR
    region, and val_utr5 extends 20 bp downstream into the CDS region.
    Both recipes preserve their own bases (the 20 bp shared positions appear
    in both).
    """
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            # Canonical transcript: exon spans CDS + 5' UTR + 3' UTR.
            _gtf_row(
                "1",
                "transcript",
                1001,
                3000,
                "+",
                'transcript_id "T1"; tag "Ensembl_canonical"; transcript_biotype "protein_coding";',
            ),
            _gtf_row(
                "1",
                "exon",
                1001,
                3000,
                "+",
                'transcript_id "T1"; transcript_biotype "protein_coding";',
            ),
            _gtf_row(
                "1",
                "CDS",
                1501,
                2500,
                "+",
                'transcript_id "T1"; transcript_biotype "protein_coding";',
            ),
        ],
    )
    ann = load_annotation(str(gtf))
    canonical = filter_to_canonical_transcripts(ann)

    # Defined region covers the whole locus: [0, 4000)
    defined = GenomicSet(pd.DataFrame({"chrom": ["1"], "start": [0], "end": [4000]}))

    cds_region = build_annotation_region(
        "val_cds",
        canonical,
        defined,
        ncrna_biotypes=["lncRNA"],
        add_flank=20,
        min_size=20,
        max_size=10_000,
        expand_min=255,
    )
    utr5_region = build_annotation_region(
        "val_utr5",
        canonical,
        defined,
        ncrna_biotypes=["lncRNA"],
        add_flank=20,
        min_size=20,
        max_size=10_000,
        expand_min=255,
    )

    cds_df = cds_region.to_pandas()
    utr5_df = utr5_region.to_pandas()
    assert len(cds_df) == 1
    assert len(utr5_df) == 1

    # CDS in 0-based: [1500, 2500). After add_flank(20): [1480, 2520).
    # CDS extends INTO the 5' UTR region [1000, 1500) by 20 bp [1480, 1500).
    assert cds_df.iloc[0]["start"] == 1480
    assert cds_df.iloc[0]["end"] == 2520

    # 5' UTR in 0-based (+ strand): [1000, 1500). After add_flank(20): [980, 1520).
    # 5' UTR extends INTO the CDS region [1500, 2500) by 20 bp [1500, 1520).
    assert utr5_df.iloc[0]["start"] == 980
    assert utr5_df.iloc[0]["end"] == 1520

    # Verify the shared 20 bp [1480, 1500) and [1500, 1520) — i.e. both
    # regions overlap each other in [1480, 1520).
    assert cds_df.iloc[0]["start"] < utr5_df.iloc[0]["end"]
    assert utr5_df.iloc[0]["start"] < cds_df.iloc[0]["end"]


def test_build_annotation_region_val_cds_drops_noncanonical(
    tmp_path: Path,
) -> None:
    """val_cds picks up only canonical CDS regions.

    The `get_cds` extractor itself doesn't know about canonical — but it
    receives the already canonical-filtered annotation, so non-canonical CDS
    rows have been removed before extraction. This guards against any future
    refactor that bypasses the pre-filter.
    """
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            # Canonical transcript T1 with one CDS row
            _gtf_row(
                "1",
                "transcript",
                1001,
                2000,
                "+",
                'transcript_id "T1"; tag "Ensembl_canonical"; transcript_biotype "protein_coding";',
            ),
            _gtf_row(
                "1",
                "exon",
                1001,
                2000,
                "+",
                'transcript_id "T1"; transcript_biotype "protein_coding";',
            ),
            _gtf_row(
                "1",
                "CDS",
                1100,
                1900,
                "+",
                'transcript_id "T1"; transcript_biotype "protein_coding";',
            ),
            # Non-canonical transcript T2 (different gene location) with CDS:
            # this CDS row should NOT appear in val_cds output.
            _gtf_row(
                "1",
                "transcript",
                10_001,
                20_000,
                "+",
                'transcript_id "T2"; transcript_biotype "protein_coding";',
            ),
            _gtf_row(
                "1",
                "exon",
                10_001,
                20_000,
                "+",
                'transcript_id "T2"; transcript_biotype "protein_coding";',
            ),
            _gtf_row(
                "1",
                "CDS",
                11_001,
                19_000,
                "+",
                'transcript_id "T2"; transcript_biotype "protein_coding";',
            ),
        ],
    )
    ann = load_annotation(str(gtf))
    canonical = filter_to_canonical_transcripts(ann)
    defined = GenomicSet(pd.DataFrame({"chrom": ["1"], "start": [0], "end": [25_000]}))
    cds_region = build_annotation_region(
        "val_cds",
        canonical,
        defined,
        ncrna_biotypes=["lncRNA"],
        add_flank=20,
        min_size=20,
        max_size=10_000,
        expand_min=255,
    )
    df = cds_region.to_pandas()
    assert len(df) == 1
    # Only the canonical CDS [1099, 1899) → +20 flank → [1079, 1919) is in output.
    # The non-canonical CDS [11000, 19000) must NOT contribute.
    assert df.iloc[0]["start"] < 5000
    assert df.iloc[0]["end"] < 5000
    # Defensive: nothing in the T2 region.
    assert not ((df["start"] > 5000) | (df["end"] > 5000)).any()


def test_build_tss_band_region_plus_strand(tmp_path: Path) -> None:
    """+ strand: band = [tx_start - flank, tx_start + flank] (TSS = tx_start)."""
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            _gtf_row(
                "1",
                "transcript",
                5001,
                5500,
                "+",
                'transcript_id "T_pc"; tag "Ensembl_canonical"; transcript_biotype "protein_coding";',
            ),
            _gtf_row(
                "1",
                "exon",
                5001,
                5500,
                "+",
                'transcript_id "T_pc"; transcript_biotype "protein_coding";',
            ),
        ],
    )
    ann = load_annotation(str(gtf))
    canonical = filter_to_canonical_transcripts(ann)
    defined = GenomicSet(pd.DataFrame({"chrom": ["1"], "start": [0], "end": [10_000]}))
    band = build_tss_band_region(canonical, defined, flank=255)
    df = band.to_pandas()
    assert len(df) == 1
    # 0-based: tx start = 5000 → band = [5000 - 255, 5000 + 255) = [4745, 5255)
    assert df.iloc[0]["start"] == 4745
    assert df.iloc[0]["end"] == 5255
    assert df.iloc[0]["end"] - df.iloc[0]["start"] == 510


def test_build_tss_band_region_minus_strand(tmp_path: Path) -> None:
    """- strand: TSS = tx_end (genomic right); band = [tx_end - flank, tx_end + flank]."""
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            _gtf_row(
                "1",
                "transcript",
                3001,
                4000,
                "-",
                'transcript_id "T_pc_minus"; tag "Ensembl_canonical"; transcript_biotype "protein_coding";',
            ),
            _gtf_row(
                "1",
                "exon",
                3001,
                4000,
                "-",
                'transcript_id "T_pc_minus"; transcript_biotype "protein_coding";',
            ),
        ],
    )
    ann = load_annotation(str(gtf))
    canonical = filter_to_canonical_transcripts(ann)
    defined = GenomicSet(pd.DataFrame({"chrom": ["1"], "start": [0], "end": [10_000]}))
    band = build_tss_band_region(canonical, defined, flank=255)
    df = band.to_pandas()
    assert len(df) == 1
    # 0-based: tx_end = 4000 (1-based [3001, 4000] → 0-based [3000, 4000)).
    # -strand TSS = 4000; band = [4000-255, 4000+255) = [3745, 4255).
    assert df.iloc[0]["start"] == 3745
    assert df.iloc[0]["end"] == 4255


def test_build_tss_band_region_excludes_lncrna(tmp_path: Path) -> None:
    """Only protein_coding canonical transcripts contribute (lncRNA, miRNA, etc. are skipped)."""
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            _gtf_row(
                "1",
                "transcript",
                5001,
                5500,
                "+",
                'transcript_id "T_pc"; tag "Ensembl_canonical"; transcript_biotype "protein_coding";',
            ),
            _gtf_row(
                "1",
                "exon",
                5001,
                5500,
                "+",
                'transcript_id "T_pc"; transcript_biotype "protein_coding";',
            ),
            # lncRNA canonical: should be excluded
            _gtf_row(
                "2",
                "transcript",
                8001,
                8500,
                "+",
                'transcript_id "T_lnc"; tag "Ensembl_canonical"; transcript_biotype "lncRNA";',
            ),
            _gtf_row(
                "2",
                "exon",
                8001,
                8500,
                "+",
                'transcript_id "T_lnc"; transcript_biotype "lncRNA";',
            ),
        ],
    )
    ann = load_annotation(str(gtf))
    canonical = filter_to_canonical_transcripts(ann)
    defined = GenomicSet(
        pd.DataFrame({"chrom": ["1", "2"], "start": [0, 0], "end": [10_000, 10_000]})
    )
    band = build_tss_band_region(canonical, defined, flank=255)
    df = band.to_pandas()
    assert len(df) == 1
    assert df.iloc[0]["chrom"] == "1"  # protein_coding chrom only


def test_build_annotation_region_unknown_recipe_raises(tmp_path: Path) -> None:
    """A non-recipe wildcard fires a clear ValueError."""
    gtf = tmp_path / "ann.gtf"
    _write_gtf(
        gtf,
        [
            _gtf_row(
                "1",
                "transcript",
                1001,
                2000,
                "+",
                'transcript_id "T1"; tag "Ensembl_canonical"; transcript_biotype "protein_coding";',
            ),
            _gtf_row(
                "1",
                "exon",
                1001,
                2000,
                "+",
                'transcript_id "T1"; transcript_biotype "protein_coding";',
            ),
        ],
    )
    ann = load_annotation(str(gtf))
    canonical = filter_to_canonical_transcripts(ann)
    defined = GenomicSet(pd.DataFrame({"chrom": ["1"], "start": [0], "end": [4000]}))
    with pytest.raises(ValueError, match="unknown filter\\+flank\\+expand recipe"):
        build_annotation_region(
            "val_promoter",  # cre recipe, not annotation
            canonical,
            defined,
            ncrna_biotypes=["lncRNA"],
        )


# ----------------------------------------------------------------------------
# build_cre_region
# ----------------------------------------------------------------------------


def _make_cre_parquet(tmp_path: Path, rows: list[tuple[str, int, int, str]]) -> Path:
    """Build a synthetic cCRE parquet with (chrom, start, end, cre_class)."""
    df = pl.DataFrame(
        {
            "chrom": [r[0] for r in rows],
            "start": [r[1] for r in rows],
            "end": [r[2] for r in rows],
            "cre_class": [r[3] for r in rows],
        }
    )
    out = tmp_path / "cre.parquet"
    df.write_parquet(out)
    return out


def test_build_cre_region_promoter_keeps_pls_only(tmp_path: Path) -> None:
    """val_promoter selects only PLS rows; pELS / dELS / CTCF / DNase dropped."""
    cre = _make_cre_parquet(
        tmp_path,
        [
            ("1", 1000, 1200, "PLS"),
            ("1", 2000, 2200, "pELS"),
            ("1", 3000, 3200, "dELS"),
            ("1", 4000, 4200, "CA-CTCF"),
            ("1", 5000, 5200, "DNase-H3K4me3"),
        ],
    )
    defined = GenomicSet(pd.DataFrame({"chrom": ["1"], "start": [0], "end": [10_000]}))
    region = build_cre_region("val_promoter", cre, defined)
    df = region.to_pandas()
    # Single PLS, resized to 255 bp → midpoint 1100, [1100-127, 1100+128) = [973, 1228).
    assert len(df) == 1
    assert df.iloc[0]["end"] - df.iloc[0]["start"] == 255


def test_build_cre_region_enhancer_keeps_pels_dels(tmp_path: Path) -> None:
    """val_enhancer selects pELS + dELS only."""
    cre = _make_cre_parquet(
        tmp_path,
        [
            ("1", 1000, 1200, "PLS"),
            ("1", 2000, 2200, "pELS"),
            ("1", 3000, 3200, "dELS"),
            ("1", 4000, 4200, "CA-CTCF"),
        ],
    )
    defined = GenomicSet(pd.DataFrame({"chrom": ["1"], "start": [0], "end": [10_000]}))
    region = build_cre_region("val_enhancer", cre, defined)
    df = region.to_pandas()
    # 2 ELS regions, each resized to 255 bp.
    assert len(df) == 2
    assert (df["end"] - df["start"] == 255).all()


def test_build_cre_region_enhancer_subtracts_exons(tmp_path: Path) -> None:
    """val_enhancer drops ELS that overlap any annotated exon."""
    cre = _make_cre_parquet(
        tmp_path,
        [
            # pELS at midpoint 5100 → [5100-127, 5100+128) = [4973, 5228) — overlaps exon
            ("1", 5000, 5200, "pELS"),
            # dELS at midpoint 9100 → [9100-127, 9100+128) = [8973, 9228) — clean
            ("1", 9000, 9200, "dELS"),
        ],
    )
    defined = GenomicSet(pd.DataFrame({"chrom": ["1"], "start": [0], "end": [20_000]}))
    # Exon mask: [4500, 5300) — overlaps the pELS, not the dELS
    exon_mask = GenomicSet(
        pd.DataFrame({"chrom": ["1"], "start": [4500], "end": [5300]})
    )
    region = build_cre_region("val_enhancer", cre, defined, subtract=exon_mask)
    df = region.to_pandas()
    # Only the dELS survives. The pELS resize [4973, 5228) was wholly inside
    # exon mask [4500, 5300) and gets fully subtracted.
    assert len(df) == 1
    # The surviving region is the (full or partial) dELS resize [8973, 9228).
    assert df.iloc[0]["start"] == 8973
    assert df.iloc[0]["end"] == 9228


def test_build_cre_region_promoter_no_subtract_keeps_all(tmp_path: Path) -> None:
    """build_cre_region with subtract=None retains every PLS, regardless of overlap.

    The snakemake rule passes a CDS mask as ``subtract`` for val_promoter,
    but the library function itself must support both subtract=None (no-op)
    and subtract=<set> (filter). This guards the no-op path.
    """
    cre = _make_cre_parquet(
        tmp_path,
        [
            ("1", 5000, 5200, "PLS"),
        ],
    )
    defined = GenomicSet(pd.DataFrame({"chrom": ["1"], "start": [0], "end": [20_000]}))
    region = build_cre_region("val_promoter", cre, defined)
    df = region.to_pandas()
    assert len(df) == 1


def test_build_cre_region_promoter_subtracts_cds_when_given(tmp_path: Path) -> None:
    """val_promoter drops PLS overlapping CDS, keeps PLS overlapping 5' UTR.

    Mirrors how the snakemake rule wires it up: pass an all-CDS mask to
    subtract; PLS that overlap CDS are dropped, PLS in non-CDS exonic
    territory (5' UTR) are kept.
    """
    cre = _make_cre_parquet(
        tmp_path,
        [
            # PLS at midpoint 5100 → resize(255) → [4973, 5228) — overlaps CDS mask
            ("1", 5000, 5200, "PLS"),
            # PLS at midpoint 9100 → [8973, 9228) — clean (no CDS here)
            ("1", 9000, 9200, "PLS"),
        ],
    )
    defined = GenomicSet(pd.DataFrame({"chrom": ["1"], "start": [0], "end": [20_000]}))
    cds_mask = GenomicSet(
        pd.DataFrame({"chrom": ["1"], "start": [4500], "end": [5300]})
    )
    region = build_cre_region("val_promoter", cre, defined, subtract=cds_mask)
    df = region.to_pandas()
    # Only the second PLS (9000-9200) survives; the first was wholly inside
    # the CDS mask.
    assert len(df) == 1
    assert df.iloc[0]["start"] == 8973
    assert df.iloc[0]["end"] == 9228


def test_build_cre_region_unknown_recipe_raises(tmp_path: Path) -> None:
    cre = _make_cre_parquet(tmp_path, [("1", 1000, 1200, "PLS")])
    defined = GenomicSet(pd.DataFrame({"chrom": ["1"], "start": [0], "end": [10_000]}))
    with pytest.raises(ValueError, match="unknown cCRE-derived recipe"):
        build_cre_region("val_cds", cre, defined)


# ----------------------------------------------------------------------------
# subsample_deterministic
# ----------------------------------------------------------------------------


def test_subsample_under_cap_keeps_all_rows_shuffled() -> None:
    """When len(df) <= max_samples, all rows are kept but in deterministic
    shuffled order — NOT input order, NOT (chrom, start) order. Guards
    against val_utr5/val_promoter coming out chrom-sorted from upstream."""
    df = pl.DataFrame(
        {
            # Sorted by (chrom, start) input — what upstream filter produces
            "chrom": [str(c) for c in [1] * 30 + [2] * 30 + [3] * 30],
            "start": list(range(90)),
            "end": [s + 255 for s in range(90)],
        }
    )
    out = subsample_deterministic(df, max_samples=10_000, seed=42)
    assert len(out) == len(df), "all rows preserved when under cap"
    in_keys = list(zip(df["chrom"].to_list(), df["start"].to_list()))
    out_keys = list(zip(out["chrom"].to_list(), out["start"].to_list()))
    assert set(out_keys) == set(in_keys), "no rows lost or duplicated"
    assert out_keys != in_keys, "expected shuffle; got identity (input order preserved)"
    assert out_keys != sorted(in_keys), "expected shuffle; got sort"


def test_subsample_over_cap_does_not_resort() -> None:
    """After sampling at the cap, rows are NOT re-sorted by (chrom, start)."""
    df = pl.DataFrame(
        {
            "chrom": [str((i % 2) + 1) for i in range(100)],
            "start": list(range(100)),
            "end": [s + 255 for s in range(100)],
        }
    )
    out = subsample_deterministic(df, max_samples=20, seed=42)
    assert len(out) == 20
    chroms = out["chrom"].to_list()
    grouped = chroms == sorted(chroms)
    assert not grouped, (
        f"subsample_deterministic re-sorted by chrom; rows came out grouped: {chroms}"
    )


def test_subsample_seed_reproducible() -> None:
    df = pl.DataFrame(
        {
            "chrom": ["1"] * 100,
            "start": list(range(100)),
            "end": [s + 255 for s in range(100)],
        }
    )
    a = subsample_deterministic(df, max_samples=10, seed=42)
    b = subsample_deterministic(df, max_samples=10, seed=42)
    assert a["start"].to_list() == b["start"].to_list()


def test_subsample_diff_seeds_differ() -> None:
    df = pl.DataFrame(
        {
            "chrom": ["1"] * 100,
            "start": list(range(100)),
            "end": [s + 255 for s in range(100)],
        }
    )
    a = subsample_deterministic(df, max_samples=10, seed=42)
    b = subsample_deterministic(df, max_samples=10, seed=99)
    # Highly likely that two different seeds give different samples
    assert a["start"].to_list() != b["start"].to_list()


# ----------------------------------------------------------------------------
# case_encode_sequences
# ----------------------------------------------------------------------------


@pytest.fixture
def synthetic_bigwig(tmp_path: Path) -> Path:
    """Tiny chr1 bigWig: [0, 30) value 2.0; [30, 60) NaN; [60, 100) value -1.0."""
    bw_path = tmp_path / "test.bw"
    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader([("chr1", 100)])
    bw.addEntries(
        ["chr1"] * 30,
        list(range(30)),
        ends=list(range(1, 31)),
        values=[2.0] * 30,
    )
    bw.addEntries(
        ["chr1"] * 40,
        list(range(60, 100)),
        ends=list(range(61, 101)),
        values=[-1.0] * 40,
    )
    bw.close()
    return bw_path


def _write_fasta(path: Path, entries: list[tuple[str, str]]) -> None:
    """Write a FASTA with (header_no_gt, sequence) entries."""
    with path.open("w") as fh:
        for header, seq in entries:
            fh.write(f">{header}\n{seq}\n")


def test_case_encode_above_threshold_uppercase(
    tmp_path: Path, synthetic_bigwig: Path
) -> None:
    fa = tmp_path / "seq.fa"
    # 30-bp sequence at chr1:[0, 30) where bigWig is 2.0
    _write_fasta(fa, [("1:0-30", "acgtacgtacgtacgtacgtacgtacgtac")])
    df = case_encode_sequences(fa, synthetic_bigwig, threshold=1.0, window_size=30)
    assert len(df) == 1
    seq = df["seq"][0]
    # All bases uppercase because phyloP_447m = 2.0 >= 1.0
    assert seq == seq.upper()
    assert len(seq) == 30


def test_case_encode_below_threshold_lowercase(
    tmp_path: Path, synthetic_bigwig: Path
) -> None:
    fa = tmp_path / "seq.fa"
    # 40-bp sequence at chr1:[60, 100) where bigWig is -1.0
    _write_fasta(fa, [("1:60-100", "ACGT" * 10)])
    df = case_encode_sequences(fa, synthetic_bigwig, threshold=1.0, window_size=40)
    seq = df["seq"][0]
    assert seq == seq.lower()


def test_case_encode_nan_lowercase(tmp_path: Path, synthetic_bigwig: Path) -> None:
    fa = tmp_path / "seq.fa"
    # 30-bp sequence at chr1:[30, 60) where bigWig is NaN throughout
    _write_fasta(fa, [("1:30-60", "ACGT" * 7 + "AC")])
    df = case_encode_sequences(fa, synthetic_bigwig, threshold=1.0, window_size=30)
    seq = df["seq"][0]
    # NaN >= threshold is False → lowercase everywhere
    assert seq == seq.lower()


def test_case_encode_threshold_inclusive(
    tmp_path: Path, synthetic_bigwig: Path
) -> None:
    """Value exactly at threshold counts as conserved (>= semantics)."""
    fa = tmp_path / "seq.fa"
    _write_fasta(fa, [("1:0-30", "acgt" * 7 + "ac")])
    df = case_encode_sequences(fa, synthetic_bigwig, threshold=2.0, window_size=30)
    seq = df["seq"][0]
    assert seq == seq.upper()


def test_case_encode_unknown_chrom_lowercase(
    tmp_path: Path, synthetic_bigwig: Path
) -> None:
    """When the chrom is missing from the bigWig, fall back to all-lowercase."""
    fa = tmp_path / "seq.fa"
    _write_fasta(fa, [("99:0-30", "ACGT" * 7 + "AC")])
    df = case_encode_sequences(fa, synthetic_bigwig, threshold=1.0, window_size=30)
    seq = df["seq"][0]
    assert seq == seq.lower()


def test_case_encode_chrom_already_prefixed(
    tmp_path: Path, synthetic_bigwig: Path
) -> None:
    """Both ``"1"`` and ``"chr1"`` resolve to the bigWig chrom."""
    fa1 = tmp_path / "bare.fa"
    _write_fasta(fa1, [("1:0-30", "acgt" * 7 + "ac")])
    df1 = case_encode_sequences(fa1, synthetic_bigwig, threshold=1.0, window_size=30)
    fa2 = tmp_path / "prefixed.fa"
    _write_fasta(fa2, [("chr1:0-30", "acgt" * 7 + "ac")])
    df2 = case_encode_sequences(fa2, synthetic_bigwig, threshold=1.0, window_size=30)
    assert df1["seq"][0] == df2["seq"][0]


def test_case_encode_id_format_chrom_colon_start_dash_end(
    tmp_path: Path, synthetic_bigwig: Path
) -> None:
    """Output id matches the FASTA header (which is ``chrom:start-end``)."""
    fa = tmp_path / "seq.fa"
    _write_fasta(fa, [("1:0-30", "ACGT" * 7 + "AC")])
    df = case_encode_sequences(fa, synthetic_bigwig, threshold=1.0, window_size=30)
    assert df["id"][0] == "1:0-30"


def test_case_encode_window_size_mismatch_assertion(
    tmp_path: Path, synthetic_bigwig: Path
) -> None:
    """Passing window_size that doesn't match the seq length triggers an assert."""
    fa = tmp_path / "seq.fa"
    _write_fasta(fa, [("1:0-30", "ACGT")])  # 4 bp seq
    with pytest.raises(AssertionError, match="sequence length"):
        case_encode_sequences(fa, synthetic_bigwig, threshold=1.0, window_size=30)


# ----------------------------------------------------------------------------
# write_hf_readme
# ----------------------------------------------------------------------------


def _readme_kwargs(**overrides):
    base = {
        "commit_sha": "abcdef0123456789" * 2 + "abcd",  # 40-char fake SHA
        "hf_owner": "bolinas-dna",
        "pipeline_version": "v1",
        "ensembl_release": 115,
        "threshold": 2.2162,
        "min_proportion_conserved": 0.20,
        "max_samples": 16384,
        "seed": 42,
        "ncrna_biotypes": ["lncRNA", "miRNA", "snoRNA"],
        "canonical_tag": "Ensembl_canonical",
    }
    base.update(overrides)
    return base


def test_write_hf_readme_includes_permalink(tmp_path: Path) -> None:
    out = tmp_path / "README.md"
    sha = "abcdef0123456789abcdef0123456789abcdef00"
    write_hf_readme("val_cds", out, **_readme_kwargs(commit_sha=sha))
    body = out.read_text()
    assert (
        f"https://github.com/Open-Athena/marin-dna/tree/{sha}/snakemake/zoonomia_projection_dataset"
        in body
    )
    # Short SHA is also referenced as link text
    assert sha[:12] in body


def test_write_hf_readme_includes_repo_name(tmp_path: Path) -> None:
    out = tmp_path / "README.md"
    write_hf_readme(
        "val_utr5",
        out,
        **_readme_kwargs(hf_owner="my-org", pipeline_version="v3"),
    )
    body = out.read_text()
    assert "`my-org/zoonomia-v3-val_utr5`" in body


def test_write_hf_readme_describes_lowercase_semantics(tmp_path: Path) -> None:
    """The case-encoding section must explicitly call out lowercase ambiguity."""
    out = tmp_path / "README.md"
    write_hf_readme("val_cds", out, **_readme_kwargs())
    body = out.read_text()
    # Uppercase + lowercase definitions
    assert "Uppercase" in body
    assert "Lowercase" in body
    # NaN -> lowercase
    assert "NaN" in body
    # Lowercase is ambiguous
    assert "ambiguous" in body.lower()


def test_write_hf_readme_includes_recipe_specific_blurb(
    tmp_path: Path,
) -> None:
    """Each recipe gets its own selection-logic description."""
    cds_path = tmp_path / "cds.md"
    enh_path = tmp_path / "enh.md"
    prom_path = tmp_path / "prom.md"
    write_hf_readme("val_cds", cds_path, **_readme_kwargs())
    write_hf_readme("val_enhancer", enh_path, **_readme_kwargs())
    write_hf_readme("val_promoter", prom_path, **_readme_kwargs())

    cds_body = cds_path.read_text()
    enh_body = enh_path.read_text()
    prom_body = prom_path.read_text()

    # CDS-specific
    assert "Protein-coding sequence" in cds_body or "splice signal" in cds_body
    # Enhancer-specific: subtracts every annotated exon
    assert "subtract" in enh_body.lower()
    assert "exon" in enh_body.lower()
    # Promoter-specific: PLS, no subtraction, mentions 5' UTR overlap is intended
    assert "PLS" in prom_body or "Promoter-Like Signature" in prom_body
    assert "no subtraction" in prom_body.lower()
    assert "5' UTR" in prom_body
    # Each blurb is recipe-specific
    assert cds_body != enh_body
    assert cds_body != prom_body


def test_write_hf_readme_ncrna_lists_biotypes(tmp_path: Path) -> None:
    out = tmp_path / "README.md"
    write_hf_readme(
        "val_ncrna",
        out,
        **_readme_kwargs(ncrna_biotypes=["lncRNA", "miRNA", "snoRNA"]),
    )
    body = out.read_text()
    assert "`lncRNA`" in body
    assert "`miRNA`" in body
    assert "`snoRNA`" in body


def test_write_hf_readme_unknown_recipe_raises(tmp_path: Path) -> None:
    out = tmp_path / "README.md"
    with pytest.raises(ValueError, match="unknown recipe"):
        write_hf_readme("val_unknown", out, **_readme_kwargs())


def test_write_hf_readme_threshold_and_min_p_appear(tmp_path: Path) -> None:
    """The actual numerical filter parameters appear in the construction section."""
    out = tmp_path / "README.md"
    write_hf_readme(
        "val_cds",
        out,
        **_readme_kwargs(threshold=2.2162, min_proportion_conserved=0.20),
    )
    body = out.read_text()
    assert "2.2162" in body
    assert "0.2" in body  # 0.20 may render as "0.2" depending on formatting
    # 51 = round(0.20 * 255)
    assert "51 of the 255" in body


def test_write_hf_readme_yaml_frontmatter(tmp_path: Path) -> None:
    """Output starts with HF dataset card YAML frontmatter."""
    out = tmp_path / "README.md"
    write_hf_readme("val_cds", out, **_readme_kwargs())
    body = out.read_text()
    # Frontmatter delimiter at start
    assert body.startswith("---\n")
    # Tags block
    assert "tags:" in body
    assert "validation" in body
