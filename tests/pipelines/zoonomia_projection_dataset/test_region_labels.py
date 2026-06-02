"""Tests for ``marin_dna.pipelines.zoonomia_projection_dataset.region_labels``.

Builds a small synthetic Ensembl-flavored GTF + cCRE parquet covering every
priority and threshold case from the plan, then asserts each 255 bp anchor
gets the expected label and frac_* columns.
"""

from pathlib import Path

import polars as pl
import pytest

from marin_dna.data.intervals import GenomicSet
from marin_dna.pipelines.zoonomia_projection_dataset.region_labels import (
    BACKGROUND_LABEL,
    REGION_LABELS,
    build_region_beds,
    label_windows,
    label_windows_bp_majority,
)

# v4 priority (issue #227): tss_region_and_utr5 promoted above ncrna_exon.
V4_PRIORITY = [
    "cds",
    "utr3",
    "tss_region_and_utr5",
    "ncrna_exon",
    "ccre_non_promoter",
]

# ============================================================================
# Synthetic GTF / cCRE / windows
# ============================================================================
#
# Chromosome "1" layout (0-based half-open everywhere):
#
#   Gene A — protein_coding, + strand
#       exon1: 1000–1500   exon2: 4000–4600
#       intron 1500–4000
#       CDS  1300–1500 + 4000–4400
#       =>  5' UTR 1000–1300, 3' UTR 4400–4600
#       TSS = 1000, band 744–1256
#
#   Gene B — processed_pseudogene, + strand
#       exon: 8000–8500     (no CDS; biotype != protein_coding)
#       TSS = 8000, band 7744–8256
#
#   Gene C — lncRNA, + strand
#       exon: 20000–20500
#       TSS = 20000, band 19744–20256
#
#   Gene E — protein_coding, + strand, very long 5' UTR
#       exon: 10000–12000
#       CDS  11800–12000   =>  5' UTR 10000–11800 (1800 bp), no 3' UTR
#       TSS = 10000, band 9744–10256
#
#   Gene D — protein_coding, - strand (isolated antisense gene)
#       exon: 60000–60900
#       CDS  60800–60900   =>  3' UTR 60000–60800, no 5' UTR
#       TSS = 60900, band 60644–61156
#
#   Gene F — protein_coding, + strand (overlaps Gene D's 3' UTR)
#       exon: 60500–60700
#       CDS  60500–60700   =>  no UTR
#       TSS = 60500, band 60244–60756
#
# cCREs (chrom 1):
#       PLS    950–1050         (near Gene A TSS — excluded from `ccre_non_promoter`)
#       PLS    50000–50200      (isolated — excluded from `ccre_non_promoter`)
#       dELS   4100–4300        (inside Gene A CDS exon2; flanked 3600–4800)
#       dELS   30000–30500      (intergenic; flanked 29500–31000)


def _gtf_row(
    chrom: str,
    feature: str,
    start_0based: int,
    end_0based: int,
    strand: str,
    attrs: dict[str, str],
) -> str:
    """Build a single GTF row.

    GTF uses 1-based inclusive starts; we accept 0-based half-open intervals
    and convert to GTF (start+1; end unchanged numerically).
    """
    attr_str = " ".join(f'{k} "{v}";' for k, v in attrs.items())
    return (
        f"{chrom}\tsynth\t{feature}\t{start_0based + 1}\t{end_0based}\t"
        f".\t{strand}\t.\t{attr_str}"
    )


def _ensembl_gtf_text() -> str:
    rows: list[str] = []

    def gene_block(
        gene_id: str,
        biotype: str,
        strand: str,
        start: int,
        end: int,
        exons: list[tuple[int, int]],
        cds: list[tuple[int, int]] | None = None,
    ) -> None:
        tx_id = f"tx_{gene_id}"
        gene_attrs = {"gene_id": gene_id, "gene_biotype": biotype}
        tx_attrs = {
            "gene_id": gene_id,
            "transcript_id": tx_id,
            "transcript_biotype": biotype,
            "tag": "Ensembl_canonical",
        }
        rows.append(_gtf_row("1", "gene", start, end, strand, gene_attrs))
        rows.append(_gtf_row("1", "transcript", start, end, strand, tx_attrs))
        for ex_start, ex_end in exons:
            rows.append(_gtf_row("1", "exon", ex_start, ex_end, strand, tx_attrs))
        if cds is not None:
            for cs_start, cs_end in cds:
                rows.append(_gtf_row("1", "CDS", cs_start, cs_end, strand, tx_attrs))

    # Gene A: PC, + strand, 2 exons + intron
    gene_block(
        "geneA",
        "protein_coding",
        "+",
        start=1000,
        end=4600,
        exons=[(1000, 1500), (4000, 4600)],
        cds=[(1300, 1500), (4000, 4400)],
    )
    # Gene B: pseudogene, + strand
    gene_block(
        "geneB",
        "processed_pseudogene",
        "+",
        start=8000,
        end=8500,
        exons=[(8000, 8500)],
        cds=None,
    )
    # Gene C: lncRNA, + strand
    gene_block(
        "geneC",
        "lncRNA",
        "+",
        start=20000,
        end=20500,
        exons=[(20000, 20500)],
        cds=None,
    )
    # Gene E: PC, + strand, long 5' UTR
    gene_block(
        "geneE",
        "protein_coding",
        "+",
        start=10000,
        end=12000,
        exons=[(10000, 12000)],
        cds=[(11800, 12000)],
    )
    # Gene D: PC, - strand
    gene_block(
        "geneD",
        "protein_coding",
        "-",
        start=60000,
        end=60900,
        exons=[(60000, 60900)],
        cds=[(60800, 60900)],
    )
    # Gene F: PC, + strand, CDS == exon (no UTR)
    gene_block(
        "geneF",
        "protein_coding",
        "+",
        start=60500,
        end=60700,
        exons=[(60500, 60700)],
        cds=[(60500, 60700)],
    )
    return "\n".join(rows) + "\n"


def _write_synthetic_cre_parquet(path: Path) -> None:
    pl.DataFrame(
        {
            "chrom": ["1", "1", "1", "1"],
            "start": [950, 50000, 4100, 30000],
            "end": [1050, 50200, 4300, 30500],
            "cre_class": ["PLS", "PLS", "dELS", "dELS"],
        }
    ).write_parquet(path)


# Window definitions: (name, start, end). All on chrom "1", 255 bp wide.
_WINDOWS: list[tuple[str, int, int]] = [
    ("w_cds100", 4050, 4305),
    ("w_cds60_utr40", 1198, 1453),
    ("w_cds10_intron90", 1474, 1729),
    ("w_cds25_intron75", 1436, 1691),
    ("w_ccre_in_cds", 4100, 4355),
    ("w_ccre_intergenic", 30000, 30255),
    ("w_utr3_antisense", 60244, 60499),
    ("w_pls_at_tss", 950, 1205),
    ("w_pls_isolated", 50000, 50255),
    ("w_pseudogene", 8000, 8255),
    ("w_long_utr5", 11000, 11255),
]


def _write_synthetic_windows_bed(path: Path) -> None:
    with open(path, "w") as fh:
        for name, start, end in _WINDOWS:
            assert end - start == 255, f"{name} not 255 bp"
            fh.write(f"1\t{start}\t{end}\t{name}\n")


# ============================================================================
# Pytest fixture
# ============================================================================


@pytest.fixture
def synth(tmp_path):
    gtf_path = tmp_path / "synth.gtf"
    gtf_path.write_text(_ensembl_gtf_text())
    cre_path = tmp_path / "cre.parquet"
    _write_synthetic_cre_parquet(cre_path)
    windows_path = tmp_path / "windows.bed"
    _write_synthetic_windows_bed(windows_path)
    # Whole-chrom defined region (no N gaps in the synthetic genome).
    defined = GenomicSet(pl.DataFrame({"chrom": ["1"], "start": [0], "end": [200_000]}))
    return {
        "gtf": gtf_path,
        "cre": cre_path,
        "windows": windows_path,
        "defined": defined,
    }


def _row(df: pl.DataFrame, name: str) -> dict:
    matches = df.filter(pl.col("name") == name)
    assert len(matches) == 1, f"window {name!r} not found in labels output"
    return matches.row(0, named=True)


# ============================================================================
# Tests
# ============================================================================


def test_priority_and_threshold_cases(synth):
    beds = build_region_beds(
        synth["gtf"],
        synth["cre"],
        synth["defined"],
        tss_radius=256,
        ccre_flank=500,
    )
    df = label_windows(
        synth["windows"],
        beds,
        functional_threshold=0.20,
        priority=list(REGION_LABELS),
    )

    # Window 100% in CDS
    r = _row(df, "w_cds100")
    assert r["label"] == "cds"
    assert r["cds_frac"] == pytest.approx(1.0)
    assert r["functional_frac"] == pytest.approx(1.0)

    # 60% CDS + 40% 5' UTR — CDS wins by priority
    r = _row(df, "w_cds60_utr40")
    assert r["label"] == "cds"
    assert r["cds_frac"] == pytest.approx(153 / 255)
    assert r["tss_region_and_utr5_frac"] == pytest.approx(102 / 255)

    # 10% CDS + 90% intron — below threshold => background
    r = _row(df, "w_cds10_intron90")
    assert r["label"] == BACKGROUND_LABEL
    assert r["cds_frac"] == pytest.approx(26 / 255)
    assert r["functional_frac"] == pytest.approx(26 / 255)
    assert r["intron_frac"] == pytest.approx(229 / 255)

    # 25% CDS + 75% intron — above threshold => cds
    r = _row(df, "w_cds25_intron75")
    assert r["label"] == "cds"
    assert r["cds_frac"] == pytest.approx(64 / 255)
    assert r["intron_frac"] == pytest.approx(191 / 255)

    # cCRE entirely inside a CDS exon — exon wins over CRE
    r = _row(df, "w_ccre_in_cds")
    assert r["label"] == "cds"
    assert r["cds_frac"] == pytest.approx(1.0)
    assert r["ccre_non_promoter_frac"] == pytest.approx(1.0)

    # cCRE in intergenic — no exon overlap, no TSS => cre
    r = _row(df, "w_ccre_intergenic")
    assert r["label"] == "ccre_non_promoter"
    assert r["ccre_non_promoter_frac"] == pytest.approx(1.0)

    # Antisense TSS-band + 3' UTR overlap, no CDS — utr3 beats tss_region
    r = _row(df, "w_utr3_antisense")
    assert r["label"] == "utr3"
    assert r["utr3_frac"] == pytest.approx(1.0)
    assert r["tss_region_and_utr5_frac"] == pytest.approx(1.0)
    assert r["cds_frac"] == pytest.approx(0.0)

    # PLS sitting on a TSS — PLS excluded from ccre_non_promoter, but TSS catches it
    r = _row(df, "w_pls_at_tss")
    assert r["label"] == "tss_region_and_utr5"
    assert r["tss_region_and_utr5_frac"] == pytest.approx(1.0)
    assert r["ccre_non_promoter_frac"] == pytest.approx(0.0)
    assert r["cds_frac"] == pytest.approx(0.0)

    # PLS far from any TSS — nothing functional => background
    r = _row(df, "w_pls_isolated")
    assert r["label"] == BACKGROUND_LABEL
    assert r["ccre_non_promoter_frac"] == pytest.approx(0.0)
    assert r["functional_frac"] == pytest.approx(0.0)

    # Pseudogene exon only — broad ncrna_exon picks it up
    r = _row(df, "w_pseudogene")
    assert r["label"] == "ncrna_exon"
    assert r["ncrna_exon_frac"] == pytest.approx(1.0)

    # Long 5' UTR > 256 bp from TSS — still tss_region_and_utr5 via UTR union
    r = _row(df, "w_long_utr5")
    assert r["label"] == "tss_region_and_utr5"
    assert r["tss_region_and_utr5_frac"] == pytest.approx(1.0)


def test_partition_invariant(synth):
    beds = build_region_beds(
        synth["gtf"],
        synth["cre"],
        synth["defined"],
        tss_radius=256,
        ccre_flank=500,
    )
    df = label_windows(
        synth["windows"],
        beds,
        functional_threshold=0.20,
        priority=list(REGION_LABELS),
    )
    valid_labels = set(REGION_LABELS) | {BACKGROUND_LABEL}
    assert set(df["label"].unique().to_list()) <= valid_labels
    # Every window gets exactly one label — counts sum to total.
    counts = df.group_by("label").len()
    assert counts["len"].sum() == len(df) == len(_WINDOWS)


def test_priority_permutation_changes_label_not_fracs(synth):
    """Shuffling `priority` re-routes ambiguous windows but per-region
    fracs (and functional_frac) must be unchanged."""
    beds = build_region_beds(
        synth["gtf"],
        synth["cre"],
        synth["defined"],
        tss_radius=256,
        ccre_flank=500,
    )
    default = label_windows(
        synth["windows"],
        beds,
        functional_threshold=0.20,
        priority=list(REGION_LABELS),
    )
    # Promote tss_region_and_utr5 above cds.
    alt = label_windows(
        synth["windows"],
        beds,
        functional_threshold=0.20,
        priority=[
            "tss_region_and_utr5",
            "cds",
            "utr3",
            "ncrna_exon",
            "ccre_non_promoter",
        ],
    )
    # Frac columns unchanged
    frac_cols = [c for c in default.columns if c.endswith("_frac")]
    for col in frac_cols:
        assert default[col].to_list() == alt[col].to_list(), (
            f"{col} should be invariant to priority order"
        )
    # CDS-only window: cds beats tss-with-zero-overlap regardless
    assert _row(alt, "w_cds100")["label"] == "cds"
    # 60% CDS + 40% UTR5: now tss_region_and_utr5 wins (it overlaps)
    assert _row(alt, "w_cds60_utr40")["label"] == "tss_region_and_utr5"


def test_multi_chrom_alignment(tmp_path):
    """Regression: ``bf.coverage`` resets its input's index in place, so a
    naive groupby-by-chrom labeler corrupts non-first-chrom alignments.
    Build a 2-chrom Ensembl GTF and confirm chr-2 windows are labelled
    correctly (not just chr-1).
    """
    # chr "1": one CDS at 1000–1500 (PC transcript).
    # chr "2": one CDS at 8000–8500 (PC transcript).
    rows: list[str] = []

    def add_pc_gene(chrom: str, gene: str, start: int, end: int, cds: tuple[int, int]):
        attrs_gene = {"gene_id": gene, "gene_biotype": "protein_coding"}
        attrs_tx = {
            "gene_id": gene,
            "transcript_id": f"tx_{gene}",
            "transcript_biotype": "protein_coding",
            "tag": "Ensembl_canonical",
        }
        rows.append(_gtf_row(chrom, "gene", start, end, "+", attrs_gene))
        rows.append(_gtf_row(chrom, "transcript", start, end, "+", attrs_tx))
        rows.append(_gtf_row(chrom, "exon", start, end, "+", attrs_tx))
        rows.append(_gtf_row(chrom, "CDS", cds[0], cds[1], "+", attrs_tx))

    add_pc_gene("1", "geneA", 1000, 1500, cds=(1000, 1500))
    add_pc_gene("2", "geneB", 8000, 8500, cds=(8000, 8500))
    gtf_path = tmp_path / "two_chrom.gtf"
    gtf_path.write_text("\n".join(rows) + "\n")

    cre_path = tmp_path / "cre.parquet"
    pl.DataFrame(
        {"chrom": ["1"], "start": [50_000], "end": [50_500], "cre_class": ["dELS"]}
    ).write_parquet(cre_path)

    # Defined for both chroms.
    defined = GenomicSet(
        pl.DataFrame({"chrom": ["1", "2"], "start": [0, 0], "end": [100_000, 100_000]})
    )

    # Windows on BOTH chroms, each entirely in their chrom's CDS.
    windows_bed = tmp_path / "windows.bed"
    windows_bed.write_text("1\t1100\t1355\tw_chr1_cds\n2\t8100\t8355\tw_chr2_cds\n")

    beds = build_region_beds(
        gtf_path,
        cre_path,
        defined,
        tss_radius=256,
        ccre_flank=500,
    )
    df = label_windows(
        windows_bed,
        beds,
        functional_threshold=0.20,
        priority=list(REGION_LABELS),
    )

    # Both windows must be labelled `cds` with cds_frac == 1.0.
    # Without the bf.coverage in-place mutation fix, the chr-2 window's
    # coverage gets misaligned (written to the chr-1 row's position in `out`),
    # and chr-2 is labelled `background` while chr-1 gets the wrong frac.
    chr1 = _row(df, "w_chr1_cds")
    chr2 = _row(df, "w_chr2_cds")
    assert chr1["label"] == "cds", f"chr1 label={chr1['label']!r}"
    assert chr1["cds_frac"] == pytest.approx(1.0), chr1
    assert chr2["label"] == "cds", f"chr2 label={chr2['label']!r}"
    assert chr2["cds_frac"] == pytest.approx(1.0), chr2


def test_refseq_gtf_rejected(tmp_path):
    """Synthetic RefSeq-style GTF (transcript_biotype "mRNA") must crash."""
    refseq_rows = [
        _gtf_row(
            "1",
            "gene",
            1000,
            2000,
            "+",
            {"gene_id": "geneA", "gene_biotype": "mRNA"},
        ),
        _gtf_row(
            "1",
            "transcript",
            1000,
            2000,
            "+",
            {"gene_id": "geneA", "transcript_id": "txA", "transcript_biotype": "mRNA"},
        ),
        _gtf_row(
            "1",
            "exon",
            1000,
            2000,
            "+",
            {"gene_id": "geneA", "transcript_id": "txA", "transcript_biotype": "mRNA"},
        ),
        _gtf_row(
            "1",
            "CDS",
            1200,
            1800,
            "+",
            {"gene_id": "geneA", "transcript_id": "txA", "transcript_biotype": "mRNA"},
        ),
    ]
    refseq_path = tmp_path / "refseq.gtf"
    refseq_path.write_text("\n".join(refseq_rows) + "\n")
    cre_path = tmp_path / "cre.parquet"
    pl.DataFrame(
        {"chrom": ["1"], "start": [3000], "end": [3500], "cre_class": ["dELS"]}
    ).write_parquet(cre_path)
    defined = GenomicSet(pl.DataFrame({"chrom": ["1"], "start": [0], "end": [10_000]}))
    with pytest.raises(AssertionError, match="protein_coding"):
        build_region_beds(
            refseq_path, cre_path, defined, tss_radius=256, ccre_flank=500
        )


def test_threshold_validates_range(synth):
    """functional_threshold must be in [0, 1]."""
    beds = build_region_beds(
        synth["gtf"],
        synth["cre"],
        synth["defined"],
        tss_radius=256,
        ccre_flank=500,
    )
    with pytest.raises(AssertionError):
        label_windows(
            synth["windows"],
            beds,
            functional_threshold=1.5,
            priority=list(REGION_LABELS),
        )


def test_priority_must_be_permutation(synth):
    beds = build_region_beds(
        synth["gtf"],
        synth["cre"],
        synth["defined"],
        tss_radius=256,
        ccre_flank=500,
    )
    with pytest.raises(AssertionError, match="permutation"):
        label_windows(
            synth["windows"],
            beds,
            functional_threshold=0.20,
            priority=["cds", "utr3", "ncrna_exon"],  # missing two
        )


# ============================================================================
# label_windows_bp_majority (v4 labeler — issue #227)
# ============================================================================


def _gene_rows(
    gene_id: str,
    biotype: str,
    strand: str,
    start: int,
    end: int,
    exons: list[tuple[int, int]],
    cds: list[tuple[int, int]] | None = None,
) -> list[str]:
    """Module-level gene-block builder for the bp-majority mini-fixtures."""
    tx_id = f"tx_{gene_id}"
    gene_attrs = {"gene_id": gene_id, "gene_biotype": biotype}
    tx_attrs = {
        "gene_id": gene_id,
        "transcript_id": tx_id,
        "transcript_biotype": biotype,
        "tag": "Ensembl_canonical",
    }
    rows = [
        _gtf_row("1", "gene", start, end, strand, gene_attrs),
        _gtf_row("1", "transcript", start, end, strand, tx_attrs),
    ]
    for ex_start, ex_end in exons:
        rows.append(_gtf_row("1", "exon", ex_start, ex_end, strand, tx_attrs))
    for cs_start, cs_end in cds or []:
        rows.append(_gtf_row("1", "CDS", cs_start, cs_end, strand, tx_attrs))
    return rows


def _write_gtf(tmp_path: Path, rows: list[str]) -> Path:
    p = tmp_path / "synth.gtf"
    p.write_text("\n".join(rows) + "\n")
    return p


def _write_cre(tmp_path: Path, cres: list[tuple[int, int, str]]) -> Path:
    p = tmp_path / "cre.parquet"
    pl.DataFrame(
        {
            "chrom": ["1"] * len(cres),
            "start": [c[0] for c in cres],
            "end": [c[1] for c in cres],
            "cre_class": [c[2] for c in cres],
        }
    ).write_parquet(p)
    return p


def _write_one_window(tmp_path: Path, name: str, start: int, end: int) -> Path:
    assert end - start == 255, f"{name} not 255 bp"
    p = tmp_path / "windows.bed"
    p.write_text(f"1\t{start}\t{end}\t{name}\n")
    return p


def _whole_chrom_defined() -> GenomicSet:
    return GenomicSet(pl.DataFrame({"chrom": ["1"], "start": [0], "end": [200_000]}))


def test_bp_majority_nested_cds_in_ccre(synth):
    """The headline #221 fix: a CDS exon nested inside a cCRE is labelled
    `cds`, not `ccre`. Under naive argmax-of-overlapping-fractions the
    enclosing cCRE (frac 1.0) would win; bp-priority gives the shared bases
    to `cds` first, leaving `ccre` 0 exclusive bp.
    """
    beds = build_region_beds(
        synth["gtf"], synth["cre"], synth["defined"], tss_radius=256, ccre_flank=500
    )
    df = label_windows_bp_majority(
        synth["windows"], beds, functional_threshold=0.20, priority=V4_PRIORITY
    )
    # w_ccre_in_cds (4100–4355) sits fully inside CDS exon2 (4000–4400) and
    # inside the flanked dELS — overlapping fracs are cds≈1.0 AND ccre≈1.0.
    r = _row(df, "w_ccre_in_cds")
    assert r["label"] == "cds"
    assert r["cds_frac"] == pytest.approx(1.0)  # disjoint: cds claims every base
    assert r["ccre_non_promoter_frac"] == pytest.approx(0.0)  # 0 exclusive bp


def test_bp_majority_true_majority_differs_from_presence(tmp_path):
    """A window 60 bp CDS + 195 bp (non-nested) cCRE: priority-on-presence
    labels it `cds` (any CDS overlap wins); window-majority labels it
    `ccre_non_promoter` (the larger disjoint area).
    """
    rows = _gene_rows(
        "genePC",
        "protein_coding",
        "+",
        start=1000,
        end=4060,
        exons=[(1000, 1100), (4000, 4060)],  # exon1 first (TSS far), exon2 internal
        cds=[(4000, 4060)],  # 60 bp coding, == exon2 (no 3' UTR)
    )
    gtf = _write_gtf(tmp_path, rows)
    cre = _write_cre(tmp_path, [(4060, 4255, "dELS")])  # adjacent, not nested
    win = _write_one_window(tmp_path, "w_cds60_ccre195", 4000, 4255)
    defined = _whole_chrom_defined()

    # ccre_flank=0 (v4) keeps the cCRE exactly 4060–4255.
    beds = build_region_beds(
        gtf, cre, defined, tss_radius=256, ccre_flank=0, tss_pc_only=True
    )

    maj = label_windows_bp_majority(
        win, beds, functional_threshold=0.20, priority=V4_PRIORITY
    )
    r = _row(maj, "w_cds60_ccre195")
    assert r["label"] == "ccre_non_promoter"
    assert r["cds_frac"] == pytest.approx(60 / 255)
    assert r["ccre_non_promoter_frac"] == pytest.approx(195 / 255)
    assert r["functional_frac"] == pytest.approx(1.0)

    # The legacy presence rule disagrees — it would label this `cds`.
    pres = label_windows(
        win, beds, functional_threshold=0.20, priority=list(REGION_LABELS)
    )
    assert _row(pres, "w_cds60_ccre195")["label"] == "cds"


def _divergent_fixture(tmp_path: Path) -> tuple[Path, Path, Path, GenomicSet]:
    """A standalone lncRNA exon+TSS, a far-away PC gene (so the Ensembl
    assertion passes), and a far-away non-PLS cCRE. Window covers only the
    lncRNA's first exon / own TSS.
    """
    rows = _gene_rows(
        "genePC", "protein_coding", "+", 1000, 1500, [(1000, 1500)], cds=[(1200, 1400)]
    ) + _gene_rows("geneL", "lncRNA", "+", 20000, 20500, [(20000, 20500)])
    gtf = _write_gtf(tmp_path, rows)
    cre = _write_cre(tmp_path, [(30000, 30200, "dELS")])
    win = _write_one_window(tmp_path, "w_lncrna_tss", 20000, 20255)
    return gtf, cre, win, _whole_chrom_defined()


def test_bp_majority_pc_only_tss_collapses_standalone_ncrna(tmp_path):
    """PC-only TSS region: a standalone lncRNA's own TSS no longer competes,
    so its first-exon window is `ncrna_exon` (not `tss_region_and_utr5`).
    """
    gtf, cre, win, defined = _divergent_fixture(tmp_path)

    beds_all = build_region_beds(
        gtf, cre, defined, tss_radius=256, ccre_flank=0, tss_pc_only=False
    )
    beds_pc = build_region_beds(
        gtf, cre, defined, tss_radius=256, ccre_flank=0, tss_pc_only=True
    )

    # All-transcript TSS band: the lncRNA's own TSS lands the window in `tss`.
    r_all = _row(
        label_windows_bp_majority(
            win, beds_all, functional_threshold=0.20, priority=V4_PRIORITY
        ),
        "w_lncrna_tss",
    )
    assert r_all["label"] == "tss_region_and_utr5"

    # PC-only TSS band: no PC TSS here, so the window is its lncRNA exon.
    r_pc = _row(
        label_windows_bp_majority(
            win, beds_pc, functional_threshold=0.20, priority=V4_PRIORITY
        ),
        "w_lncrna_tss",
    )
    assert r_pc["label"] == "ncrna_exon"
    assert r_pc["ncrna_exon_frac"] == pytest.approx(1.0)
    assert r_pc["tss_region_and_utr5_frac"] == pytest.approx(0.0)


def test_bp_majority_priority_flip_tss_over_ncrna(tmp_path):
    """When `tss` and `ncrna` both cover a window, the priority order decides:
    v4 (`tss > ncrna`) → `tss_region_and_utr5`; v3 order → `ncrna_exon`.
    Per-label disjoint fracs are *not* invariant to the order (unlike the
    overlapping fracs of `label_windows`).
    """
    gtf, cre, win, defined = _divergent_fixture(tmp_path)
    # All-transcript band so both classes cover the window fully.
    beds = build_region_beds(
        gtf, cre, defined, tss_radius=256, ccre_flank=0, tss_pc_only=False
    )

    v4 = _row(
        label_windows_bp_majority(
            win, beds, functional_threshold=0.20, priority=V4_PRIORITY
        ),
        "w_lncrna_tss",
    )
    assert v4["label"] == "tss_region_and_utr5"

    v3_order = _row(
        label_windows_bp_majority(
            win, beds, functional_threshold=0.20, priority=list(REGION_LABELS)
        ),
        "w_lncrna_tss",
    )
    assert v3_order["label"] == "ncrna_exon"


def test_bp_majority_disjoint_fracs_partition_functional(synth):
    """Invariant: the five disjoint per-label fracs sum to functional_frac
    (≤ 1.0) for every window — the defining property of a true partition.
    """
    beds = build_region_beds(
        synth["gtf"], synth["cre"], synth["defined"], tss_radius=256, ccre_flank=500
    )
    df = label_windows_bp_majority(
        synth["windows"], beds, functional_threshold=0.20, priority=V4_PRIORITY
    )
    frac_cols = [f"{lbl}_frac" for lbl in REGION_LABELS]
    summed = df.select(frac_cols).to_numpy().sum(axis=1)
    fnc = df["functional_frac"].to_numpy()
    assert summed == pytest.approx(fnc, abs=1e-9)
    assert (fnc <= 1.0 + 1e-9).all()
    valid = set(REGION_LABELS) | {BACKGROUND_LABEL}
    assert set(df["label"].unique().to_list()) <= valid


def test_bp_majority_background_below_threshold(synth):
    """Windows under the functional threshold (or with no functional overlap)
    are `background`, same gate as label_windows.
    """
    beds = build_region_beds(
        synth["gtf"], synth["cre"], synth["defined"], tss_radius=256, ccre_flank=500
    )
    df = label_windows_bp_majority(
        synth["windows"], beds, functional_threshold=0.20, priority=V4_PRIORITY
    )
    assert _row(df, "w_cds10_intron90")["label"] == BACKGROUND_LABEL  # 10% < 20%
    assert _row(df, "w_pls_isolated")["label"] == BACKGROUND_LABEL  # nothing functional


# ============================================================================
# write_subset_hf_readme tests
# ============================================================================


from marin_dna.pipelines.zoonomia_projection_dataset.region_labels import (  # noqa: E402
    write_subset_hf_readme,
)


# Synthetic six-label composition TSV (counts chosen to be distinct so the
# tests can hand-verify which row each subset's card pulls).
_SUBSET_LABEL_COUNTS: dict[str, tuple[str, int]] = {
    "v3_cds": ("cds", 402_393),
    "v3_utr3": ("utr3", 54_828),
    "v3_ncrna_exon": ("ncrna_exon", 93_064),
    "v3_tss_region_and_utr5": ("tss_region_and_utr5", 40_823),
    "v3_ccre_non_promoter": ("ccre_non_promoter", 468_131),
    "v3_bg": ("background", 77_615),
}


def _write_synth_composition_tsv(path: Path) -> int:
    """Mirror ``rule region_label_composition``'s output schema.

    Returns the total anchor count, so callers can assert against
    ``n_total`` in the rendered README.
    """
    labels = [lc[0] for lc in _SUBSET_LABEL_COUNTS.values()]
    counts = [lc[1] for lc in _SUBSET_LABEL_COUNTS.values()]
    n_total = sum(counts)
    rows = []
    for lbl, n in zip(labels, counts):
        rows.append((lbl, n, None, None, None, None, n / n_total))
    # The real rule also appends background_intronic / background_intergenic
    # rows that split background; include them so the loader's "ignore the
    # split rows" path is exercised.
    bg_intronic = 54_141
    bg_intergenic = 77_615 - 54_141
    rows.append(
        (
            "background_intronic",
            bg_intronic,
            None,
            None,
            None,
            None,
            bg_intronic / n_total,
        )
    )
    rows.append(
        (
            "background_intergenic",
            bg_intergenic,
            None,
            None,
            None,
            None,
            bg_intergenic / n_total,
        )
    )

    df = pl.DataFrame(
        rows,
        schema=[
            ("label", pl.Utf8),
            ("n_windows", pl.Int64),
            ("mean_functional_frac", pl.Float64),
            ("mean_gene_body_frac", pl.Float64),
            ("mean_intron_frac", pl.Float64),
            ("mean_intergenic_frac", pl.Float64),
            ("fraction_of_total", pl.Float64),
        ],
        orient="row",
    )
    df.write_csv(path, separator="\t")
    return n_total


@pytest.fixture
def synth_composition(tmp_path: Path) -> tuple[Path, int]:
    p = tmp_path / "min0.20.composition.tsv"
    n_total = _write_synth_composition_tsv(p)
    return p, n_total


@pytest.mark.parametrize("subset", list(_SUBSET_LABEL_COUNTS))
def test_write_subset_hf_readme_renders_per_subset_card(
    subset: str, tmp_path: Path, synth_composition: tuple[Path, int]
) -> None:
    composition_tsv, n_total = synth_composition
    out = tmp_path / f"{subset}.README.md"
    label, count = _SUBSET_LABEL_COUNTS[subset]

    # Total samples = anchors × per-anchor projection fanout × 2 (RC).
    # Use a distinct number so the assertion uniquely matches.
    n_samples = 12_345_678
    write_subset_hf_readme(
        subset,
        out,
        commit_sha="0123456789abcdef0123456789abcdef01234567",
        hf_owner="bolinas-dna",
        pipeline_version="v1",
        ensembl_release=115,
        functional_threshold=0.20,
        tss_radius=256,
        ccre_flank=500,
        priority=[
            "cds",
            "utr3",
            "ncrna_exon",
            "tss_region_and_utr5",
            "ccre_non_promoter",
        ],
        composition_tsv=composition_tsv,
        n_samples=n_samples,
    )

    body = out.read_text()
    # Repo name + label
    assert f"bolinas-dna/zoonomia-v1-{subset}" in body
    assert f"`{label}`" in body
    # Commit-pinned permalink (short SHA in heading, full SHA in URL)
    assert "0123456789ab" in body
    assert (
        "https://github.com/Open-Athena/marin-dna/tree/"
        "0123456789abcdef0123456789abcdef01234567/"
        "snakemake/zoonomia_projection_dataset" in body
    )
    # Partition stats (anchor count for this subset + grand total + samples)
    assert f"{count:,}" in body
    assert f"{n_total:,}" in body
    assert f"{n_samples:,}" in body
    # Priority chain (background appended)
    assert (
        "`cds` > `utr3` > `ncrna_exon` > `tss_region_and_utr5` > `ccre_non_promoter` > `background`"
        in body
    )
    # YAML tags minimal (biology/genomics/DNA only — no extras).
    front_matter, _, _ = body.partition("---\n\n")
    assert front_matter.count("- ") == 3, (
        f"expected exactly 3 YAML tags, got: {front_matter!r}"
    )
    # Sibling links: five other subsets (not the current one)
    for other in _SUBSET_LABEL_COUNTS:
        link = f"https://huggingface.co/datasets/bolinas-dna/zoonomia-v1-{other}"
        if other == subset:
            # Cross-link to v1 / v2 is allowed, but no self-link.
            assert (
                f"-v1-{subset}](https://huggingface.co/datasets/bolinas-dna/zoonomia-v1-{subset})"
                not in body.split("Five sibling v3 subsets")[1]
            )
        else:
            assert link in body, f"missing sibling link for {other}"


def test_write_subset_hf_readme_v3_specific_placeholders(
    tmp_path: Path, synth_composition: tuple[Path, int]
) -> None:
    """v3_tss_region_and_utr5's blurb uses {tss_radius}; v3_ccre_non_promoter's
    uses {ccre_flank}. Render both and assert the values appear.
    """
    composition_tsv, _ = synth_composition

    out_tss = tmp_path / "tss.md"
    write_subset_hf_readme(
        "v3_tss_region_and_utr5",
        out_tss,
        commit_sha="a" * 40,
        hf_owner="bolinas-dna",
        pipeline_version="v1",
        ensembl_release=115,
        functional_threshold=0.20,
        tss_radius=999,
        ccre_flank=500,
        priority=[
            "cds",
            "utr3",
            "ncrna_exon",
            "tss_region_and_utr5",
            "ccre_non_promoter",
        ],
        composition_tsv=composition_tsv,
        n_samples=1,
    )
    assert "999 bp" in out_tss.read_text()

    out_ccre = tmp_path / "ccre.md"
    write_subset_hf_readme(
        "v3_ccre_non_promoter",
        out_ccre,
        commit_sha="a" * 40,
        hf_owner="bolinas-dna",
        pipeline_version="v1",
        ensembl_release=115,
        functional_threshold=0.20,
        tss_radius=256,
        ccre_flank=777,
        priority=[
            "cds",
            "utr3",
            "ncrna_exon",
            "tss_region_and_utr5",
            "ccre_non_promoter",
        ],
        composition_tsv=composition_tsv,
        n_samples=1,
    )
    assert "777 bp" in out_ccre.read_text()


def test_write_subset_hf_readme_rejects_unknown_subset(
    tmp_path: Path, synth_composition: tuple[Path, int]
) -> None:
    composition_tsv, _ = synth_composition
    with pytest.raises(ValueError, match="unknown subset"):
        write_subset_hf_readme(
            "v3_bogus",
            tmp_path / "bogus.md",
            commit_sha="a" * 40,
            hf_owner="bolinas-dna",
            pipeline_version="v1",
            ensembl_release=115,
            functional_threshold=0.20,
            tss_radius=256,
            ccre_flank=500,
            priority=[
                "cds",
                "utr3",
                "ncrna_exon",
                "tss_region_and_utr5",
                "ccre_non_promoter",
            ],
            composition_tsv=composition_tsv,
            n_samples=1,
        )


def test_write_subset_hf_readme_requires_complete_composition(
    tmp_path: Path,
) -> None:
    """Composition TSV missing a canonical label should fail loudly, not
    silently drop the subset's anchor count from the rendered card.
    """
    incomplete = tmp_path / "incomplete.tsv"
    pl.DataFrame(
        {
            "label": ["cds", "utr3"],
            "n_windows": [100, 50],
            "mean_functional_frac": [None, None],
            "mean_gene_body_frac": [None, None],
            "mean_intron_frac": [None, None],
            "mean_intergenic_frac": [None, None],
            "fraction_of_total": [0.67, 0.33],
        }
    ).write_csv(incomplete, separator="\t")

    with pytest.raises(AssertionError, match="missing labels"):
        write_subset_hf_readme(
            "v3_cds",
            tmp_path / "out.md",
            commit_sha="a" * 40,
            hf_owner="bolinas-dna",
            pipeline_version="v1",
            ensembl_release=115,
            functional_threshold=0.20,
            tss_radius=256,
            ccre_flank=500,
            priority=[
                "cds",
                "utr3",
                "ncrna_exon",
                "tss_region_and_utr5",
                "ccre_non_promoter",
            ],
            composition_tsv=incomplete,
            n_samples=1,
        )


# ============================================================================
# region_label_composition_table + v4 dataset card (issue #227)
# ============================================================================


from marin_dna.pipelines.zoonomia_projection_dataset.region_labels import (  # noqa: E402
    _SUBSET_TO_LABEL_V4,
    region_label_composition_table,
    write_subset_hf_readme_v4,
)


def test_region_label_composition_table():
    """Per-label counts + the background intronic/intergenic subsplit, in the
    schema the card loader (_read_composition) expects.
    """
    df = pl.DataFrame(
        {
            "label": [
                "cds",
                "cds",
                "background",
                "background",
                "background",
                "ccre_non_promoter",
            ],
            "functional_frac": [1.0, 0.8, 0.05, 0.0, 0.1, 0.9],
            # bg gene_body_frac: 0.9>0.5 intronic, 0.0<=0.5 intergenic, 0.7>0.5 intronic
            "gene_body_frac": [1.0, 1.0, 0.9, 0.0, 0.7, 0.2],
            "intron_frac": [0.0, 0.2, 0.9, 0.0, 0.7, 0.1],
            "intergenic_frac": [0.0, 0.0, 0.1, 1.0, 0.3, 0.8],
        }
    )
    out = region_label_composition_table(df)
    rows = {r["label"]: r for r in out.iter_rows(named=True)}

    assert rows["cds"]["n_windows"] == 2
    assert rows["cds"]["fraction_of_total"] == pytest.approx(2 / 6)
    assert rows["ccre_non_promoter"]["n_windows"] == 1
    assert rows["background"]["n_windows"] == 3
    # background subsplit by gene-body coverage > 0.5
    assert rows["background_intronic"]["n_windows"] == 2
    assert rows["background_intergenic"]["n_windows"] == 1
    assert set(out.columns) == {
        "label",
        "n_windows",
        "mean_functional_frac",
        "mean_gene_body_frac",
        "mean_intron_frac",
        "mean_intergenic_frac",
        "fraction_of_total",
    }


@pytest.mark.parametrize("subset", list(_SUBSET_TO_LABEL_V4))
def test_write_subset_hf_readme_v4_renders_card(
    subset: str, tmp_path: Path, synth_composition: tuple[Path, int]
) -> None:
    composition_tsv, n_total = synth_composition
    out = tmp_path / f"{subset}.README.md"
    label = _SUBSET_TO_LABEL_V4[subset]
    n_samples = 99_887_766

    write_subset_hf_readme_v4(
        subset,
        out,
        commit_sha="0123456789abcdef0123456789abcdef01234567",
        hf_owner="bolinas-dna",
        pipeline_version="v1",
        ensembl_release=115,
        functional_threshold=0.20,
        tss_radius=256,
        ccre_flank=0,
        priority=V4_PRIORITY,
        composition_tsv=composition_tsv,
        n_samples=n_samples,
    )
    body = out.read_text()

    assert f"bolinas-dna/zoonomia-v1-{subset}" in body
    assert f"`{label}`" in body
    assert "0123456789ab" in body
    assert f"{n_samples:,}" in body
    assert f"{n_total:,}" in body
    # v4-specific: bp-priority + window-majority scheme and the flipped chain.
    assert "Base-pair priority" in body
    assert "Window majority" in body
    assert (
        "`cds` > `utr3` > `tss_region_and_utr5` > `ncrna_exon` > "
        "`ccre_non_promoter` > `background`" in body
    )
    assert "issues/221" in body  # links the design issue
    # Exactly the three canonical tags.
    front_matter, _, _ = body.partition("---\n\n")
    assert front_matter.count("- ") == 3
    # No self-link among the five siblings.
    assert "Five sibling v4 subsets" in body


def test_write_subset_hf_readme_v4_pc_only_and_flank_wording(
    tmp_path: Path, synth_composition: tuple[Path, int]
) -> None:
    """The TSS card states PC-only; the cCRE card states no-flank."""
    composition_tsv, _ = synth_composition

    out_tss = tmp_path / "tss.md"
    write_subset_hf_readme_v4(
        "v4_tss_region_and_utr5",
        out_tss,
        commit_sha="a" * 40,
        hf_owner="bolinas-dna",
        pipeline_version="v1",
        ensembl_release=115,
        functional_threshold=0.20,
        tss_radius=256,
        ccre_flank=0,
        priority=V4_PRIORITY,
        composition_tsv=composition_tsv,
        n_samples=1,
    )
    tss_body = out_tss.read_text()
    assert "protein-coding-only" in tss_body
    assert "256 bp" in tss_body

    out_ccre = tmp_path / "ccre.md"
    write_subset_hf_readme_v4(
        "v4_ccre_non_promoter",
        out_ccre,
        commit_sha="a" * 40,
        hf_owner="bolinas-dna",
        pipeline_version="v1",
        ensembl_release=115,
        functional_threshold=0.20,
        tss_radius=256,
        ccre_flank=0,
        priority=V4_PRIORITY,
        composition_tsv=composition_tsv,
        n_samples=1,
    )
    assert "no flank" in out_ccre.read_text()


def test_write_subset_hf_readme_v4_rejects_unknown_subset(
    tmp_path: Path, synth_composition: tuple[Path, int]
) -> None:
    composition_tsv, _ = synth_composition
    with pytest.raises(ValueError, match="unknown subset"):
        write_subset_hf_readme_v4(
            "v4_bogus",
            tmp_path / "bogus.md",
            commit_sha="a" * 40,
            hf_owner="bolinas-dna",
            pipeline_version="v1",
            ensembl_release=115,
            functional_threshold=0.20,
            tss_radius=256,
            ccre_flank=0,
            priority=V4_PRIORITY,
            composition_tsv=composition_tsv,
            n_samples=1,
        )
