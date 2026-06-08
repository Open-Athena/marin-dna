"""Tests for SGE (saturation genome editing) dataset construction."""

from pathlib import Path

import polars as pl
import pytest

from marin_dna.data.utils import load_annotation
from marin_dna.pipelines.evals.sge import (
    _CALIBRATION_SCHEMA,
    _select_numeric_calibration,
    annotate_sge_variants,
    attach_assay_facts,
    attach_author_class_harmonized,
    attach_calibrated_class,
    attach_function_direction,
    build_mavedb_metadata,
    extract_assay_facts,
    extract_score_calibrations,
    load_mavedb_genomic_scoreset,
    load_mavedb_transcript_scoreset,
    normalize_brca1_findlay,
    recode_hgvs_c_to_genomic,
)
from marin_dna.pipelines.evals.trait_intervals import get_exon, get_tss


# --------------------------------------------------------------------------- #
# normalize_brca1_findlay
# --------------------------------------------------------------------------- #
class TestNormalizeBrca1Findlay:
    def _raw(self) -> pl.DataFrame:
        # Columns mirror the Findlay xlsx (read with header=2).
        return pl.DataFrame(
            {
                "gene": ["BRCA1", "BRCA1", "BRCA1"],
                "chromosome": [17, 17, 17],
                "position (hg19)": [41276135, 41276134, 41276133],
                "reference": ["T", "C", "A"],
                "alt": ["G", "T", "AT"],  # last is an indel -> dropped by filter_snp
                "consequence": ["Splice region", "Missense", "Intronic"],
                "function.score.mean": [-0.37, -1.5, 0.02],
                "func.class": ["FUNC", "LOF", "FUNC"],
                "p.nonfunctional": [5e-5, 0.99, 1e-6],
                "function.score.r1": [-0.5, -1.6, 0.0],
                "function.score.r2": [-0.2, -1.4, 0.04],
                "mean.rna.score": [None, None, None],
            }
        )

    def test_schema_and_renames(self) -> None:
        out = normalize_brca1_findlay(self._raw(), mavedb_urn="urn:test:1")
        assert out.height == 2  # indel dropped by filter_snp
        assert out["chrom"].dtype == pl.Utf8 and out["chrom"].to_list() == ["17", "17"]
        assert out["pos"].dtype == pl.Int64
        assert out["author_function_score_mean"].to_list() == [-0.37, -1.5]
        assert out["author_func_class"].to_list() == ["FUNC", "LOF"]
        assert out["assay"].unique().to_list() == ["sge"]
        assert out["mavedb_urn"].unique().to_list() == ["urn:test:1"]
        # Standard coords come first, then every original column under author_.
        assert out.columns[:7] == [
            "chrom",
            "pos",
            "ref",
            "alt",
            "gene",
            "assay",
            "mavedb_urn",
        ]
        for c in (
            "author_p_nonfunctional",
            "author_function_score_r1",
            "author_function_score_r2",
            "author_mean_rna_score",
            "author_consequence",
            "author_position_hg19",  # original hg19 coord preserved
        ):
            assert c in out.columns

    def test_author_consequence_distinct_from_pipeline_consequence(self) -> None:
        # Findlay's own consequence is preserved under a distinct name so the
        # later VEP `consequence` column doesn't collide with it.
        out = normalize_brca1_findlay(self._raw(), mavedb_urn="urn:test:1")
        assert "consequence" not in out.columns
        assert out["author_consequence"].to_list() == ["Splice region", "Missense"]

    def test_bad_class_raises(self) -> None:
        raw = self._raw().with_columns(pl.lit("WEIRD").alias("func.class"))
        with pytest.raises(AssertionError, match="func.class"):
            normalize_brca1_findlay(raw, mavedb_urn="urn:test:1")

    def test_null_score_raises(self) -> None:
        raw = self._raw().with_columns(
            pl.when(pl.col("alt") == "G")
            .then(None)
            .otherwise(pl.col("function.score.mean"))
            .alias("function.score.mean")
        )
        with pytest.raises(AssertionError, match="null function score"):
            normalize_brca1_findlay(raw, mavedb_urn="urn:test:1")

    def test_null_position_raises(self) -> None:
        # A blank hg19 coordinate must fail loud, not silently vanish in liftover.
        raw = self._raw().with_columns(
            pl.when(pl.col("alt") == "G")
            .then(None)
            .otherwise(pl.col("position (hg19)"))
            .alias("position (hg19)")
        )
        with pytest.raises(AssertionError, match="null pos"):
            normalize_brca1_findlay(raw, mavedb_urn="urn:test:1")

    def test_null_func_class_tolerated(self) -> None:
        # A null func.class is preserved-but-non-signal metadata; it must NOT crash the
        # build (only an *unexpected* category should).
        raw = self._raw().with_columns(
            pl.when(pl.col("alt") == "G")
            .then(None)
            .otherwise(pl.col("func.class"))
            .alias("func.class")
        )
        out = normalize_brca1_findlay(raw, mavedb_urn="urn:test:1")
        assert out.height == 2  # both SNVs kept (indel dropped); null class tolerated
        assert out["author_func_class"].null_count() == 1


# --------------------------------------------------------------------------- #
# load_mavedb_genomic_scoreset
# --------------------------------------------------------------------------- #
class TestLoadMavedbGenomic:
    def _csv(self, tmp_path: Path) -> Path:
        # Genome-targeted MaveDB scores CSV (NC_…:g. hgvs_nt). Mix of SNVs (kept),
        # an intronic SNV (kept — no transcript mapping needed), a deletion, a
        # delins, an MNV, and a null-score SNV (all dropped).
        text = (
            "accession,hgvs_nt,hgvs_pro,score,functional_consequence\n"
            "x#1,NC_000002.12:g.214728667A>G,p.?,-1.0,abnormal\n"
            "x#2,NC_000002.12:g.214767000C>T,p.?,0.1,normal\n"  # deep intronic SNV
            "x#3,NC_000023.11:g.100A>G,p.?,-0.5,abnormal\n"  # chrX
            "x#4,NC_000002.12:g.214728632_214728634del,p.?,-2.0,abnormal\n"  # del
            "x#5,NC_000002.12:g.214728700_214728702delinsAAG,p.?,-1.5,abnormal\n"  # delins
            "x#6,NC_000002.12:g.214728680G>A,p.?,,normal\n"  # null score
            "x#7,NC_000002.12:g.214728690AC>GT,p.?,0.3,normal\n"  # MNV
        )
        p = tmp_path / "scores.csv"
        p.write_text(text)
        return p

    def test_parses_snvs_drops_nonsnv_and_nullscore(self, tmp_path: Path) -> None:
        V = load_mavedb_genomic_scoreset(
            self._csv(tmp_path), gene="BARD1", mavedb_urn="urn:test:bard1"
        )
        assert V.height == 3  # x#1, x#2, x#3
        assert V["chrom"].to_list() == ["2", "2", "X"]
        assert V["pos"].to_list() == [214728667, 214767000, 100]
        assert V["ref"].to_list() == ["A", "C", "A"]
        assert V["alt"].to_list() == ["G", "T", "G"]
        assert V["function_score"].to_list() == [-1.0, 0.1, -0.5]
        # del/delins/MNV positions never appear.
        assert 214728632 not in V["pos"].to_list()
        assert 214728690 not in V["pos"].to_list()

    def test_provenance_and_author_columns(self, tmp_path: Path) -> None:
        V = load_mavedb_genomic_scoreset(
            self._csv(tmp_path), gene="BARD1", mavedb_urn="urn:test:bard1"
        )
        assert V["gene"].unique().to_list() == ["BARD1"]
        assert V["assay"].unique().to_list() == ["sge"]
        assert V["mavedb_urn"].unique().to_list() == ["urn:test:bard1"]
        assert V.columns[:8] == [
            "chrom",
            "pos",
            "ref",
            "alt",
            "gene",
            "assay",
            "mavedb_urn",
            "function_score",
        ]
        # Original columns preserved verbatim, author_-prefixed.
        for c in ("author_hgvs_nt", "author_score", "author_functional_consequence"):
            assert c in V.columns
        # function_score mirrors the study's `score` column.
        assert V["function_score"].to_list() == V["author_score"].to_list()

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.csv"
        p.write_text("accession,hgvs_nt,hgvs_pro\nx#1,NC_000002.12:g.1A>G,p.?\n")
        with pytest.raises(AssertionError, match="missing 'score'"):
            load_mavedb_genomic_scoreset(p, gene="BARD1", mavedb_urn="urn:test:bard1")

    def test_nonnumeric_score_dropped_not_crash(self, tmp_path: Path) -> None:
        # A non-numeric score token must coerce to null and drop the row (strict=False),
        # not abort the whole load.
        p = tmp_path / "s.csv"
        p.write_text(
            "accession,hgvs_nt,score\n"
            "x#1,NC_000002.12:g.100A>G,-1.0\n"
            "x#2,NC_000002.12:g.200C>T,NA\n"  # non-numeric -> null -> dropped
        )
        V = load_mavedb_genomic_scoreset(p, gene="BARD1", mavedb_urn="urn:x")
        assert V.height == 1
        assert V["pos"].to_list() == [100]

    def test_multivariant_semicolon_dropped(self, tmp_path: Path) -> None:
        # A ';'-joined multi-variant hgvs_nt must drop cleanly, not form a chimera with
        # chrom from the first sub-variant and pos/ref/alt from the last.
        p = tmp_path / "s.csv"
        p.write_text(
            "accession,hgvs_nt,score\n"
            "x#1,NC_000002.12:g.100A>G,-1.0\n"
            "x#2,NC_000002.12:g.100A>G;NC_000003.11:g.5C>T,0.2\n"  # chimera -> dropped
        )
        V = load_mavedb_genomic_scoreset(p, gene="BARD1", mavedb_urn="urn:x")
        assert V.height == 1
        assert V["chrom"].to_list() == ["2"]
        assert V["pos"].to_list() == [100]


# --------------------------------------------------------------------------- #
# recode_hgvs_c_to_genomic + load_mavedb_transcript_scoreset (transcript-targeted)
# --------------------------------------------------------------------------- #
# Mock c.->genomic mapper (the pyhgvs+cdot impl is exercised by the real build):
# c. HGVS -> (chrom, pos, ref, alt) | None.
_RECODE = {
    "ENST:c.7436A>C": ("13", 32356428, "A", "C"),
    "ENST:c.7436-10T>A": ("13", 32356418, "T", "A"),  # intronic
    "ENST:c.5A>G": ("13", 31000000, "A", "G"),
    "ENST:c.bad": None,  # unmappable
}


def _fake_mapper(hgvs_c: str) -> tuple | None:
    return _RECODE.get(hgvs_c)


class TestRecodeAndTranscriptLoader:
    def test_recode_maps_and_drops_unmapped(self) -> None:
        out = recode_hgvs_c_to_genomic(
            ["ENST:c.7436A>C", "ENST:c.7436-10T>A", "ENST:c.bad"],
            mapper=_fake_mapper,
        )
        assert out.height == 2  # c.bad dropped (mapper -> None)
        assert out["chrom"].to_list() == ["13", "13"]
        assert out["pos"].to_list() == [32356428, 32356418]  # intronic mapped
        assert out["ref"].to_list() == ["A", "T"]
        assert out["alt"].to_list() == ["C", "A"]

    def test_recode_floor_guard_raises_on_mass_unmap(self) -> None:
        # A large unmapped fraction (e.g. a cdot.cc outage swallowed by the mapper)
        # must fail loud rather than emit a near-empty recode parquet.
        many = [f"ENST:c.bad{i}A>G" for i in range(10)]  # none in _RECODE -> all None
        with pytest.raises(AssertionError, match="mapped"):
            recode_hgvs_c_to_genomic(many, mapper=_fake_mapper)

    def test_transcript_loader_join_keeps_intronic_drops_unrecoded(
        self, tmp_path: Path
    ) -> None:
        csv = tmp_path / "scores.csv"
        csv.write_text(
            "accession,hgvs_nt,hgvs_pro,score,functional_classification\n"
            "u#1,ENST:c.7436A>C,p.?,-1.0,abnormal\n"
            "u#2,ENST:c.7436-10T>A,p.?,0.2,normal\n"  # intronic -> recoded -> kept
            "u#3,ENST:c.9del,p.?,-2.0,abnormal\n"  # indel -> not recoded -> dropped
            "u#4,ENST:c.5A>G,p.?,,normal\n"  # recoded but null score -> dropped
        )
        recoded = recode_hgvs_c_to_genomic(
            ["ENST:c.7436A>C", "ENST:c.7436-10T>A", "ENST:c.5A>G"], mapper=_fake_mapper
        )
        V = load_mavedb_transcript_scoreset(
            csv, recoded, gene="BRCA2", mavedb_urn="urn:test:brca2"
        )
        assert V.height == 2  # u#1 + u#2 (u#3 unrecoded, u#4 null score)
        assert sorted(V["pos"].to_list()) == [32356418, 32356428]
        assert V["gene"].unique().to_list() == ["BRCA2"]
        assert "author_functional_classification" in V.columns
        assert V["function_score"].to_list() == V["author_score"].to_list()
        # the intronic variant is retained.
        assert V.filter(pl.col("author_hgvs_nt") == "ENST:c.7436-10T>A").height == 1


# --------------------------------------------------------------------------- #
# annotate_sge_variants
# --------------------------------------------------------------------------- #
class FakeGenome:
    """(chrom, 1-based-pos) -> base; mimics Genome.__call__(chrom, start, end)."""

    def __init__(self, table: dict[tuple[str, int], str]) -> None:
        self._table = table

    def __call__(self, chrom: str, start: int, end: int, strand: str = "+") -> str:
        assert end == start + 1, "fake genome only supports single-base lookups"
        return self._table[(chrom, end)]


SYNTHETIC_GTF = """\
1\thavana\ttranscript\t1001\t2000\t.\t+\t.\tgene_id "ENSG_PLUS"; transcript_id "T1"; transcript_biotype "protein_coding";
1\thavana\texon\t1001\t1100\t.\t+\t.\tgene_id "ENSG_PLUS"; transcript_id "T1"; transcript_biotype "protein_coding";
1\thavana\texon\t1500\t1600\t.\t+\t.\tgene_id "ENSG_PLUS"; transcript_id "T1"; transcript_biotype "protein_coding";
1\thavana\ttranscript\t5000\t6000\t.\t+\t.\tgene_id "ENSG_LNCRNA"; transcript_id "T3"; transcript_biotype "lncRNA";
1\thavana\texon\t5000\t6000\t.\t+\t.\tgene_id "ENSG_LNCRNA"; transcript_id "T3"; transcript_biotype "lncRNA";
"""


@pytest.fixture
def intervals(tmp_path: Path) -> dict[str, pl.DataFrame]:
    gtf = tmp_path / "synthetic.gtf"
    gtf.write_text(SYNTHETIC_GTF)
    ann = load_annotation(str(gtf))
    nc = pl.col("transcript_biotype") != "protein_coding"
    return {
        "tss_pc": get_tss(ann),
        "tss_nc": get_tss(ann, biotype_filter=nc),
        "exon_pc": get_exon(ann),
        "exon_nc": get_exon(ann, biotype_filter=nc),
    }


def _write_consequences(tmp_path: Path, rows: list[dict]) -> str:
    path = tmp_path / "1.parquet"
    pl.DataFrame(
        rows,
        schema={
            "chrom": pl.String,
            "pos": pl.Int64,
            "ref": pl.String,
            "alt": pl.String,
            "consequence": pl.String,
            "consequence_cre": pl.String,
        },
    ).write_parquet(path)
    return str(path)


def _sge_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": ["1", "1", "1"],
            "pos": [1150, 1700, 1750],
            "ref": ["A", "C", "G"],
            "alt": ["T", "G", "A"],
            "gene": ["G1", "G1", "G1"],
            "author_function_score_mean": [-1.0, 0.1, -2.0],
            "author_func_class": ["LOF", "FUNC", "LOF"],
            "assay": ["sge", "sge", "sge"],
            "mavedb_urn": ["urn:test:1", "urn:test:1", "urn:test:1"],
        }
    )


def _cons_rows(V: pl.DataFrame, consequences: list[str]) -> list[dict]:
    return [
        {
            "chrom": "1",
            "pos": p,
            "ref": r,
            "alt": a,
            "consequence": c,
            "consequence_cre": c,
        }
        for p, r, a, c in zip(
            V["pos"].to_list(),
            V["ref"].to_list(),
            V["alt"].to_list(),
            consequences,
        )
    ]


def _annotate(
    tmp_path,
    intervals,
    V,
    consequences,
    exclude,
    *,
    lift=False,
    consequence_groups=None,
    consequence_group_allowlist=None,
):
    genome = FakeGenome(
        {("1", p): r for p, r in zip(V["pos"].to_list(), V["ref"].to_list())}
    )
    cons = _write_consequences(tmp_path, _cons_rows(V, consequences))
    return annotate_sge_variants(
        V,
        genome=genome,
        consequence_paths=[cons],
        chroms=["1"],
        exon_pc=intervals["exon_pc"],
        exon_nc=intervals["exon_nc"],
        tss_pc=intervals["tss_pc"],
        tss_nc=intervals["tss_nc"],
        exon_proximal_dist=30,
        tss_proximal_dist=1000,
        exclude_consequences=exclude,
        consequence_groups=consequence_groups or {},
        consequence_group_allowlist=consequence_group_allowlist,
        lift=lift,
        name="test",
    )


class TestAnnotateSgeVariants:
    def test_exclude_drops_high_impact_keeps_rest(self, tmp_path, intervals) -> None:
        V = _sge_frame()
        out = _annotate(
            tmp_path,
            intervals,
            V,
            ["missense_variant", "stop_gained", "intron_variant"],
            exclude=["stop_gained"],
        )
        # The stop_gained variant (pos 1700) is dropped; the missense + intron stay.
        assert out.height == 2
        assert 1700 not in out["pos"].to_list()
        assert set(out["consequence"].to_list()) == {
            "missense_variant",
            "intron_variant",
        }

    def test_intronic_retained(self, tmp_path, intervals) -> None:
        V = _sge_frame()
        out = _annotate(
            tmp_path,
            intervals,
            V,
            ["missense_variant", "synonymous_variant", "intron_variant"],
            exclude=["stop_gained", "splice_donor_variant"],
        )
        # Nothing in the exclude set is present, so all three are kept.
        assert out.height == 3
        assert "intron_variant" in out["consequence"].to_list()

    def test_author_columns_preserved(self, tmp_path, intervals) -> None:
        V = _sge_frame()
        out = _annotate(
            tmp_path,
            intervals,
            V,
            ["missense_variant", "synonymous_variant", "intron_variant"],
            exclude=[],
        )
        for c in (
            "author_function_score_mean",
            "author_func_class",
            "assay",
            "mavedb_urn",
            "gene",
        ):
            assert c in out.columns
        # The author function score travels with its variant (sorted by COORDINATES).
        got = dict(
            zip(out["pos"].to_list(), out["author_function_score_mean"].to_list())
        )
        assert got == {1150: -1.0, 1700: 0.1, 1750: -2.0}
        for c in ("consequence", "consequence_final", "distance_exon", "distance_tss"):
            assert c in out.columns

    def test_ref_alt_swap_raises(self, tmp_path, intervals) -> None:
        # Genome base at pos 1700 is "C" but the variant lists ref="G" — an SGE
        # function score is tied to ref->alt, so a swap must fail loudly rather
        # than silently re-orient.
        V = pl.DataFrame(
            {
                "chrom": ["1"],
                "pos": [1700],
                "ref": ["G"],
                "alt": ["C"],
                "gene": ["G1"],
                "function_score": [-1.0],
                "functional_class": ["LOF"],
                "assay": ["sge"],
                "source": ["test"],
            }
        )
        genome = FakeGenome({("1", 1700): "C"})
        cons = _write_consequences(
            tmp_path,
            [
                {
                    "chrom": "1",
                    "pos": 1700,
                    "ref": "C",
                    "alt": "G",
                    "consequence": "intron_variant",
                    "consequence_cre": "intron_variant",
                }
            ],
        )
        with pytest.raises(AssertionError, match="ref/alt swap"):
            annotate_sge_variants(
                V,
                genome=genome,
                consequence_paths=[cons],
                chroms=["1"],
                exon_pc=intervals["exon_pc"],
                exon_nc=intervals["exon_nc"],
                tss_pc=intervals["tss_pc"],
                tss_nc=intervals["tss_nc"],
                exon_proximal_dist=30,
                tss_proximal_dist=1000,
                exclude_consequences=[],
                consequence_groups={},
                lift=False,
                name="test",
            )

    def test_null_consequence_raises(self, tmp_path, intervals) -> None:
        V = _sge_frame()
        # Provide a consequence row for only the first two variants.
        genome = FakeGenome(
            {("1", p): r for p, r in zip(V["pos"].to_list(), V["ref"].to_list())}
        )
        cons = _write_consequences(
            tmp_path, _cons_rows(V, ["missense_variant", "intron_variant", "x"])[:2]
        )
        with pytest.raises(AssertionError, match="null consequence"):
            annotate_sge_variants(
                V,
                genome=genome,
                consequence_paths=[cons],
                chroms=["1"],
                exon_pc=intervals["exon_pc"],
                exon_nc=intervals["exon_nc"],
                tss_pc=intervals["tss_pc"],
                tss_nc=intervals["tss_nc"],
                exon_proximal_dist=30,
                tss_proximal_dist=1000,
                exclude_consequences=[],
                consequence_groups={},
                lift=False,
                name="test",
            )


# --------------------------------------------------------------------------- #
# MaveDB study-level metadata: assay facts (keywords) + score calibrations
# --------------------------------------------------------------------------- #
# Canned MaveDB score-set record (the fields the extractors read). Mirrors the
# real API: experiment.keywords for assay facts; scoreCalibrations -> a list of
# threshold schemes each with functionalClassifications.
_SCORE_SET = {
    "experiment": {
        "keywords": [
            {"keyword": {"key": "Phenotypic Assay Method", "label": "Cell fitness"}},
            {
                "keyword": {
                    "key": "Phenotypic Assay Mechanism",
                    "label": "Loss of function",
                }
            },
            {
                "keyword": {
                    "key": "Endogenous Locus Library Method Mechanism",
                    "label": "Nuclease",
                }
            },
            {"keyword": {"key": "No Label", "label": None}},  # malformed -> skipped
        ]
    },
    "scoreCalibrations": [
        {
            "title": "Investigator-provided functional classes",
            "researchUseOnly": False,
            "baselineScore": 0.0,
            "calibrationMetadata": None,
            "thresholdSources": [
                {"identifier": "30209399", "dbName": "PubMed"},
                {"identifier": "10.1038/x", "dbName": "DOI"},  # non-PubMed -> skipped
            ],
            "functionalClassifications": [
                {
                    "label": "Functional",
                    "functionalClassification": "normal",
                    "range": [-0.748, None],
                    "inclusiveLowerBound": True,
                    "inclusiveUpperBound": False,
                    "variantCount": 2821,
                    "acmgClassification": None,
                },
                {
                    "label": "Non-functional",
                    "functionalClassification": "abnormal",
                    "range": [None, -1.328],
                    "inclusiveLowerBound": False,
                    "inclusiveUpperBound": True,
                    "variantCount": 823,
                    "acmgClassification": {
                        "criterion": "PS3",
                        "evidenceStrength": "STRONG",
                        "points": 5,
                    },
                },
            ],
        },
        {
            "title": "Empty IGVF calibration",
            "researchUseOnly": False,
            "baselineScore": None,
            "calibrationMetadata": {"prior_probability_pathogenicity": 0.2285},
            "thresholdSources": [],
            "functionalClassifications": [],  # no classes -> one class-null row
        },
    ],
}


class TestExtractAssayFacts:
    def test_keywords_to_assay_columns(self) -> None:
        facts = extract_assay_facts(_SCORE_SET)
        assert facts == {
            "assay_phenotypic_assay_method": "Cell fitness",
            "assay_phenotypic_assay_mechanism": "Loss of function",
            "assay_endogenous_locus_library_method_mechanism": "Nuclease",
        }

    def test_unannotated_returns_empty(self) -> None:
        # BRCA2-like: no experiment / no keywords -> {} (not an error).
        assert extract_assay_facts({}) == {}
        assert extract_assay_facts({"experiment": {"keywords": []}}) == {}

    def test_keys_sorted_for_deterministic_columns(self) -> None:
        # Column order must be deterministic regardless of MaveDB's keyword ordering.
        ss = {
            "experiment": {
                "keywords": [
                    {"keyword": {"key": "Zeta Method", "label": "z"}},
                    {"keyword": {"key": "Alpha Method", "label": "a"}},
                ]
            }
        }
        assert list(extract_assay_facts(ss)) == [
            "assay_alpha_method",
            "assay_zeta_method",
        ]


class TestExtractScoreCalibrations:
    def test_flattens_classes_and_empty_calibration(self) -> None:
        rows = extract_score_calibrations(_SCORE_SET, gene="BRCA1", mavedb_urn="urn:x")
        # 2 classes from calibration 1 + 1 class-null row from the empty calibration.
        assert len(rows) == 3
        assert {r["calibration_title"] for r in rows} == {
            "Investigator-provided functional classes",
            "Empty IGVF calibration",
        }
        functional = next(r for r in rows if r["class_label"] == "Functional")
        assert functional["go_classification"] == "normal"
        assert functional["range_lower"] == -0.748
        assert functional["range_upper"] is None
        assert functional["variant_count"] == 2821
        assert functional["acmg_criterion"] is None
        # Only the PubMed threshold source is kept.
        assert functional["threshold_source_pmids"] == "30209399"
        nonfunc = next(r for r in rows if r["class_label"] == "Non-functional")
        assert nonfunc["acmg_criterion"] == "PS3"
        assert nonfunc["acmg_evidence_strength"] == "STRONG"
        assert nonfunc["acmg_points"] == 5
        # The class-less calibration still records its scheme, with null class fields.
        empty = next(
            r for r in rows if r["calibration_title"] == "Empty IGVF calibration"
        )
        assert empty["class_label"] is None
        assert empty["variant_count"] is None
        assert empty["prior_probability_pathogenicity"] == 0.2285
        # Every row carries the full schema keys (so pl.DataFrame(schema=...) is safe).
        for r in rows:
            assert set(r) == set(_CALIBRATION_SCHEMA)

    def test_no_calibrations_returns_empty(self) -> None:
        assert extract_score_calibrations({}, gene="BRCA2", mavedb_urn="urn:y") == []


class TestBuildMavedbMetadata:
    def _get_fn(self):
        store = {
            "https://api.mavedb.org/api/v1/score-sets/urn:a": _SCORE_SET,
            "https://api.mavedb.org/api/v1/score-sets/urn:b": {},  # unannotated
        }
        return lambda url: store[url]

    def test_facts_table_unions_and_calibrations_flatten(self) -> None:
        facts, calib = build_mavedb_metadata(
            [("GENEA", "urn:a"), ("GENEB", "urn:b")], get_fn=self._get_fn()
        )
        # One assay-facts row per gene; GENEB (unannotated) has null assay_ cells.
        assert facts.height == 2
        assert set(facts["gene"]) == {"GENEA", "GENEB"}
        assert "assay_phenotypic_assay_method" in facts.columns
        b = facts.filter(pl.col("gene") == "GENEB")
        assert b["assay_phenotypic_assay_method"].item() is None
        a = facts.filter(pl.col("gene") == "GENEA")
        assert a["assay_phenotypic_assay_mechanism"].item() == "Loss of function"
        # Calibrations: only GENEA contributes (3 rows); schema matches.
        assert calib.height == 3
        assert set(calib["gene"]) == {"GENEA"}
        assert calib.schema == _CALIBRATION_SCHEMA


class TestAttachAssayFacts:
    def _data(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "chrom": ["2", "16"],
                "pos": [1, 2],
                "ref": ["A", "C"],
                "alt": ["G", "T"],
                "gene": ["BARD1", "PALB2"],
                "mavedb_urn": ["urn:a", "urn:b"],
            }
        )

    def _facts(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "gene": ["BARD1", "PALB2"],
                "mavedb_urn": ["urn:a", "urn:b"],
                "assay_phenotypic_assay_method": ["Cell proliferation", "Cell fitness"],
            }
        )

    def test_join_adds_assay_columns_by_urn(self) -> None:
        out = attach_assay_facts(self._data(), self._facts())
        assert out.height == 2
        # Only mavedb_urn + assay_ columns come from the facts side (no duplicate gene).
        assert out.columns.count("gene") == 1
        assert out["assay_phenotypic_assay_method"].to_list() == [
            "Cell proliferation",
            "Cell fitness",
        ]

    def test_uncovered_urn_raises(self) -> None:
        data = self._data().with_columns(pl.lit("urn:missing").alias("mavedb_urn"))
        with pytest.raises(AssertionError, match="absent from assay_facts"):
            attach_assay_facts(data, self._facts())


# --------------------------------------------------------------------------- #
# consequence_group / subset + the build-time allowlist (#297 A + E)
# --------------------------------------------------------------------------- #
class TestConsequenceGroupAndAllowlist:
    # A small grouping map; consequence_final values absent from it keep their own
    # value (same `.replace(...)` semantics as trait_intervals.build_dataset).
    GROUPS = {
        "splicing": ["splice_region_variant", "exon_proximal"],
        "distal": ["intron_variant"],
    }

    def test_consequence_group_and_subset_invariant(self, tmp_path, intervals) -> None:
        out = _annotate(
            tmp_path,
            intervals,
            _sge_frame(),
            ["missense_variant", "intron_variant", "splice_region_variant"],
            exclude=[],
            consequence_groups=self.GROUPS,
        )
        assert {"consequence_group", "subset"}.issubset(out.columns)
        # subset is a verbatim copy of consequence_group.
        assert out["subset"].to_list() == out["consequence_group"].to_list()
        # group = map.get(consequence_final, consequence_final), robust to whatever
        # consequence_final the distance recategorization produced.
        cmap = {c: g for g, cs in self.GROUPS.items() for c in cs}
        for cf, cg in zip(
            out["consequence_final"].to_list(), out["consequence_group"].to_list()
        ):
            assert cg == cmap.get(cf, cf)

    def test_allowlist_keeps_only_listed_groups(self, tmp_path, intervals) -> None:
        args = (
            tmp_path,
            intervals,
            _sge_frame(),
            ["missense_variant", "intron_variant", "splice_region_variant"],
        )
        out_all = _annotate(*args, exclude=[], consequence_groups=self.GROUPS)
        keep = [out_all["consequence_group"].to_list()[0]]
        out_keep = _annotate(
            *args,
            exclude=[],
            consequence_groups=self.GROUPS,
            consequence_group_allowlist=keep,
        )
        assert set(out_keep["consequence_group"].to_list()) == set(keep)
        assert (
            out_keep.height
            == out_all.filter(pl.col("consequence_group").is_in(keep)).height
        )
        assert out_keep.height < out_all.height  # at least one group was dropped

    def test_allowlist_none_keeps_all(self, tmp_path, intervals) -> None:
        args = (
            tmp_path,
            intervals,
            _sge_frame(),
            ["missense_variant", "intron_variant", "splice_region_variant"],
        )
        a = _annotate(*args, exclude=[], consequence_groups=self.GROUPS)
        b = _annotate(
            *args,
            exclude=[],
            consequence_groups=self.GROUPS,
            consequence_group_allowlist=None,
        )
        assert a.height == b.height == 3


# --------------------------------------------------------------------------- #
# Calibration / author-class / direction attach helpers (#297 B + C + D)
# --------------------------------------------------------------------------- #
def _cal_frame(rows: list[dict]) -> pl.DataFrame:
    """Build a calibrations frame with the columns the selection / labeling helpers
    read (a slice of `_CALIBRATION_SCHEMA`)."""
    return pl.DataFrame(
        rows,
        schema={
            "gene": pl.Utf8,
            "calibration_title": pl.Utf8,
            "go_classification": pl.Utf8,
            "range_lower": pl.Float64,
            "range_upper": pl.Float64,
            "inclusive_lower": pl.Boolean,
            "inclusive_upper": pl.Boolean,
            "acmg_evidence_strength": pl.Utf8,
        },
    )


def _cls_row(gene, title, go, lo, hi, strength=None, *, inc_lo=True, inc_hi=True):
    return dict(
        gene=gene,
        calibration_title=title,
        go_classification=go,
        range_lower=lo,
        range_upper=hi,
        inclusive_lower=inc_lo,
        inclusive_upper=inc_hi,
        acmg_evidence_strength=strength,
    )


# A numeric ExCALIBR scheme for "GENEN": normal = fs >= 0, abnormal = fs <= -1,
# the (-1, 0) gap = intermediate. abnormal range below normal -> direction +1.
_GENEN_CAL = _cal_frame(
    [
        _cls_row("GENEN", "ExCALIBR calibration", "normal", 0.0, None, "BS3_STRONG"),
        _cls_row("GENEN", "ExCALIBR calibration", "abnormal", None, -1.0, "STRONG"),
    ]
)


class TestSelectNumericCalibration:
    def test_prefers_primary_over_dated_snapshot(self) -> None:
        # The dated snapshot has MORE rows, but the live ExCALIBR must still win.
        rows = []
        for title in ("ExCALIBR calibration", "ExCALIBR calibration (ClinVar 2018)"):
            rows.append(_cls_row("VHL", title, "normal", 0.0, None, "BS3_STRONG"))
            rows.append(_cls_row("VHL", title, "abnormal", None, -1.0, "STRONG"))
        rows.append(
            _cls_row(
                "VHL",
                "ExCALIBR calibration (ClinVar 2018)",
                "abnormal",
                None,
                -2.0,
                "VERY_STRONG",
            )
        )
        sel = _select_numeric_calibration(_cal_frame(rows))
        assert sel is not None and sel[0] == "ExCALIBR calibration"

    def test_falls_back_when_excalibr_lacks_normal_class(self) -> None:
        # CTCF-like: ExCALIBR has only abnormal classes -> not qualifying -> investigator.
        rows = [
            _cls_row("CTCF", "ExCALIBR calibration", "abnormal", None, 0.0, "STRONG"),
            _cls_row(
                "CTCF", "Investigator-provided functional classes", "normal", 0.0, None
            ),
            _cls_row(
                "CTCF",
                "Investigator-provided functional classes",
                "abnormal",
                None,
                -0.5,
            ),
        ]
        sel = _select_numeric_calibration(_cal_frame(rows))
        assert sel is not None
        assert sel[0] == "Investigator-provided functional classes"

    def test_none_when_only_categorical(self) -> None:
        # DDX3X-like: classes exist but ranges are null (categorical) -> no numeric scheme.
        rows = [
            _cls_row(
                "DDX3X",
                "Investigator-provided functional classes",
                "normal",
                None,
                None,
            ),
            _cls_row(
                "DDX3X",
                "Investigator-provided functional classes",
                "abnormal",
                None,
                None,
            ),
        ]
        assert _select_numeric_calibration(_cal_frame(rows)) is None


class TestAttachAuthorClassHarmonized:
    def test_maps_each_vocabulary(self) -> None:
        data = pl.DataFrame(
            {
                "gene": ["BRCA1", "RAD51C", "DDX3X", "BARD1", "BAP1"],
                "author_func_class": ["LOF", None, None, None, None],
                "author_functional_classification": [
                    None,
                    "fast depleted",
                    None,
                    None,
                    None,
                ],
                "author_sge_prediction_of_variant_function_in_ndd_context": [
                    None,
                    None,
                    "abnormal",
                    None,
                    None,
                ],
                "author_functional_consequence": [
                    None,
                    None,
                    None,
                    "functionally_normal",
                    None,
                ],
            }
        )
        out = attach_author_class_harmonized(data)
        # BRCA1 LOF, RAD51C fast-depleted, DDX3X abnormal -> abnormal; BARD1
        # functionally_normal -> normal; BAP1 (no class) -> null.
        assert out["author_class_harmonized"].to_list() == [
            "abnormal",
            "abnormal",
            "abnormal",
            "normal",
            None,
        ]

    def test_raises_on_unmapped_value(self) -> None:
        data = pl.DataFrame({"gene": ["BRCA1"], "author_func_class": ["WEIRD"]})
        with pytest.raises(AssertionError, match="unmapped author class"):
            attach_author_class_harmonized(data)

    def test_raises_on_stray_unmapped_gene(self) -> None:
        # An unmapped gene must not carry a non-null value in a known class column.
        data = pl.DataFrame(
            {
                "gene": ["BAP1"],
                "author_functional_consequence": ["functionally_abnormal"],
            }
        )
        with pytest.raises(
            AssertionError, match="non-null author_functional_consequence"
        ):
            attach_author_class_harmonized(data)


class TestAttachCalibratedClass:
    def test_numeric_ranges_label_and_strength(self) -> None:
        data = pl.DataFrame(
            {
                "gene": ["GENEN", "GENEN", "GENEN"],
                "function_score": [0.5, -2.0, -0.5],  # normal / abnormal / gap
                "author_class_harmonized": [None, None, None],
            }
        )
        out = attach_calibrated_class(data, _GENEN_CAL)
        assert out["calibrated_class"].to_list() == [
            "normal",
            "abnormal",
            "intermediate",
        ]
        assert out["acmg_strength"].to_list() == ["BS3_STRONG", "STRONG", None]
        assert out["calibration_scheme"].to_list() == ["ExCALIBR calibration"] * 3

    def test_categorical_gene_uses_author_class(self) -> None:
        # DDX3X-like: no numeric scheme for this gene -> inherit author class.
        data = pl.DataFrame(
            {
                "gene": ["DDX", "DDX"],
                "function_score": [0.1, 0.9],
                "author_class_harmonized": ["normal", "abnormal"],
            }
        )
        out = attach_calibrated_class(data, _GENEN_CAL)  # no DDX rows in the cal
        assert out["calibrated_class"].to_list() == ["normal", "abnormal"]
        assert out["calibration_scheme"].to_list() == ["author_class", "author_class"]
        assert out["acmg_strength"].to_list() == [None, None]

    def test_no_calibration_no_class_is_null(self) -> None:
        data = pl.DataFrame(
            {
                "gene": ["BRCA2"],
                "function_score": [0.3],
                "author_class_harmonized": [None],
            }
        )
        out = attach_calibrated_class(data, _GENEN_CAL)
        assert out["calibrated_class"].to_list() == [None]
        assert out["calibration_scheme"].to_list() == [None]
        assert out["acmg_strength"].to_list() == [None]

    def test_requires_author_class_first(self) -> None:
        data = pl.DataFrame({"gene": ["GENEN"], "function_score": [0.5]})
        with pytest.raises(
            AssertionError, match="attach_author_class_harmonized first"
        ):
            attach_calibrated_class(data, _GENEN_CAL)


class TestAttachFunctionDirection:
    def test_numeric_positive_direction(self) -> None:
        # abnormal range below normal -> +1; aligned == raw score.
        data = pl.DataFrame(
            {
                "gene": ["GENEN", "GENEN"],
                "function_score": [0.5, -2.0],
                "author_class_harmonized": [None, None],
            }
        )
        out = attach_function_direction(data, _GENEN_CAL)
        assert out["function_direction"].to_list() == [1, 1]
        assert out["function_score_aligned"].to_list() == [0.5, -2.0]

    def test_numeric_negative_direction(self) -> None:
        # abnormal range ABOVE normal -> -1; aligned flips sign.
        cal = _cal_frame(
            [
                _cls_row(
                    "GENEP", "ExCALIBR calibration", "normal", None, 0.0, "BS3_STRONG"
                ),
                _cls_row(
                    "GENEP", "ExCALIBR calibration", "abnormal", 1.0, None, "STRONG"
                ),
            ]
        )
        data = pl.DataFrame(
            {
                "gene": ["GENEP", "GENEP"],
                "function_score": [0.2, 3.0],
                "author_class_harmonized": [None, None],
            }
        )
        out = attach_function_direction(data, cal)
        assert out["function_direction"].to_list() == [-1, -1]
        assert out["function_score_aligned"].to_list() == [-0.2, -3.0]

    def test_categorical_direction_from_author_class(self) -> None:
        # DDX3X-like: abnormal author class has the HIGHER function_score -> -1.
        data = pl.DataFrame(
            {
                "gene": ["DDX", "DDX"],
                "function_score": [0.1, 0.9],
                "author_class_harmonized": ["normal", "abnormal"],
            }
        )
        out = attach_function_direction(data, _GENEN_CAL)  # no DDX rows in the cal
        assert out["function_direction"].to_list() == [-1, -1]
        assert out["function_score_aligned"].to_list() == [-0.1, -0.9]

    def test_null_when_no_signal(self) -> None:
        # BRCA2-like: no numeric scheme, no author class -> null direction + aligned.
        data = pl.DataFrame(
            {
                "gene": ["BRCA2"],
                "function_score": [0.3],
                "author_class_harmonized": [None],
            }
        )
        out = attach_function_direction(data, _GENEN_CAL)
        assert out["function_direction"].to_list() == [None]
        assert out["function_score_aligned"].to_list() == [None]
