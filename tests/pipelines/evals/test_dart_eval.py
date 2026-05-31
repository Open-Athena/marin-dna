"""Tests for DART-Eval caQTL/dsQTL parsing + annotation."""

from pathlib import Path

import polars as pl
import pytest

from marin_dna.data.utils import load_annotation
from marin_dna.pipelines.evals.dart_eval import (
    annotate_variants,
    parse_caqtl,
    parse_dsqtl,
)
from marin_dna.pipelines.evals.trait_intervals import get_exon, get_tss


# --------------------------------------------------------------------------- #
# parse_caqtl
# --------------------------------------------------------------------------- #
class TestParseCaqtl:
    def _raw(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "chr_hg38": ["chr1", "chr2", "chr3", "chr4", "chr5"],
                "pos_hg38": [100, 200, 300, 400, 500],
                "allele1": ["A", "C", "A", "A", "AT"],
                "allele2": ["T", "G", "T", "T", "T"],
                "IsUsed": [1, 1, 0, 1, 1],
                "in_peaks": [1, 1, 1, 0, 1],
                # caQTL `label` is a boolean flag in the real TSV.
                "label": [True, False, True, True, False],
                "beta": [0.5, -0.2, 0.1, 0.1, 0.0],
                "pval": [0.01, 0.5, 0.2, 0.3, 0.9],
                "se": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )

    def test_filters_and_maps(self) -> None:
        out = parse_caqtl(self._raw())
        # r3 dropped (IsUsed=0), r4 dropped (in_peaks=0), r5 dropped (indel).
        assert out.height == 2
        assert out["chrom"].to_list() == ["1", "2"]  # "chr" stripped
        assert out["label"].to_list() == [True, False]  # boolean label preserved
        assert out["effect"].to_list() == [
            0.5,
            -0.2,
        ]  # signed; effect_size=abs added later
        assert out.columns[:6] == ["chrom", "pos", "ref", "alt", "label", "effect"]

    def test_pval_se_carried(self) -> None:
        out = parse_caqtl(self._raw())
        assert out["pval"].to_list() == [0.01, 0.5]
        assert out["se"].to_list() == [0.1, 0.2]

    def test_pval_se_null_when_absent(self) -> None:
        out = parse_caqtl(self._raw().drop("pval", "se"))
        assert out["pval"].null_count() == out.height
        assert out["se"].null_count() == out.height

    def test_label_is_boolean(self) -> None:
        assert parse_caqtl(self._raw())["label"].dtype == pl.Boolean

    def test_boolean_flag_dtype(self) -> None:
        df = self._raw().with_columns(
            pl.col("IsUsed").cast(pl.Boolean), pl.col("in_peaks").cast(pl.Boolean)
        )
        assert parse_caqtl(df).height == 2

    def test_string_flag_dtype(self) -> None:
        df = self._raw().with_columns(
            pl.when(pl.col("IsUsed") == 1)
            .then(pl.lit("True"))
            .otherwise(pl.lit("False"))
            .alias("IsUsed")
        )
        assert parse_caqtl(df).height == 2

    def test_non_boolean_label_raises(self) -> None:
        # caQTL `label` must be boolean in the real TSV; an int-coded label is
        # schema drift and should fail loudly.
        df = self._raw().with_columns(pl.col("label").cast(pl.Int64))
        with pytest.raises(AssertionError, match="expected boolean"):
            parse_caqtl(df)

    def test_missing_column_raises(self) -> None:
        with pytest.raises(AssertionError, match="missing expected columns"):
            parse_caqtl(self._raw().drop("label"))

    def test_missing_effect_is_null(self) -> None:
        out = parse_caqtl(self._raw().drop("beta"))
        assert out["effect"].null_count() == out.height

    def test_lowercase_alleles_uppercased(self) -> None:
        df = self._raw().with_columns(
            pl.col("allele1").str.to_lowercase(), pl.col("allele2").str.to_lowercase()
        )
        out = parse_caqtl(df)
        assert out["ref"].to_list() == ["A", "C"]
        assert out["alt"].to_list() == ["T", "G"]


# --------------------------------------------------------------------------- #
# parse_dsqtl
# --------------------------------------------------------------------------- #
class TestParseDsqtl:
    def _raw(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "var.chrom": ["chr1", "chr2", "chr3", "chr4"],
                "var.pos": [100, 200, 300, 400],
                "var.allele1": ["A", "C", "A", "G"],
                "var.allele2": ["T", "G", "T", "GC"],
                "var.isused": [1, 1, 0, 1],
                "var.label": [1, -1, 1, -1],
                "obs.estimate": [0.3, -0.1, 0.2, 0.4],
            }
        )

    def test_filters_and_maps(self) -> None:
        out = parse_dsqtl(self._raw())
        # r3 dropped (isused=0), r4 dropped (indel G->GC).
        assert out.height == 2
        assert out["chrom"].to_list() == ["1", "2"]
        assert out["label"].to_list() == [True, False]  # 1->True, -1->False
        assert out["effect"].to_list() == [0.3, -0.1]

    def test_negative_one_is_false(self) -> None:
        out = parse_dsqtl(self._raw())
        assert out.filter(pl.col("chrom") == "2")["label"][0] is False

    def test_unexpected_label_zero_raises(self) -> None:
        df = self._raw().with_columns(
            pl.when(pl.col("var.pos") == 100)
            .then(0)
            .otherwise(pl.col("var.label"))
            .alias("var.label")
        )
        with pytest.raises(AssertionError, match="unexpected values"):
            parse_dsqtl(df)

    def test_missing_column_raises(self) -> None:
        with pytest.raises(AssertionError, match="missing expected columns"):
            parse_dsqtl(self._raw().drop("var.label"))


# --------------------------------------------------------------------------- #
# annotate_variants
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


def _variants() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": ["1", "1", "1"],
            "pos": [1150, 1700, 50000],
            "ref": ["A", "C", "G"],
            "alt": ["T", "G", "A"],
            "label": [True, False, True],
            "effect": [0.5, -0.2, 0.1],
        }
    )


def _annotate(tmp_path, intervals, V, cons_rows, *, lift=False):
    genome = FakeGenome(
        {("1", p): r for p, r in zip(V["pos"].to_list(), V["ref"].to_list())}
    )
    cons = _write_consequences(tmp_path, cons_rows)
    return annotate_variants(
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
        lift=lift,
    )


class TestAnnotateVariants:
    def _full_cons(self, V: pl.DataFrame) -> list[dict]:
        return [
            {
                "chrom": "1",
                "pos": p,
                "ref": r,
                "alt": a,
                "consequence": "intron_variant",
                "consequence_cre": "intron_variant",
            }
            for p, r, a in zip(
                V["pos"].to_list(), V["ref"].to_list(), V["alt"].to_list()
            )
        ]

    def test_no_drop_and_columns(self, tmp_path, intervals) -> None:
        V = _variants()
        out = _annotate(tmp_path, intervals, V, self._full_cons(V))
        assert out.height == 3  # nothing dropped (all refs match, all in cons set)
        for c in (
            "consequence",
            "consequence_cre",
            "consequence_final",
            "distance_exon_pc",
            "distance_exon",
            "distance_tss_pc",
            "distance_tss",
            "label",
            "effect",
            "effect_size",
        ):
            assert c in out.columns
        assert out["label"].dtype == pl.Boolean
        assert out["consequence"].null_count() == 0
        # No swap (genome returns the ref base) -> effect unchanged; effect_size=|effect|.
        assert out.sort("pos")["effect"].to_list() == [0.5, -0.2, 0.1]
        assert out.sort("pos")["effect_size"].to_list() == [0.5, 0.2, 0.1]

    def test_null_consequence_raises(self, tmp_path, intervals) -> None:
        V = _variants()
        # Omit the last variant from the consequence set -> null consequence.
        partial = self._full_cons(V)[:2]
        with pytest.raises(AssertionError, match="null consequence"):
            _annotate(tmp_path, intervals, V, partial)

    def test_ref_alt_swap(self, tmp_path, intervals) -> None:
        # Genome says base at ("1", 1700) is "C" but variant lists ref="G";
        # check_ref_alt should swap so ref="C", alt="G".
        V = pl.DataFrame(
            {
                "chrom": ["1"],
                "pos": [1700],
                "ref": ["G"],
                "alt": ["C"],
                "label": [True],
                "effect": [1.0],
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
        out = annotate_variants(
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
            lift=False,
        )
        assert out["ref"][0] == "C"
        assert out["alt"][0] == "G"
        # alt changed C->G (swapped) -> effect flips sign (+1.0 -> -1.0) to stay
        # signed relative to the final alt; effect_size = |effect| = 1.0.
        assert out["effect"][0] == -1.0
        assert out["effect_size"][0] == 1.0

    def test_low_retention_raises(self, tmp_path, intervals) -> None:
        # Genome base never matches either allele -> check_ref_alt drops all.
        V = _variants()
        genome = FakeGenome({("1", p): "N" for p in V["pos"].to_list()})
        cons = _write_consequences(tmp_path, self._full_cons(V))
        with pytest.raises(AssertionError, match="coordinate-base"):
            annotate_variants(
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
                lift=False,
            )
