"""Tests for SGE (saturation genome editing) dataset construction."""

from pathlib import Path

import polars as pl
import pytest

from marin_dna.data.utils import load_annotation
from marin_dna.pipelines.evals.sge import (
    annotate_sge_variants,
    normalize_brca1_findlay,
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
        out = normalize_brca1_findlay(self._raw())
        assert out.height == 2  # indel dropped by filter_snp
        assert out["chrom"].dtype == pl.Utf8 and out["chrom"].to_list() == ["17", "17"]
        assert out["pos"].dtype == pl.Int64
        assert out["author_function_score_mean"].to_list() == [-0.37, -1.5]
        assert out["author_func_class"].to_list() == ["FUNC", "LOF"]
        assert out["assay"].unique().to_list() == ["sge"]
        assert out["source"].unique().to_list() == ["findlay2018"]
        # Standard coords come first, then every original column under author_.
        assert out.columns[:7] == [
            "chrom",
            "pos",
            "ref",
            "alt",
            "gene",
            "assay",
            "source",
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
        out = normalize_brca1_findlay(self._raw())
        assert "consequence" not in out.columns
        assert out["author_consequence"].to_list() == ["Splice region", "Missense"]

    def test_bad_class_raises(self) -> None:
        raw = self._raw().with_columns(pl.lit("WEIRD").alias("func.class"))
        with pytest.raises(AssertionError, match="func.class"):
            normalize_brca1_findlay(raw)

    def test_null_score_raises(self) -> None:
        raw = self._raw().with_columns(
            pl.when(pl.col("alt") == "G")
            .then(None)
            .otherwise(pl.col("function.score.mean"))
            .alias("function.score.mean")
        )
        with pytest.raises(AssertionError, match="null function score"):
            normalize_brca1_findlay(raw)


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
            "source": ["test", "test", "test"],
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


def _annotate(tmp_path, intervals, V, consequences, exclude, *, lift=False):
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
        assert set(out["consequence"].to_list()) == {"missense_variant", "intron_variant"}

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
            "source",
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
                lift=False,
                name="test",
            )
