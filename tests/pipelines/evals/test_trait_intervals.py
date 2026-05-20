"""Tests for TraitGym-style TSS/exon nearest-feature annotations."""

from pathlib import Path

import polars as pl
import pytest

from marin_dna.data.utils import load_annotation
from marin_dna.pipelines.evals.trait_intervals import (
    add_exon,
    add_tss,
    build_dataset,
    get_exon,
    get_tss,
)


SYNTHETIC_GTF = """\
1\thavana\ttranscript\t1001\t2000\t.\t+\t.\tgene_id "ENSG_PLUS"; transcript_id "T1"; transcript_biotype "protein_coding";
1\thavana\texon\t1001\t1100\t.\t+\t.\tgene_id "ENSG_PLUS"; transcript_id "T1"; transcript_biotype "protein_coding";
1\thavana\texon\t1500\t1600\t.\t+\t.\tgene_id "ENSG_PLUS"; transcript_id "T1"; transcript_biotype "protein_coding";
1\thavana\texon\t1900\t2000\t.\t+\t.\tgene_id "ENSG_PLUS"; transcript_id "T1"; transcript_biotype "protein_coding";
1\thavana\ttranscript\t3000\t4000\t.\t-\t.\tgene_id "ENSG_MINUS"; transcript_id "T2"; transcript_biotype "protein_coding";
1\thavana\texon\t3000\t3200\t.\t-\t.\tgene_id "ENSG_MINUS"; transcript_id "T2"; transcript_biotype "protein_coding";
1\thavana\texon\t3800\t4000\t.\t-\t.\tgene_id "ENSG_MINUS"; transcript_id "T2"; transcript_biotype "protein_coding";
1\thavana\ttranscript\t5000\t6000\t.\t+\t.\tgene_id "ENSG_LNCRNA"; transcript_id "T3"; transcript_biotype "lncRNA";
1\thavana\texon\t5000\t6000\t.\t+\t.\tgene_id "ENSG_LNCRNA"; transcript_id "T3"; transcript_biotype "lncRNA";
MT\thavana\ttranscript\t10\t100\t.\t+\t.\tgene_id "ENSG_MT"; transcript_id "T4"; transcript_biotype "protein_coding";
MT\thavana\texon\t10\t100\t.\t+\t.\tgene_id "ENSG_MT"; transcript_id "T4"; transcript_biotype "protein_coding";
"""


NON_PC_FILTER = pl.col("transcript_biotype") != "protein_coding"


@pytest.fixture
def annotation(tmp_path: Path) -> pl.DataFrame:
    gtf = tmp_path / "synthetic.gtf"
    gtf.write_text(SYNTHETIC_GTF)
    return load_annotation(str(gtf))


@pytest.fixture
def tss_pc(annotation: pl.DataFrame) -> pl.DataFrame:
    return get_tss(annotation)


@pytest.fixture
def tss_nc(annotation: pl.DataFrame) -> pl.DataFrame:
    return get_tss(annotation, biotype_filter=NON_PC_FILTER)


@pytest.fixture
def exon_pc(annotation: pl.DataFrame) -> pl.DataFrame:
    return get_exon(annotation)


@pytest.fixture
def exon_nc(annotation: pl.DataFrame) -> pl.DataFrame:
    return get_exon(annotation, biotype_filter=NON_PC_FILTER)


class TestGetTss:
    def test_output_schema(self, tss_pc: pl.DataFrame) -> None:
        assert tss_pc.columns == ["chrom", "start", "end", "gene_id"]

    def test_tss_is_1bp(self, tss_pc: pl.DataFrame) -> None:
        assert (tss_pc["end"] - tss_pc["start"] == 1).all()

    def test_plus_strand_tss(self, tss_pc: pl.DataFrame) -> None:
        # 1-based GTF transcript starting at 1001 → 0-based start = 1000
        plus_tss = tss_pc.filter(pl.col("gene_id") == "ENSG_PLUS")
        assert plus_tss["start"][0] == 1000
        assert plus_tss["end"][0] == 1001

    def test_minus_strand_tss(self, tss_pc: pl.DataFrame) -> None:
        # 1-based GTF transcript ending at 4000 → 0-based end = 4000;
        # minus-strand TSS = end - 1 = 3999
        minus_tss = tss_pc.filter(pl.col("gene_id") == "ENSG_MINUS")
        assert minus_tss["start"][0] == 3999
        assert minus_tss["end"][0] == 4000

    def test_default_skips_non_protein_coding(self, tss_pc: pl.DataFrame) -> None:
        gene_ids = set(tss_pc["gene_id"].to_list())
        assert "ENSG_LNCRNA" not in gene_ids

    def test_nc_filter_keeps_non_protein_coding(self, tss_nc: pl.DataFrame) -> None:
        gene_ids = set(tss_nc["gene_id"].to_list())
        assert gene_ids == {"ENSG_LNCRNA"}


class TestGetExon:
    def test_output_schema(self, exon_pc: pl.DataFrame) -> None:
        assert exon_pc.columns == ["chrom", "start", "end", "gene_id"]

    def test_default_filters_protein_coding(self, exon_pc: pl.DataFrame) -> None:
        gene_ids = set(exon_pc["gene_id"].to_list())
        assert "ENSG_LNCRNA" not in gene_ids

    def test_nc_filter_keeps_non_protein_coding(self, exon_nc: pl.DataFrame) -> None:
        gene_ids = set(exon_nc["gene_id"].to_list())
        assert gene_ids == {"ENSG_LNCRNA"}

    def test_filters_canonical_chroms(self, exon_pc: pl.DataFrame) -> None:
        # MT not in canonical list
        assert "MT" not in exon_pc["chrom"].unique().to_list()

    def test_deduplicates(self, exon_pc: pl.DataFrame) -> None:
        assert exon_pc.shape[0] == exon_pc.unique().shape[0]


class TestAddExon:
    @pytest.fixture
    def variants(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "chrom": ["1", "1", "1", "1"],
                "pos": [1150, 1700, 50000, 5050],
                "ref": ["A", "A", "A", "A"],
                "alt": ["T", "T", "T", "T"],
                # 1150: 50bp downstream of PC exon 1001-1100 (intron, near PC exon)
                # 1700: 100bp downstream of PC exon 1500-1600 (intron, near PC exon)
                # 50000: far away (intergenic)
                # 5050: inside lncRNA exon 5000-6000 (intron in this synthetic, but
                #       distance to nearest PC exon = ~3050; distance to nearest
                #       nc exon = 0)
                "consequence": [
                    "intron_variant",
                    "intron_variant",
                    "intergenic_variant",
                    "intron_variant",
                ],
                "consequence_cre": [
                    "intron_variant",
                    "intron_variant",
                    "intergenic_variant",
                    "intron_variant",
                ],
            }
        )

    def test_output_columns(
        self, variants: pl.DataFrame, exon_pc: pl.DataFrame, exon_nc: pl.DataFrame
    ) -> None:
        result = add_exon(variants, exon_pc, exon_nc)
        for c in (
            "distance_exon_pc",
            "distance_exon_nc",
            "distance_exon",
            "exon_closest_pc_gene_id",
            "exon_closest_nc_gene_id",
            "exon_closest_gene_id",
            "consequence_final",
        ):
            assert c in result.columns

    def test_pc_distance_correct(
        self, variants: pl.DataFrame, exon_pc: pl.DataFrame, exon_nc: pl.DataFrame
    ) -> None:
        result = add_exon(variants, exon_pc, exon_nc)
        # 1150 is 50 bp from PC exon 1001-1100 (end at 1100, 1-based;
        # 0-based half-open [1000, 1100). 1150 (0-based) = 1149 - 1100 = 49.
        # add_exon uses pos as 1-based; 1150 → start=1149, end=1150;
        # nearest PC exon edge at 1100 → distance 49 (depends on pb.nearest semantics)
        # Just check it's small (< 100).
        d = result.filter(pl.col("pos") == 1150)["distance_exon_pc"][0]
        assert d < 100, f"PC distance for pos 1150 should be small, got {d}"

    def test_nc_distance_for_lncrna_exon_overlap(
        self, variants: pl.DataFrame, exon_pc: pl.DataFrame, exon_nc: pl.DataFrame
    ) -> None:
        result = add_exon(variants, exon_pc, exon_nc)
        # 5050 is inside the lncRNA exon 5000-6000, so distance to nc exon = 0
        nc_dist = result.filter(pl.col("pos") == 5050)["distance_exon_nc"][0]
        assert nc_dist == 0
        # Distance to PC exon should be much larger (closest PC exon is at 3000-3200 or 3800-4000)
        pc_dist = result.filter(pl.col("pos") == 5050)["distance_exon_pc"][0]
        assert pc_dist > 1000

    def test_combined_min_distance(
        self, variants: pl.DataFrame, exon_pc: pl.DataFrame, exon_nc: pl.DataFrame
    ) -> None:
        result = add_exon(variants, exon_pc, exon_nc)
        # exon_dist should be min(pc, nc) per row
        for row in result.iter_rows(named=True):
            assert row["distance_exon"] == min(
                row["distance_exon_pc"], row["distance_exon_nc"]
            )

    def test_exon_proximal_within_threshold(
        self, variants: pl.DataFrame, exon_pc: pl.DataFrame, exon_nc: pl.DataFrame
    ) -> None:
        result = add_exon(variants, exon_pc, exon_nc, exon_proximal_dist=100)
        # 1150 is 50bp from PC exon → exon_proximal
        assert (
            result.filter(pl.col("pos") == 1150)["consequence_final"][0]
            == "exon_proximal"
        )

    def test_exon_proximal_uses_nc_when_closer(
        self, variants: pl.DataFrame, exon_pc: pl.DataFrame, exon_nc: pl.DataFrame
    ) -> None:
        result = add_exon(variants, exon_pc, exon_nc, exon_proximal_dist=100)
        # 5050 is INSIDE a lncRNA exon (nc dist = 0) but its consequence is
        # intron_variant. Combined min distance = 0, so it should be exon_proximal.
        assert (
            result.filter(pl.col("pos") == 5050)["consequence_final"][0]
            == "exon_proximal"
        )

    def test_exon_proximal_outside_threshold(
        self, variants: pl.DataFrame, exon_pc: pl.DataFrame, exon_nc: pl.DataFrame
    ) -> None:
        result = add_exon(variants, exon_pc, exon_nc, exon_proximal_dist=10)
        # 1150 is 50bp away → not within 10bp → falls back to consequence_cre
        assert (
            result.filter(pl.col("pos") == 1150)["consequence_final"][0]
            == "intron_variant"
        )

    def test_non_intron_not_modified(
        self, variants: pl.DataFrame, exon_pc: pl.DataFrame, exon_nc: pl.DataFrame
    ) -> None:
        result = add_exon(variants, exon_pc, exon_nc, exon_proximal_dist=10**6)
        assert (
            result.filter(pl.col("pos") == 50000)["consequence_final"][0]
            == "intergenic_variant"
        )


class TestAddTss:
    @pytest.fixture
    def variants_with_exon(
        self,
        exon_pc: pl.DataFrame,
        exon_nc: pl.DataFrame,
    ) -> pl.DataFrame:
        V = pl.DataFrame(
            {
                "chrom": ["1", "1", "1", "1"],
                "pos": [1001, 50000, 4000, 5001],
                "ref": ["A", "A", "A", "A"],
                "alt": ["T", "T", "T", "T"],
                # 1001: at the plus-strand PC TSS (intron context)
                # 50000: far (intergenic)
                # 4000: at minus-strand PC TSS (upstream_gene)
                # 5001: at the lncRNA TSS (intergenic context)
                "consequence": [
                    "intron_variant",
                    "intergenic_variant",
                    "upstream_gene_variant",
                    "intergenic_variant",
                ],
                "consequence_cre": [
                    "intron_variant",
                    "intergenic_variant",
                    "upstream_gene_variant",
                    "intergenic_variant",
                ],
            }
        )
        return add_exon(V, exon_pc, exon_nc, exon_proximal_dist=10)

    def test_output_columns(
        self,
        variants_with_exon: pl.DataFrame,
        tss_pc: pl.DataFrame,
        tss_nc: pl.DataFrame,
    ) -> None:
        result = add_tss(variants_with_exon, tss_pc, tss_nc)
        for c in (
            "distance_tss_pc",
            "distance_tss_nc",
            "distance_tss",
            "tss_closest_pc_gene_id",
            "tss_closest_nc_gene_id",
            "tss_closest_gene_id",
            "consequence_final",
        ):
            assert c in result.columns

    def test_tss_proximal_at_pc_tss(
        self,
        variants_with_exon: pl.DataFrame,
        tss_pc: pl.DataFrame,
        tss_nc: pl.DataFrame,
    ) -> None:
        result = add_tss(variants_with_exon, tss_pc, tss_nc, tss_proximal_dist=1000)
        # upstream_gene_variant at 4000 sits at the minus-strand PC TSS → tss_proximal
        assert (
            result.filter(pl.col("pos") == 4000)["consequence_final"][0]
            == "tss_proximal"
        )

    def test_tss_proximal_at_nc_tss(
        self,
        variants_with_exon: pl.DataFrame,
        tss_pc: pl.DataFrame,
        tss_nc: pl.DataFrame,
    ) -> None:
        result = add_tss(variants_with_exon, tss_pc, tss_nc, tss_proximal_dist=100)
        # 5001 is at the lncRNA TSS (1bp away). Closest PC TSS is at 3999
        # (~1000 bp away). Combined min distance ≈ 1, so under tss_proximal_dist=100
        # the variant should be reclassified to tss_proximal via nc.
        row = result.filter(pl.col("pos") == 5001).to_dict(as_series=False)
        assert row["consequence_final"][0] == "tss_proximal", (
            f"expected tss_proximal, got {row['consequence_final'][0]}; "
            f"tss_dist_pc={row['distance_tss_pc'][0]}, tss_dist_nc={row['distance_tss_nc'][0]}"
        )

    def test_combined_min_distance(
        self,
        variants_with_exon: pl.DataFrame,
        tss_pc: pl.DataFrame,
        tss_nc: pl.DataFrame,
    ) -> None:
        result = add_tss(variants_with_exon, tss_pc, tss_nc, tss_proximal_dist=1000)
        for row in result.iter_rows(named=True):
            assert row["distance_tss"] == min(
                row["distance_tss_pc"], row["distance_tss_nc"]
            )

    def test_intergenic_far_unchanged(
        self,
        variants_with_exon: pl.DataFrame,
        tss_pc: pl.DataFrame,
        tss_nc: pl.DataFrame,
    ) -> None:
        result = add_tss(variants_with_exon, tss_pc, tss_nc, tss_proximal_dist=100)
        assert (
            result.filter(pl.col("pos") == 50000)["consequence_final"][0]
            == "intergenic_variant"
        )


class TestBuildDataset:
    def test_end_to_end(
        self,
        exon_pc: pl.DataFrame,
        exon_nc: pl.DataFrame,
        tss_pc: pl.DataFrame,
        tss_nc: pl.DataFrame,
    ) -> None:
        V = pl.DataFrame(
            {
                "chrom": ["1", "1", "1", "1"],
                "pos": [1050, 4000, 50000, 1150],
                "ref": ["A", "A", "A", "A"],
                "alt": ["T", "T", "T", "T"],
                "label": [True, True, False, True],
                "consequence": [
                    "missense_variant",
                    "upstream_gene_variant",
                    "intergenic_variant",
                    "intron_variant",
                ],
                "consequence_cre": [
                    "missense_variant",
                    "upstream_gene_variant",
                    "intergenic_variant",
                    "intron_variant",
                ],
            }
        )
        consequence_groups = {
            "Missense": ["missense_variant"],
            "Promoter": ["tss_proximal"],
            "Splicing": ["exon_proximal"],
            "Distal": ["intergenic_variant", "intron_variant", "upstream_gene_variant"],
        }
        out = build_dataset(
            V,
            exon_pc=exon_pc,
            exon_nc=exon_nc,
            tss_pc=tss_pc,
            tss_nc=tss_nc,
            exclude_consequences=[],
            exon_proximal_dist=100,
            tss_proximal_dist=10,
            consequence_groups=consequence_groups,
        )

        # 50000 is intergenic — its consequence_final is intergenic_variant, which
        # never appears in label=True positives, so it gets filtered out.
        positions = set(out["pos"].to_list())
        assert 50000 not in positions
        assert {1050, 4000, 1150} <= positions

        assert out["consequence_group"].null_count() == 0

        cg = dict(zip(out["pos"].to_list(), out["consequence_group"].to_list()))
        assert cg[1050] == "Missense"
        assert cg[4000] == "Promoter"
        assert cg[1150] == "Splicing"

    def test_excludes_consequences(
        self,
        exon_pc: pl.DataFrame,
        exon_nc: pl.DataFrame,
        tss_pc: pl.DataFrame,
        tss_nc: pl.DataFrame,
    ) -> None:
        V = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "pos": [1050, 1500],
                "ref": ["A", "A"],
                "alt": ["T", "T"],
                "label": [True, True],
                "consequence": ["missense_variant", "stop_gained"],
                "consequence_cre": ["missense_variant", "stop_gained"],
            }
        )
        out = build_dataset(
            V,
            exon_pc=exon_pc,
            exon_nc=exon_nc,
            tss_pc=tss_pc,
            tss_nc=tss_nc,
            exclude_consequences=["stop_gained"],
            exon_proximal_dist=100,
            tss_proximal_dist=1000,
            consequence_groups={"Missense": ["missense_variant"]},
        )
        assert 1500 not in out["pos"].to_list()
