"""TSS/exon nearest-feature annotations for trait-pipeline curation.

Ported from TraitGym (commit e59d612e9; src/traitgym/intervals.py).

GTF parsing reuses ``marin_dna.data.utils.load_annotation`` (functionally identical
to the TraitGym version), so callers should pass an annotation DataFrame loaded
via that helper.

Now supports dual-biotype TSS/exon distance computation: nearest distance to
protein-coding transcripts AND nearest distance to non-protein-coding
transcripts (lncRNA, pseudogene, TEC, small_ncRNA, etc.). The ``consequence_final``
``tss_proximal`` / ``exon_proximal`` recategorization uses the minimum of the
two distances. Per-biotype distance and closest-gene columns are kept on the
output as metadata so downstream consumers can filter however they like.
"""

import polars as pl
import polars_bio as pb

from marin_dna.pipelines.evals.variants import CHROMS, COORDINATES, NON_EXONIC


def get_tss(
    ann: pl.DataFrame,
    biotype_filter: pl.Expr | None = None,
) -> pl.DataFrame:
    """Extract transcription start sites from an annotation DataFrame.

    Returns ``[chrom, start, end, gene_id]`` 1bp intervals from transcripts whose
    ``transcript_biotype`` matches ``biotype_filter``. Default keeps only
    ``transcript_biotype == "protein_coding"`` (canonical coding transcripts).
    """
    if biotype_filter is None:
        biotype_filter = pl.col("transcript_biotype") == "protein_coding"
    return (
        ann.filter(pl.col("feature") == "transcript")
        .with_columns(
            pl.col("attribute").str.extract(r'gene_id "([^;]*)";').alias("gene_id"),
            pl.col("attribute")
            .str.extract(r'transcript_biotype "([^;]*)";')
            .alias("transcript_biotype"),
        )
        .filter(biotype_filter)
        .with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("start"))
            .otherwise(pl.col("end") - 1)
            .alias("tss_start"),
        )
        .with_columns((pl.col("tss_start") + 1).alias("tss_end"))
        .select(
            pl.col("chrom"),
            pl.col("tss_start").alias("start"),
            pl.col("tss_end").alias("end"),
            pl.col("gene_id"),
        )
        .unique()
        .sort(["chrom", "start", "end"])
    )


def get_exon(
    ann: pl.DataFrame,
    biotype_filter: pl.Expr | None = None,
) -> pl.DataFrame:
    """Extract exons from an annotation DataFrame.

    Returns ``[chrom, start, end, gene_id]`` for exons of transcripts whose
    ``transcript_biotype`` matches ``biotype_filter``, restricted to canonical
    chromosomes. Default keeps only ``transcript_biotype == "protein_coding"``.
    """
    if biotype_filter is None:
        biotype_filter = pl.col("transcript_biotype") == "protein_coding"
    return (
        ann.filter(pl.col("feature") == "exon")
        .with_columns(
            pl.col("attribute").str.extract(r'gene_id "([^;]*)";').alias("gene_id"),
            pl.col("attribute")
            .str.extract(r'transcript_biotype "([^;]*)";')
            .alias("transcript_biotype"),
        )
        .filter(
            biotype_filter,
            pl.col("chrom").is_in(CHROMS),
        )
        .select(["chrom", "start", "end", "gene_id"])
        .unique()
        .sort(["chrom", "start", "end"])
    )


def _nearest(
    V: pl.DataFrame,
    ref: pl.DataFrame,
    dist_name: str,
    gene_name: str,
    suffix: str,
) -> pl.DataFrame:
    """Wrap ``pb.nearest`` and rename outputs to caller-chosen final names so
    multiple nearest joins can chain without column collisions."""
    return (
        pb.nearest(
            V,
            ref,
            cols1=("chrom", "start", "end"),
            cols2=("chrom", "start", "end"),
            suffixes=("", suffix),
            output_type="polars.DataFrame",
        )
        .rename({"distance": dist_name, f"gene_id{suffix}": gene_name})
        .drop(f"chrom{suffix}", f"start{suffix}", f"end{suffix}")
    )


def add_exon(
    V: pl.DataFrame,
    exon_pc: pl.DataFrame,
    exon_nc: pl.DataFrame,
    exon_proximal_dist: int = 100,
) -> pl.DataFrame:
    """Attach per-biotype exon distances + closest gene IDs, plus combined
    ``distance_exon`` (= min(pc, nc)) and ``exon_closest_gene_id`` (the gene id
    from whichever biotype is closer). Updates ``consequence_final`` to
    ``"exon_proximal"`` when consequence == intron_variant and the combined
    distance <= ``exon_proximal_dist``.
    """
    V_intervals = V.with_columns(
        (pl.col("pos") - 1).alias("start"),
        pl.col("pos").alias("end"),
    )
    res = _nearest(
        V_intervals, exon_pc, "distance_exon_pc", "exon_closest_pc_gene_id", "_pc"
    )
    res = _nearest(res, exon_nc, "distance_exon_nc", "exon_closest_nc_gene_id", "_nc")
    return (
        res.with_columns(
            pl.min_horizontal("distance_exon_pc", "distance_exon_nc").alias(
                "distance_exon"
            ),
            pl.when(pl.col("distance_exon_pc") <= pl.col("distance_exon_nc"))
            .then(pl.col("exon_closest_pc_gene_id"))
            .otherwise(pl.col("exon_closest_nc_gene_id"))
            .alias("exon_closest_gene_id"),
        )
        .with_columns(
            pl.when(
                pl.col("consequence") == "intron_variant",
                pl.col("distance_exon") <= exon_proximal_dist,
            )
            .then(pl.lit("exon_proximal"))
            .otherwise(pl.col("consequence_cre"))
            .alias("consequence_final"),
        )
        .drop("start", "end")
    )


def add_tss(
    V: pl.DataFrame,
    tss_pc: pl.DataFrame,
    tss_nc: pl.DataFrame,
    tss_proximal_dist: int = 1000,
) -> pl.DataFrame:
    """Attach per-biotype TSS distances + closest gene IDs, plus combined
    ``distance_tss`` (= min(pc, nc)) and ``tss_closest_gene_id`` (the gene id
    from whichever biotype is closer). Updates ``consequence_final`` to
    ``"tss_proximal"`` when consequence ∈ NON_EXONIC and the combined
    distance <= ``tss_proximal_dist``.
    """
    V_intervals = V.with_columns(
        (pl.col("pos") - 1).alias("start"),
        pl.col("pos").alias("end"),
    )
    res = _nearest(
        V_intervals, tss_pc, "distance_tss_pc", "tss_closest_pc_gene_id", "_pc"
    )
    res = _nearest(res, tss_nc, "distance_tss_nc", "tss_closest_nc_gene_id", "_nc")
    return (
        res.with_columns(
            pl.min_horizontal("distance_tss_pc", "distance_tss_nc").alias(
                "distance_tss"
            ),
            pl.when(pl.col("distance_tss_pc") <= pl.col("distance_tss_nc"))
            .then(pl.col("tss_closest_pc_gene_id"))
            .otherwise(pl.col("tss_closest_nc_gene_id"))
            .alias("tss_closest_gene_id"),
        )
        .with_columns(
            pl.when(
                pl.col("consequence").is_in(NON_EXONIC),
                pl.col("distance_tss") <= tss_proximal_dist,
            )
            .then(pl.lit("tss_proximal"))
            .otherwise(pl.col("consequence_final"))
            .alias("consequence_final"),
        )
        .drop("start", "end")
    )


def build_dataset(
    V: pl.DataFrame,
    exon_pc: pl.DataFrame,
    exon_nc: pl.DataFrame,
    tss_pc: pl.DataFrame,
    tss_nc: pl.DataFrame,
    exclude_consequences: list[str],
    exon_proximal_dist: int,
    tss_proximal_dist: int,
    consequence_groups: dict[str, list[str]],
) -> pl.DataFrame:
    """Build a dataset with final consequence annotations and groups.

    Filters out ``exclude_consequences``; attaches PC + nc exon/TSS distances
    and combined min-distance final consequence; restricts to consequences
    observed in positives; adds the ``consequence_group`` column. Output is
    sorted by COORDINATES.
    """
    V = (
        V.filter(~pl.col("consequence").is_in(exclude_consequences))
        .pipe(add_exon, exon_pc, exon_nc, exon_proximal_dist)
        .pipe(add_tss, tss_pc, tss_nc, tss_proximal_dist)
    )
    consequence_final_pos = V.filter("label")["consequence_final"].unique()
    V = V.filter(pl.col("consequence_final").is_in(consequence_final_pos))
    consequence_to_group = {
        c: group
        for group, consequences in consequence_groups.items()
        for c in consequences
    }
    V = V.with_columns(
        pl.col("consequence_final")
        .replace(consequence_to_group)
        .alias("consequence_group")
    ).sort(COORDINATES)
    assert V["consequence_group"].null_count() == 0
    return V
