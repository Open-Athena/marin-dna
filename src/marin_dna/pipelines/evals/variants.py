"""Variant constants and coordinate utilities for trait-pipeline curation.

Ported from TraitGym (commit e59d612e9; src/traitgym/variants.py).
"""

import polars as pl
from liftover import get_lifter

from marin_dna.data.dna import COMPLEMENT, NUCLEOTIDES, reverse_complement  # noqa: F401  re-exported

COORDINATES = ["chrom", "pos", "ref", "alt"]
CHROMS = sorted([str(i) for i in range(1, 23)] + ["X", "Y"])
NON_EXONIC = [
    "intergenic_variant",
    "intron_variant",
    "upstream_gene_variant",
    "downstream_gene_variant",
]


def filter_snp(V: pl.DataFrame) -> pl.DataFrame:
    return V.filter(pl.col("ref").is_in(NUCLEOTIDES) & pl.col("alt").is_in(NUCLEOTIDES))


def filter_chroms(V: pl.DataFrame) -> pl.DataFrame:
    return V.filter(pl.col("chrom").is_in(CHROMS))


def lift_hg19_to_hg38(V: pl.DataFrame) -> pl.DataFrame:
    """Lift variants from hg19 to hg38, sets pos=-1 for unmapped."""
    converter = get_lifter("hg19", "hg38", one_based=True)
    original_columns = V.columns

    def get_new_coords(row: dict) -> dict:
        try:
            res = converter[row["chrom"]][row["pos"]]
            assert len(res) == 1
            chrom, pos, strand = res[0]
            chrom = chrom.replace("chr", "")
            ref = row["ref"]
            alt = row["alt"]
            if strand == "-":
                ref = reverse_complement(ref)
                alt = reverse_complement(alt)
            return {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt}
        except Exception:
            return {
                "chrom": row["chrom"],
                "pos": -1,
                "ref": row["ref"],
                "alt": row["alt"],
            }

    return (
        V.with_columns(
            pl.struct(["chrom", "pos", "ref", "alt"])
            .map_elements(
                get_new_coords,
                return_dtype=pl.Struct(
                    {
                        "chrom": pl.Utf8,
                        "pos": pl.Int64,
                        "ref": pl.Utf8,
                        "alt": pl.Utf8,
                    }
                ),
            )
            .alias("_coords")
        )
        .drop("chrom", "pos", "ref", "alt")
        .unnest("_coords")
        .select(original_columns)
    )


def attach_per_chrom_consequences(
    V: pl.DataFrame,
    consequence_paths: list[str],
    chroms: list[str],
) -> pl.DataFrame:
    """Left-join per-chrom consequences onto a variants frame, one chrom at
    a time, using parquet predicate pushdown on ``pos`` to avoid scanning
    non-matching row groups.

    Used by ``complex_traits_annotate`` and ``eqtl_annotate`` — both have
    ~1-10M variants spread across 24 chroms, joining against ~30M-row
    per-chrom consequence parquets. A naive cross-join materializes the
    right side; per-chrom + ``pos.is_in()`` lets parquet skip irrelevant
    row groups, cutting peak memory by ~30×.

    Args:
        V: variants frame with at least ``chrom`` and ``pos`` columns.
        consequence_paths: list of parquet paths, one per chrom in the same
            order as ``chroms`` — each parquet must have the four
            ``COORDINATES`` columns + the consequence columns to attach.
        chroms: per-path chromosome label list, parallel to
            ``consequence_paths``. Length must match ``consequence_paths``.

    Returns:
        ``pl.DataFrame`` with original ``V`` columns plus the consequence
        columns from the per-chrom parquets. Chroms with zero matching
        variants are skipped.
    """
    assert len(consequence_paths) == len(chroms), (
        f"path/chrom length mismatch: {len(consequence_paths)} vs {len(chroms)}"
    )
    results: list[pl.DataFrame] = []
    for path, chrom in zip(consequence_paths, chroms):
        pos_chrom = V.filter(pl.col("chrom") == chrom)
        if pos_chrom.height == 0:
            continue
        cons_subset = (
            pl.scan_parquet(path)
            .filter(pl.col("pos").is_in(pos_chrom["pos"].unique().to_list()))
            .collect()
        )
        results.append(pos_chrom.join(cons_subset, on=COORDINATES, how="left"))
    return pl.concat(results)


def check_ref_alt(V: pl.DataFrame, genome) -> pl.DataFrame:
    """Check ref/alt against the reference genome.

    For each variant: look up the reference nucleotide. If ref doesn't match,
    swap ref and alt. Drop variants where neither matches.
    """

    def get_ref_nuc(row: dict) -> str:
        # biofoundation.data.Genome is callable with 0-based half-open
        # [start, end) coords; pos is 1-based, so the single base at pos is
        # the half-open interval [pos-1, pos).
        return genome(row["chrom"], row["pos"] - 1, row["pos"]).upper()

    original_columns = V.columns

    return (
        V.with_columns(
            pl.struct(["chrom", "pos"])
            .map_elements(get_ref_nuc, return_dtype=pl.Utf8)
            .alias("_ref_nuc")
        )
        .with_columns(_needs_swap=(pl.col("ref") != pl.col("_ref_nuc")))
        .with_columns(
            ref=pl.when(pl.col("_needs_swap"))
            .then(pl.col("alt"))
            .otherwise(pl.col("ref")),
            alt=pl.when(pl.col("_needs_swap"))
            .then(pl.col("ref"))
            .otherwise(pl.col("alt")),
        )
        .filter(pl.col("ref") == pl.col("_ref_nuc"))
        .select(original_columns)
    )
