"""Saturation genome editing (SGE) variant-effect dataset construction (issue #289).

Builds the ``bolinas-dna/evals_sge`` HF artifact: per-SNV experimental function
scores from endogenous saturation-genome-editing assays, normalized to GRCh38 with
the standard consequence/distance annotation.

How it differs from the other eval datasets:

- vs the matched-pair datasets (``mendelian_traits`` / ``complex_traits``): **no
  matching, no subsetting** — every assayed SNV is kept.
- vs the QTL datasets (``caqtl`` / ``dsqtl``, which keep all consequences): the
  HIGH-impact ``exclude_consequences`` (canonical splice, nonsense, frameshift, …)
  are **dropped**. Those are trivially-deleterious and not the discriminative
  signal an SGE benchmark is about.

Every original author column is preserved verbatim under an ``author_`` prefix
(namespaced so nothing collides with the pipeline's annotation columns —
``consequence``, ``consequence_final``, ``distance_*``, …); no binary label is
imposed here (an eval-time decision).

Phase-1 source: BRCA1 (Findlay et al. 2018, *Nature* 562:217-222), via the
Evo2-bundled supplementary table ``41586_2018_461_MOESM3_ESM.xlsx``. It carries
hg19 genomic coordinates for **all** SNVs including intronic ones (unlike MaveDB,
whose cDNA->genome mapping drops intronic variants), so it is the cleanest BRCA1
source; coordinates are lifted hg19->GRCh38 here.
"""

import re
from pathlib import Path

import pandas as pd
import polars as pl

from marin_dna.data.genome import Genome
from marin_dna.pipelines.evals.trait_intervals import add_exon, add_tss
from marin_dna.pipelines.evals.variants import (
    COORDINATES,
    attach_per_chrom_consequences,
    check_ref_alt,
    filter_chroms,
    filter_snp,
    lift_hg19_to_hg38,
)

# Findlay 2018 BRCA1 SGE xlsx columns that become the standard pipeline coords;
# every other column is preserved verbatim under an ``author_`` prefix.
_BRCA1_COORD_SOURCE: dict[str, str] = {
    "chromosome": "chrom",
    "position (hg19)": "pos",
    "reference": "ref",
    "alt": "alt",
    "gene": "gene",
}


def author_slug(name: str) -> str:
    """``author_``-prefixed, identifier-safe column name (lowercased; each run of
    non-alphanumerics -> ``_``). Namespaces source columns so they never collide
    with pipeline columns (``consequence``, ``consequence_final``, ``distance_*``)."""
    return "author_" + re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()


def normalize_brca1_findlay(raw: pl.DataFrame) -> pl.DataFrame:
    """Normalize the Findlay 2018 BRCA1 SGE table (already header-resolved) to the
    SGE schema.

    Emits the standard ``chrom, pos, ref, alt, gene, assay, source`` columns plus
    **every original column preserved verbatim under an ``author_`` prefix** — so no
    author metadata is lost and nothing collides with the pipeline's annotation
    columns. ``chrom`` is a string and ``pos`` is 1-based hg19 (lifted to GRCh38
    downstream in :func:`annotate_sge_variants`). The headline author variables are
    ``author_function_score_mean`` (continuous) and ``author_func_class`` (discrete:
    FUNC/INT/LOF).
    """
    slugs = [author_slug(c) for c in raw.columns]
    assert len(set(slugs)) == len(slugs), (
        f"BRCA1: author column slugs collide: {sorted(slugs)}"
    )
    author = raw.rename(dict(zip(raw.columns, slugs)))
    for src in _BRCA1_COORD_SOURCE:
        assert author_slug(src) in author.columns, (
            f"BRCA1: missing expected source column {src!r} ({author_slug(src)})"
        )
    out = (
        author.with_columns(
            pl.col("author_chromosome").cast(pl.Utf8).alias("chrom"),
            pl.col("author_position_hg19").cast(pl.Int64).alias("pos"),
            pl.col("author_reference").cast(pl.Utf8).alias("ref"),
            pl.col("author_alt").cast(pl.Utf8).alias("alt"),
            pl.col("author_gene").cast(pl.Utf8).alias("gene"),
            assay=pl.lit("sge"),
            source=pl.lit("findlay2018"),
        )
        .pipe(filter_snp)
        # Standard pipeline columns first, then every preserved author column.
        .select(
            "chrom",
            "pos",
            "ref",
            "alt",
            "gene",
            "assay",
            "source",
            pl.col("^author_.*$"),
        )
    )
    assert out["author_function_score_mean"].null_count() == 0, (
        "BRCA1: null function score (author_function_score_mean) after normalization"
    )
    assert set(out["author_func_class"].unique()) <= {"FUNC", "INT", "LOF"}, (
        f"BRCA1: unexpected func.class values: {set(out['author_func_class'].unique())}"
    )
    return out


def read_brca1_findlay(xlsx_path: str | Path) -> pl.DataFrame:
    """Read + normalize the Findlay 2018 BRCA1 SGE supplementary xlsx.

    The sheet has two super-header rows above the real column header (row index 2),
    so it is read with ``header=2``. Returns the :func:`normalize_brca1_findlay`
    schema.
    """
    raw = pd.read_excel(xlsx_path, header=2)
    return normalize_brca1_findlay(pl.from_pandas(raw))


def annotate_sge_variants(
    V: pl.DataFrame,
    *,
    genome: Genome,
    consequence_paths: list[str],
    chroms: list[str],
    exon_pc: pl.DataFrame,
    exon_nc: pl.DataFrame,
    tss_pc: pl.DataFrame,
    tss_nc: pl.DataFrame,
    exon_proximal_dist: int,
    tss_proximal_dist: int,
    exclude_consequences: list[str],
    lift: bool,
    name: str = "",
) -> pl.DataFrame:
    """Liftover (optional) + ref/alt validation + consequence/distance annotation +
    HIGH-impact ``exclude_consequences`` drop, with **no** matching or subsampling.

    Mirrors :func:`marin_dna.pipelines.evals.dart_eval.annotate_variants` but (a)
    drops ``exclude_consequences`` (the QTL path keeps them; SGE drops them), (b)
    carries no signed ``effect`` (SGE has a direction-tied function score, not a
    QTL effect), and (c) **asserts zero ref/alt swaps**: an SGE function score is
    tied to the ref->alt substitution as the author defined it, so a swap (author
    ref != genome) signals a coordinate/strand/build problem, not a benign
    re-orientation.

    Args:
        V: normalized SGE frame (``chrom, pos, ref, alt`` + author columns); ``pos``
            is 1-based (hg19 if ``lift`` else GRCh38).
        genome: GRCh38 reference for :func:`check_ref_alt`.
        consequence_paths / chroms: per-chrom VEP-consequence parquet paths and
            their parallel chromosome labels (only the chroms present in ``V`` are
            needed).
        exon_pc / exon_nc / tss_pc / tss_nc: nearest-feature interval frames.
        exon_proximal_dist / tss_proximal_dist: proximity thresholds for the
            ``consequence_final`` recategorization.
        exclude_consequences: VEP consequences to drop (canonical-LOF HIGH-impact).
        lift: if True, lift hg19->GRCh38 first.
        name: label for log/assert messages.
    """
    assert V.height > 0, f"{name}: empty input frame"
    n_in = V.height
    n_strand_flip = 0
    if lift:
        # Keep the pre-lift ref to count strand-flips: for an SNV, lift_hg19_to_hg38
        # only changes `ref` when the chain maps to the minus strand (RC). This is
        # informational — a strand-RC preserves ref/alt roles, so the function score
        # is carried unchanged (see the swap note below).
        V = V.with_columns(pl.col("ref").alias("_pre_lift_ref"))
        V = V.pipe(lift_hg19_to_hg38).filter(pl.col("pos") != -1)
        n_strand_flip = V.filter(pl.col("ref") != pl.col("_pre_lift_ref")).height
        V = V.drop("_pre_lift_ref")
    n_lift = V.height
    V = V.pipe(filter_chroms)
    n_chrom = V.height
    # Two distinct allele transforms, with different consequences for an
    # alt-vs-ref quantity like the SGE function score:
    #   - liftover strand-RC (above): RCs both alleles but PRESERVES ref/alt
    #     roles (ref stays the WT allele, complemented), so the physical variant
    #     and its score are unchanged — no flip.
    #   - check_ref_alt swap (author ref != genome): re-labels which allele is
    #     ref. For SGE this is invalid, not a benign re-orientation: the score is
    #     "effect of the alt (variant) allele vs WT", so a swap would nonsensically
    #     call the WT allele the variant. We use Findlay's +strand GENOMIC
    #     `reference`/`alt` (not the mRNA-strand `transcript_*`), so ref should
    #     match the +strand genome and no swap should ever fire. Assert it.
    V = V.with_columns(pl.col("ref").alias("_pre_ref"))
    V = check_ref_alt(V, genome)
    n_ref = V.height
    n_swapped = V.filter(pl.col("ref") != pl.col("_pre_ref")).height
    V = V.drop("_pre_ref")
    print(
        f"[sge annotate {name}] attrition: in={n_in} after_lift={n_lift} "
        f"after_chrom_filter={n_chrom} after_ref_alt={n_ref} "
        f"lift_strand_flipped={n_strand_flip} ref_alt_swapped={n_swapped}"
    )
    assert n_swapped == 0, (
        f"{name}: {n_swapped} ref/alt swaps in check_ref_alt — an SGE function "
        "score is tied to ref->alt, so a swap signals a coordinate/strand/build "
        "mismatch, not a benign re-orientation"
    )
    assert n_ref >= 0.9 * n_lift, (
        f"check_ref_alt kept only {n_ref}/{n_lift} variants for {name!r} — suspect a "
        "coordinate-base (0- vs 1-based) or genome-build mismatch"
    )
    V = attach_per_chrom_consequences(V, consequence_paths, chroms)
    assert V["consequence"].null_count() == 0, f"{name}: variants with null consequence"
    # Drop the trivially-deleterious HIGH-impact consequences (canonical splice,
    # nonsense, frameshift, …) before distance recategorization.
    n_pre_excl = V.height
    V = V.filter(~pl.col("consequence").is_in(exclude_consequences))
    print(
        f"[sge annotate {name}] exclude_consequences dropped "
        f"{n_pre_excl - V.height} HIGH-impact variants ({V.height} kept)"
    )
    assert V.height > 0, f"{name}: all variants dropped by exclude_consequences"
    V = (
        V.pipe(add_exon, exon_pc, exon_nc, exon_proximal_dist)
        .pipe(add_tss, tss_pc, tss_nc, tss_proximal_dist)
        .sort(COORDINATES)
    )
    assert (V["pos"] > 0).all(), f"{name}: non-positive positions after annotation"
    assert V["consequence_final"].null_count() == 0, (
        f"{name}: null consequence_final after annotation"
    )
    return V
