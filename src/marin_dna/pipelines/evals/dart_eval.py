"""DART-Eval caQTL / dsQTL variant-effect-prediction datasets.

Parsing + annotation for the two DART-Eval Task-5 chromatin-accessibility QTL
benchmarks, brought into the ``snakemake/evals`` pipeline with train/test
splits and **no matching / no subsampling** (rules in
``snakemake/evals/workflow/rules/dart_eval.smk``).

Sources (DART-Eval, Patel *et al.* NeurIPS 2024 D&B;
https://github.com/kundajelab/DART-Eval):

- **caQTL** — African caQTLs (DeGorter *et al.* 2023), Synapse ``syn60756043``,
  file ``Afr.CaQTLS.tsv``. Native **GRCh38** (``chr_hg38`` / ``pos_hg38``).
- **dsQTL** — Yoruban dsQTLs (Degner *et al.* 2012), Synapse ``syn60756039``,
  file ``yoruban.dsqtls.benchmarking.tsv``. **hg19** → lifted to GRCh38.

Positives are statistically significant QTLs falling within accessible peaks;
negatives are control variants in peaks. Every positive and negative is kept
at its natural ratio — there is no class balancing.

The TSV column names / encodings below come from DART-Eval's
``variant_tasks.py``; ``parse_caqtl`` / ``parse_dsqtl`` assert loudly if the
real file deviates so a schema drift can't silently corrupt the dataset.
"""

from __future__ import annotations

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

# Expected raw-TSV columns (DART-Eval variant_tasks.py).
CAQTL_REQUIRED = [
    "chr_hg38",
    "pos_hg38",
    "allele1",
    "allele2",
    "IsUsed",
    "in_peaks",
    "label",
]
DSQTL_REQUIRED = [
    "var.chrom",
    "var.pos",
    "var.allele1",
    "var.allele2",
    "var.isused",
    "var.label",
]

# polars numeric dtypes (membership test is version-robust).
_NUMERIC_DTYPES = frozenset(
    {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
)


def _strip_chr(col: pl.Expr) -> pl.Expr:
    """Drop a leading ``chr`` so chromosome names match our ``CHROMS``
    convention (``"1"``..``"22"``, ``"X"``, ``"Y"``)."""
    return col.cast(pl.Utf8).str.replace(r"^chr", "")


def _require_columns(df: pl.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    assert not missing, (
        f"{name} TSV missing expected columns {missing}; got {df.columns}"
    )


def _truthy(df: pl.DataFrame, col: str) -> pl.Expr:
    """Boolean expression for a flag column that may be stored as bool, 0/1
    integer, or a string like ``"True"``/``"1"``."""
    dt = df.schema[col]
    if dt == pl.Boolean:
        return pl.col(col)
    if dt in _NUMERIC_DTYPES:
        return pl.col(col) != 0
    return pl.col(col).cast(pl.Utf8).str.to_lowercase().is_in(["1", "true", "t", "yes"])


def _assert_label_values(
    df: pl.DataFrame, col: str, allowed: set[int], name: str
) -> None:
    raw = df.get_column(col)
    as_int = raw.cast(pl.Int64, strict=False)
    # Casting must not introduce nulls — that would mean a non-integer label.
    assert as_int.null_count() == raw.null_count(), (
        f"{name} {col!r} has non-integer label values"
    )
    vals = set(as_int.drop_nulls().unique().to_list())
    extra = vals - allowed
    assert not extra, (
        f"{name} {col!r} has unexpected values {sorted(extra)} "
        f"(allowed {sorted(allowed)}) after the used/in-peaks filter"
    )


def _assemble(
    df: pl.DataFrame,
    *,
    chrom_col: str,
    pos_col: str,
    a1_col: str,
    a2_col: str,
    label_expr: pl.Expr,
    effect_col: str,
    name: str,
    extra_cols: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Project the raw frame onto the standard schema. ``allele1``/``allele2``
    become ``ref``/``alt`` provisionally — `check_ref_alt` (in
    ``annotate_variants``) reorients them against the reference. ``label_expr``
    is the boolean positive/negative indicator built by the caller (the two
    datasets encode the label differently).

    ``effect`` is the signed study effect, parsed as the effect of ``allele2``
    (= ``alt`` here); ``annotate_variants`` flips its sign for variants whose
    ref/alt get swapped so it stays signed relative to the final ``alt``, then
    derives ``effect_size = abs(effect)`` (the unsigned magnitude). ``extra_cols``
    maps output name -> source column for dataset-specific passthrough floats
    (e.g. caQTL ``pval``/``se``), cast to Float64."""
    cols = [
        _strip_chr(pl.col(chrom_col)).alias("chrom"),
        pl.col(pos_col).cast(pl.Int64).alias("pos"),
        pl.col(a1_col).cast(pl.Utf8).str.to_uppercase().alias("ref"),
        pl.col(a2_col).cast(pl.Utf8).str.to_uppercase().alias("alt"),
        label_expr.alias("label"),
    ]
    for out_name, src in {"effect": effect_col, **(extra_cols or {})}.items():
        if src in df.columns:
            cols.append(pl.col(src).cast(pl.Float64).alias(out_name))
        else:
            print(
                f"WARNING [dart_eval {name}]: column {src!r} absent; "
                f"{out_name} set to null"
            )
            cols.append(pl.lit(None, dtype=pl.Float64).alias(out_name))
    return df.select(cols)


def parse_caqtl(df: pl.DataFrame) -> pl.DataFrame:
    """Parse a raw ``Afr.CaQTLS.tsv`` frame into the standard schema.

    Restricts to DART-Eval's benchmark set (``IsUsed`` and ``in_peaks`` — this
    reproduces the published 6,821 positive / 77,999 control counts), uses the
    boolean ``label`` (True = significant caQTL, False = control), and keeps
    SNVs only. Native GRCh38.
    """
    _require_columns(df, CAQTL_REQUIRED, "caQTL")
    df = df.filter(_truthy(df, "IsUsed") & _truthy(df, "in_peaks"))
    # `label` is a boolean flag (True = significant caQTL, False = control).
    assert df.schema["label"] == pl.Boolean, (
        f"caQTL: expected boolean `label`, got {df.schema['label']}"
    )
    out = _assemble(
        df,
        chrom_col="chr_hg38",
        pos_col="pos_hg38",
        a1_col="allele1",
        a2_col="allele2",
        label_expr=pl.col("label"),
        effect_col="beta",
        name="caqtl",
        # caQTL-only reference columns (not used by the eval, kept for the
        # record): association p-value and the standard error of beta.
        extra_cols={"pval": "pval", "se": "se"},
    )
    assert out["label"].null_count() == 0, "caQTL: null labels after parse"
    return out.pipe(filter_snp)


def parse_dsqtl(df: pl.DataFrame) -> pl.DataFrame:
    """Parse a raw ``yoruban.dsqtls.benchmarking.tsv`` frame into the standard
    schema.

    Restricts to ``var.isused`` (reproduces the published 560 positive / 26,813
    control counts), maps ``var.label`` 1→positive / −1→negative, and keeps
    SNVs only. hg19 coordinates (lifted to GRCh38 in ``annotate_variants``).
    """
    _require_columns(df, DSQTL_REQUIRED, "dsQTL")
    df = df.filter(_truthy(df, "var.isused"))
    _assert_label_values(df, "var.label", {1, -1}, "dsQTL")
    out = _assemble(
        df,
        chrom_col="var.chrom",
        pos_col="var.pos",
        a1_col="var.allele1",
        a2_col="var.allele2",
        label_expr=pl.col("var.label") == 1,
        effect_col="obs.estimate",
        name="dsqtl",
    )
    assert out["label"].null_count() == 0, "dsQTL: null labels after parse"
    return out.pipe(filter_snp)


def annotate_variants(
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
    lift: bool,
    name: str = "",
) -> pl.DataFrame:
    """Liftover (optional) + ref/alt orientation + consequence/distance
    annotations, with **no** matching-style filtering or subsampling.

    Mirrors the annotation portion of ``complex_traits_annotate`` +
    ``complex_traits_dataset_all`` but deliberately skips ``build_dataset``,
    which drops ``exclude_consequences``, restricts negatives to consequences
    seen in positives, and asserts every consequence maps to a group — all
    matching-prep behaviours that would violate "no subsampling".

    Drops are limited to build-correctness QC: unmapped liftover positions,
    non-canonical contigs, and variants whose alleles don't match the
    reference. These are reported (printed) and guarded by a retention assert.

    Args:
        V: parsed frame (``chrom, pos, ref, alt, label, effect``); ``pos``
            is 1-based.
        genome: reference for ``check_ref_alt`` (GRCh38).
        consequence_paths: per-chrom consequence parquet paths, parallel to
            ``chroms``.
        chroms: per-path chromosome labels, parallel to ``consequence_paths``.
        exon_pc/exon_nc/tss_pc/tss_nc: nearest-feature interval frames.
        exon_proximal_dist/tss_proximal_dist: proximity thresholds for the
            ``consequence_final`` recategorization.
        lift: if True, lift hg19→GRCh38 first (dsQTL); else assume GRCh38
            (caQTL).
        name: label for log/assert messages.
    """
    assert V.height > 0, f"{name}: empty input frame"
    n_in = V.height
    if lift:
        V = V.pipe(lift_hg19_to_hg38).filter(pl.col("pos") != -1)
    n_lift = V.height
    V = V.pipe(filter_chroms)
    n_chrom = V.height
    # Record alt before check_ref_alt may swap ref<->alt. `effect` is parsed as
    # the effect of the study allele we assigned to alt (allele2), so when a
    # swap moves the effect allele to ref, flip the sign to keep `effect` signed
    # relative to the FINAL alt (i.e. effect of alt vs ref — what an alt-vs-ref /
    # LLR model score correlates against). Liftover RCs both alleles but
    # preserves ref/alt roles, so only the swap flips the sign. A high flip rate
    # is expected when the study codes allele1 as the non-reference allele
    # (e.g. dsQTL, ~81%).
    V = V.with_columns(pl.col("alt").alias("_pre_alt"))
    V = check_ref_alt(V, genome)
    n_ref = V.height
    swapped = pl.col("alt") != pl.col("_pre_alt")
    n_flipped = V.filter(swapped).height
    V = V.with_columns(
        pl.when(swapped)
        .then(-pl.col("effect"))
        .otherwise(pl.col("effect"))
        .alias("effect")
    ).drop("_pre_alt")
    # Magnitude alongside the signed effect (orientation-independent).
    V = V.with_columns(pl.col("effect").abs().alias("effect_size"))
    print(
        f"[dart_eval annotate {name}] attrition: in={n_in} "
        f"after_lift={n_lift} after_chrom_filter={n_chrom} after_ref_alt={n_ref} "
        f"effect_sign_flipped={n_flipped}"
    )
    # All drops above are build-correctness QC, not class balancing. A uniform
    # coordinate-base (0- vs 1-based) or genome-build error would make ref_alt
    # reject ~everything — guard with a loose retention floor.
    assert n_ref >= 0.5 * n_in, (
        f"check_ref_alt kept only {n_ref}/{n_in} variants for {name!r} — "
        "suspect a coordinate-base (0- vs 1-based) or genome-build mismatch"
    )
    V = attach_per_chrom_consequences(V, consequence_paths, chroms)
    # songlab/hg38-variant-consequences covers all possible SNVs, so every
    # (already SNV-filtered) variant must receive a consequence.
    assert V["consequence"].null_count() == 0, f"{name}: variants with null consequence"
    assert V["consequence_cre"].null_count() == 0, (
        f"{name}: variants with null consequence_cre"
    )
    V = (
        V.pipe(add_exon, exon_pc, exon_nc, exon_proximal_dist)
        .pipe(add_tss, tss_pc, tss_nc, tss_proximal_dist)
        .sort(COORDINATES)
    )
    assert V["label"].dtype == pl.Boolean, f"{name}: label is not Boolean"
    assert (V["pos"] > 0).all(), f"{name}: non-positive positions after annotation"
    assert V["consequence_final"].null_count() == 0, (
        f"{name}: null consequence_final after annotation"
    )
    return V
