"""Canonical caQTL / dsQTL variant datasets from the **standardized** ChromBPNet
benchmark files (Synapse ``syn64126763``).

These are the variant sets used by *both* the ChromBPNet and AlphaGenome papers
for the supervised accessibility-QTL benchmark (issue #309 / #262). Each
standardized TSV carries **precomputed** per-variant baseline scores — ChromBPNet
GM12878 DNase (``encsr000emt``) + ATAC (``encsr637xsc``) ``logfc``/``ips`` and
Enformer's recomputed local-2 kb ``local_logfc`` — which we carry through into the
dataset so the ChromBPNet / Enformer baselines are free downstream (no model run).

The official protocol (#262): **causality** = auPRC on ``|score|`` over all
``var.isused`` variants; **direction** = signed Pearson of the score vs the study
``effect`` over positives only. Both are computed downstream by
``marin_dna.pipelines.chrombpnet_eval.metrics.compute_supervised_qtl_metrics``;
chromosome splits (all / AG-test / odd-even) are a metrics-time slice, not baked in.

Build (productionizes the #262 one-off ``standardized_qtl.py`` +
``ag_qtl_run.py::build_variants``):

1. ``load_standardized_qtl`` — restrict to ``var.isused`` (positives −log10p>5 ∪
   controls −log10p<3, dropping the ambiguous middle), project onto
   ``[chrom, pos, ref, alt, label, effect]`` (``ref``/``alt`` = the file's
   ``allele1``/``allele2``) plus the canonical baseline score columns.
2. ``build_qtl_dataset`` — dsQTL: lift hg19→hg38 (drop unmapped) + ``filter_chroms``;
   both: ``check_ref_alt`` genome-orient. **On a ref/alt swap the sign of ``effect``
   AND every carried signed score column is flipped** so the whole row stays in the
   genome-oriented alt-vs-ref frame (see ``SIGNED_SCORE_COLS``).

Schema quirks handled below: the per-score column **infix** differs between files
(``variantscore`` for caQTL, ``varscore`` for dsQTL); both use ``obs.label``
(significance), ``var.isused`` (benchmark membership), and a per-file effect column
(caQTL ``obs.beta`` / dsQTL ``obs.estimate``). caQTL is native hg38; dsQTL is hg19.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from marin_dna.data.genome import Genome
from marin_dna.pipelines.evals.variants import (
    check_ref_alt,
    filter_chroms,
    lift_hg19_to_hg38,
)

# ENCODE experiment id -> human-readable GM12878 assay label.
ENCODE_ASSAY: dict[str, str] = {
    "encsr000emt": "GM12878 DNase",
    "encsr637xsc": "GM12878 ATAC",
}


@dataclass(frozen=True)
class StandardizedQTL:
    """One standardized benchmark file (Synapse ``syn64126763`` children)."""

    name: str
    synapse_id: str
    build: str  # "hg38" (caQTL) | "hg19" (dsQTL, lifted)
    chrom_col: str
    pos_col: str
    effect_col: str
    score_infix: str  # "variantscore" (caqtl) | "varscore" (dsqtl)


STANDARDIZED_QTL: dict[str, StandardizedQTL] = {
    "caqtl": StandardizedQTL(
        name="caqtl",
        synapse_id="syn64126781",
        build="hg38",
        chrom_col="var.chr",
        pos_col="var.pos_hg38",
        effect_col="obs.beta",
        score_infix="variantscore",
    ),
    "dsqtl": StandardizedQTL(
        name="dsqtl",
        synapse_id="syn64126779",
        build="hg19",
        chrom_col="var.chr",
        pos_col="var.pos_hg19",
        effect_col="obs.estimate",
        score_infix="varscore",
    ),
}

# Canonical (file-independent) baseline score column -> source-column template, with
# ``{inf}`` filled by the per-file ``score_infix``. ChromBPNet uses **IPS** for
# causality and **logFC** for direction; Enformer uses its local-2 kb ``local_logfc``
# for both. ``encsr637xsc`` = GM12878 ATAC, ``encsr000emt`` = GM12878 DNase.
SCORE_COL_TEMPLATES: dict[str, str] = {
    "chrombpnet_atac_ips": "pred.chrombpnet.encsr637xsc.{inf}.ips",
    "chrombpnet_atac_logfc": "pred.chrombpnet.encsr637xsc.{inf}.logfc",
    "chrombpnet_dnase_ips": "pred.chrombpnet.encsr000emt.{inf}.ips",
    "chrombpnet_dnase_logfc": "pred.chrombpnet.encsr000emt.{inf}.logfc",
    "enformer_dnase_local_logfc": "pred.enformer.encsr000emt.{inf}.local_logfc",
}

# The headline baselines every standardized file must carry (#262): ChromBPNet
# GM12878-ATAC IPS (causality) + logFC (direction), and Enformer local-logfc. The
# GM12878-DNase ChromBPNet columns are carried when present but not required.
REQUIRED_SCORE_COLS: tuple[str, ...] = (
    "chrombpnet_atac_ips",
    "chrombpnet_atac_logfc",
    "enformer_dnase_local_logfc",
)

# Every carried score column is a **signed** allelic score in the alt-vs-ref frame
# (logFC = log2(alt/ref) accessibility; IPS = logFC × JSD × AAQ, so sign(IPS) =
# sign(logFC)). When ``check_ref_alt`` swaps ref/alt the allelic direction inverts,
# so each of these must be sign-flipped alongside ``effect`` to stay aligned. (The
# causality metric uses ``|score|`` and is swap-invariant; only direction is sign-
# sensitive, but flipping all of them keeps every row internally consistent.)
SIGNED_SCORE_COLS: tuple[str, ...] = tuple(SCORE_COL_TEMPLATES)

# Acceptance counts after the ``var.isused`` filter (#262; = AlphaGenome random
# baselines 0.0852 / 0.0200 on its test chroms). Asserted in ``build_qtl_dataset``.
EXPECTED_USED_COUNTS: dict[str, dict[str, int]] = {
    "caqtl": {"pos": 6821, "ctrl": 72205},
    "dsqtl": {"pos": 560, "ctrl": 26813},
}


def _truthy_expr(df: pl.DataFrame, col: str) -> pl.Expr:
    """Boolean expression for a flag column stored as bool, 0/1, or ``"True"``/``"1"``
    (used for ``var.isused``)."""
    dt = df.schema[col]
    if dt == pl.Boolean:
        return pl.col(col)
    if dt.is_numeric():
        return pl.col(col) != 0
    return pl.col(col).cast(pl.Utf8).str.to_lowercase().is_in(["1", "true", "t", "yes"])


def _positive_label_expr(df: pl.DataFrame, col: str = "obs.label") -> pl.Expr:
    """Positive (= significant QTL) boolean from ``obs.label``.

    The two standardized files encode the label differently: caQTL uses a boolean
    (``True`` = significant), dsQTL uses an integer **``1`` = significant, ``-1`` =
    control**. A plain truthiness test would mislabel dsQTL's ``-1`` controls as
    positive — map numeric labels with ``== 1`` instead (``-1``/``0`` → negative).
    Asserts numeric labels live in ``{-1, 0, 1}`` so an unexpected encoding fails loud.
    """
    dt = df.schema[col]
    if dt == pl.Boolean:
        return pl.col(col)
    if dt.is_numeric():
        vals = set(df.get_column(col).drop_nulls().unique().to_list())
        assert vals <= {-1, 0, 1}, f"unexpected {col!r} values {sorted(vals)}"
        return pl.col(col) == 1
    return pl.col(col).cast(pl.Utf8).str.to_lowercase().is_in(["1", "true", "t", "yes"])


def load_standardized_qtl(path: str, spec: StandardizedQTL) -> pl.DataFrame:
    """Parse a standardized benchmark TSV into the dataset's native-build schema.

    Restricts to ``var.isused`` (the benchmark set) and projects onto
    ``[chrom, pos, ref, alt, label, effect]`` plus the carried baseline score
    columns (renamed to canonical, file-independent names — any source column
    absent in this file is simply omitted, except the ``REQUIRED_SCORE_COLS`` which
    must be present). ``ref``/``alt`` are the file's ``allele1``/``allele2`` (not yet
    genome-oriented — that happens in ``build_qtl_dataset``). ``label`` is the
    boolean ``obs.label``; ``effect`` is the signed study effect.

    Coordinates are the file's native build (caQTL hg38, dsQTL hg19); ``pos`` is
    1-based and ``chrom`` has any leading ``chr`` stripped.
    """
    df = pl.read_csv(path, separator="\t", infer_schema_length=None)
    required = [
        "var.isused",
        "obs.label",
        "var.allele1",
        "var.allele2",
        spec.chrom_col,
        spec.pos_col,
        spec.effect_col,
    ]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"{spec.name}: standardized TSV missing columns {missing}"

    df = df.filter(_truthy_expr(df, "var.isused"))
    assert df.height > 0, f"{spec.name}: no var.isused variants"

    out_cols = [
        pl.col(spec.chrom_col).cast(pl.Utf8).str.replace(r"^chr", "").alias("chrom"),
        pl.col(spec.pos_col).cast(pl.Int64).alias("pos"),
        pl.col("var.allele1").cast(pl.Utf8).str.to_uppercase().alias("ref"),
        pl.col("var.allele2").cast(pl.Utf8).str.to_uppercase().alias("alt"),
        _positive_label_expr(df).alias("label"),
        pl.col(spec.effect_col).cast(pl.Float64).alias("effect"),
    ]
    inf = spec.score_infix
    for canon, template in SCORE_COL_TEMPLATES.items():
        src = template.format(inf=inf)
        if src in df.columns:
            out_cols.append(pl.col(src).cast(pl.Float64).alias(canon))
        else:
            assert canon not in REQUIRED_SCORE_COLS, (
                f"{spec.name}: required baseline column {src!r} ({canon}) absent"
            )
    out = df.select(out_cols)
    assert out["label"].dtype == pl.Boolean, f"{spec.name}: label is not Boolean"
    # Every positive must carry a measured effect (controls legitimately may not).
    # CSV-empty floats read as null, not NaN, so check both.
    eff_pos = out.filter(pl.col("label")).get_column("effect")
    n_nan_pos = int((eff_pos.is_null() | eff_pos.is_nan()).sum())
    assert n_nan_pos == 0, f"{spec.name}: {n_nan_pos} positives with NaN effect"
    return out


def orient_variants(V: pl.DataFrame, genome: Genome) -> tuple[pl.DataFrame, int]:
    """Genome-orient ``ref``/``alt`` and keep the signed columns aligned.

    Runs ``check_ref_alt`` (swaps ref↔alt where ``allele1`` ≠ the reference base, and
    drops variants matching neither allele), then sign-flips ``effect`` and every
    carried column in ``SIGNED_SCORE_COLS`` on the same swap mask so each stays signed
    relative to the final ``alt``. Returns ``(oriented_frame, n_swapped)``.

    This is the #310 correctness fix: #262's per-variant build dropped the score
    columns, so carrying them through orientation is new — without the matched flip a
    swapped variant's carried score would stay in the *old* allele frame and
    anti-correlate with the oriented ``effect``. ``|score|`` (causality) is swap-
    invariant; signed-Pearson (direction) is invariant only when score and effect flip
    together, which this guarantees.
    """
    V = V.with_columns(pl.col("alt").alias("_pre_alt"))
    V = check_ref_alt(V, genome)
    swapped = pl.col("alt") != pl.col("_pre_alt")
    n_swapped = V.filter(swapped).height
    flip_cols = ["effect", *(c for c in SIGNED_SCORE_COLS if c in V.columns)]
    V = V.with_columns(
        *(
            pl.when(swapped).then(-pl.col(c)).otherwise(pl.col(c)).alias(c)
            for c in flip_cols
        )
    ).drop("_pre_alt")
    return V, n_swapped


def build_qtl_dataset(name: str, tsv_path: str, genome: Genome) -> pl.DataFrame:
    """Build the canonical genome-oriented hg38 dataset for ``name`` (caqtl/dsqtl).

    Returns ``[chrom, pos, ref, alt, label, effect, <carried score columns>]`` on
    GRCh38, with ``effect`` and every carried signed score column oriented to the
    genome ``alt`` allele.

    Steps: load (``var.isused``) → dsQTL lift hg19→hg38 (drop unmapped) →
    ``filter_chroms`` (canonical 1..22,X,Y) → ``check_ref_alt`` (swap ref/alt where
    ``allele1`` ≠ genome reference). On a swap, ``effect`` and ``SIGNED_SCORE_COLS``
    are sign-flipped so they stay signed relative to the final ``alt`` (this is the
    invariant that lets the carried baselines reproduce the #262 numbers regardless
    of how the source coded allele1/allele2).
    """
    spec = STANDARDIZED_QTL[name]
    V = load_standardized_qtl(tsv_path, spec)

    # Acceptance: the var.isused positive/control counts (#262).
    n_pos = int(V.get_column("label").sum())
    n_ctrl = V.height - n_pos
    exp = EXPECTED_USED_COUNTS[name]
    assert (n_pos, n_ctrl) == (exp["pos"], exp["ctrl"]), (
        f"{name}: var.isused counts {n_pos} pos / {n_ctrl} ctrl != "
        f"expected {exp['pos']} / {exp['ctrl']} (#262)"
    )
    n_used = V.height

    if spec.build == "hg19":
        V = V.pipe(lift_hg19_to_hg38).filter(pl.col("pos") != -1)
        lift_cov = V.height / n_used
        assert lift_cov >= 0.999, f"{name}: lift coverage {lift_cov:.4f} < 0.999"
    # Liftover can map to alt/patch contigs absent from the primary assembly;
    # restrict to canonical {1..22,X,Y} before the genome lookup.
    V = V.pipe(filter_chroms)
    n_before_orient = V.height

    # Genome-orient: check_ref_alt swaps ref<->alt where allele1 != reference. Flip
    # effect + every signed score column on the same swap mask so the whole row stays
    # in the alt-vs-ref frame. (Liftover RCs both alleles but preserves ref/alt roles,
    # so only the swap flips signs.)
    V, n_swapped = orient_variants(V, genome)
    orient_cov = V.height / n_before_orient
    assert orient_cov >= 0.999, (
        f"{name}: genome-orient coverage {orient_cov:.4f} < 0.999"
    )
    flipped = [c for c in SIGNED_SCORE_COLS if c in V.columns]

    print(
        f"[chrombpnet_qtl {name}] used={n_used} oriented={V.height} "
        f"(orient_cov={orient_cov:.4f}) ref_alt_swapped={n_swapped} "
        f"signed_cols_flipped={flipped}"
    )

    # Final invariants.
    assert V["label"].dtype == pl.Boolean, f"{name}: label is not Boolean"
    assert (V["pos"] > 0).all(), f"{name}: non-positive positions after orient"
    eff_pos = V.filter(pl.col("label")).get_column("effect")
    n_nan_pos = int((eff_pos.is_null() | eff_pos.is_nan()).sum())
    assert n_nan_pos == 0, f"{name}: {n_nan_pos} positives with NaN effect after orient"
    final_pos = int(V.get_column("label").sum())
    assert final_pos >= 2, f"{name}: only {final_pos} positives survived"
    return V.sort(["chrom", "pos", "ref", "alt"])
