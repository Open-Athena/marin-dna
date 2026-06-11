"""Issue #283 — composition of the v4 `ccre_non_promoter` ("distal" enhancer) arm.

Reproduces the statistics in the H1 diagnostic comment on
https://github.com/Open-Athena/marin-dna/issues/283 :

  1. Overlap of the arm's training windows with other functional elements
     (CDS / UTR / ncRNA exon / TSS+5'UTR) — the H1 "contamination" check.
  2. Genomic context (intronic / intergenic / exon-overlap).
  3. cCRE-class mix inside the conserved arm vs the raw registry baseline.

All numbers come from the v4 region-labels parquet (one row per 255 bp human
anchor, with the disjoint priority-resolved `*_frac` columns produced by
`label_windows_bp_majority`) and the cCRE registry parquet. Read-only; ~1 GB
RAM; single process.

    uv run python scripts/issue283/h1_composition.py
"""

import bioframe as bf
import polars as pl

PREFIX = "s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results"
REGION_LABELS = f"{PREFIX}/human/intervals/region_labels/v4/min0.20.parquet"
CRE = f"{PREFIX}/human/intervals/cre/all.parquet"

NONCCRE = ["cds_frac", "utr3_frac", "ncrna_exon_frac", "tss_region_and_utr5_frac"]
ELEM_NAME = {
    "cds_frac": "cds",
    "ncrna_exon_frac": "ncrna_exon",
    "tss_region_and_utr5_frac": "tss_region_and_utr5",
    "utr3_frac": "utr3",
}
ENHANCER_CLASSES = {"dELS", "pELS"}  # mirrors marin_dna.data.utils.ENHANCER_CRE_CLASSES


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def overlap_with_functional_elements(ccre: pl.DataFrame) -> None:
    """(1) H1 — how much of the arm overlaps other functional elements."""
    n = len(ccre)
    ccre = ccre.with_columns(sum(pl.col(c) for c in NONCCRE).alias("nonccre"))
    ov = ccre.filter(pl.col("nonccre") > 0)
    n_ov = len(ov)

    section("(1) Overlap with other functional elements (H1)")
    print(f"arm windows: {n}")
    print(f"overlap another functional element: {n_ov} ({n_ov / n:.1%})")
    print(f"pure cCRE (no overlap):             {n - n_ov} ({(n - n_ov) / n:.1%})")
    print(
        f"\nwhole-arm: windows touching CDS (cds_frac>0): {(ccre['cds_frac'] > 0).mean():.1%}"
    )
    print(
        f"whole-arm: non-cCRE share of functional bp:   "
        f"{ov['nonccre'].sum() / ccre['functional_frac'].sum():.1%}"
    )
    has_cds = ov.filter(pl.col("cds_frac") > 0)
    print(
        f"mean CDS coverage when a window clips CDS:    {has_cds['cds_frac'].mean():.1%}"
    )

    print("\nper-element breakdown among the overlapping windows:")
    tot_bp = ov["nonccre"].sum()
    print(f"  {'element':22s} {'% overlappers':>14s} {'bp share':>10s}")
    for c in NONCCRE:
        frac_win = (ov[c] > 0).mean()
        bp_share = ov[c].sum() / tot_bp
        print(f"  {ELEM_NAME[c]:22s} {frac_win:>14.1%} {bp_share:>10.1%}")


def genomic_context(ccre: pl.DataFrame) -> None:
    """(2) intronic / intergenic / exon-overlap split."""
    n = len(ccre)
    ccre = ccre.with_columns(
        pl.when(sum(pl.col(c) for c in NONCCRE) > 0)
        .then(pl.lit("exon_overlap"))
        .when(pl.col("intron_frac") >= pl.col("intergenic_frac"))
        .then(pl.lit("intronic"))
        .otherwise(pl.lit("intergenic"))
        .alias("context")
    )
    section("(2) Genomic context")
    tab = ccre.group_by("context").agg(pl.len().alias("n")).sort("n", descending=True)
    for r in tab.iter_rows(named=True):
        print(f"  {r['context']:14s} {r['n']:>8d}  {r['n'] / n:>6.1%}")


def cre_class_mix(ccre: pl.DataFrame) -> None:
    """(3) cCRE-class mix inside the arm vs the raw registry baseline."""
    cre = (
        pl.read_parquet(CRE)
        .filter(pl.col("cre_class") != "PLS")
        .select(["chrom", "start", "end", "cre_class"])
    )

    section("(3) cCRE-class mix")
    # Registry baseline: enhancer-like share of all non-PLS cCRE bp.
    reg = cre.with_columns((pl.col("end") - pl.col("start")).alias("bp"))
    reg_enh = reg.filter(pl.col("cre_class").is_in(list(ENHANCER_CLASSES)))["bp"].sum()
    print(
        f"registry baseline (non-PLS): enhancer dELS+pELS = {reg_enh / reg['bp'].sum():.1%} of cCRE bp"
    )

    # Arm: intersect each window with the per-class cCRE elements.
    win = ccre.select(["chrom", "start", "end"]).with_row_index("wid").to_pandas()
    win["chrom"] = win["chrom"].astype(str)
    crep = cre.to_pandas()
    crep["chrom"] = crep["chrom"].astype(str)
    ov = bf.overlap(win, crep, how="inner", return_overlap=True, suffixes=("", "_cre"))
    ccol = [c for c in ov.columns if "class" in c][0]
    ov["bp"] = ov["overlap_end"] - ov["overlap_start"]
    assert (ov["bp"] > 0).all()

    g = pl.from_pandas(
        ov.groupby(["wid", ccol], observed=True)["bp"].sum().reset_index()
    )
    n = len(win)
    per_class = (
        g.group_by(ccol)
        .agg(pl.col("wid").n_unique().alias("n_win"), pl.col("bp").sum().alias("bp"))
        .sort("bp", descending=True)
    )
    tot_bp = per_class["bp"].sum()
    enh_bp = per_class.filter(pl.col(ccol).is_in(list(ENHANCER_CLASSES)))["bp"].sum()

    # Per-window dominant class.
    idx = ov.groupby("wid", observed=True)["bp"].idxmax()
    dom = pl.from_pandas(ov.loc[idx])
    dom_enh = dom.filter(pl.col(ccol).is_in(list(ENHANCER_CLASSES))).height

    print(
        f"arm:              enhancer dELS+pELS = {enh_bp / tot_bp:.1%} of cCRE bp "
        f"({dom_enh / n:.1%} of windows by dominant class)"
    )
    print(f"\n  {'cre_class':14s} {'% windows':>10s} {'bp share':>10s}")
    for r in per_class.iter_rows(named=True):
        print(f"  {r[ccol]:14s} {r['n_win'] / n:>10.1%} {r['bp'] / tot_bp:>10.1%}")


def main() -> None:
    df = pl.read_parquet(REGION_LABELS)
    ccre = df.filter(pl.col("label") == "ccre_non_promoter")
    assert len(ccre) > 0, "no ccre_non_promoter windows"
    overlap_with_functional_elements(ccre)
    genomic_context(ccre)
    cre_class_mix(ccre)


if __name__ == "__main__":
    main()
