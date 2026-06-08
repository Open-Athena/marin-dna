"""Stage 2 of issue #296 — annotate the per-token cache, measure stratified LL gaps.

Reads a per-token loss cache (``per_token_loss.compute_hf_per_token_loss`` output
for a ``val_cds`` run), annotates every base from the Ensembl r115 GTF, and
computes the conserved/non-conserved **LL gap within each of an additive (not
necessarily disjoint) set of strata** — the headline being ``LLgap | {codon 1,2}``
vs the vanilla all-token gap.

Strata: ``all_token`` (= vanilla LL gap), ``coding``, ``codon_1/2/3``, ``codon_12``,
``codon_3_4fold`` / ``codon_3_not4fold`` (4-fold-degenerate wobble vs constrained),
``splicing`` (intron ≤ 20 bp from a junction) with ``splice_donor`` / ``splice_acceptor``,
``utr5``, ``utr3``, and ``other_noncoding``.

Correctness gate (model-free): the 3rd (wobble) codon position must be the least
phyloP-conserved — breaks if codon assignment is shifted, independent of model.

CPU-only, no model. One-off (issue #296).

    uv run python scripts/issue296/annotate_and_measure.py \
        --cache scratch/issue296/per_token/<model>/val_cds.parquet \
        --model-name <model> --out-dir scratch/issue296
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import bioframe as bf
import polars as pl

from marin_dna.data.utils import load_annotation
from marin_dna.pipelines.evals.per_token_annotate import (
    cds_codon_positions,
    flag_fourfold_degenerate,
    intron_splice_regions,
)
from marin_dna.pipelines.zoonomia_projection_dataset.validation import (
    filter_to_canonical_transcripts,
    get_ensembl_3_prime_utr,
    get_ensembl_5_prime_utr,
)

DEFAULT_GTF_S3 = (
    "s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results/annotation/"
    "Homo_sapiens.GRCh38.115.gtf.gz"
)
SPLICE_FLANK = 20  # bp into the intron from a junction (matches val_cds add_flank)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, help="Per-token loss parquet (val_cds).")
    p.add_argument("--model-name", required=True)
    p.add_argument(
        "--gtf", default=DEFAULT_GTF_S3, help="Ensembl r115 GTF (s3:// or local)."
    )
    p.add_argument("--out-dir", default="scratch/issue296")
    return p.parse_args()


def _local_gtf(gtf: str, out_dir: Path) -> str:
    if not gtf.startswith("s3://"):
        return gtf
    dest = out_dir / "annotation" / Path(gtf).name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[stage2] downloading GTF → {dest}")
        subprocess.run(["aws", "s3", "cp", "--no-progress", gtf, str(dest)], check=True)
    return str(dest)


def _parse_window_ids(cache: pl.DataFrame) -> pl.DataFrame:
    """``window_id`` = ``chrom:start-end`` → add ``chrom`` + ``genomic_pos``."""
    return cache.with_columns(
        chrom=pl.col("window_id").str.split(":").list.first(),
        _win_start=pl.col("window_id")
        .str.split(":")
        .list.last()
        .str.split("-")
        .list.first()
        .cast(pl.Int64),
    ).with_columns(genomic_pos=pl.col("_win_start") + pl.col("pos_in_window"))


def _canonical(gtf_local: str) -> pl.DataFrame:
    return filter_to_canonical_transcripts(load_annotation(gtf_local))


def _feature(canon: pl.DataFrame, feature: str, cols: list[str]) -> pl.DataFrame:
    tid = (
        pl.col("attribute").str.extract(r'transcript_id "(.*?)"').alias("transcript_id")
    )
    return (
        canon.filter(pl.col("feature") == feature)
        .with_columns(tid)
        .filter(pl.col("transcript_id").is_not_null())
        .select(cols)
    )


def _transcripts_over_windows(cds: pl.DataFrame, windows: pl.DataFrame) -> list[str]:
    cds_pd = cds.select(["chrom", "start", "end", "transcript_id"]).to_pandas()
    win_pd = windows.select(["chrom", "start", "end"]).unique().to_pandas()
    ov = bf.overlap(cds_pd, win_pd, how="inner", suffixes=("", "_w"))
    return list(set(ov["transcript_id"].unique()))


def _resolve_unique(per_pos: pl.DataFrame, col: str) -> pl.DataFrame:
    """One value of ``col`` per (chrom, genomic_pos); cross-transcript conflicts
    → null (kept out of that stratum)."""
    return (
        per_pos.group_by(["chrom", "genomic_pos"])
        .agg(pl.col(col).unique().alias("_u"))
        .with_columns(
            **{
                col: pl.when(pl.col("_u").list.len() == 1)
                .then(pl.col("_u").list.first())
                .otherwise(None)
            }
        )
        .select(["chrom", "genomic_pos", col])
    )


def _overlap_flag(
    positions: pl.DataFrame, regions: pl.DataFrame, col: str
) -> pl.DataFrame:
    """Add boolean ``col`` to ``positions`` ([chrom, genomic_pos]) — True iff the
    position falls in any of ``regions`` ([chrom, start, end])."""
    if len(regions) == 0:
        return positions.with_columns(**{col: pl.lit(False)})
    pos_pd = (
        positions.with_columns(
            start=pl.col("genomic_pos"), end=pl.col("genomic_pos") + 1
        )
        .select(["chrom", "start", "end", "genomic_pos"])
        .to_pandas()
    )
    ov = bf.overlap(
        pos_pd,
        regions.select(["chrom", "start", "end"]).to_pandas(),
        how="inner",
        suffixes=("", "_r"),
    )
    hits = (
        pl.from_pandas(ov[["chrom", "genomic_pos"]])
        .unique()
        .with_columns(**{col: pl.lit(True)})
    )
    return positions.join(hits, on=["chrom", "genomic_pos"], how="left").with_columns(
        pl.col(col).fill_null(False)
    )


def _overlap_value(
    positions: pl.DataFrame, regions: pl.DataFrame, value_col: str, out_col: str
) -> pl.DataFrame:
    """Add ``out_col`` to ``positions`` = ``regions[value_col]`` of an overlapping
    region (first match; positions in no region → null)."""
    if len(regions) == 0:
        return positions.with_columns(**{out_col: pl.lit(None, dtype=pl.Utf8)})
    pos_pd = (
        positions.with_columns(
            start=pl.col("genomic_pos"), end=pl.col("genomic_pos") + 1
        )
        .select(["chrom", "start", "end", "genomic_pos"])
        .to_pandas()
    )
    reg = regions.select(["chrom", "start", "end", value_col]).rename(
        {value_col: out_col}
    )
    ov = bf.overlap(pos_pd, reg.to_pandas(), how="inner", suffixes=("", "_r"))
    # bioframe suffixes the second frame's columns; the value lands as out_col or
    # out_col + "_r" depending on version — accept either.
    val_col = out_col if out_col in ov.columns else f"{out_col}_r"
    assert val_col in ov.columns, (out_col, list(ov.columns))
    hits = (
        pl.from_pandas(ov[["chrom", "genomic_pos", val_col]])
        .rename({val_col: out_col})
        .unique(subset=["chrom", "genomic_pos"], keep="first")
    )
    return positions.join(hits, on=["chrom", "genomic_pos"], how="left")


def _stratum_stats(df: pl.DataFrame) -> dict[str, float]:
    """Per-stratum stats: overall ``mean_loss`` (all tokens) **and** the
    conserved/non-conserved LL gap. ``mean_loss`` = mean ``−log p`` over every
    token in the stratum (how well the model predicts the region); ``gap`` =
    ``LL_upper − LL_lower`` (the conservation delta)."""
    up = df.filter(pl.col("is_upper"))
    lo = df.filter(~pl.col("is_upper"))
    n_u, n_l = len(up), len(lo)
    n = n_u + n_l
    ll_u = -up["loss"].mean() if n_u else float("nan")
    ll_l = -lo["loss"].mean() if n_l else float("nan")
    return {
        "n": n,
        "n_upper": n_u,
        "n_lower": n_l,
        "mean_loss": float(df["loss"].mean()) if n else float("nan"),
        "LL_upper": ll_u,
        "LL_lower": ll_l,
        "gap": (ll_u - ll_l) if (n_u and n_l) else float("nan"),
    }


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)

    cache = _parse_window_ids(pl.read_parquet(args.cache))
    print(f"[stage2] cache: {len(cache):,} token rows for {args.model_name}")
    windows = cache.select(["chrom", pl.col("_win_start").alias("start")]).with_columns(
        end=pl.col("start") + 255
    )
    ref = cache.select(["chrom", "genomic_pos", "ref_base"]).unique()

    canon = _canonical(_local_gtf(args.gtf, out_dir))
    cds_all = _feature(
        canon, "CDS", ["chrom", "start", "end", "strand", "transcript_id", "frame"]
    )
    exon_all = _feature(
        canon, "exon", ["chrom", "start", "end", "strand", "transcript_id"]
    )
    keep = _transcripts_over_windows(cds_all, windows)
    cds = cds_all.filter(pl.col("transcript_id").is_in(keep))
    exons = exon_all.filter(pl.col("transcript_id").is_in(keep))
    print(
        f"[stage2] {len(keep)} transcripts ({len(cds)} CDS, {len(exons)} exon segments)"
    )

    # Codon position, 4-fold degeneracy, and the CDS gene strand (per coding base).
    per_base = cds_codon_positions(cds)  # [.. strand, codon_pos, codon_id]
    codon = _resolve_unique(per_base, "codon_pos")
    coding_strand = _resolve_unique(per_base, "strand").rename(
        {"strand": "coding_strand"}
    )
    coding_ref = per_base.join(ref, on=["chrom", "genomic_pos"], how="left")
    fourfold = _resolve_unique(flag_fourfold_degenerate(coding_ref), "is_4fold")

    ann = (
        cache.join(codon, on=["chrom", "genomic_pos"], how="left")
        .join(fourfold, on=["chrom", "genomic_pos"], how="left")
        .join(coding_strand, on=["chrom", "genomic_pos"], how="left")
    )
    nc = (
        ann.filter(pl.col("codon_pos").is_null())
        .select(["chrom", "genomic_pos"])
        .unique()
    )

    # `splicing` = the broad intronic window (≤20bp of a junction). donor/acceptor
    # = the 2-position canonical dinucleotides (GT / AG) — intron_splice_regions
    # with flank=2 gives exactly the 2 bases abutting the junction on each side.
    splice_broad = intron_splice_regions(exons, flank=SPLICE_FLANK)
    splice_canon = intron_splice_regions(exons, flank=2)
    nc = _overlap_flag(nc, splice_broad, "is_splice")  # broad: intron ≤20bp
    nc = _overlap_flag(nc, splice_canon.filter(pl.col("side") == "donor"), "is_donor")
    nc = _overlap_flag(
        nc, splice_canon.filter(pl.col("side") == "acceptor"), "is_acceptor"
    )
    nc = _overlap_value(nc, splice_broad, "strand", "splice_strand")  # gene strand
    nc = _overlap_flag(nc, get_ensembl_5_prime_utr(canon).to_polars(), "is_utr5")
    nc = _overlap_flag(nc, get_ensembl_3_prime_utr(canon).to_polars(), "is_utr3")

    ann = ann.join(nc, on=["chrom", "genomic_pos"], how="left").with_columns(
        *[
            pl.col(c).fill_null(False)
            for c in ["is_splice", "is_donor", "is_acceptor", "is_utr5", "is_utr3"]
        ]
    )
    # Gene strand at this position: CDS strand if coding, else the intron's strand.
    # The model reads the FORWARD strand, so on a + gene a splice donor is reached
    # CDS→donor (CDS-primed) while on a − gene the same donor is intron→donor.
    ann = ann.with_columns(
        gene_strand=pl.coalesce(pl.col("coding_strand"), pl.col("splice_strand"))
    )

    # --- Gate: model-free conservation-by-codon signature -------------------
    by_codon = (
        ann.filter(pl.col("codon_pos").is_not_null())
        .group_by("codon_pos")
        .agg(
            n=pl.len(),
            frac_conserved=pl.col("is_upper").mean(),
            mean_loss=pl.col("loss").mean(),
        )
        .sort("codon_pos")
    )
    print("\n[stage2] per-codon-position (CDS bases):")
    print(by_codon)
    fc = {r["codon_pos"]: r["frac_conserved"] for r in by_codon.iter_rows(named=True)}
    assert fc[3] < fc[1] and fc[3] < fc[2], f"codon gate FAILED: {fc}"
    print(
        f"[stage2] GATE OK — wobble least conserved (pos3={fc[3]:.3f} < "
        f"pos1={fc[1]:.3f}, pos2={fc[2]:.3f})"
    )

    # --- Additive strata (overlap allowed) ----------------------------------
    cp = pl.col("codon_pos")
    strata = {
        "all_token": pl.lit(True),
        "coding": cp.is_not_null(),
        "codon_1": cp == 1,
        "codon_2": cp == 2,
        "codon_3": cp == 3,
        "codon_12": cp.is_in([1, 2]),
        "codon_3_4fold": (cp == 3) & (pl.col("is_4fold") == True),  # noqa: E712
        "codon_3_not4fold": (cp == 3) & (pl.col("is_4fold") == False),  # noqa: E712
        "splicing": pl.col("is_splice"),
        "splice_donor": pl.col("is_donor"),
        "splice_acceptor": pl.col("is_acceptor"),
        "utr5": pl.col("is_utr5"),
        "utr3": pl.col("is_utr3"),
        "other_noncoding": cp.is_null()
        & ~pl.col("is_splice")
        & ~pl.col("is_utr5")
        & ~pl.col("is_utr3"),
        # Split by gene strand: the FWD-reading model approaches a + gene's donor
        # CDS-primed (CDS→donor) but a − gene's donor intron-primed (intron→donor)
        # — and symmetrically for acceptors. codon_12 by strand checks sense vs
        # antisense coding readout.
        "splice_donor_plus": pl.col("is_donor") & (pl.col("gene_strand") == "+"),
        "splice_donor_minus": pl.col("is_donor") & (pl.col("gene_strand") == "-"),
        "splice_acceptor_plus": pl.col("is_acceptor") & (pl.col("gene_strand") == "+"),
        "splice_acceptor_minus": pl.col("is_acceptor") & (pl.col("gene_strand") == "-"),
        "codon_12_plus": cp.is_in([1, 2]) & (pl.col("gene_strand") == "+"),
        "codon_12_minus": cp.is_in([1, 2]) & (pl.col("gene_strand") == "-"),
    }
    rows = [
        {"model": args.model_name, "stratum": name, **_stratum_stats(ann.filter(mask))}
        for name, mask in strata.items()
    ]
    summary = pl.DataFrame(rows)
    with pl.Config(tbl_rows=-1):
        print(
            "\n[stage2] stratified LL gaps "
            "(gap = LL_upper − LL_lower, > 0 ⇒ conserved easier):"
        )
        print(summary)

    # Per-(region, gene strand) breakdown for the strand figure — every region
    # that carries a gene strand (coding + splice; all_token / deep intron have
    # no strand and are omitted).
    strand_regions = {
        "codon_1": cp == 1,
        "codon_2": cp == 2,
        "codon_3": cp == 3,
        "codon_3_4fold": (cp == 3) & (pl.col("is_4fold") == True),  # noqa: E712
        "splicing": pl.col("is_splice"),
        "splice_donor": pl.col("is_donor"),
        "splice_acceptor": pl.col("is_acceptor"),
    }
    strand_rows = [
        {
            "model": args.model_name,
            "stratum": name,
            "gene_strand": strand,
            **_stratum_stats(ann.filter(mask & (pl.col("gene_strand") == strand))),
        }
        for name, mask in strand_regions.items()
        for strand in ("+", "-")
    ]
    by_strand = pl.DataFrame(strand_rows)

    out = out_dir / "stage2" / args.model_name
    out.mkdir(parents=True, exist_ok=True)
    by_codon.write_parquet(out / "val_cds_by_codon.parquet")
    summary.write_parquet(out / "val_cds_stratum_ll_gap.parquet")
    by_strand.write_parquet(out / "val_cds_by_strand.parquet")
    print(f"\n[stage2] wrote → {out}")


if __name__ == "__main__":
    main()
