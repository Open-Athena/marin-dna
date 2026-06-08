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


def _ll_gap(df: pl.DataFrame) -> dict[str, float]:
    up = df.filter(pl.col("is_upper"))
    lo = df.filter(~pl.col("is_upper"))
    n_u, n_l = len(up), len(lo)
    ll_u = -up["loss"].mean() if n_u else float("nan")
    ll_l = -lo["loss"].mean() if n_l else float("nan")
    return {
        "n_upper": n_u,
        "n_lower": n_l,
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

    # Codon position + 4-fold degeneracy (per coding base).
    per_base = cds_codon_positions(cds)  # [.. codon_pos, codon_id]
    codon = _resolve_unique(per_base, "codon_pos")
    coding_ref = per_base.join(ref, on=["chrom", "genomic_pos"], how="left")
    fourfold = _resolve_unique(flag_fourfold_degenerate(coding_ref), "is_4fold")

    # Non-coding region flags (splice donor/acceptor, UTRs) on the non-coding set.
    ann = cache.join(codon, on=["chrom", "genomic_pos"], how="left").join(
        fourfold, on=["chrom", "genomic_pos"], how="left"
    )
    nc = (
        ann.filter(pl.col("codon_pos").is_null())
        .select(["chrom", "genomic_pos"])
        .unique()
    )

    splice = intron_splice_regions(exons, flank=SPLICE_FLANK)
    nc = _overlap_flag(nc, splice, "is_splice")
    nc = _overlap_flag(nc, splice.filter(pl.col("side") == "donor"), "is_donor")
    nc = _overlap_flag(nc, splice.filter(pl.col("side") == "acceptor"), "is_acceptor")
    nc = _overlap_flag(nc, get_ensembl_5_prime_utr(canon).to_polars(), "is_utr5")
    nc = _overlap_flag(nc, get_ensembl_3_prime_utr(canon).to_polars(), "is_utr3")

    ann = ann.join(nc, on=["chrom", "genomic_pos"], how="left").with_columns(
        *[
            pl.col(c).fill_null(False)
            for c in ["is_splice", "is_donor", "is_acceptor", "is_utr5", "is_utr3"]
        ]
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
    }
    rows = [
        {"model": args.model_name, "stratum": name, **_ll_gap(ann.filter(mask))}
        for name, mask in strata.items()
    ]
    summary = pl.DataFrame(rows)
    with pl.Config(tbl_rows=-1):
        print(
            "\n[stage2] stratified LL gaps "
            "(gap = LL_upper − LL_lower, > 0 ⇒ conserved easier):"
        )
        print(summary)

    out = out_dir / "stage2" / args.model_name
    out.mkdir(parents=True, exist_ok=True)
    by_codon.write_parquet(out / "val_cds_by_codon.parquet")
    summary.write_parquet(out / "val_cds_stratum_ll_gap.parquet")
    print(f"\n[stage2] wrote → {out}")


if __name__ == "__main__":
    main()
