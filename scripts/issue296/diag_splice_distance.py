"""Diagnostic (issue #296): per-token loss vs distance-into-intron at splice sites.

Checks whether the splice strata's high mean loss is dilution (the ≤20 bp window
is mostly ordinary intronic flank, with only the 2 bp canonical donor `GT` /
acceptor `AG` being invariant) or a bug. For each cache position that lands in an
intron of a canonical transcript, compute its distance to the nearest exon
boundary (0 = the base abutting the exon = the canonical G of GT/AG) and report
mean loss + conservation + the modal reference base by distance.

    uv run python scripts/issue296/diag_splice_distance.py \
        --cache scratch/issue296/per_token/<model>/val_cds.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import bioframe as bf
import polars as pl

from marin_dna.data.utils import load_annotation
from marin_dna.pipelines.zoonomia_projection_dataset.validation import (
    filter_to_canonical_transcripts,
)

GTF = "scratch/issue296/annotation/Homo_sapiens.GRCh38.115.gtf.gz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gtf", default=GTF)
    ap.add_argument("--max-dist", type=int, default=20)
    args = ap.parse_args()

    cache = pl.read_parquet(args.cache).with_columns(
        chrom=pl.col("window_id").str.split(":").list.first(),
        _ws=pl.col("window_id").str.split(":").list.last().str.split("-").list.first().cast(pl.Int64),
    ).with_columns(genomic_pos=pl.col("_ws") + pl.col("pos_in_window"))

    canon = filter_to_canonical_transcripts(load_annotation(args.gtf))
    exons = (
        canon.filter(pl.col("feature") == "exon")
        .with_columns(
            pl.col("attribute").str.extract(r'transcript_id "(.*?)"').alias("tid")
        )
        .filter(pl.col("tid").is_not_null())
        .select(["chrom", "start", "end", "strand", "tid"])
    )
    # introns = gaps between consecutive exons of a transcript
    introns = (
        exons.sort(["tid", "start"])
        .with_columns(_ns=pl.col("start").shift(-1).over("tid"))
        .filter(pl.col("_ns").is_not_null() & (pl.col("_ns") > pl.col("end")))
        .select(pl.col("chrom"), pl.col("end").alias("istart"), pl.col("_ns").alias("iend"))
    )
    # restrict introns to those overlapping the windows (keep it small)
    win = cache.select(["chrom", pl.col("_ws").alias("start")]).with_columns(end=pl.col("start") + 255).unique()
    iv = introns.rename({"istart": "start", "iend": "end"})
    keep = bf.overlap(iv.to_pandas(), win.select(["chrom", "start", "end"]).to_pandas(), how="inner", suffixes=("", "_w"))
    introns = pl.from_pandas(keep[["chrom", "start", "end"]]).unique().rename({"start": "istart", "end": "iend"})

    # point-in-intron join: cache positions falling inside an intron
    pos_pd = cache.select(["chrom", "genomic_pos"]).unique().with_columns(
        start=pl.col("genomic_pos"), end=pl.col("genomic_pos") + 1
    ).select(["chrom", "start", "end", "genomic_pos"]).to_pandas()
    ov = bf.overlap(
        pos_pd, introns.rename({"istart": "start", "iend": "end"}).to_pandas(),
        how="inner", suffixes=("", "_i"),
    )
    j = pl.from_pandas(ov)
    icols = {c for c in j.columns if c.endswith("_i")}
    istart_c = "start_i" if "start_i" in icols else "istart"
    iend_c = "end_i" if "end_i" in icols else "iend"
    j = j.select(
        "chrom", "genomic_pos",
        pl.col(istart_c).alias("istart"), pl.col(iend_c).alias("iend"),
    ).with_columns(
        dist=pl.min_horizontal(
            pl.col("genomic_pos") - pl.col("istart"),
            pl.col("iend") - 1 - pl.col("genomic_pos"),
        )
    ).filter(pl.col("dist") < args.max_dist)
    # one distance per position (nearest intron edge across overlapping introns)
    j = j.group_by(["chrom", "genomic_pos"]).agg(pl.col("dist").min())

    df = cache.join(j, on=["chrom", "genomic_pos"], how="inner")
    print(f"[diag] {len(df):,} intronic cache positions within {args.max_dist}bp of an exon "
          f"({Path(args.cache).parent.name})")
    tab = (
        df.group_by("dist")
        .agg(
            n=pl.len(),
            mean_loss=pl.col("loss").mean(),
            frac_conserved=pl.col("is_upper").mean(),
            modal_base=pl.col("ref_base").mode().first(),
        )
        .sort("dist")
    )
    with pl.Config(tbl_rows=-1):
        print(tab)


if __name__ == "__main__":
    main()
