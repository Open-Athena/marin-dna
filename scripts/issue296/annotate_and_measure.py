"""Stage 2 of issue #296 — annotate the per-token cache, measure stratified LL gaps.

Reads a per-token loss cache (the ``per_token_loss.compute_hf_per_token_loss``
output for a ``val_cds`` run), attaches codon position from the Ensembl r115 GTF,
and computes the conserved/non-conserved **LL gap restricted to strata** — the
headline being ``LLgap | {codon pos 1,2}`` vs the vanilla all-token gap.

Two correctness gates run here:
  * the GTF-``frame`` cross-check inside ``cds_codon_positions`` (off-by-one), and
  * a **model-free** conservation-by-codon signature — the 3rd (wobble) position
    must be the least phyloP-conserved. If codon assignment were shifted, this
    breaks regardless of the model.

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
from marin_dna.pipelines.evals.per_token_annotate import cds_codon_positions
from marin_dna.pipelines.zoonomia_projection_dataset.validation import (
    filter_to_canonical_transcripts,
)

DEFAULT_GTF_S3 = (
    "s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results/annotation/"
    "Homo_sapiens.GRCh38.115.gtf.gz"
)


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


def _canonical_cds(gtf_local: str) -> pl.DataFrame:
    """Canonical-transcript CDS segments ``[chrom,start,end,strand,transcript_id,frame]``."""
    ann = load_annotation(gtf_local)
    canon = filter_to_canonical_transcripts(ann)
    cds = (
        canon.filter(pl.col("feature") == "CDS")
        .with_columns(
            pl.col("attribute")
            .str.extract(r'transcript_id "(.*?)"')
            .alias("transcript_id")
        )
        .filter(pl.col("transcript_id").is_not_null())
        .select(["chrom", "start", "end", "strand", "transcript_id", "frame"])
    )
    assert len(cds) > 0, "no canonical CDS segments extracted"
    return cds


def _cds_for_windows(cds: pl.DataFrame, windows: pl.DataFrame) -> pl.DataFrame:
    """Keep **whole transcripts** whose CDS overlaps any window.

    Filtering by *segment* would corrupt the codon numbering — ``offset`` is
    cumulative over a transcript's CDS, so a dropped upstream segment shifts the
    frame. So we find transcript_ids overlapping the windows (bioframe) and keep
    all their segments.
    """
    cds_pd = cds.select(["chrom", "start", "end", "transcript_id"]).to_pandas()
    win_pd = windows.select(["chrom", "start", "end"]).unique().to_pandas()
    ov = bf.overlap(cds_pd, win_pd, how="inner", suffixes=("", "_w"))
    keep = set(ov["transcript_id"].unique())
    out = cds.filter(pl.col("transcript_id").is_in(list(keep)))
    print(
        f"[stage2] {len(keep)} canonical transcripts overlap the windows "
        f"({len(out)} CDS segments)"
    )
    return out


def _resolve_codon_per_pos(per_base: pl.DataFrame) -> pl.DataFrame:
    """One codon_pos per (chrom, genomic_pos); conflicts across overlapping
    transcripts → null (dropped from codon strata)."""
    return (
        per_base.group_by(["chrom", "genomic_pos"])
        .agg(pl.col("codon_pos").unique().alias("_cps"))
        .with_columns(
            codon_pos=pl.when(pl.col("_cps").list.len() == 1)
            .then(pl.col("_cps").list.first())
            .otherwise(None)
        )
        .select(["chrom", "genomic_pos", "codon_pos"])
    )


def _ll_gap(df: pl.DataFrame) -> dict[str, float]:
    """Conserved − non-conserved mean log p (= −loss) over ``df``."""
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

    cds = _cds_for_windows(_canonical_cds(_local_gtf(args.gtf, out_dir)), windows)
    per_base = cds_codon_positions(cds)  # runs the GTF-frame cross-check
    codon = _resolve_codon_per_pos(per_base)

    ann = cache.join(codon, on=["chrom", "genomic_pos"], how="left")

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
    assert fc[3] < fc[1] and fc[3] < fc[2], (
        f"codon off-by-one gate FAILED: wobble (pos3) is not the least conserved: {fc}"
    )
    print(
        f"[stage2] GATE OK — wobble least conserved (pos3={fc[3]:.3f} < "
        f"pos1={fc[1]:.3f}, pos2={fc[2]:.3f})"
    )

    # --- Stratified LL gaps -------------------------------------------------
    strata = {
        "all_token": pl.lit(True),
        "coding": pl.col("codon_pos").is_not_null(),
        "codon_12": pl.col("codon_pos").is_in([1, 2]),
        "codon_3": pl.col("codon_pos") == 3,
        "non_coding": pl.col("codon_pos").is_null(),
    }
    rows = []
    for name, mask in strata.items():
        rows.append(
            {"model": args.model_name, "stratum": name, **_ll_gap(ann.filter(mask))}
        )
    summary = pl.DataFrame(rows)
    print(
        "\n[stage2] stratified LL gaps (gap = LL_upper − LL_lower, > 0 ⇒ conserved easier):"
    )
    print(summary)

    out = out_dir / "stage2" / args.model_name
    out.mkdir(parents=True, exist_ok=True)
    by_codon.write_parquet(out / "val_cds_by_codon.parquet")
    summary.write_parquet(out / "val_cds_stratum_ll_gap.parquet")
    print(f"\n[stage2] wrote → {out}")


if __name__ == "__main__":
    main()
