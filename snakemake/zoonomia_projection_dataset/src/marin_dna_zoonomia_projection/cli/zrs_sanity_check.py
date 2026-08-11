"""ZRS sanity check for the smoke-tier cross-mammal projection.

Reads the per-species Mus_musculus Parquet from the smoke run and asserts
that the two ZRS cCRE queries lift to mm10 chr5 with substantial overlap
to the published mouse cCRE coordinates. Failing this check fails the
smoke job and blocks the full launch.

The ZRS (Zone of Polarizing Activity Regulatory Sequence) is the
canonical limb-development enhancer in the LMBR1 intron, with deep
human/mouse synteny. The cCRE pairs come from the SCREEN Registry V4
GRCh38-cCREs and mm10-cCREs BEDs, used previously in issue #120 for
hg38↔mm10 cCRE ortholog recovery. The Mus_musculus assembly inside the
447-mammalian HAL is GCF_000001635.26 (GRCm38.p6 = mm10), so these mm10
expectations are build-aligned.

| query (hg38)            | expected mm10 ortholog              |
|-------------------------|-------------------------------------|
| zrs_EH38E2604086        | chr5:29315086-29315432 (cCRE        |
| (chr7:156791361-156791613) |   EM10E1584494)                  |
| zrs_EH38E2604087        | chr5:29316458-29316801 (cCRE        |
| (chr7:156792617-156792954) |   EM10E1584495)                  |

Each lifted (resized to 255 bp) interval must overlap its expected mm10
cCRE by at least ``--min-overlap`` bp (default 50). Asserts:

- exactly one row exists for each query in the Mus_musculus Parquet,
- ``t_chrom == "chr5"``,
- ``t_strand`` matches the input strand (lift may flip; we don't assert),
- the lifted interval overlaps the expected mm10 cCRE by ≥ min_overlap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

# (query_name, mm10_chrom, mm10_start, mm10_end). 0-based half-open.
ZRS_EXPECTATIONS: list[tuple[str, str, int, int]] = [
    ("zrs_EH38E2604086", "chr5", 29315086, 29315432),
    ("zrs_EH38E2604087", "chr5", 29316458, 29316801),
]


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Length of intersection of [a_start, a_end) and [b_start, b_end)."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mus-parquet",
        type=Path,
        required=True,
        help="Path to results/projection/min{p}/per_species/Mus_musculus.parquet",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the PASS/FAIL report.",
    )
    p.add_argument(
        "--min-overlap",
        type=int,
        default=50,
        help="Minimum required overlap (bp) with the expected mm10 cCRE.",
    )
    args = p.parse_args()

    df = pl.read_parquet(args.mus_parquet)

    lines: list[str] = []
    failed: list[str] = []
    for query, ex_chrom, ex_start, ex_end in ZRS_EXPECTATIONS:
        matched = df.filter(pl.col("query_name") == query)
        if matched.height == 0:
            msg = f"FAIL {query}: no row in {args.mus_parquet}"
            failed.append(msg)
            lines.append(msg)
            continue
        if matched.height != 1:
            msg = (
                f"FAIL {query}: expected exactly 1 row, got {matched.height} "
                f"(query should yield 0 or 1 row per species after filter)"
            )
            failed.append(msg)
            lines.append(msg)
            continue

        row = matched.row(0, named=True)
        t_chrom: str = row["t_chrom"]
        t_start: int = row["t_start"]
        t_end: int = row["t_end"]
        t_strand: str = row["t_strand"]
        if t_chrom != ex_chrom:
            msg = (
                f"FAIL {query}: lifted to {t_chrom} but expected {ex_chrom} "
                f"(at {t_chrom}:{t_start}-{t_end})"
            )
            failed.append(msg)
            lines.append(msg)
            continue
        ov = _overlap(t_start, t_end, ex_start, ex_end)
        if ov < args.min_overlap:
            msg = (
                f"FAIL {query}: only {ov} bp overlap with expected "
                f"{ex_chrom}:{ex_start}-{ex_end} "
                f"(lifted {t_chrom}:{t_start}-{t_end} strand={t_strand}); "
                f"required ≥ {args.min_overlap} bp"
            )
            failed.append(msg)
            lines.append(msg)
            continue
        msg = (
            f"PASS {query}: lifted to {t_chrom}:{t_start}-{t_end} "
            f"strand={t_strand}, {ov} bp overlap with expected "
            f"{ex_chrom}:{ex_start}-{ex_end}"
        )
        lines.append(msg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")

    for line in lines:
        print(line, file=sys.stderr)

    if failed:
        print(
            f"\n{len(failed)} ZRS sanity check(s) FAILED — projection "
            f"output is suspect, blocking the full launch.",
            file=sys.stderr,
        )
        return 1
    print(
        f"\nAll {len(ZRS_EXPECTATIONS)} ZRS sanity checks passed.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
