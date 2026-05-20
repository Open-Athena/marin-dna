"""halLiftover wrapper + output parser.

``halLiftover`` (Cactus binary distribution) projects intervals from one
alignment leaf onto another via a HAL file. We call it once per target
species; output is a 6-column BED ``(chrom, start, end, name, score,
strand)`` where ``name`` carries the source interval id we set in column
4 of the input BED. Multiple output rows per source name = the
projection split across alignment blocks.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import polars as pl


HALLIFTOVER_BED_COLUMNS: list[str] = [
    "t_chrom",
    "t_start",
    "t_end",
    "query_name",
    "score",
    "t_strand",
]

HALLIFTOVER_BED_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "t_chrom": pl.Utf8,
    "t_start": pl.Int64,
    "t_end": pl.Int64,
    "query_name": pl.Utf8,
    "score": pl.Int64,
    "t_strand": pl.Utf8,
}


def run_halliftover(
    hal_path: str | Path,
    src_species: str,
    src_bed: str | Path,
    tgt_species: str,
    out_bed: str | Path,
    *,
    no_dupes: bool = True,
) -> float:
    """Run ``halLiftover [--noDupes] <hal> <src> <bed> <tgt> <out>``.

    Returns wall-clock seconds. ``halLiftover`` is single-threaded per
    call; parallelize by spawning multiple calls (e.g. one per target
    species). Pass ``cmd`` as a ``list[str]`` to ``subprocess.run`` —
    no shell, no ``pipefail`` edge cases.
    """
    cmd: list[str] = ["halLiftover"]
    if no_dupes:
        cmd.append("--noDupes")
    cmd += [
        str(hal_path),
        src_species,
        str(src_bed),
        tgt_species,
        str(out_bed),
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - t0


def parse_halliftover_bed(path: str | Path, species: str) -> pl.DataFrame:
    """Parse a halLiftover output BED into a Polars frame.

    Empty file → empty schema-correct frame (mirrors
    ``marin_dna.pipelines.alignment.mmseqs2.parse_mmseqs2_hits``). Adds a constant
    ``species`` column. ``t_src_size`` is NOT in halLiftover's output and
    must be joined separately from a chrom.sizes table (see
    ``halStats --chromSizes`` and :func:`attach_src_size`).
    """
    path = Path(path)
    if path.stat().st_size == 0:
        return pl.DataFrame(schema={**HALLIFTOVER_BED_SCHEMA, "species": pl.Utf8})
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=HALLIFTOVER_BED_COLUMNS,
        schema_overrides=HALLIFTOVER_BED_SCHEMA,
    )
    return df.with_columns(species=pl.lit(species))


def attach_src_size(records: pl.DataFrame, chrom_sizes_tsv: str | Path) -> pl.DataFrame:
    """Inner-join a ``chrom\\tsize`` TSV onto records as ``t_src_size``.

    The TSV is the output of ``halStats --chromSizes <species> <hal>``;
    chrom names must match the ``t_chrom`` column. Rows whose chrom is
    missing from the TSV are dropped — schema-invariant assertions
    downstream surface that as a loud failure.
    """
    sizes = pl.read_csv(
        chrom_sizes_tsv,
        separator="\t",
        has_header=False,
        new_columns=["t_chrom", "t_src_size"],
        schema_overrides={"t_chrom": pl.Utf8, "t_src_size": pl.Int64},
    )
    return records.join(sizes, on="t_chrom", how="inner")


def write_chrom_sizes(hal_path: str | Path, species: str, out_path: str | Path) -> None:
    """Run ``halStats --chromSizes <species> <hal>`` and write to ``out_path``."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        subprocess.run(
            ["halStats", "--chromSizes", species, str(hal_path)],
            check=True,
            stdout=f,
        )
