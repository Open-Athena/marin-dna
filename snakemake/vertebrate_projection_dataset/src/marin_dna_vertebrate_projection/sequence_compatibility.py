"""Checks that sequence archives match the alignment coordinate source."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def validate_projected_twobit_sizes(
    accepted_path: str | Path,
    chrom_sizes_path: str | Path,
    output_path: str | Path,
) -> None:
    """Require every MAF target chromosome size to equal its 2bit size."""
    projected = (
        pl.scan_parquet(accepted_path)
        .select("t_chrom", "t_src_size")
        .unique()
        .collect(engine="streaming")
    )
    assert projected.group_by("t_chrom").len()["len"].max() in {None, 1}, (
        "alignment reports multiple source sizes for one target chromosome"
    )
    sizes = pl.read_csv(
        chrom_sizes_path,
        separator="\t",
        has_header=False,
        new_columns=["t_chrom", "twobit_src_size"],
        schema_overrides={"t_chrom": pl.String, "twobit_src_size": pl.Int64},
    )
    assert sizes["t_chrom"].n_unique() == sizes.height
    compared = projected.join(sizes, on="t_chrom", how="left", validate="m:1")
    missing = compared.filter(pl.col("twobit_src_size").is_null())
    assert missing.is_empty(), (
        f"2bit is missing projected chromosomes: {missing['t_chrom'].to_list()}"
    )
    mismatched = compared.filter(pl.col("t_src_size") != pl.col("twobit_src_size"))
    assert mismatched.is_empty(), (
        "2bit chromosome sizes disagree with MAF source sizes: "
        f"{mismatched.to_dicts()[:10]}"
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "accepted_parquet": str(accepted_path),
                "checked_chromosomes": projected.height,
            },
            sort_keys=True,
        )
        + "\n"
    )
