"""Per-species sequence extraction at projected coordinates.

The Snakemake rule converts each per-species projection Parquet into a
6-column BED, then runs ``bedtools getfasta -s`` against the species
FASTA (output of ``hal2fasta``) to extract strand-aware sequences. The
helpers here are small Polars-backed glue, kept out of the rule for
testability.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def parquet_to_bed6(parquet_path: str | Path, out_bed: str | Path) -> int:
    """Materialize a per-species projection Parquet as a BED6 file.

    Columns: ``t_chrom\\tt_start\\tt_end\\tquery_name\\t0\\tt_strand``.
    No header. The score column is always ``0`` (the projection has no
    score concept — the BED6 column is required so ``bedtools getfasta -s``
    sees a strand). Returns the number of rows written.

    Empty Parquet → empty BED, returns 0.
    """
    df = pl.read_parquet(parquet_path)
    out = Path(out_bed)
    out.parent.mkdir(parents=True, exist_ok=True)
    if df.is_empty():
        out.write_text("")
        return 0
    df.select(
        "t_chrom",
        "t_start",
        "t_end",
        "query_name",
        pl.lit(0).alias("score"),
        "t_strand",
    ).write_csv(out, separator="\t", include_header=False)
    return df.height


def parse_bedtools_getfasta_output(fasta_path: str | Path) -> list[str]:
    """Parse a single-line-per-record FASTA from ``bedtools getfasta -nameOnly``.

    bedtools getfasta emits two lines per BED row in BED order:
    a header ``>{name}({strand})`` and a single line of sequence. We
    rely on the single-line-per-record convention (no width wrapping
    at default settings) for fast row-aligned consumption — the
    returned list of sequences is in BED row order, ready to
    ``zip(parquet_rows, sequences)``.

    Asserts every other line starts with ``>``; raises on a malformed
    file.
    """
    seqs: list[str] = []
    with Path(fasta_path).open() as f:
        for line_no, line in enumerate(f):
            stripped = line.rstrip("\n")
            if line_no % 2 == 0:
                assert stripped.startswith(">"), (
                    f"expected header at line {line_no + 1}, got: {stripped[:60]!r}"
                )
            else:
                # Empty sequence lines would silently flow through and
                # only fail downstream in attach_sequences_to_parquet's
                # length check — fail here instead so the diagnostic
                # points at the actual malformed FASTA row.
                assert stripped, f"empty sequence at line {line_no + 1}"
                seqs.append(stripped)
    return seqs


def attach_sequences_to_parquet(
    proj_parquet: str | Path,
    sequences: list[str],
    out_parquet: str | Path,
    *,
    target_len: int,
) -> int:
    """Append a ``sequence`` column to a per-species projection Parquet.

    ``sequences`` must align row-for-row with ``proj_parquet`` (the BED
    written by :func:`parquet_to_bed6` and consumed by
    ``bedtools getfasta`` is in the same order). Asserts every sequence
    is exactly ``target_len`` characters and that the count matches.

    Empty Parquet → empty Parquet with the extended schema.
    """
    df = pl.read_parquet(proj_parquet)
    out = Path(out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    extended_schema: dict[str, type[pl.DataType] | pl.DataType] = {
        **df.schema,
        "sequence": pl.Utf8,
    }
    if df.is_empty():
        assert len(sequences) == 0
        pl.DataFrame(schema=extended_schema).write_parquet(out)
        return 0
    assert len(sequences) == df.height, (
        f"sequence count ({len(sequences)}) != Parquet rows ({df.height})"
    )
    seq_series = pl.Series("sequence", sequences)
    # Vectorised length check — one pass in Rust over the column instead
    # of a Python `for` loop. Keeps the loud-failure invariant.
    if not (seq_series.str.len_bytes() == target_len).all():
        bad_idx = (seq_series.str.len_bytes() != target_len).arg_true().to_list()[:3]
        bad = [(i, len(sequences[i])) for i in bad_idx]
        raise AssertionError(
            f"unexpected sequence length(s); expected {target_len} bp, "
            f"first offenders (row, len): {bad}"
        )
    df_with_seq = df.with_columns(sequence=seq_series)
    df_with_seq.write_parquet(out)
    return df.height
