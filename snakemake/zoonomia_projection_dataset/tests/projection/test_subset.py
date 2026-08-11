"""Tests for ``marin_dna_zoonomia_projection.projection.subset``."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from marin_dna_zoonomia_projection.projection.subset import (
    filter_to_species,
    filter_to_subset,
    load_query_names,
    load_species,
)


def test_load_query_names_skips_blanks_and_comments(tmp_path: Path) -> None:
    p = tmp_path / "names.txt"
    p.write_text(
        "# header comment\n"
        "win_1_000000001\n"
        "\n"
        "win_2_000000099\n"
        "    \n"
        "# trailing comment\n"
        "win_X_000000777\n"
    )
    assert load_query_names(p) == {
        "win_1_000000001",
        "win_2_000000099",
        "win_X_000000777",
    }


def _make_all_species_parquet(tmp_path: Path) -> Path:
    """Build a tiny 6-row, 3-species, 2-query all-species Parquet."""
    df = pl.DataFrame(
        [
            # win_1: present for all 3 species
            {
                "query_name": "win_1",
                "species": "A",
                "t_chrom": "chr1",
                "sequence": "ACGT",
            },
            {
                "query_name": "win_1",
                "species": "B",
                "t_chrom": "chr1",
                "sequence": "AAAA",
            },
            {
                "query_name": "win_1",
                "species": "C",
                "t_chrom": "chr1",
                "sequence": "TTTT",
            },
            # win_2: present for A and B
            {
                "query_name": "win_2",
                "species": "A",
                "t_chrom": "chr2",
                "sequence": "GGGG",
            },
            {
                "query_name": "win_2",
                "species": "B",
                "t_chrom": "chr2",
                "sequence": "CCCC",
            },
            # win_3: present for A only — should be excluded by the subset
            {
                "query_name": "win_3",
                "species": "A",
                "t_chrom": "chr3",
                "sequence": "NNNN",
            },
        ],
        schema={
            "query_name": pl.Utf8,
            "species": pl.Utf8,
            "t_chrom": pl.Utf8,
            "sequence": pl.Utf8,
        },
    )
    p = tmp_path / "all.parquet"
    df.write_parquet(p)
    return p


def test_filter_to_subset_keeps_all_species_for_subset_keys(tmp_path: Path) -> None:
    src = _make_all_species_parquet(tmp_path)
    out = tmp_path / "subset.parquet"
    filter_to_subset(src, {"win_1", "win_2"}, out)

    got = pl.read_parquet(out).sort(["query_name", "species"])
    # win_1 → 3 species, win_2 → 2 species, win_3 dropped.
    assert got.height == 5
    assert set(got["query_name"].unique()) == {"win_1", "win_2"}
    # All three species kept for win_1 (the unique trait of the
    # query_name-based subset: keep all species' rows together).
    assert set(got.filter(pl.col("query_name") == "win_1")["species"]) == {
        "A",
        "B",
        "C",
    }


def test_filter_to_subset_via_text_file(tmp_path: Path) -> None:
    src = _make_all_species_parquet(tmp_path)
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("# only win_3\nwin_3\n")
    out = tmp_path / "subset.parquet"
    filter_to_subset(src, keys_file, out)

    got = pl.read_parquet(out)
    assert got.height == 1
    row = got.row(0, named=True)
    assert row["query_name"] == "win_3"
    assert row["species"] == "A"


def test_filter_to_subset_empty_keys(tmp_path: Path) -> None:
    src = _make_all_species_parquet(tmp_path)
    out = tmp_path / "subset.parquet"
    filter_to_subset(src, set(), out)
    got = pl.read_parquet(out)
    assert got.height == 0
    # Schema preserved.
    assert "query_name" in got.columns
    assert "species" in got.columns
    assert "sequence" in got.columns


def _write_species_tsv(tmp_path: Path, species: list[str]) -> Path:
    """Write a minimal species TSV (header + species column) like the configs."""
    p = tmp_path / "species.tsv"
    lines = [
        "species\tfamily\torder\taccession\tassembly_level\tcontig_n50\tquality_source"
    ]
    for s in species:
        lines.append(f"{s}\tFamX\tOrderX\tGCA_0\tScaffold\t1000\tzoonomia_supp_st2")
    p.write_text("\n".join(lines) + "\n")
    return p


def test_load_species_reads_species_column(tmp_path: Path) -> None:
    tsv = _write_species_tsv(tmp_path, ["A", "C"])
    assert load_species(tsv) == {"A", "C"}


def test_filter_to_species_keeps_only_cohort_species(tmp_path: Path) -> None:
    src = _make_all_species_parquet(tmp_path)  # species A (×3), B (×2), C (×1)
    out = tmp_path / "cohort.parquet"
    filter_to_species(src, {"A", "C"}, out)

    got = pl.read_parquet(out)
    assert set(got["species"].unique()) == {"A", "C"}
    # A has 3 rows (win_1/2/3), C has 1 (win_1); B (×2) dropped.
    assert got.height == 4
    # Orthogonal to the query_name axis: every query_name A/C touch survives.
    assert set(got["query_name"].unique()) == {"win_1", "win_2", "win_3"}


def test_filter_to_species_via_tsv(tmp_path: Path) -> None:
    src = _make_all_species_parquet(tmp_path)
    tsv = _write_species_tsv(tmp_path, ["B"])
    out = tmp_path / "cohort.parquet"
    filter_to_species(src, tsv, out)

    got = pl.read_parquet(out)
    assert set(got["species"].unique()) == {"B"}
    assert got.height == 2  # B is present for win_1 and win_2


def test_filter_to_species_rejects_species_not_in_source(tmp_path: Path) -> None:
    """A cohort leaf absent from the source must fail loudly (not silently shrink)."""
    src = _make_all_species_parquet(tmp_path)
    out = tmp_path / "cohort.parquet"
    with pytest.raises(AssertionError, match="not a subset"):
        filter_to_species(src, {"A", "Z_absent"}, out)


def test_filter_to_species_compose_with_query_name_subset(tmp_path: Path) -> None:
    """Species filter ∘ query_name filter commute (the v4_cds × order pattern)."""
    src = _make_all_species_parquet(tmp_path)
    iv = tmp_path / "iv.parquet"  # intervals subset: win_1, win_2 (all species)
    filter_to_subset(src, {"win_1", "win_2"}, iv)
    out = tmp_path / "iv_cohort.parquet"
    filter_to_species(iv, {"A", "C"}, out)

    got = pl.read_parquet(out)
    # win_1 → A,C ; win_2 → A (C absent for win_2). 3 rows.
    assert got.height == 3
    assert set(got["species"].unique()) == {"A", "C"}
    assert set(got["query_name"].unique()) == {"win_1", "win_2"}
