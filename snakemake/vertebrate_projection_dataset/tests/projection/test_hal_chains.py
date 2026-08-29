from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.projection.hal_chains import (
    _parse_gnu_time,
    build_direction_matched_hal_to_chain_pipeline,
    build_hal_to_chain_pipeline,
    validate_chain_direction,
    write_chain_parity_audit,
    write_exact_chain_parity_audit,
    write_regional_smoke_beds,
    write_uniform_grid_center_bed,
)


def _write_bed(path: Path, rows: list[tuple[object, ...]]) -> None:
    path.write_text("".join("\t".join(map(str, row)) + "\n" for row in rows))


def test_build_hal_to_chain_pipeline_matches_released_cactus_direction() -> None:
    observed = build_hal_to_chain_pipeline(
        hal_path="alignment.hal",
        query_genome="Mus_musculus",
        query_bed="mouse.bed",
        target_genome="Homo_sapiens",
        target_twobit="human.2bit",
        query_twobit="mouse.2bit",
        output_chain="human_to_mouse.chain.gz",
        no_dupes=True,
        linear_gap="medium",
    )
    assert observed == (
        "halLiftover --noDupes --outPSL alignment.hal Mus_musculus mouse.bed "
        "Homo_sapiens /dev/stdout | pslPosTarget /dev/stdin /dev/stdout | "
        "axtChain -psl -verbose=0 -linearGap=medium /dev/stdin human.2bit "
        "mouse.2bit /dev/stdout | gzip -n -c > human_to_mouse.chain.gz"
    )


def test_build_direction_matched_pipeline_swaps_psl_and_keeps_negative_scores() -> None:
    observed = build_direction_matched_hal_to_chain_pipeline(
        hal_path="alignment.hal",
        source_genome="Homo_sapiens",
        source_bed="human.bed",
        destination_genome="Papio_anubis",
        source_twobit="human.2bit",
        destination_twobit="baboon.2bit",
        output_chain="human_to_baboon.chain.gz",
        min_score=-1_000_000,
        linear_gap="medium",
    )
    assert observed == (
        "halLiftover --noDupes --outPSL alignment.hal Homo_sapiens human.bed "
        "Papio_anubis /dev/stdout | pslSwap /dev/stdin /dev/stdout | "
        "pslPosTarget /dev/stdin /dev/stdout | axtChain -psl -verbose=0 "
        "-linearGap=medium -minScore=-1000000 /dev/stdin human.2bit "
        "baboon.2bit /dev/stdout | gzip -n -c > human_to_baboon.chain.gz"
    )


def test_validate_chain_direction_checks_names_sizes_and_aligned_bases(
    tmp_path: Path,
) -> None:
    target_sizes = tmp_path / "human.sizes"
    query_sizes = tmp_path / "mouse.sizes"
    target_sizes.write_text("chr1\t1000\n")
    query_sizes.write_text("chrM\t800\n")
    chain = tmp_path / "human_to_mouse.chain.gz"
    with gzip.open(chain, "wt") as handle:
        handle.write(
            "##matrix=axtChain 16 91 114 91\n"
            "# gap penalties\n"
            "chain 100 chr1 1000 + 10 40 chrM 800 - 20 50 1\n"
            "10\t2\t3\n"
            "18\n\n"
        )
    assert validate_chain_direction(
        chain,
        target_chrom_sizes=target_sizes,
        query_chrom_sizes=query_sizes,
    ) == {"chain_count": 1, "aligned_block_bases": 28}


def test_parse_gnu_time_preserves_colons_in_elapsed_value(tmp_path: Path) -> None:
    path = tmp_path / "time.txt"
    path.write_text(
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03\n"
        "\tMaximum resident set size (kbytes): 1234\n"
    )
    assert _parse_gnu_time(path) == {
        "Elapsed (wall clock) time (h:mm:ss or m:ss)": "1:02:03",
        "Maximum resident set size (kbytes)": "1234",
    }


def test_write_chain_parity_audit_classifies_every_query(tmp_path: Path) -> None:
    input_bed = tmp_path / "input.bed"
    direct_bed = tmp_path / "direct.bed"
    chain_bed = tmp_path / "chain.bed"
    summary = tmp_path / "summary.json"
    discrepancies = tmp_path / "discrepancies.parquet"
    _write_bed(
        input_bed,
        [
            ("chr1", 1, 2, "exact-mapped", 0, "+"),
            ("chr1", 2, 3, "exact-unmapped", 0, "+"),
            ("chr1", 3, 4, "chain-only", 0, "+"),
            ("chr1", 4, 5, "direct-only", 0, "+"),
            ("chr1", 5, 6, "conflict", 0, "+"),
        ],
    )
    _write_bed(
        direct_bed,
        [
            ("target", 11, 12, "exact-mapped", 0, "+"),
            ("target", 14, 15, "direct-only", 0, "+"),
            ("target", 15, 16, "conflict", 0, "+"),
        ],
    )
    _write_bed(
        chain_bed,
        [
            ("target", 11, 12, "exact-mapped", 0, "+"),
            ("target", 13, 14, "chain-only", 0, "+"),
            ("target", 25, 26, "conflict", 0, "+"),
        ],
    )
    observed = write_chain_parity_audit(
        input_bed=input_bed,
        direct_bed=direct_bed,
        chain_bed=chain_bed,
        summary_path=summary,
        discrepancies_path=discrepancies,
        expected_queries=5,
    )
    assert observed["parity_counts"] == {
        "exact_mapped": 1,
        "exact_unmapped": 1,
        "chain_only": 1,
        "direct_only": 1,
        "mapping_conflict": 1,
    }
    assert observed["exact_fraction"] == 0.4
    assert json.loads(summary.read_text()) == observed
    assert set(pl.read_parquet(discrepancies)["query_name"]) == {
        "chain-only",
        "direct-only",
        "conflict",
    }


def test_write_exact_chain_parity_audit_accepts_empty_discrepancies(
    tmp_path: Path,
) -> None:
    input_bed = tmp_path / "input.bed"
    direct_bed = tmp_path / "direct.bed"
    chain_bed = tmp_path / "chain.bed"
    summary = tmp_path / "summary.json"
    discrepancies = tmp_path / "discrepancies.parquet"
    rows = [("chr1", 1, 2, "mapped", 0, "+")]
    _write_bed(input_bed, rows)
    _write_bed(direct_bed, [("target", 11, 12, "mapped", 0, "+")])
    _write_bed(chain_bed, [("target", 11, 12, "mapped", 0, "+")])
    observed = write_exact_chain_parity_audit(
        input_bed=input_bed,
        direct_bed=direct_bed,
        chain_bed=chain_bed,
        summary_path=summary,
        discrepancies_path=discrepancies,
        expected_queries=1,
    )
    assert observed["exact_fraction"] == 1.0
    assert pl.read_parquet(discrepancies).height == 0


def test_write_exact_chain_parity_audit_removes_failed_gate_outputs(
    tmp_path: Path,
) -> None:
    input_bed = tmp_path / "input.bed"
    direct_bed = tmp_path / "direct.bed"
    chain_bed = tmp_path / "chain.bed"
    summary = tmp_path / "summary.json"
    discrepancies = tmp_path / "discrepancies.parquet"
    rows = [("chr1", 1, 2, "mapped", 0, "+")]
    _write_bed(input_bed, rows)
    _write_bed(direct_bed, [("target", 11, 12, "mapped", 0, "+")])
    chain_bed.write_text("")
    try:
        write_exact_chain_parity_audit(
            input_bed=input_bed,
            direct_bed=direct_bed,
            chain_bed=chain_bed,
            summary_path=summary,
            discrepancies_path=discrepancies,
            expected_queries=1,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("non-exact parity unexpectedly passed")
    assert not summary.exists()
    assert not discrepancies.exists()


def test_write_regional_smoke_beds_uses_global_grid_phase(tmp_path: Path) -> None:
    region_bed = tmp_path / "regions.bed"
    centers_bed = tmp_path / "centers.bed"
    write_regional_smoke_beds(
        regions=[
            {"name": "one", "chrom": "chr1", "start": 100, "end": 300},
            {"name": "two", "chrom": "chr2", "start": 500, "end": 600},
        ],
        region_bed_path=region_bed,
        centers_bed_path=centers_bed,
        step_size=128,
        center_offset=127,
        expected_queries=3,
    )
    assert region_bed.read_text() == (
        "chr1\t100\t300\tone\t0\t+\n" "chr2\t500\t600\ttwo\t0\t+\n"
    )
    assert centers_bed.read_text() == (
        "chr1\t127\t128\tsmoke_one_000000000127\t0\t+\n"
        "chr1\t255\t256\tsmoke_one_000000000255\t0\t+\n"
        "chr2\t511\t512\tsmoke_two_000000000511\t0\t+\n"
    )


def test_write_uniform_grid_center_bed_is_half_open_and_skips_undefined(
    tmp_path: Path,
) -> None:
    sizes = tmp_path / "sizes.tsv"
    undefined = tmp_path / "undefined.bed"
    output = tmp_path / "centers.bed"
    sizes.write_text("chr1\t11\n")
    undefined.write_text("chr1\t5\t6\n")
    write_uniform_grid_center_bed(
        chrom_sizes_path=sizes,
        undefined_bed_path=undefined,
        output_bed_path=output,
        standard_chroms=["chr1"],
        window_size=5,
        step_size=2,
        expected_queries=2,
    )
    assert output.read_text() == (
        "chr1\t2\t3\twin_chr1_000000001\t0\t+\n"
        "chr1\t8\t9\twin_chr1_000000004\t0\t+\n"
    )
