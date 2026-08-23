from pathlib import Path

import polars as pl
import pytest
from marin_dna_vertebrate_projection.contract import (
    apply_projection_contract,
)
from marin_dna_vertebrate_projection.pipeline_io import (
    write_contract_outputs_for_alignment,
)
from marin_dna_vertebrate_projection.projection.center import (
    project_requests_from_maf,
    read_projection_requests,
    write_hal_request_bed6,
    write_maf_request_candidates,
)
from marin_dna_vertebrate_projection.projection.requests import (
    build_projection_requests,
)

from ..helpers import species_manifest


def _anchor() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_name": ["anchor-1"],
            "source_chrom": ["chr1"],
            "source_start": [100],
            "source_end": [355],
            "region_label": ["cds"],
        }
    )


def test_hal_bed_uses_landmark_while_request_keeps_source_anchor(
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "center-1.parquet"
    bed_path = tmp_path / "center-1.bed"
    build_projection_requests(_anchor()).write_parquet(request_path)

    request = read_projection_requests(request_path).row(0, named=True)
    assert (request["source_start"], request["source_end"]) == (100, 355)
    assert (request["projection_start"], request["projection_end"]) == (227, 228)

    write_hal_request_bed6(request_path, bed_path)
    assert bed_path.read_text() == "chr1\t227\t228\tanchor-1\t0\t+\n"


def test_request_reader_rejects_off_center_landmark(tmp_path: Path) -> None:
    request_path = tmp_path / "off-center.parquet"
    off_center = build_projection_requests(_anchor()).with_columns(
        projection_start=pl.col("projection_start") + 1,
        projection_end=pl.col("projection_end") + 1,
    )
    off_center.write_parquet(request_path)

    with pytest.raises(AssertionError):
        read_projection_requests(request_path)


def test_center_one_maf_projection_retains_anchor_and_resizes_target(
    tmp_path: Path,
) -> None:
    maf = tmp_path / "center.maf"
    maf.write_text(
        "##maf version=1\n\n"
        "a score=1\n"
        "s hg38.chr1 227 1 + 1000 A\n"
        "s galGal4.chr2 300 1 + 1000 C\n"
        "s xenTro7.scaf 499 1 - 1000 G\n"
    )
    requests = build_projection_requests(_anchor())
    fragments = project_requests_from_maf(maf, requests, species_manifest())

    assert fragments.height == 2
    assert set(fragments.select("source_start", "source_end").iter_rows()) == {
        (100, 355)
    }
    assert set(
        fragments.select("source_fragment_start", "source_fragment_end").iter_rows()
    ) == {(227, 228)}
    assert set(fragments["t_strand"].to_list()) == {"+", "-"}

    result = apply_projection_contract(
        fragments,
    )
    assert result.accepted.height == 2
    assert (
        result.accepted["pre_resize_t_end"] - result.accepted["pre_resize_t_start"] == 1
    ).all()
    assert (result.accepted["t_end"] - result.accepted["t_start"] == 255).all()
    target_midpoints = (
        result.accepted["pre_resize_t_start"] + result.accepted["pre_resize_t_end"]
    ) // 2
    assert (target_midpoints - result.accepted["t_start"] == 127).all()
    assert result.rejected.is_empty()


def test_streaming_maf_and_contract_writers_use_policy_landmark(
    tmp_path: Path,
) -> None:
    maf = tmp_path / "center.maf"
    maf.write_text(
        "##maf version=1\n\n"
        "a score=1\n"
        "s hg38.chr1 227 1 + 1000 A\n"
        "s galGal4.chr2 300 1 + 1000 C\n"
        "s xenTro7.scaf 499 1 - 1000 G\n"
    )
    requests_path = tmp_path / "requests.parquet"
    build_projection_requests(_anchor()).write_parquet(requests_path)
    manifest_path = tmp_path / "species.tsv"
    species_manifest().write_csv(manifest_path, separator="\t")
    fragments_path = tmp_path / "fragments.parquet"

    write_maf_request_candidates(
        maf,
        requests_path,
        manifest_path,
        fragments_path,
        rows_per_batch=1,
    )

    fragments = pl.read_parquet(fragments_path)
    assert fragments["alignment_name"].to_list() == ["galGal4", "xenTro7"]
    assert set(
        fragments.select("source_fragment_start", "source_fragment_end").iter_rows()
    ) == {(227, 228)}

    accepted_path = tmp_path / "accepted.parquet"
    rejected_path = tmp_path / "rejected.parquet"
    write_contract_outputs_for_alignment(
        fragments_path,
        "galGal4",
        accepted_path,
        rejected_path,
    )
    assert pl.read_parquet(accepted_path).height == 1
    assert pl.read_parquet(rejected_path).is_empty()
