from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from marin_dna_vertebrate_projection.issue_473.batched_core import (
    CORE_RUNS,
    decode_query_name,
    encode_query_name,
    split_batched_hal_output,
    write_batched_hal_request_bed6,
    write_batched_maf_request_candidates,
    write_batched_prefill_manifest,
)
from marin_dna_vertebrate_projection.issue_473.policy import (
    FULL_WINDOW_POLICY,
    build_projection_requests,
    centered_landmark_policy,
)
from marin_dna_vertebrate_projection.issue_473.projection import (
    write_maf_request_candidates,
)

from .helpers import species_manifest


def _request_paths(tmp_path: Path) -> dict[str, Path]:
    center_anchor = pl.DataFrame(
        {
            "query_name": ["center|anchor"],
            "source_chrom": ["chr1"],
            "source_start": [100],
            "source_end": [355],
            "region_label": ["cds"],
        }
    )
    enhancer_anchor = pl.DataFrame(
        {
            "query_name": ["enhancer:anchor"],
            "source_chrom": ["chr1"],
            "source_start": [500],
            "source_end": [755],
            "region_label": ["ccre_enhancer_centered"],
        }
    )
    paths = {
        "full_center_1": tmp_path / "center.parquet",
        "full_enhancer_full_window": tmp_path / "enhancer.parquet",
    }
    build_projection_requests(center_anchor, centered_landmark_policy(1)).write_parquet(
        paths["full_center_1"]
    )
    build_projection_requests(enhancer_anchor, FULL_WINDOW_POLICY).write_parquet(
        paths["full_enhancer_full_window"]
    )
    return paths


def test_batched_hal_names_round_trip_and_split_exactly(tmp_path: Path) -> None:
    requests = _request_paths(tmp_path)
    bed = tmp_path / "batched.bed"
    manifest = tmp_path / "requests.json"
    write_batched_hal_request_bed6(requests, bed, manifest)

    center_name = encode_query_name("full_center_1", "center|anchor")
    enhancer_name = encode_query_name("full_enhancer_full_window", "enhancer:anchor")
    assert decode_query_name(center_name) == ("full_center_1", "center|anchor")
    assert bed.read_text() == (
        f"chr1\t227\t228\t{center_name}\t0\t+\nchr1\t500\t755\t{enhancer_name}\t0\t+\n"
    )
    request_receipt = json.loads(manifest.read_text())
    assert request_receipt["runs"] == list(CORE_RUNS)
    assert request_receipt["output"]["rows"] == 2

    combined = tmp_path / "combined.bed"
    combined.write_text(
        f"chrA\t10\t11\t{center_name}\t0\t+\n"
        f"chrB\t20\t30\t{enhancer_name}\t0\t-\n"
        f"chrA\t12\t13\t{center_name}\t0\t+\n"
    )
    outputs = {
        "full_center_1": tmp_path / "center.raw.bed",
        "full_enhancer_full_window": tmp_path / "enhancer.raw.bed",
    }
    summary = split_batched_hal_output(combined, outputs)
    assert outputs["full_center_1"].read_text() == (
        "chrA\t10\t11\tcenter|anchor\t0\t+\nchrA\t12\t13\tcenter|anchor\t0\t+\n"
    )
    assert outputs["full_enhancer_full_window"].read_text() == (
        "chrB\t20\t30\tenhancer:anchor\t0\t-\n"
    )
    assert summary["full_center_1"]["rows"] == 2
    assert summary["full_enhancer_full_window"]["rows"] == 1


def test_batched_maf_matches_two_independent_scans(tmp_path: Path) -> None:
    requests = _request_paths(tmp_path)
    maf = tmp_path / "chr1.maf"
    source_full = "A" * 255
    target_full = "C" * 255
    maf.write_text(
        "##maf version=1\n\n"
        "a score=1\n"
        "s hg38.chr1 227 1 + 2000 A\n"
        "s galGal4.chr2 300 1 + 2000 C\n\n"
        "a score=2\n"
        f"s hg38.chr1 500 255 + 2000 {source_full}\n"
        f"s galGal4.chr2 700 255 + 2000 {target_full}\n"
    )
    manifest = tmp_path / "species.tsv"
    species_manifest().write_csv(manifest, separator="\t")
    batched_outputs = {
        "full_center_1": tmp_path / "batched-center.parquet",
        "full_enhancer_full_window": tmp_path / "batched-enhancer.parquet",
    }
    receipt = tmp_path / "maf-receipt.json"
    write_batched_maf_request_candidates(
        maf,
        requests,
        manifest,
        batched_outputs,
        receipt,
        rows_per_batch=1,
    )

    for run, request_path in requests.items():
        independent = tmp_path / f"independent-{run}.parquet"
        write_maf_request_candidates(
            maf, request_path, manifest, independent, rows_per_batch=1
        )
        expected = pl.read_parquet(independent).sort(
            "query_name", "alignment_name", "mapping_id", "fragment_id"
        )
        observed = pl.read_parquet(batched_outputs[run]).sort(
            "query_name", "alignment_name", "mapping_id", "fragment_id"
        )
        assert observed.equals(expected)

    maf_receipt = json.loads(receipt.read_text())
    assert maf_receipt["runs"] == list(CORE_RUNS)
    assert maf_receipt["outputs"]["full_center_1"]["rows"] == 1
    assert maf_receipt["outputs"]["full_enhancer_full_window"]["rows"] == 1


def test_prefill_manifest_requires_exact_species_and_chromosomes(
    tmp_path: Path,
) -> None:
    hal = tmp_path / "hal.json"
    maf = tmp_path / "maf.json"
    output_fields = {
        run: {"path": run, "rows": 1, "bytes": 1, "sha256": "a" * 64}
        for run in CORE_RUNS
    }
    hal.write_text(
        json.dumps(
            {
                "kind": "issue_473_batched_core_hal",
                "runs": list(CORE_RUNS),
                "target_species": "Mus_musculus",
                "outputs": output_fields,
            }
        )
    )
    maf.write_text(
        json.dumps(
            {
                "kind": "issue_473_batched_core_maf",
                "runs": list(CORE_RUNS),
                "maf_path": "/staged/chr1.maf.gz",
                "outputs": output_fields,
            }
        )
    )
    manifest = tmp_path / "prefill.json"
    write_batched_prefill_manifest(
        [hal, maf],
        manifest,
        expected_species=["Mus_musculus"],
        expected_chroms=["chr1"],
    )
    payload = json.loads(manifest.read_text())
    assert payload["hal_receipts"] == 1
    assert payload["maf_receipts"] == 1
