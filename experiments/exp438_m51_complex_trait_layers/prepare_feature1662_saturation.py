"""Build the frozen, outcome-blind feature-1662 saturation context panel."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from common import ISSUE, assert_commit, sha256_file, write_json
from saturation_common import (
    CONTEXTS_PER_CODON_POSITION,
    DESIGN_RUN_ID,
    NUCLEOTIDES,
    SELECTION_HASH_NAMESPACE,
    complement,
    parse_codon_change,
    selection_hash,
    translate_codon,
)
from transfer_common import validate_test_panel

API_ROOT = "https://rest.ensembl.org"
VEP_OPTIONS = {"canonical": "1", "pick": "1"}
BATCH_SIZE = 50
MAX_WORKERS = 4
MAX_ATTEMPTS = 4
REQUEST_TIMEOUT_SECONDS = 60


def variant_input(row: dict[str, Any]) -> str:
    assert len(row["ref"]) == len(row["alt"]) == 1
    return (
        f"{row['chrom']} {row['pos']} {row['panel_row']} "
        f"{row['ref']} {row['alt']} . . ."
    )


def request_json(url: str, *, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS
            ) as response:
                assert response.status == 200
                return json.loads(response.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if isinstance(error, urllib.error.HTTPError) and error.code not in {
                429,
                500,
                502,
                503,
                504,
            }:
                raise
            if attempt + 1 < MAX_ATTEMPTS:
                retry_after = 2**attempt
                if isinstance(error, urllib.error.HTTPError):
                    retry_after = float(error.headers.get("Retry-After", retry_after))
                time.sleep(max(1.0, retry_after))
    assert last_error is not None
    raise last_error


def request_batch(variants: list[str]) -> list[dict[str, Any]]:
    assert 0 < len(variants) <= BATCH_SIZE
    endpoint = (
        f"{API_ROOT}/vep/homo_sapiens/region?{urllib.parse.urlencode(VEP_OPTIONS)}"
    )
    result = request_json(endpoint, payload={"variants": variants})
    assert isinstance(result, list) and len(result) == len(variants)
    return result


def request_with_recursive_split(variants: list[str]) -> list[dict[str, Any]]:
    try:
        return request_batch(variants)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        if len(variants) == 1:
            raise
        midpoint = len(variants) // 2
        return request_with_recursive_split(
            variants[:midpoint]
        ) + request_with_recursive_split(variants[midpoint:])


def annotate_parallel(frame: pl.DataFrame) -> list[dict[str, Any]]:
    rows = frame.select("panel_row", "chrom", "pos", "ref", "alt").to_dicts()
    batches = [
        rows[start : start + BATCH_SIZE] for start in range(0, len(rows), BATCH_SIZE)
    ]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                request_with_recursive_split,
                [variant_input(row) for row in batch],
            ): index
            for index, batch in enumerate(batches)
        }
        completed: dict[int, list[dict[str, Any]]] = {}
        for future in as_completed(futures):
            index = futures[future]
            completed[index] = future.result()
            print(f"VEP batches {len(completed)}/{len(batches)}", flush=True)
    assert set(completed) == set(range(len(batches)))
    for index in range(len(batches)):
        results.extend(completed[index])
    assert len(results) == frame.height
    assert {int(result["id"]) for result in results} == set(frame["panel_row"])
    return results


def flatten_vep_result(result: dict[str, Any]) -> dict[str, Any]:
    consequences = result.get("transcript_consequences", [])
    transcript = consequences[0] if len(consequences) == 1 else {}
    return {
        "panel_row": int(result["id"]),
        "assembly_name": result.get("assembly_name"),
        "most_severe_consequence": result.get("most_severe_consequence"),
        "transcript_consequence_count": len(consequences),
        "gene_id": transcript.get("gene_id"),
        "gene_symbol": transcript.get("gene_symbol"),
        "transcript_id": transcript.get("transcript_id"),
        "canonical": transcript.get("canonical") == 1,
        "mane_select": transcript.get("mane_select"),
        "biotype": transcript.get("biotype"),
        "strand": transcript.get("strand"),
        "codons": transcript.get("codons"),
        "amino_acids": transcript.get("amino_acids"),
        "cds_start": transcript.get("cds_start"),
        "hgvsc": transcript.get("hgvsc"),
        "hgvsp": transcript.get("hgvsp"),
    }


def eligibility_metadata(row: dict[str, Any]) -> dict[str, Any]:
    reason: str | None = None
    ref_codon: str | None = None
    official_alt_codon: str | None = None
    focal_position: int | None = None
    ref_aa: str | None = None
    official_alt_aa: str | None = None
    try:
        if row["assembly_name"] != "GRCh38":
            reason = "assembly"
        elif row["transcript_consequence_count"] != 1:
            reason = "picked_transcript_count"
        elif row["biotype"] != "protein_coding":
            reason = "biotype"
        elif row["strand"] not in {-1, 1}:
            reason = "strand"
        elif not row["transcript_id"] or not row["codons"]:
            reason = "missing_transcript_or_codon"
        else:
            ref_codon, official_alt_codon, changed_index = parse_codon_change(
                str(row["codons"])
            )
            focal_position = changed_index + 1
            ref_aa = translate_codon(ref_codon)
            official_alt_aa = translate_codon(official_alt_codon)
            if ref_aa == "*":
                reason = "reference_stop"
            else:
                expected_ref = str(row["ref"]).upper()
                expected_alt = str(row["alt"]).upper()
                if row["strand"] == -1:
                    expected_ref = complement(expected_ref)
                    expected_alt = complement(expected_alt)
                if (
                    ref_codon[changed_index] != expected_ref
                    or official_alt_codon[changed_index] != expected_alt
                ):
                    reason = "allele_orientation"
                elif set(ref_codon + official_alt_codon) > set(NUCLEOTIDES):
                    reason = "ambiguous_codon"
    except (AssertionError, KeyError, TypeError, ValueError):
        reason = "codon_parse"
    return {
        "eligibility_reason": reason,
        "ref_codon": ref_codon,
        "official_alt_codon": official_alt_codon,
        "focal_codon_position": focal_position,
        "ref_amino_acid": ref_aa,
        "official_alt_amino_acid": official_alt_aa,
    }


def annotate_eligibility(frame: pl.DataFrame) -> pl.DataFrame:
    metadata = [eligibility_metadata(row) for row in frame.iter_rows(named=True)]
    assert len(metadata) == frame.height
    return frame.hstack(pl.DataFrame(metadata))


def select_contexts(eligible: pl.DataFrame) -> pl.DataFrame:
    assert "label" not in eligible.columns
    rows = eligible.with_columns(
        pl.Series(
            "selection_hash",
            [selection_hash(row) for row in eligible.iter_rows(named=True)],
            pl.String,
        )
    )
    selected_parts: list[pl.DataFrame] = []
    for position in (1, 2, 3):
        part = rows.filter(pl.col("focal_codon_position") == position).sort(
            "selection_hash"
        )
        assert part.height >= CONTEXTS_PER_CODON_POSITION, (position, part.height)
        selected_parts.append(part.head(CONTEXTS_PER_CODON_POSITION))
    selected = (
        pl.concat(selected_parts)
        .sort("focal_codon_position", "selection_hash")
        .with_row_index("context_index")
    )
    assert selected.height == 3 * CONTEXTS_PER_CODON_POSITION
    assert selected["context_index"].to_list() == list(range(selected.height))
    assert selected["panel_row"].n_unique() == selected.height
    assert (
        selected.group_by("focal_codon_position")
        .len()
        .sort("focal_codon_position")["len"]
        .to_list()
        == [CONTEXTS_PER_CODON_POSITION] * 3
    )
    return selected


def validate_design(design_dir: Path) -> tuple[dict[str, Any], pl.DataFrame]:
    manifest_path = design_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == ISSUE
    assert manifest["run_id"] == DESIGN_RUN_ID
    assert manifest["analysis_status"] == "post_hoc_mechanistic_design"
    for name, metadata in manifest["artifacts"].items():
        path = design_dir / name
        assert path.is_file() and path.stat().st_size == metadata["bytes"]
        assert sha256_file(path) == metadata["sha256"]
    eligible = pl.read_parquet(design_dir / "eligible_contexts.parquet")
    selected = pl.read_parquet(design_dir / "selected_contexts.parquet")
    assert "label" not in eligible.columns and "label" not in selected.columns
    expected = select_contexts(eligible)
    assert selected.equals(expected, null_equal=True)
    return manifest, selected


def prepare(panel_path: Path, output_dir: Path) -> dict[str, Any]:
    assert panel_path.is_file() and not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    input_manifest = validate_test_panel(panel_path)
    panel = (
        pl.read_parquet(panel_path)
        .with_row_index("panel_row")
        .filter(pl.col("subset") == "missense_variant")
        .select("panel_row", "chrom", "pos", "ref", "alt", "subset")
    )
    assert panel.height == 2_040 and "label" not in panel.columns
    raw_results = annotate_parallel(panel)
    annotation = pl.DataFrame([flatten_vep_result(result) for result in raw_results])
    assert annotation.height == panel.height
    assert annotation["panel_row"].n_unique() == annotation.height
    merged = panel.join(annotation, on="panel_row", how="inner", validate="1:1")
    annotated = annotate_eligibility(merged)
    eligible = annotated.filter(pl.col("eligibility_reason").is_null())
    assert eligible.height > 0
    selected = select_contexts(eligible)
    release = request_json(f"{API_ROOT}/info/data?")

    output_dir.mkdir(parents=True)
    (output_dir / "vep_raw.json").write_text(
        json.dumps(raw_results, indent=2, sort_keys=True) + "\n"
    )
    annotated.write_parquet(output_dir / "vep_annotations.parquet", compression="zstd")
    eligible.write_parquet(output_dir / "eligible_contexts.parquet", compression="zstd")
    selected.write_parquet(output_dir / "selected_contexts.parquet", compression="zstd")
    reason_counts = (
        annotated.with_columns(pl.col("eligibility_reason").fill_null("eligible"))
        .group_by("eligibility_reason")
        .len()
        .sort("eligibility_reason")
        .to_dicts()
    )
    eligible_counts = (
        eligible.group_by("focal_codon_position")
        .len()
        .sort("focal_codon_position")
        .to_dicts()
    )
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": DESIGN_RUN_ID,
        "analysis_status": "post_hoc_mechanistic_design",
        "experiment_commit": experiment_commit,
        "input_panel": input_manifest,
        "api_root": API_ROOT,
        "api_release": release,
        "vep_options": VEP_OPTIONS,
        "selection": {
            "source_subset": "missense_variant",
            "source_rows": panel.height,
            "contexts_per_codon_position": CONTEXTS_PER_CODON_POSITION,
            "selected_rows": selected.height,
            "hash_key": (
                f"{SELECTION_HASH_NAMESPACE}|panel_row|chrom|pos|ref|alt|transcript_id"
            ),
            "forbidden_fields": [
                "label",
                "SAE activation",
                "SIFT",
                "PolyPhen",
                "BLOSUM62",
                "clinical annotation",
            ],
            "eligibility_reason_counts": reason_counts,
            "eligible_codon_position_counts": eligible_counts,
        },
    }
    write_json(output_dir / "results.json", result)
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {**result, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", manifest)
    validate_design(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.panel, args.output_dir)
    print(json.dumps(result["selection"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
