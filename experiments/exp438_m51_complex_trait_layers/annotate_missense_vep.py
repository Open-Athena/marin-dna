"""Annotate feature-candidate missense variants with the official Ensembl VEP API.

The annotation and downstream mechanism checks are explicitly post-hoc. The
script retains the raw API response and a compact picked-transcript table so
the exact external annotation can be audited later.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from common import ISSUE, write_json

API_ROOT = "https://rest.ensembl.org"
BATCH_SIZE = 50
EXPECTED_MISSENSE_ROWS = 2_500
MAX_ATTEMPTS = 5
VEP_OPTIONS = {
    "Blosum62": "1",
    "canonical": "1",
    "pick": "1",
    "polyphen": "b",
    "sift": "b",
}


def request_json(url: str, *, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                assert response.status == 200
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if (
                error.code not in {429, 500, 502, 503, 504}
                or attempt + 1 == MAX_ATTEMPTS
            ):
                raise
            retry_after = float(error.headers.get("Retry-After", 2**attempt))
            time.sleep(max(1.0, retry_after))
        except urllib.error.URLError:
            if attempt + 1 == MAX_ATTEMPTS:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def variant_input(row: dict[str, Any]) -> str:
    assert len(row["ref"]) == len(row["alt"]) == 1
    return (
        f"{row['chrom']} {row['pos']} {row['panel_row']} "
        f"{row['ref']} {row['alt']} . . ."
    )


def annotate_variants(frame: pl.DataFrame) -> tuple[list[dict[str, Any]], Any]:
    rows = frame.select("panel_row", "chrom", "pos", "ref", "alt").to_dicts()
    endpoint = (
        f"{API_ROOT}/vep/homo_sapiens/region?{urllib.parse.urlencode(VEP_OPTIONS)}"
    )
    results: list[dict[str, Any]] = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        response = request_json(
            endpoint, payload={"variants": [variant_input(row) for row in batch]}
        )
        assert isinstance(response, list) and len(response) == len(batch)
        results.extend(response)
        print(f"VEP {min(start + BATCH_SIZE, len(rows))}/{len(rows)}", flush=True)
        time.sleep(0.25)
    release = request_json(f"{API_ROOT}/info/data?")
    return results, release


def joined(values: list[Any] | None) -> str | None:
    if not values:
        return None
    return ",".join(sorted({str(value) for value in values}))


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    consequences = result.get("transcript_consequences", [])
    assert len(consequences) == 1
    transcript = consequences[0]
    colocated = result.get("colocated_variants", [])
    clinical_significance = joined(
        [term for variant in colocated for term in variant.get("clin_sig", [])]
    )
    panel_row = int(result["id"])
    return {
        "panel_row": panel_row,
        "assembly_name": result["assembly_name"],
        "most_severe_consequence": result["most_severe_consequence"],
        "gene_id": transcript.get("gene_id"),
        "gene_symbol": transcript.get("gene_symbol"),
        "transcript_id": transcript.get("transcript_id"),
        "canonical": transcript.get("canonical") == 1,
        "mane_select": transcript.get("mane_select"),
        "biotype": transcript.get("biotype"),
        "impact": transcript.get("impact"),
        "exon": transcript.get("exon"),
        "hgvsc": transcript.get("hgvsc"),
        "hgvsp": transcript.get("hgvsp"),
        "codons": transcript.get("codons"),
        "amino_acids": transcript.get("amino_acids"),
        "protein_start": transcript.get("protein_start"),
        "cds_start": transcript.get("cds_start"),
        "sift_score": transcript.get("sift_score"),
        "sift_prediction": transcript.get("sift_prediction"),
        "polyphen_score": transcript.get("polyphen_score"),
        "polyphen_prediction": transcript.get("polyphen_prediction"),
        "blosum62": transcript.get("blosum62"),
        "alphamissense_score": transcript.get("alphamissense_score"),
        "alphamissense_class": transcript.get("alphamissense_class"),
        "domains_json": json.dumps(transcript.get("domains", []), sort_keys=True),
        "clinical_significance": clinical_significance,
        "known_variant_ids": joined(
            [variant.get("id") for variant in colocated if variant.get("id")]
        ),
    }


def run(candidate_path: Path, output_dir: Path) -> None:
    assert candidate_path.is_file()
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates = pl.read_parquet(candidate_path)
    missense = candidates.filter(pl.col("subset") == "missense_variant")
    assert missense.height == EXPECTED_MISSENSE_ROWS
    assert missense["panel_row"].n_unique() == missense.height
    raw_results, release = annotate_variants(missense)
    assert len(raw_results) == missense.height
    annotation = pl.DataFrame([flatten_result(result) for result in raw_results])
    assert annotation.height == missense.height
    assert annotation["panel_row"].n_unique() == annotation.height
    assert set(annotation["assembly_name"].unique()) == {"GRCh38"}
    merged = missense.join(annotation, on="panel_row", how="inner", validate="1:1")
    assert merged.height == missense.height

    (output_dir / "vep_raw.json").write_text(
        json.dumps(raw_results, indent=2, sort_keys=True) + "\n"
    )
    annotation.write_parquet(output_dir / "vep_annotations.parquet", compression="zstd")
    merged.write_parquet(output_dir / "missense_annotated.parquet", compression="zstd")
    write_json(
        output_dir / "results.json",
        {
            "issue": ISSUE,
            "analysis_status": "post_hoc_descriptive",
            "annotated_at": datetime.now(UTC).isoformat(),
            "api_root": API_ROOT,
            "api_release": release,
            "vep_options": VEP_OPTIONS,
            "rows": merged.height,
            "assembly_name": "GRCh38",
            "fields_with_values": {
                column: int(merged[column].is_not_null().sum())
                for column in (
                    "sift_score",
                    "polyphen_score",
                    "blosum62",
                    "alphamissense_score",
                    "clinical_significance",
                )
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.candidate_table, args.output_dir)


if __name__ == "__main__":
    main()
