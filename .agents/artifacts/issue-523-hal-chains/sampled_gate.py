"""Issue #523 bounded chain-reader validation against saved direct HAL outputs.

Run only on approved remote compute. No HAL access or chain generation.
Preparation downloads saved metadata/BEDs, samples 10,000 queries, and writes
a pinned config for the maintained chain-only workflow. Auditing reads that
workflow's outputs and compares exact raw mapping multisets.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import boto3
import polars as pl
import yaml

from marin_dna_vertebrate_projection.projection.chains import file_sha256

BUCKET = "oa-bolinas"
BASE = "snakemake/vertebrate_projection_dataset/results/phylop-uniform-v1/2162b6aa8299a9748eeb8031318b49072bb8c3fc/94d512050de327f96fda1105ce9c6ae5562944e402802516c7cde54795d8cdd1/full"
CHAINS = "snakemake/vertebrate_projection_dataset/results/hal-chains-directional-ramp-v2/b86897b7050bc9fdf397dd6abfb3af11fc876f86/d035c2561f6be3b11449647adfe9ce865884aef7da8b7e06b817d8a75c7f37f9/full/chains"
TARGETS = {"Papio_anubis", "Mus_musculus", "Loxodonta_africana"}


def bed_rows(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                fields = line.rstrip("\n").split("\t")
                assert len(fields) == 6, fields
                yield fields


def prepare(args) -> None:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    s3 = boto3.client("s3")
    objects = {}

    def fetch(key: str, name: str) -> Path:
        path = output / name
        head = s3.head_object(Bucket=BUCKET, Key=key)
        s3.download_file(BUCKET, key, str(path))
        assert path.stat().st_size == head["ContentLength"]
        objects[name] = {"s3_uri": f"s3://{BUCKET}/{key}", "bytes": path.stat().st_size,
                         "sha256": file_sha256(path), "etag": head["ETag"],
                         "version_id": head.get("VersionId")}
        return path

    requests = fetch(f"{BASE}/anchors/projection_requests.parquet", "requests.parquet")
    input_bed = fetch(f"{BASE}/hal/input.bed", "input.bed")
    direct = fetch(f"{BASE}/hal/raw/{args.species}.bed", "direct_hal.bed")
    source = fetch(f"{BASE}/reference/hg38.chrom.sizes", "source.sizes")
    target = fetch(f"{BASE}/hal/chrom_sizes/{args.species}.tsv", "target.sizes")
    generation = fetch(f"{CHAINS}/{args.species}/chain_generation.json", "generation.json")
    generated = json.loads(generation.read_text())
    assert generated["destination_genome"] == args.species
    assert generated["source_genome"] == "Homo_sapiens"
    assert generated["recipe"] == "direction_matched_no_dupes_psl_swap"
    chain_uri = f"s3://{BUCKET}/{CHAINS}/{args.species}/human_to_species.chain.gz"
    mapped_names = {row[3] for row in bed_rows(direct)}
    frame = pl.read_parquet(requests)
    assert frame.height == 1_136_854
    frame = frame.with_columns(
        pl.col("query_name").is_in(mapped_names).alias("direct_mapped"),
        pl.col("query_name").map_elements(
            lambda name: hashlib.sha256(f"523-v1:{name}".encode()).hexdigest(),
            return_dtype=pl.String,
        ).alias("sample_hash"),
    ).sort("sample_hash", "query_name")
    strata = ["source_chrom", "region_label", "direct_mapped"]
    nstrata = frame.select(strata).unique().height
    assert 0 < nstrata < 10_000
    selected = frame.group_by(strata, maintain_order=True).head(10_000 // nstrata).select(frame.columns)
    fill = frame.filter(~pl.col("query_name").is_in(selected["query_name"].implode())).head(10_000 - selected.height)
    selected = pl.concat([selected, fill]).sort("source_chrom", "source_start", "query_name")
    assert selected.height == selected["query_name"].n_unique() == 10_000
    assert selected.select(strata).unique().height == nstrata
    expected = {r["query_name"]: (r["source_chrom"], str(r["projection_start"]), str(r["projection_end"])) for r in selected.to_dicts()}
    found = 0
    input_count = 0
    with (output / "sample.input.bed").open("w") as handle:
        for row in bed_rows(input_bed):
            input_count += 1
            if row[3] in expected:
                assert tuple(row[:3]) == expected[row[3]] and row[4:] == ["0", "+"]
                handle.write("\t".join(row) + "\n")
                found += 1
    assert input_count == 1_136_854 and found == 10_000
    with (output / "sample.direct_hal.bed").open("w") as handle:
        for row in bed_rows(direct):
            if row[3] in expected:
                handle.write("\t".join(row) + "\n")
    selected.write_parquet(output / "sample.design.parquet")
    selected.select("query_name", "source_chrom", "source_start", "source_end", "region_label").write_parquet(output / "anchors.parquet")
    species = pl.read_csv(args.species_manifest, separator="\t").filter(pl.col("alignment_name") == args.species)
    assert species.height == 1
    asset = {
        "alignment_name": args.species, "assembly": species["assembly"][0],
        "source_assembly": "hg38", "chain_origin": "Zoonomia-447-2022v1:direction_matched_no_dupes_psl_swap:minScore=-1000000",
        "chain": chain_uri, "chain_sha256": generated["chain_sha256"],
        "source_sizes": objects[source.name]["s3_uri"], "source_sizes_sha256": file_sha256(source),
        "target_sizes": objects[target.name]["s3_uri"], "target_sizes_sha256": file_sha256(target),
    }
    pl.DataFrame([asset]).write_csv(output / "assets.tsv", separator="\t")
    # Samples are issue-owned research inputs, not copies of source datasets.
    owner = f"issues/523/chain-reader-sampled-validation/{args.research_commit}/{args.species}"
    config = {
        "anchors": f"s3://{BUCKET}/{owner}/anchors.parquet",
        "anchors_sha256": file_sha256(output / "anchors.parquet"),
        "species_selected": str(Path(args.species_manifest).resolve()),
        "chain_assets": str(output / "assets.tsv"),
        "table_mem_mb": 4000, "liftover_mem_mb": 24000,
    }
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    metadata = {"research_commit": args.research_commit, "species": args.species,
        "sample_queries": 10_000, "population_queries": frame.height,
        "sample_design": "equal allocation to chromosome x region x direct-mapped strata, then SHA256-ranked fill; not population-weighted",
        "strata": selected.group_by(strata).len().sort(strata).to_dicts(),
        "input_objects": objects, "chain": asset,
        "owner": f"s3://{BUCKET}/{owner}",
        "regional_gate": "not repeated: saved regional raw direct-HAL BEDs not recovered"}
    (output / "sample.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for name in ["anchors.parquet", "sample.input.bed", "sample.direct_hal.bed", "sample.design.parquet", "assets.tsv", "config.yaml", "sample.metadata.json"]:
        s3.upload_file(str(output / name), BUCKET, f"{owner}/{name}")
    print(json.dumps({"species": args.species, "sample_queries": 10_000,
                      "strata": nstrata, "direct_mapped": int(selected["direct_mapped"].sum()),
                      "config": str(output / "config.yaml"), "owner": metadata["owner"]}), flush=True)


def audit(args) -> None:
    sample = Path(args.sample)
    result = Path(args.result)
    names = {r[3] for r in bed_rows(sample / "sample.input.bed")}
    direct, chain = defaultdict(Counter), defaultdict(Counter)
    for path, target in [(sample / "sample.direct_hal.bed", direct), (result / "mapped.bed", chain)]:
        for row in bed_rows(path):
            assert row[3] in names
            target[row[3]][(row[0], int(row[1]), int(row[2]), row[5])] += 1
    unmapped = {r[3] for r in bed_rows(result / "unmapped.bed")}
    assert set(chain).isdisjoint(unmapped) and set(chain) | unmapped == names
    counts = Counter()
    mismatches = []
    for name in sorted(names):
        a, b = direct[name], chain[name]
        kind = ("exact_mapped" if a else "exact_unmapped") if a == b else (
            "chain_only" if not a else "direct_only" if not b else "coordinate_or_multiplicity_conflict")
        counts[kind] += 1
        if a != b:
            mismatches.append({"query_name": name, "category": kind,
                               "direct": [[list(k), v] for k, v in a.items()],
                               "chain": [[list(k), v] for k, v in b.items()]})
    summary = {"input_queries": len(names), **dict(counts),
               "exact_queries": counts["exact_mapped"] + counts["exact_unmapped"],
               "exact_fraction": (counts["exact_mapped"] + counts["exact_unmapped"]) / len(names),
               "sample_metadata_sha256": file_sha256(sample / "sample.metadata.json"),
               "scope": "stratified sample, not genome-wide equivalence or an unbiased genome-wide agreement estimate"}
    (sample / "parity.json").write_text(json.dumps(summary, indent=2) + "\n")
    (sample / "discrepancies.json").write_text(json.dumps(mismatches, indent=2) + "\n")
    metadata = json.loads((sample / "sample.metadata.json").read_text())
    key = metadata["owner"].removeprefix(f"s3://{BUCKET}/")
    s3 = boto3.client("s3")
    for name in ["parity.json", "discrepancies.json"]:
        s3.upload_file(str(sample / name), BUCKET, f"{key}/{name}")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--species", choices=sorted(TARGETS), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--species-manifest", required=True)
    p.add_argument("--research-commit", required=True)
    p = commands.add_parser("audit")
    p.add_argument("--sample", required=True)
    p.add_argument("--result", required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else audit)(args)
