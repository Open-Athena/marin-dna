"""Pin and prepare the expanded panel without accessing validation labels."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

import boto3
import py2bit
from boto3.s3.transfer import TransferConfig

from window_conservation.local_run import extract
from window_conservation.prepare import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    target = root / "extension"
    data = target / "data"
    data.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path("config/extension.json").read_text())
    receipt_path = data / "manifest.json"
    receipt = (
        json.loads(receipt_path.read_text())
        if receipt_path.exists()
        else {"sources": [], "annotations": []}
    )
    client = boto3.client("s3", region_name="us-east-2")
    transfer = TransferConfig(use_threads=False)
    paths = (root / "data/genomes.list").read_text().splitlines()
    for source in protocol["additional_species_in_order"]:
        name = source["name"]
        fasta = data / f"{name}.fa"
        paths.append(str(fasta))
        previous = next((r for r in receipt["sources"] if r["name"] == name), None)
        if previous:
            assert fasta.stat().st_size == previous["fasta_bytes"]
            continue
        # Refuse a download if it would consume the workspace reserve.
        assert shutil.disk_usage(data).free > source["bytes"] * 5 + 6 * 2**30
        bucket, key = source["uri"].removeprefix("s3://").split("/", 1)
        head = client.head_object(
            Bucket=bucket, Key=key, ExpectedBucketOwner="836683583872"
        )
        assert (
            head["ContentLength"] == source["bytes"]
            and head["ETag"].strip('"') == source["etag"]
        )
        twobit = data / f"{name}.2bit"
        client.download_file(bucket, key, str(twobit), Config=transfer)
        assert twobit.stat().st_size == source["bytes"]
        genome = py2bit.open(str(twobit), True)
        chroms = genome.chroms()
        with fasta.open("w") as out:
            for chrom, length in chroms.items():
                out.write(f">{chrom}\n")
                for start in range(0, length, 2**20):
                    end = min(length, start + 2**20)
                    sequence = genome.sequence(chrom, start, end)
                    assert len(sequence) == end - start
                    out.write(sequence + "\n")
        genome.close()
        receipt["sources"].append(
            {
                **source,
                "sha256": sha256(twobit),
                "fasta_sha256": sha256(fasta),
                "fasta_bytes": fasta.stat().st_size,
                "bases": sum(chroms.values()),
                "contigs": len(chroms),
            }
        )
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        twobit.unlink()  # Re-downloadable upstream input; checksums remain.
        print("prepared", name, sum(chroms.values()), flush=True)
    (data / "genomes.list").write_text("\n".join(paths) + "\n")
    extract(root / "data/human.fa", data / "query.fa", {"chr1", "chr3"})
    prefix = "staging/vertebrate_projection_dataset/v1/06549d8f7f3ba76151b9c54a5e52d3e3f4402a2d/full/anchors/"
    for filename in ["Homo_sapiens.GRCh38.115.gtf.gz", "ccre.bare.parquet"]:
        path = data / filename
        head = client.head_object(
            Bucket="oa-bolinas",
            Key=prefix + filename,
            ExpectedBucketOwner="836683583872",
        )
        if not path.exists():
            client.download_file(
                "oa-bolinas", prefix + filename, str(path), Config=transfer
            )
        assert path.stat().st_size == head["ContentLength"]
        if not any(r["name"] == filename for r in receipt["annotations"]):
            receipt["annotations"].append(
                {
                    "name": filename,
                    "uri": "s3://oa-bolinas/" + prefix + filename,
                    "etag": head["ETag"],
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    path = data / "rmsk.txt.gz"
    url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz"
    if not path.exists():
        temporary = path.with_suffix(".part")
        with urlopen(url, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, 2**20)
        temporary.rename(path)
    if not any(r["name"] == path.name for r in receipt["annotations"]):
        receipt["annotations"].append(
            {
                "name": path.name,
                "uri": url,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    receipt["query_sha256"] = sha256(data / "query.fa")
    receipt["pilot_input_manifest_sha256"] = sha256(root / "data/manifest.json")
    receipt["annotation_coordinates"] = (
        "GTF 1-based closed converted at parse; cCRE bare chromosomes explicitly mapped 1/2/3 to chr1/chr2/chr3; RepeatMasker UCSC coordinates retained. All internal intervals 0-based half-open."
    )
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("extension inputs ready; no chr3 conservation labels read", flush=True)


if __name__ == "__main__":
    main()
