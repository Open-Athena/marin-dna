"""Build the immutable UCSC hg38 MultiZ 100-way S3 mirror manifest."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

import polars as pl


BASE_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way"
S3_PREFIX = "s3://oa-bolinas/staging/multiz100way/hg38/ucsc-2015-05-12"
PRIMARY_CHROMS = tuple([f"chr{chrom}" for chrom in range(1, 23)] + ["chrX", "chrY"])
METADATA_FILES = (
    "README.txt",
    "hg38.100way.nh",
    "hg38.100way.scientificNames.nh",
    "md5sum.txt",
    "maf/md5sum.txt",
)


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "marin-dna/417"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _content_length(url: str) -> int:
    request = urllib.request.Request(
        url, headers={"User-Agent": "marin-dna/417"}, method="HEAD"
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        value = response.headers.get("Content-Length")
    assert value is not None, f"missing Content-Length for {url}"
    size = int(value)
    assert size > 0
    return size


def _parse_md5(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        checksum, filename = line.split(maxsplit=1)
        result[filename.strip()] = checksum
    return result


def build_manifest() -> pl.DataFrame:
    root_md5_bytes = _download(f"{BASE_URL}/md5sum.txt")
    maf_md5_bytes = _download(f"{BASE_URL}/maf/md5sum.txt")
    root_md5 = _parse_md5(root_md5_bytes.decode())
    maf_md5 = _parse_md5(maf_md5_bytes.decode())

    rows: list[dict[str, object]] = []
    for chrom in PRIMARY_CHROMS:
        filename = f"{chrom}.maf.gz"
        source_url = f"{BASE_URL}/maf/{filename}"
        assert filename in maf_md5, f"missing UCSC MD5 for {filename}"
        rows.append(
            {
                "kind": "primary_chromosome_maf",
                "chrom": chrom,
                "source_url": source_url,
                "s3_uri": f"{S3_PREFIX}/maf/{filename}",
                "byte_size": _content_length(source_url),
                "md5": maf_md5[filename],
            }
        )

    metadata_content = {
        "md5sum.txt": root_md5_bytes,
        "maf/md5sum.txt": maf_md5_bytes,
    }
    for filename in METADATA_FILES:
        source_url = f"{BASE_URL}/{filename}"
        content = metadata_content.get(filename) or _download(source_url)
        expected_md5 = root_md5.get(filename)
        observed_md5 = hashlib.md5(content).hexdigest()
        if expected_md5 is not None:
            assert observed_md5 == expected_md5
        rows.append(
            {
                "kind": "source_metadata",
                "chrom": "",
                "source_url": source_url,
                "s3_uri": f"{S3_PREFIX}/metadata/{filename}",
                "byte_size": len(content),
                "md5": observed_md5,
            }
        )

    result = pl.DataFrame(rows).sort("kind", "chrom", "source_url")
    assert result["source_url"].n_unique() == result.height
    assert result["s3_uri"].n_unique() == result.height
    assert result.filter(pl.col("kind") == "primary_chromosome_maf").height == 24
    assert (result["byte_size"] > 0).all()
    assert (result["md5"].str.len_chars() == 32).all()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(args.output, separator="\t")
    print(f"wrote {manifest.height} mirror objects to {args.output}")


if __name__ == "__main__":
    main()
