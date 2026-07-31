"""Build the frozen balanced chr21 panel for issue 422.

The source dataset uses 1-based GRCh38 positions. This sampler does not convert
coordinates; conversion to 0-based coordinates happens later at the FASTA boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

HF_REPOSITORY = "songlab/hg38-variant-consequences"
HF_REVISION = "eb3022cc6797b9369cca16af72ff3c4197df343a"
CHROMOSOME = "21"
SOURCE_FILENAME = "21.parquet"
SOURCE_SHA256 = "3be2188e6d9555058061710d58b94ccff2e7294cb39b53067b2531a2d7925347"

BLOCK_SIZE = 1_000_000
HASH_SEEDS = (422, 288, 21, 5)
OVERSAMPLE_FACTOR = 8
QUOTAS: dict[str, int] = {
    "discovery": 256,
    "validation": 128,
    "test": 128,
}
EXPECTED_EXCLUDED_CLASSES = {
    "coding_sequence_variant",
    "incomplete_terminal_codon_variant",
    "stop_retained_variant",
}
EXPECTED_RETAINED_CLASS_COUNT = 35

SOURCE_COLUMNS = [
    "chrom",
    "pos",
    "ref",
    "alt",
    "consequence",
    "consequence_cre",
]
KEY_COLUMNS = ["chrom", "pos", "ref", "alt"]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for_position(pos: int) -> tuple[int, str]:
    """Return the 1 Mb block and deterministic split for a 1-based position."""

    assert pos >= 1
    block = (pos - 1) // BLOCK_SIZE
    fold = block % 5
    if fold <= 2:
        split = "discovery"
    elif fold == 3:
        split = "validation"
    else:
        split = "test"
    return block, split


def _with_split(frame: pl.LazyFrame) -> pl.LazyFrame:
    block = ((pl.col("pos") - 1) // BLOCK_SIZE).alias("block_id")
    return frame.with_columns(block).with_columns(
        pl.when((pl.col("block_id") % 5) <= 2)
        .then(pl.lit("discovery"))
        .when((pl.col("block_id") % 5) == 3)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )


def _validate_source_schema(schema: pl.Schema) -> None:
    missing = set(SOURCE_COLUMNS) - set(schema.names())
    assert not missing, f"source missing columns: {sorted(missing)}"
    assert schema["pos"].is_integer(), schema["pos"]
    for column in ("chrom", "ref", "alt", "consequence", "consequence_cre"):
        assert schema[column] == pl.String, (column, schema[column])


def _class_split_counts(frame: pl.LazyFrame) -> pl.DataFrame:
    return (
        _with_split(frame)
        .group_by(["consequence_cre", "split"])
        .len(name="n_available")
        .collect(engine="streaming")
    )


def _sampling_plan(
    counts: pl.DataFrame,
    quotas: dict[str, int],
) -> tuple[pl.DataFrame, list[str], list[str]]:
    assert set(counts.columns) == {"consequence_cre", "split", "n_available"}
    assert set(quotas) == {"discovery", "validation", "test"}
    assert all(quota > 0 for quota in quotas.values())

    classes = sorted(counts["consequence_cre"].unique().to_list())
    available = {
        (row["consequence_cre"], row["split"]): int(row["n_available"])
        for row in counts.iter_rows(named=True)
    }
    retained = [
        consequence
        for consequence in classes
        if all(
            available.get((consequence, split), 0) >= quota
            for split, quota in quotas.items()
        )
    ]
    excluded = sorted(set(classes) - set(retained))
    rows = [
        {
            "consequence_cre": consequence,
            "split": split,
            "n_available": available[(consequence, split)],
            "quota": quota,
        }
        for consequence in retained
        for split, quota in quotas.items()
    ]
    plan = pl.DataFrame(rows).with_columns(
        pl.col("n_available").cast(pl.UInt64),
        pl.col("quota").cast(pl.UInt64),
    )
    return plan, retained, excluded


def sample_balanced_panel(
    frame: pl.LazyFrame,
    *,
    quotas: dict[str, int] = QUOTAS,
    oversample_factor: int = OVERSAMPLE_FACTOR,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str], list[str]]:
    """Return an exact deterministic balanced panel and its availability counts.

    A deterministic Polars struct hash creates a small oversampled candidate set
    in one streaming scan. Exact per-class/split quotas are then selected by hash
    rank with genomic keys as tie breakers. The persisted panel, rather than the
    hash implementation, is the stable input for comparisons across SAE versions.
    """

    assert oversample_factor >= 2
    _validate_source_schema(frame.collect_schema())
    counts = _class_split_counts(frame)
    plan, retained, excluded = _sampling_plan(counts, quotas)
    assert retained, "no consequence_cre classes meet all split quotas"

    ranked = _with_split(frame.select(SOURCE_COLUMNS)).with_columns(
        pl.struct(KEY_COLUMNS).hash(*HASH_SEEDS).alias("sample_hash")
    )
    candidates = (
        ranked.join(plan.lazy(), on=["consequence_cre", "split"], how="inner")
        .filter(
            (pl.col("sample_hash") % pl.col("n_available"))
            < pl.min_horizontal(
                pl.col("n_available"),
                pl.col("quota") * oversample_factor,
            )
        )
        .select(
            SOURCE_COLUMNS
            + ["block_id", "split", "sample_hash", "n_available", "quota"]
        )
        .collect(engine="streaming")
    )

    selected_parts: list[pl.DataFrame] = []
    for split, quota in quotas.items():
        split_candidates = candidates.filter(pl.col("split") == split)
        candidate_counts = split_candidates.group_by("consequence_cre").len()
        too_small = candidate_counts.filter(pl.col("len") < quota)
        assert too_small.is_empty(), (
            "deterministic prefilter returned fewer candidates than quota; "
            f"increase oversample_factor: {too_small.to_dicts()}"
        )
        selected_parts.append(
            split_candidates.sort(
                ["consequence_cre", "sample_hash", *KEY_COLUMNS]
            )
            .group_by("consequence_cre", maintain_order=True)
            .head(quota)
        )

    panel = (
        pl.concat(selected_parts)
        .sort(["consequence_cre", "split", "sample_hash", *KEY_COLUMNS])
        .with_row_index("panel_row")
        .select(
            [
                "panel_row",
                *SOURCE_COLUMNS,
                "block_id",
                "split",
                "sample_hash",
            ]
        )
    )
    expected_rows = len(retained) * sum(quotas.values())
    assert panel.height == expected_rows, (panel.height, expected_rows)
    assert panel.select(pl.struct(KEY_COLUMNS).n_unique()).item() == panel.height
    assert panel["chrom"].unique().to_list() == [CHROMOSOME]
    assert panel.filter(pl.col("pos") < 1).is_empty()
    assert panel.filter(pl.col("ref") == pl.col("alt")).is_empty()
    assert panel.filter(
        ~pl.col("ref").is_in(["A", "C", "G", "T"])
        | ~pl.col("alt").is_in(["A", "C", "G", "T"])
    ).is_empty()

    observed = panel.group_by(["consequence_cre", "split"]).len()
    expected = plan.select("consequence_cre", "split", pl.col("quota").alias("len"))
    assert observed.sort(["consequence_cre", "split"]).equals(
        expected.sort(["consequence_cre", "split"])
    )
    return panel, counts, retained, excluded


def build_panel(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Validate the pinned source, write the sampled panel, and return a manifest."""

    assert input_path.is_file(), input_path
    source_sha256 = sha256_file(input_path)
    assert source_sha256 == SOURCE_SHA256, (source_sha256, SOURCE_SHA256)

    panel, counts, retained, excluded = sample_balanced_panel(
        pl.scan_parquet(input_path)
    )
    assert len(retained) == EXPECTED_RETAINED_CLASS_COUNT, len(retained)
    assert set(excluded) == EXPECTED_EXCLUDED_CLASSES, excluded

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(output_path)
    output_sha256 = sha256_file(output_path)

    availability = (
        counts.filter(pl.col("consequence_cre").is_in(retained))
        .sort(["consequence_cre", "split"])
        .to_dicts()
    )
    manifest: dict[str, Any] = {
        "source": {
            "hf_repository": HF_REPOSITORY,
            "hf_revision": HF_REVISION,
            "filename": SOURCE_FILENAME,
            "sha256": source_sha256,
        },
        "sampling": {
            "chromosome": CHROMOSOME,
            "coordinate_system": "1-based VCF-style; convert only at FASTA boundary",
            "block_size": BLOCK_SIZE,
            "split_rule": "((pos - 1) // 1000000) % 5: 0-2 discovery, 3 validation, 4 test",
            "hash": "polars struct hash over chrom,pos,ref,alt",
            "hash_seeds": list(HASH_SEEDS),
            "oversample_factor": OVERSAMPLE_FACTOR,
            "quotas": QUOTAS,
            "retained_classes": retained,
            "excluded_classes": excluded,
            "availability": availability,
            "polars_version": pl.__version__,
        },
        "output": {
            "path": str(output_path),
            "rows": panel.height,
            "classes": len(retained),
            "sha256": output_sha256,
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_panel(args.input, args.output)
    print(json.dumps(manifest["output"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
