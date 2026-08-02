"""Materialize the two official accessibility-QTL panels for issue #434.

Input positions are VCF-style 1-based coordinates. They are converted exactly
once to 0-based, half-open intervals at the FASTA boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import polars as pl
from huggingface_hub import hf_hub_download
from marin_dna.data.genome import Genome

ISSUE = 434
WINDOW_BP = 255
FOCAL_INDEX = 127
NUCLEOTIDES = frozenset("ACGT")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo_id: str
    revision: str
    split_rows: dict[str, int]
    split_positives: dict[str, int]


DATASETS = (
    DatasetSpec(
        name="caqtl",
        repo_id="marin-dna/evals_caqtl",
        revision="27a24296f50ed55afdc412d1612df680d13138d6",
        split_rows={"train": 38_616, "test": 40_410},
        split_positives={"train": 3_173, "test": 3_648},
    ),
    DatasetSpec(
        name="dsqtl",
        repo_id="marin-dna/evals_dsqtl",
        revision="4a3bf152cd7c28be290adde48a402ec40992cb62",
        split_rows={"train": 15_018, "test": 12_328},
        split_positives={"train": 309, "test": 250},
    ),
)
EXPECTED_ROWS = sum(sum(spec.split_rows.values()) for spec in DATASETS)
EXPECTED_POSITIVES = {
    spec.name: sum(spec.split_positives.values()) for spec in DATASETS
}
SCOPE_FULL = "full"
SCOPE_DSQTL_DIRECTION_PILOT = "dsqtl-positive-direction-pilot"
SCOPES = (SCOPE_FULL, SCOPE_DSQTL_DIRECTION_PILOT)

REQUIRED_COLUMNS = {
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "effect",
    "chrombpnet_atac_ips",
    "chrombpnet_atac_logfc",
    "chrombpnet_dnase_ips",
    "chrombpnet_dnase_logfc",
    "enformer_dnase_local_logfc",
}


class GenomeLike(Protocol):
    def __call__(self, chrom: str, start: int, end: int, strand: str = "+") -> str: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def variant_sequences(reference: str, ref: str, alt: str) -> tuple[str, str]:
    reference = reference.upper()
    ref = ref.upper()
    alt = alt.upper()
    assert len(reference) == WINDOW_BP
    assert set(reference) <= NUCLEOTIDES
    assert len(ref) == len(alt) == 1
    assert ref in NUCLEOTIDES and alt in NUCLEOTIDES and ref != alt
    assert reference[FOCAL_INDEX] == ref
    alternate = reference[:FOCAL_INDEX] + alt + reference[FOCAL_INDEX + 1 :]
    assert len(alternate) == WINDOW_BP and alternate[FOCAL_INDEX] == alt
    assert sum(a != b for a, b in zip(reference, alternate, strict=True)) == 1
    return reference, alternate


def validate_split(frame: pl.DataFrame, spec: DatasetSpec, split: str) -> None:
    assert split in spec.split_rows
    assert REQUIRED_COLUMNS <= set(frame.columns), REQUIRED_COLUMNS - set(frame.columns)
    assert frame.height == spec.split_rows[split]
    assert frame["label"].cast(pl.UInt8).sum() == spec.split_positives[split]
    assert frame.filter(pl.col("pos") < 1).is_empty()
    assert frame.filter(pl.col("ref") == pl.col("alt")).is_empty()
    assert frame.filter(
        ~pl.col("ref").str.to_uppercase().is_in(sorted(NUCLEOTIDES))
        | ~pl.col("alt").str.to_uppercase().is_in(sorted(NUCLEOTIDES))
    ).is_empty()
    assert frame.filter(pl.col("label") & pl.col("effect").is_null()).is_empty()
    assert (
        frame.select(pl.struct("chrom", "pos", "ref", "alt").n_unique()).item()
        == frame.height
    )


def load_official_panel(
    *, scope: str = SCOPE_FULL
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    assert scope in SCOPES
    frames: list[pl.DataFrame] = []
    inputs: list[dict[str, Any]] = []
    for spec in DATASETS:
        if scope == SCOPE_DSQTL_DIRECTION_PILOT and spec.name != "dsqtl":
            continue
        for split in ("train", "test"):
            path = Path(
                hf_hub_download(
                    repo_id=spec.repo_id,
                    repo_type="dataset",
                    revision=spec.revision,
                    filename=f"{split}.parquet",
                )
            )
            frame = pl.read_parquet(path).with_columns(
                pl.col("chrom").cast(pl.String),
                pl.col("ref").cast(pl.String).str.to_uppercase(),
                pl.col("alt").cast(pl.String).str.to_uppercase(),
            )
            validate_split(frame, spec, split)
            frames.append(
                frame.with_columns(
                    pl.lit(spec.name).alias("dataset"),
                    pl.lit(split).alias("official_split"),
                )
            )
            inputs.append(
                {
                    "dataset": spec.name,
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "split": split,
                    "rows": frame.height,
                    "positives": int(frame["label"].cast(pl.UInt8).sum()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    panel = pl.concat(frames, how="diagonal_relaxed")
    if scope == SCOPE_DSQTL_DIRECTION_PILOT:
        panel = panel.filter(pl.col("label"))
        assert panel.height == EXPECTED_POSITIVES["dsqtl"]
        assert panel["dataset"].unique().to_list() == ["dsqtl"]
        assert panel["label"].all()
        assert panel["effect"].is_not_null().all()
    else:
        assert panel.height == EXPECTED_ROWS
    panel = panel.with_row_index("panel_row")
    assert (
        panel.select(
            pl.struct("dataset", "chrom", "pos", "ref", "alt").n_unique()
        ).item()
        == panel.height
    )
    observed = dict(panel.group_by("dataset").agg(pl.col("label").sum()).iter_rows())
    expected_positives = (
        {"dsqtl": EXPECTED_POSITIVES["dsqtl"]}
        if scope == SCOPE_DSQTL_DIRECTION_PILOT
        else EXPECTED_POSITIVES
    )
    assert observed == expected_positives
    assert panel["panel_row"].to_list() == list(range(panel.height))
    return panel, inputs


def materialize_sequences(frame: pl.DataFrame, genome: GenomeLike) -> pl.DataFrame:
    ref_sequences: list[str] = []
    alt_sequences: list[str] = []
    for row in frame.select("chrom", "pos", "ref", "alt").iter_rows(named=True):
        pos0 = int(row["pos"]) - 1
        start = pos0 - FOCAL_INDEX
        end = pos0 + FOCAL_INDEX + 1
        assert start >= 0 and end - start == WINDOW_BP
        reference = genome(str(row["chrom"]), start, end, "+")
        ref_sequence, alt_sequence = variant_sequences(
            reference, str(row["ref"]), str(row["alt"])
        )
        ref_sequences.append(ref_sequence)
        alt_sequences.append(alt_sequence)
    result = frame.with_columns(
        pl.Series("ref_sequence", ref_sequences),
        pl.Series("alt_sequence", alt_sequences),
    )
    assert result.filter(
        (pl.col("ref_sequence").str.len_chars() != WINDOW_BP)
        | (pl.col("alt_sequence").str.len_chars() != WINDOW_BP)
    ).is_empty()
    return result


def build(*, fasta_path: Path, output_dir: Path, scope: str) -> dict[str, Any]:
    assert fasta_path.is_file() and Path(f"{fasta_path}.fai").is_file()
    assert not output_dir.exists()
    assert scope in SCOPES
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert len(experiment_commit) == 40
    assert all(character in "0123456789abcdef" for character in experiment_commit)
    frame, dataset_inputs = load_official_panel(scope=scope)
    chroms = set(frame["chrom"].unique().to_list())
    genome = Genome(fasta_path, subset_chroms=chroms)
    assert set(genome.chroms) == chroms
    frame = materialize_sequences(frame, genome)
    output_dir.mkdir(parents=True)
    panel_path = output_dir / "panel.parquet"
    frame.write_parquet(panel_path, compression="zstd", statistics=True)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", ""),
        "experiment_commit": experiment_commit,
        "scope": scope,
        "coordinate_boundary": "VCF pos1 -> pos0 = pos1 - 1",
        "window_bp": WINDOW_BP,
        "focal_index": FOCAL_INDEX,
        "rows": frame.height,
        "dataset_rows": dict(frame.group_by("dataset").len().iter_rows()),
        "dataset_positives": dict(
            frame.group_by("dataset").agg(pl.col("label").sum()).iter_rows()
        ),
        "datasets": [
            asdict(spec)
            for spec in DATASETS
            if spec.name in set(frame["dataset"].unique().to_list())
        ],
        "dataset_files": dataset_inputs,
        "reference": {
            "path": str(fasta_path),
            "bytes": fasta_path.stat().st_size,
            "sha256": sha256_file(fasta_path),
            "fai_sha256": sha256_file(Path(f"{fasta_path}.fai")),
            "assembly": "Ensembl release 115 GRCh38 soft-masked primary assembly",
        },
        "panel": {
            "path": str(panel_path),
            "bytes": panel_path.stat().st_size,
            "sha256": sha256_file(panel_path),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, default=SCOPE_FULL)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                fasta_path=args.fasta,
                output_dir=args.output_dir,
                scope=args.scope,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
