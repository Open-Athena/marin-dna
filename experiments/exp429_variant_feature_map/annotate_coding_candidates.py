"""Add transcript/codon annotations to strong issue #429 coding candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import polars as pl

from analyze import sha256_file, write_json
from sample_panel import assert_current_commit

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp428_replicate_codon_context.panel import (
    annotate_candidates,
)

ISSUE = 429
CODING_CLASSES = frozenset({"stop_gained", "synonymous_variant"})
EXPECTED_ROWS_PER_CLASS = 1_024


def summarize_annotations(
    annotated: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Summarize coverage, codon phase, and unambiguous codon pairs."""

    required = {
        "class",
        "is_top",
        "matching_transcript_count",
        "consensus_strand",
        "consensus_codon_position",
        "consensus_ref_codon",
        "consensus_alt_codon",
        "transcript_substitution",
    }
    assert required <= set(annotated.columns)
    groups: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    codons: list[dict[str, Any]] = []
    for class_name in annotated["class"].unique(maintain_order=True):
        class_frame = annotated.filter(pl.col("class") == class_name)
        for subset, frame in (
            ("top", class_frame.filter(pl.col("is_top"))),
            ("remainder", class_frame.filter(~pl.col("is_top"))),
            ("all", class_frame),
        ):
            matching = frame.filter(pl.col("matching_transcript_count") > 0)
            consensus = matching.filter(
                pl.col("consensus_codon_position").is_not_null()
                & pl.col("consensus_ref_codon").is_not_null()
                & pl.col("consensus_alt_codon").is_not_null()
            )
            groups.append(
                {
                    "class": class_name,
                    "subset": subset,
                    "rows": frame.height,
                    "matching_consequence_rows": matching.height,
                    "matching_consequence_fraction": matching.height / frame.height,
                    "unambiguous_codon_rows": consensus.height,
                    "unambiguous_codon_fraction": consensus.height / frame.height,
                }
            )
            position_counts = (
                consensus.group_by("consensus_codon_position")
                .len()
                .sort("consensus_codon_position")
            )
            for row in position_counts.iter_rows(named=True):
                positions.append(
                    {
                        "class": class_name,
                        "subset": subset,
                        "codon_position": int(row["consensus_codon_position"]),
                        "count": int(row["len"]),
                        "total_unambiguous": consensus.height,
                        "fraction": row["len"] / consensus.height,
                    }
                )
            codon_counts = (
                consensus.group_by(
                    "consensus_ref_codon",
                    "consensus_alt_codon",
                    "transcript_substitution",
                )
                .len()
                .sort("len", descending=True)
            )
            for row in codon_counts.iter_rows(named=True):
                codons.append(
                    {
                        "class": class_name,
                        "subset": subset,
                        "ref_codon": row["consensus_ref_codon"],
                        "alt_codon": row["consensus_alt_codon"],
                        "transcript_substitution": row["transcript_substitution"],
                        "count": int(row["len"]),
                        "total_unambiguous": consensus.height,
                        "fraction": row["len"] / consensus.height,
                    }
                )
    return (
        pl.DataFrame(groups).sort(["class", "subset"]),
        pl.DataFrame(positions).sort(["class", "subset", "codon_position"]),
        pl.DataFrame(codons).sort(
            ["class", "subset", "count"], descending=[False, False, True]
        ),
    )


def annotate_coding_candidates(
    *,
    contexts_path: Path,
    inspection_manifest_path: Path,
    gtf_path: Path,
    fasta_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Reuse the tested #428 transcript reconstruction on fixed #429 candidates."""

    assert not output_dir.exists()
    annotation_commit = os.environ.get("CODING_ANNOTATION_COMMIT", "")
    assert_current_commit(annotation_commit)
    inspection_manifest = json.loads(inspection_manifest_path.read_text())
    assert inspection_manifest["artifacts"][contexts_path.name][
        "sha256"
    ] == sha256_file(contexts_path)
    contexts = pl.read_parquet(contexts_path).filter(
        pl.col("class").is_in(sorted(CODING_CLASSES))
    )
    assert contexts.height == len(CODING_CLASSES) * EXPECTED_ROWS_PER_CLASS
    assert contexts.select("panel_row").n_unique() == contexts.height
    annotated, metadata = annotate_candidates(
        contexts, gtf_path=gtf_path, fasta_path=fasta_path
    )
    assert annotated.height == contexts.height
    summaries, positions, codons = summarize_annotations(annotated)

    output_dir.mkdir(parents=True, exist_ok=False)
    annotated.write_parquet(output_dir / "coding_candidate_annotations.parquet")
    summaries.write_parquet(output_dir / "coding_candidate_summary.parquet")
    positions.write_parquet(output_dir / "coding_candidate_codon_positions.parquet")
    codons.write_parquet(output_dir / "coding_candidate_codons.parquet")
    summary = {
        "issue": ISSUE,
        "coding_annotation_commit": annotation_commit,
        "candidate_inspection_manifest_sha256": sha256_file(inspection_manifest_path),
        "contexts_sha256": sha256_file(contexts_path),
        "gtf": {"path": str(gtf_path), "sha256": sha256_file(gtf_path)},
        "fasta": {"path": str(fasta_path), "sha256": sha256_file(fasta_path)},
        "protocol": {
            "classes": sorted(CODING_CLASSES),
            "rows": contexts.height,
            "annotation": "reuse commit-pinned exp428 Ensembl-109 transcript/codon reconstruction",
            "sequence_selection": "discovery-only candidates frozen by validation AP in candidate inspection",
        },
        "annotation_metadata": metadata,
    }
    write_json(output_dir / "results.json", summary)
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {**summary, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--inspection-manifest", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = annotate_coding_candidates(
        contexts_path=args.contexts,
        inspection_manifest_path=args.inspection_manifest,
        gtf_path=args.gtf,
        fasta_path=args.fasta,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
