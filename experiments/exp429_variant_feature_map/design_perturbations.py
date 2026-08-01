"""Design matched splice-motif and coding-codon perturbations for issue #429."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from marin_dna.data.dna import reverse_complement
from marin_dna.data.genome import Genome

from analyze import sha256_file, write_json
from sample_panel import assert_current_commit

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp428_replicate_codon_context.panel import (  # noqa: E402
    COMPLEMENT,
    GENETIC_CODE,
)

ISSUE = 429
WINDOW_BP = 255
FOCAL_INDEX = 127
CONTEXTS_PER_GROUP = 16
SPLICE_POSITIONS = tuple(range(-12, 5))
NUCLEOTIDES = tuple("ACGT")
CODONS = tuple("".join(value) for value in itertools.product(NUCLEOTIDES, repeat=3))
SPLICE_FEATURES = {
    "splice_acceptor_variant": 11698,
    "splice_donor_5th_base_variant": 11681,
}
CODING_FEATURES = {"stop_gained": 3312, "synonymous_variant": 6072}

assert len(CODONS) == 64 and set(CODONS) == set(GENETIC_CODE)


def perturbation_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Construct a mixed splice/coding frame without prefix-biased inference."""

    assert rows
    return pl.DataFrame(rows, infer_schema_length=None).with_row_index(
        "perturbation_row"
    )


def select_contexts(
    frame: pl.DataFrame, *, class_name: str, contexts_per_group: int
) -> pl.DataFrame:
    """Select strongest contexts and deterministic rank-spaced controls."""

    assert contexts_per_group > 0
    class_frame = frame.filter(pl.col("class") == class_name).sort("rank")
    top = class_frame.filter(pl.col("is_top")).head(contexts_per_group)
    remainder = class_frame.filter(~pl.col("is_top")).sort("rank")
    assert top.height == contexts_per_group
    assert remainder.height >= contexts_per_group
    control_indices = np.linspace(
        0, remainder.height - 1, contexts_per_group, dtype=np.int64
    )
    assert len(set(control_indices.tolist())) == contexts_per_group
    controls = (
        remainder.with_row_index("_control_index")
        .filter(pl.col("_control_index").is_in(control_indices.tolist()))
        .drop("_control_index")
    )
    assert controls.height == contexts_per_group
    return pl.concat(
        (
            top.with_columns(pl.lit("top").alias("context_group")),
            controls.with_columns(pl.lit("rank_spaced_control").alias("context_group")),
        ),
        how="vertical",
    ).sort(["context_group", "rank"])


def reference_window(genome: Genome, row: Mapping[str, Any]) -> tuple[str, int, int]:
    """Load and validate a forward genomic window around a 1-based variant."""

    pos0 = int(row["pos"]) - 1
    start0 = pos0 - FOCAL_INDEX
    end0 = start0 + WINDOW_BP
    assert start0 >= 0 and end0 - start0 == WINDOW_BP
    sequence = genome(str(row["chrom"]), start0, end0, "+").upper()
    assert len(sequence) == WINDOW_BP and set(sequence) <= set(NUCLEOTIDES)
    assert sequence[FOCAL_INDEX] == row["ref"]
    return sequence, start0, end0


def oriented_base(base: str, orientation: str) -> str:
    """Express one genomic-forward base in a declared response orientation."""

    assert base in NUCLEOTIDES
    assert orientation in {"forward", "reverse_complement"}
    return base if orientation == "forward" else base.translate(COMPLEMENT)


def splice_saturation_rows(
    source: Mapping[str, Any],
    reference_sequence: str,
    *,
    window_start0: int,
    window_end0: int,
    relative_positions: Iterable[int],
) -> list[dict[str, Any]]:
    """Saturate positions in the feature's selected response orientation."""

    orientation = str(source["response_orientation"])
    assert orientation in {"forward", "reverse_complement"}
    oriented_context = reference_sequence[FOCAL_INDEX - 15 : FOCAL_INDEX + 16]
    if orientation == "reverse_complement":
        oriented_context = reverse_complement(oriented_context)
    assert oriented_context == source["ref_context"]

    rows: list[dict[str, Any]] = []
    for relative_position in relative_positions:
        assert -FOCAL_INDEX <= relative_position <= FOCAL_INDEX
        genomic_offset = (
            relative_position if orientation == "forward" else -relative_position
        )
        sequence_index = FOCAL_INDEX + genomic_offset
        genomic_source = reference_sequence[sequence_index]
        response_source = oriented_base(genomic_source, orientation)
        for response_alternate in NUCLEOTIDES:
            if response_alternate == response_source:
                continue
            genomic_alternate = oriented_base(response_alternate, orientation)
            alternate_sequence = (
                reference_sequence[:sequence_index]
                + genomic_alternate
                + reference_sequence[sequence_index + 1 :]
            )
            assert (
                sum(
                    ref != alt
                    for ref, alt in zip(
                        reference_sequence, alternate_sequence, strict=True
                    )
                )
                == 1
            )
            rows.append(
                {
                    "perturbation_type": "splice_saturation",
                    "class": source["class"],
                    "feature_id": int(source["feature_id"]),
                    "source_panel_row": int(source["panel_row"]),
                    "source_rank": int(source["rank"]),
                    "context_group": source["context_group"],
                    "designed_orientation": orientation,
                    "chrom": source["chrom"],
                    "source_pos1": int(source["pos"]),
                    "window_start0": window_start0,
                    "window_end0": window_end0,
                    "relative_position": int(relative_position),
                    "source_state": response_source,
                    "alternate_state": response_alternate,
                    "reference_codon": None,
                    "alternate_codon": None,
                    "reference_amino_acid": None,
                    "alternate_amino_acid": None,
                    "expected_consequence": "single_base_saturation",
                    "edit_distance": 1,
                    "reference_sequence": reference_sequence,
                    "alternate_sequence": alternate_sequence,
                }
            )
    return rows


def coding_codon_start_index(codon_position: int, strand: str) -> int:
    """Return the genomic-forward window index of a transcript codon's left edge."""

    assert codon_position in {1, 2, 3} and strand in {"+", "-"}
    if strand == "+":
        return FOCAL_INDEX - (codon_position - 1)
    return FOCAL_INDEX - (3 - codon_position)


def codon_consequence(reference_codon: str, alternate_codon: str) -> str:
    """Classify a designed full-codon replacement relative to its source codon."""

    reference_amino_acid = GENETIC_CODE[reference_codon]
    alternate_amino_acid = GENETIC_CODE[alternate_codon]
    assert reference_amino_acid != "*"
    if alternate_amino_acid == "*":
        return "stop_gained"
    if alternate_amino_acid == reference_amino_acid:
        return "synonymous_variant"
    return "missense_variant"


def codon_sweep_rows(
    source: Mapping[str, Any],
    reference_sequence: str,
    *,
    window_start0: int,
    window_end0: int,
) -> list[dict[str, Any]]:
    """Replace one transcript-oriented codon with every other codon."""

    strand = str(source["consensus_strand"])
    codon_position = int(source["consensus_codon_position"])
    codon_start = coding_codon_start_index(codon_position, strand)
    genomic_reference_codon = reference_sequence[codon_start : codon_start + 3]
    transcript_reference_codon = (
        genomic_reference_codon
        if strand == "+"
        else reverse_complement(genomic_reference_codon)
    )
    assert transcript_reference_codon == source["consensus_ref_codon"]
    assert transcript_reference_codon[codon_position - 1] == oriented_base(
        str(source["ref"]),
        "forward" if strand == "+" else "reverse_complement",
    )
    reference_amino_acid = GENETIC_CODE[transcript_reference_codon]
    assert reference_amino_acid != "*"

    rows: list[dict[str, Any]] = []
    for alternate_codon in CODONS:
        if alternate_codon == transcript_reference_codon:
            continue
        genomic_alternate_codon = (
            alternate_codon if strand == "+" else reverse_complement(alternate_codon)
        )
        alternate_sequence = (
            reference_sequence[:codon_start]
            + genomic_alternate_codon
            + reference_sequence[codon_start + 3 :]
        )
        edit_distance = sum(
            ref != alt
            for ref, alt in zip(reference_sequence, alternate_sequence, strict=True)
        )
        assert edit_distance == sum(
            ref != alt
            for ref, alt in zip(
                transcript_reference_codon, alternate_codon, strict=True
            )
        )
        assert 1 <= edit_distance <= 3
        rows.append(
            {
                "perturbation_type": "codon_sweep",
                "class": source["class"],
                "feature_id": int(source["feature_id"]),
                "source_panel_row": int(source["panel_row"]),
                "source_rank": int(source["rank"]),
                "context_group": source["context_group"],
                "designed_orientation": (
                    "transcript_forward" if strand == "+" else "transcript_reverse"
                ),
                "chrom": source["chrom"],
                "source_pos1": int(source["pos"]),
                "window_start0": window_start0,
                "window_end0": window_end0,
                "relative_position": None,
                "source_state": transcript_reference_codon,
                "alternate_state": alternate_codon,
                "reference_codon": transcript_reference_codon,
                "alternate_codon": alternate_codon,
                "reference_amino_acid": reference_amino_acid,
                "alternate_amino_acid": GENETIC_CODE[alternate_codon],
                "expected_consequence": codon_consequence(
                    transcript_reference_codon, alternate_codon
                ),
                "edit_distance": edit_distance,
                "reference_sequence": reference_sequence,
                "alternate_sequence": alternate_sequence,
            }
        )
    assert len(rows) == len(CODONS) - 1
    return rows


def design_perturbations(
    *,
    candidate_contexts_path: Path,
    inspection_manifest_path: Path,
    coding_annotations_path: Path,
    coding_manifest_path: Path,
    fasta_path: Path,
    output_dir: Path,
    contexts_per_group: int,
) -> dict[str, Any]:
    """Build a commit-pinned matched perturbation panel from discovery contexts."""

    assert contexts_per_group > 0 and not output_dir.exists()
    design_commit = os.environ.get("PERTURBATION_DESIGN_COMMIT", "")
    assert_current_commit(design_commit)
    inspection_manifest = json.loads(inspection_manifest_path.read_text())
    coding_manifest = json.loads(coding_manifest_path.read_text())
    assert inspection_manifest["artifacts"][candidate_contexts_path.name][
        "sha256"
    ] == sha256_file(candidate_contexts_path)
    assert coding_manifest["artifacts"][coding_annotations_path.name][
        "sha256"
    ] == sha256_file(coding_annotations_path)
    assert coding_manifest["candidate_inspection_manifest_sha256"] == sha256_file(
        inspection_manifest_path
    )
    assert fasta_path.is_file() and Path(f"{fasta_path}.fai").is_file()
    assert Path(f"{fasta_path}.gzi").is_file()

    candidate_contexts = pl.read_parquet(candidate_contexts_path)
    coding_annotations = pl.read_parquet(coding_annotations_path)
    assert candidate_contexts.height == 4 * 1_024
    assert coding_annotations.height == 2 * 1_024
    assert set(candidate_contexts["class"].unique()) == (
        set(SPLICE_FEATURES) | set(CODING_FEATURES)
    )
    assert set(coding_annotations["class"].unique()) == set(CODING_FEATURES)
    genome = Genome(fasta_path, subset_chroms={"21"})
    assert set(genome.chroms) == {"21"}

    rows: list[dict[str, Any]] = []
    for class_name, feature_id in SPLICE_FEATURES.items():
        selected = select_contexts(
            candidate_contexts,
            class_name=class_name,
            contexts_per_group=contexts_per_group,
        )
        assert selected["feature_id"].unique().to_list() == [feature_id]
        for source in selected.iter_rows(named=True):
            sequence, start0, end0 = reference_window(genome, source)
            rows.extend(
                splice_saturation_rows(
                    source,
                    sequence,
                    window_start0=start0,
                    window_end0=end0,
                    relative_positions=SPLICE_POSITIONS,
                )
            )
    for class_name, feature_id in CODING_FEATURES.items():
        selected = select_contexts(
            coding_annotations,
            class_name=class_name,
            contexts_per_group=contexts_per_group,
        )
        assert selected["feature_id"].unique().to_list() == [feature_id]
        for source in selected.iter_rows(named=True):
            sequence, start0, end0 = reference_window(genome, source)
            rows.extend(
                codon_sweep_rows(
                    source,
                    sequence,
                    window_start0=start0,
                    window_end0=end0,
                )
            )

    panel = perturbation_frame(rows)
    expected_splice_rows = (
        len(SPLICE_FEATURES) * 2 * contexts_per_group * len(SPLICE_POSITIONS) * 3
    )
    expected_coding_rows = (
        len(CODING_FEATURES) * 2 * contexts_per_group * (len(CODONS) - 1)
    )
    assert panel.height == expected_splice_rows + expected_coding_rows
    assert panel["perturbation_row"].to_list() == list(range(panel.height))
    assert panel.filter(pl.col("edit_distance") < 1).is_empty()
    assert panel.filter(pl.col("edit_distance") > 3).is_empty()
    assert set(panel["context_group"].unique()) == {
        "top",
        "rank_spaced_control",
    }
    counts = (
        panel.group_by("perturbation_type", "class", "context_group")
        .len()
        .sort(["perturbation_type", "class", "context_group"])
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    panel_path = output_dir / "perturbation_panel.parquet"
    panel.write_parquet(panel_path)
    result = {
        "issue": ISSUE,
        "design_commit": design_commit,
        "sources": {
            "candidate_contexts": {
                "path": str(candidate_contexts_path),
                "sha256": sha256_file(candidate_contexts_path),
            },
            "inspection_manifest": {
                "path": str(inspection_manifest_path),
                "sha256": sha256_file(inspection_manifest_path),
            },
            "coding_annotations": {
                "path": str(coding_annotations_path),
                "sha256": sha256_file(coding_annotations_path),
            },
            "coding_manifest": {
                "path": str(coding_manifest_path),
                "sha256": sha256_file(coding_manifest_path),
            },
            "fasta": {"path": str(fasta_path), "sha256": sha256_file(fasta_path)},
        },
        "protocol": {
            "contexts_per_group": contexts_per_group,
            "context_groups": ["top", "rank_spaced_control"],
            "splice_positions": list(SPLICE_POSITIONS),
            "splice_alternates_per_position": 3,
            "coding_states": len(CODONS) - 1,
            "coding_design": "replace the transcript-oriented source codon with every other genetic-code codon",
            "coordinate_system": "source pos is 1-based; window_start0/window_end0 are 0-based half-open",
        },
        "counts": counts.to_dicts(),
        "rows": panel.height,
    }
    write_json(output_dir / "results.json", result)
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {**result, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-contexts", type=Path, required=True)
    parser.add_argument("--inspection-manifest", type=Path, required=True)
    parser.add_argument("--coding-annotations", type=Path, required=True)
    parser.add_argument("--coding-manifest", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contexts-per-group", type=int, default=CONTEXTS_PER_GROUP)
    args = parser.parse_args()
    manifest = design_perturbations(
        candidate_contexts_path=args.candidate_contexts,
        inspection_manifest_path=args.inspection_manifest,
        coding_annotations_path=args.coding_annotations,
        coding_manifest_path=args.coding_manifest,
        fasta_path=args.fasta,
        output_dir=args.output_dir,
        contexts_per_group=args.contexts_per_group,
    )
    print(json.dumps(manifest["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
