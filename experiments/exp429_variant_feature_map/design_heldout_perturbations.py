"""Design a hash-sampled causal replication panel on untouched issue #429 test blocks.

The first perturbation panel used discovery-derived feature-responsive contexts.
This replication instead selects source variants by the frozen panel hash alone,
after requiring an unambiguous Ensembl transcript orientation.  GTF coordinates
are converted from 1-based closed to 0-based half-open at the parser boundary.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import polars as pl
from marin_dna.data.dna import reverse_complement
from marin_dna.data.genome import Genome

from analyze import sha256_file, write_json
from design_perturbations import (
    CODING_FEATURES,
    FOCAL_INDEX,
    SPLICE_FEATURES,
    SPLICE_POSITIONS,
    coding_codon_start_index,
    codon_consequence,
    perturbation_frame,
    reference_window,
    splice_saturation_rows,
)
from sample_panel import assert_current_commit

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp428_replicate_codon_context.panel import (  # noqa: E402
    GENETIC_CODE,
    annotate_candidates,
)

ISSUE = 429
SOURCE_SPLIT = "test"
CONTEXT_GROUP = "untouched_test_hash"
CONTEXTS_PER_CLASS = 64
TARGET_CLASSES = tuple(SPLICE_FEATURES) + tuple(CODING_FEATURES)
NUCLEOTIDES = tuple("ACGT")


@dataclass(frozen=True, order=True)
class Intron:
    """One 0-based, half-open protein-coding intron."""

    chrom: str
    start0: int
    end0: int
    strand: Literal["+", "-"]

    def __post_init__(self) -> None:
        assert self.start0 >= 0 and self.end0 > self.start0
        assert self.strand in {"+", "-"}


def parse_gtf_attributes(value: str) -> dict[str, str]:
    """Parse the simple key/value grammar used by Ensembl GTF attributes."""

    attributes: dict[str, str] = {}
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        key, raw_value = item.split(" ", 1)
        attributes[key] = raw_value.strip().strip('"')
    return attributes


def load_protein_coding_introns(gtf_path: Path, *, chrom: str) -> list[Intron]:
    """Load unique protein-coding introns from an Ensembl GRCh38 GTF."""

    assert gtf_path.is_file()
    transcript_exons: dict[str, tuple[str, str, list[tuple[int, int]]]] = {}
    saw_grch38 = False
    with gzip.open(gtf_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                saw_grch38 |= "genome-build GRCh38" in line
                continue
            fields = line.rstrip("\n").split("\t")
            assert len(fields) == 9
            row_chrom, _, feature, start1, end1, _, strand, _, attrs = fields
            if row_chrom != chrom or feature != "exon":
                continue
            attributes = parse_gtf_attributes(attrs)
            if attributes.get("transcript_biotype") != "protein_coding":
                continue
            assert strand in {"+", "-"}
            transcript_id = attributes["transcript_id"]
            start0 = int(start1) - 1
            end0 = int(end1)
            assert 0 <= start0 < end0
            if transcript_id not in transcript_exons:
                transcript_exons[transcript_id] = (row_chrom, strand, [])
            tx_chrom, tx_strand, exons = transcript_exons[transcript_id]
            assert tx_chrom == row_chrom and tx_strand == strand
            exons.append((start0, end0))
    assert saw_grch38 and transcript_exons

    introns: set[Intron] = set()
    for tx_chrom, strand, exons in transcript_exons.values():
        for left, right in pairwise(sorted(set(exons))):
            if right[0] > left[1]:
                introns.add(Intron(tx_chrom, left[1], right[0], strand))
    assert introns
    return sorted(introns)


def splice_class_positions(intron: Intron, class_name: str) -> tuple[int, ...]:
    """Return 0-based reference positions assigned to one VEP splice class."""

    if class_name == "splice_acceptor_variant":
        if intron.strand == "+":
            return (intron.end0 - 2, intron.end0 - 1)
        return (intron.start0, intron.start0 + 1)
    if class_name == "splice_donor_5th_base_variant":
        if intron.strand == "+":
            return (intron.start0 + 4,)
        return (intron.end0 - 5,)
    raise AssertionError(class_name)


def splice_strand_index(
    introns: list[Intron], *, class_name: str
) -> dict[int, set[str]]:
    """Map eligible reference positions to all matching transcript strands."""

    assert class_name in SPLICE_FEATURES
    strands: dict[int, set[str]] = defaultdict(set)
    for intron in introns:
        for position0 in splice_class_positions(intron, class_name):
            strands[position0].add(intron.strand)
    return dict(strands)


def deterministic_sources(
    frame: pl.DataFrame, *, contexts_per_class: int
) -> pl.DataFrame:
    """Select an exact hash-ranked sample independently of SAE activations."""

    assert contexts_per_class > 0
    required = {"consequence_cre", "sample_hash", "chrom", "pos", "ref", "alt"}
    assert required <= set(frame.columns)
    selected: list[pl.DataFrame] = []
    for class_name in TARGET_CLASSES:
        class_frame = frame.filter(pl.col("consequence_cre") == class_name).sort(
            ["sample_hash", "chrom", "pos", "ref", "alt"]
        )
        assert class_frame.height >= contexts_per_class, (
            class_name,
            class_frame.height,
            contexts_per_class,
        )
        selected.append(
            class_frame.head(contexts_per_class).with_row_index("source_rank", offset=1)
        )
    result = pl.concat(selected, how="diagonal_relaxed")
    assert result.height == len(TARGET_CLASSES) * contexts_per_class
    assert result.select("panel_row").n_unique() == result.height
    return result.sort(["consequence_cre", "source_rank"])


def annotate_splice_sources(frame: pl.DataFrame, *, gtf_path: Path) -> pl.DataFrame:
    """Attach a unique protein-coding transcript strand to splice variants."""

    assert set(frame["chrom"].unique()) == {"21"}
    introns = load_protein_coding_introns(gtf_path, chrom="21")
    indices = {
        class_name: splice_strand_index(introns, class_name=class_name)
        for class_name in SPLICE_FEATURES
    }
    strands: list[str | None] = []
    matching_strand_counts: list[int] = []
    for row in frame.iter_rows(named=True):
        matches = indices[row["consequence_cre"]].get(int(row["pos"]) - 1, set())
        strands.append(next(iter(matches)) if len(matches) == 1 else None)
        matching_strand_counts.append(len(matches))
    return frame.with_columns(
        pl.Series("consensus_strand", strands, dtype=pl.String),
        pl.Series(
            "matching_splice_strand_count", matching_strand_counts, dtype=pl.Int64
        ),
    )


def one_edit_codon_rows(
    source: Mapping[str, Any],
    reference_sequence: str,
    *,
    window_start0: int,
    window_end0: int,
) -> list[dict[str, Any]]:
    """Replace the transcript-oriented source codon by each one-edit codon."""

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
    rows: list[dict[str, Any]] = []
    for codon_index in range(3):
        for alternate_base in NUCLEOTIDES:
            if alternate_base == transcript_reference_codon[codon_index]:
                continue
            alternate_codon = (
                transcript_reference_codon[:codon_index]
                + alternate_base
                + transcript_reference_codon[codon_index + 1 :]
            )
            genomic_alternate_codon = (
                alternate_codon
                if strand == "+"
                else reverse_complement(alternate_codon)
            )
            alternate_sequence = (
                reference_sequence[:codon_start]
                + genomic_alternate_codon
                + reference_sequence[codon_start + 3 :]
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
                    "perturbation_type": "coding_one_edit",
                    "class": source["consequence_cre"],
                    "feature_id": CODING_FEATURES[source["consequence_cre"]],
                    "source_panel_row": int(source["panel_row"]),
                    "source_rank": int(source["source_rank"]),
                    "context_group": CONTEXT_GROUP,
                    "designed_orientation": (
                        "forward" if strand == "+" else "reverse_complement"
                    ),
                    "chrom": source["chrom"],
                    "source_pos1": int(source["pos"]),
                    "window_start0": window_start0,
                    "window_end0": window_end0,
                    "relative_position": codon_index - (codon_position - 1),
                    "source_state": transcript_reference_codon[codon_index],
                    "alternate_state": alternate_base,
                    "reference_codon": transcript_reference_codon,
                    "alternate_codon": alternate_codon,
                    "reference_amino_acid": GENETIC_CODE[transcript_reference_codon],
                    "alternate_amino_acid": GENETIC_CODE[alternate_codon],
                    "expected_consequence": codon_consequence(
                        transcript_reference_codon, alternate_codon
                    ),
                    "edit_distance": 1,
                    "reference_sequence": reference_sequence,
                    "alternate_sequence": alternate_sequence,
                }
            )
    assert len(rows) == 9
    return rows


def design_heldout_perturbations(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    gtf_path: Path,
    fasta_path: Path,
    output_dir: Path,
    contexts_per_class: int,
) -> dict[str, Any]:
    """Build the frozen, hash-sampled test-context replication panel."""

    assert contexts_per_class > 0 and not output_dir.exists()
    design_commit = os.environ.get("HELDOUT_DESIGN_COMMIT", "")
    assert_current_commit(design_commit)
    panel_manifest = json.loads(panel_manifest_path.read_text())
    assert panel_manifest["output"]["sha256"] == sha256_file(panel_path)
    assert fasta_path.is_file() and Path(f"{fasta_path}.fai").is_file()
    assert Path(f"{fasta_path}.gzi").is_file() and gtf_path.is_file()

    full_panel = pl.read_parquet(panel_path)
    test = full_panel.filter(
        (pl.col("split") == SOURCE_SPLIT)
        & pl.col("consequence_cre").is_in(TARGET_CLASSES)
    )
    assert set(test["consequence_cre"].unique()) == set(TARGET_CLASSES)

    splice = annotate_splice_sources(
        test.filter(pl.col("consequence_cre").is_in(tuple(SPLICE_FEATURES))),
        gtf_path=gtf_path,
    ).filter(pl.col("matching_splice_strand_count") == 1)
    coding_input = test.filter(pl.col("consequence_cre").is_in(tuple(CODING_FEATURES)))
    coding, coding_metadata = annotate_candidates(
        coding_input, gtf_path=gtf_path, fasta_path=fasta_path
    )
    coding = coding.filter(
        (pl.col("matching_transcript_count") > 0)
        & (pl.col("consensus_strand").is_not_null())
        & (pl.col("consensus_codon_position").is_not_null())
        & (pl.col("consensus_ref_codon").is_not_null())
    )
    assert not coding.select(
        "consensus_strand", "consensus_codon_position", "consensus_ref_codon"
    ).null_count().sum_horizontal().item()
    eligible = pl.concat((splice, coding), how="diagonal_relaxed")
    sources = deterministic_sources(eligible, contexts_per_class=contexts_per_class)
    assert set(sources["split"].unique()) == {SOURCE_SPLIT}

    genome = Genome(fasta_path, subset_chroms={"21"})
    rows: list[dict[str, Any]] = []
    for source in sources.iter_rows(named=True):
        sequence, start0, end0 = reference_window(genome, source)
        class_name = str(source["consequence_cre"])
        if class_name in SPLICE_FEATURES:
            orientation = (
                "forward" if source["consensus_strand"] == "+" else "reverse_complement"
            )
            context = sequence[FOCAL_INDEX - 15 : FOCAL_INDEX + 16]
            if orientation == "reverse_complement":
                context = reverse_complement(context)
            splice_source = {
                **source,
                "class": class_name,
                "feature_id": SPLICE_FEATURES[class_name],
                "rank": int(source["source_rank"]),
                "context_group": CONTEXT_GROUP,
                "response_orientation": orientation,
                "ref_context": context,
            }
            source_rows = splice_saturation_rows(
                splice_source,
                sequence,
                window_start0=start0,
                window_end0=end0,
                relative_positions=SPLICE_POSITIONS,
            )
        else:
            source_rows = one_edit_codon_rows(
                source,
                sequence,
                window_start0=start0,
                window_end0=end0,
            )
        for row in source_rows:
            row["source_split"] = SOURCE_SPLIT
            row["source_sample_hash"] = int(source["sample_hash"])
            row["source_annotation_strand"] = source["consensus_strand"]
        rows.extend(source_rows)

    perturbations = perturbation_frame(rows)
    expected_splice = (
        len(SPLICE_FEATURES) * contexts_per_class * len(SPLICE_POSITIONS) * 3
    )
    expected_coding = len(CODING_FEATURES) * contexts_per_class * 9
    assert perturbations.height == expected_splice + expected_coding
    assert set(perturbations["context_group"].unique()) == {CONTEXT_GROUP}
    assert set(perturbations["source_split"].unique()) == {SOURCE_SPLIT}
    assert set(perturbations["edit_distance"].unique()) == {1}

    output_dir.mkdir(parents=True, exist_ok=False)
    sources_path = output_dir / "heldout_sources.parquet"
    perturbations_path = output_dir / "perturbation_panel.parquet"
    sources.write_parquet(sources_path)
    perturbations.write_parquet(perturbations_path)
    summary = {
        "issue": ISSUE,
        "design_commit": design_commit,
        "sources": {
            "panel": {"path": str(panel_path), "sha256": sha256_file(panel_path)},
            "panel_manifest": {
                "path": str(panel_manifest_path),
                "sha256": sha256_file(panel_manifest_path),
            },
            "gtf": {"path": str(gtf_path), "sha256": sha256_file(gtf_path)},
            "fasta": {"path": str(fasta_path), "sha256": sha256_file(fasta_path)},
        },
        "protocol": {
            "source_split": SOURCE_SPLIT,
            "source_selection": "lowest frozen panel sample_hash after unambiguous Ensembl protein-coding annotation; no SAE activation ranking",
            "contexts_per_class": contexts_per_class,
            "context_group": CONTEXT_GROUP,
            "splice_positions": list(SPLICE_POSITIONS),
            "coding_design": "all nine one-nucleotide codon substitutions",
            "coordinate_system": "source pos is 1-based; all derived coordinates are 0-based half-open",
        },
        "annotation_metadata": coding_metadata,
        "eligible_rows": (
            eligible.group_by("consequence_cre").len().sort("consequence_cre")
        ).to_dicts(),
        "source_rows": sources.height,
        "rows": perturbations.height,
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
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contexts-per-class", type=int, default=CONTEXTS_PER_CLASS)
    args = parser.parse_args()
    manifest = design_heldout_perturbations(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        gtf_path=args.gtf,
        fasta_path=args.fasta,
        output_dir=args.output_dir,
        contexts_per_class=args.contexts_per_class,
    )
    print(json.dumps(manifest["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
