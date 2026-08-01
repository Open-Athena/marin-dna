"""Test whether selected coding SAE features reflect strand- and frame-aware biology."""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import os
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from marin_dna.data.genome import Genome
from sklearn.metrics import roc_auc_score

from analyze import column_values, make_views, sha256_file, write_json
from controls import (
    bootstrap_matched_substitution_auc,
    extract_contexts,
    matched_substitution_auc,
)
from pairwise import (
    BOOTSTRAPS,
    NEGATIVE_CLASS,
    PAIR_CLASSES,
    POSITIVE_CLASS,
    bootstrap_pairwise_metrics,
    load_selected_activations,
)
from train import D_SAE, ISSUE, assert_commit

CHROM = "21"
RANDOM_SEED = 4_262
GTF_RELEASE = 109
GTF_FTP_BSD_CHECKSUM = 26_235
GTF_FTP_BLOCKS = 52_988
GTF_BYTES = 54_258_835
SELECTED_FEATURES = (
    ("block19-5m", "signed_mean", 12_658, 1),
    ("block19-25m", "signed_mean", 13_637, 1),
    ("block19-5m", "max_abs", 11_064, 1),
)
COMPLEMENT = str.maketrans("ACGT", "TGCA")
GENETIC_CODE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


@dataclass(frozen=True)
class RawCdsSegment:
    start0: int
    end0: int
    phase: int


@dataclass(frozen=True)
class CdsSegment:
    start0: int
    end0: int
    phase: int
    coding_start: int

    @property
    def length(self) -> int:
        return self.end0 - self.start0


@dataclass(frozen=True)
class TranscriptCds:
    transcript_id: str
    gene_id: str
    gene_name: str
    strand: str
    tags: tuple[str, ...]
    segments: tuple[CdsSegment, ...]
    phase_consistent: bool

    @property
    def coding_start(self) -> int:
        return self.segments[0].coding_start

    @property
    def coding_end(self) -> int:
        final = self.segments[-1]
        return final.coding_start + final.length

    def coding_offset(self, position0: int) -> int | None:
        for segment in self.segments:
            if segment.start0 <= position0 < segment.end0:
                local = (
                    position0 - segment.start0
                    if self.strand == "+"
                    else segment.end0 - 1 - position0
                )
                return segment.coding_start + local
        return None

    def genomic_position0(self, coding_offset: int) -> int | None:
        for segment in self.segments:
            local = coding_offset - segment.coding_start
            if 0 <= local < segment.length:
                return (
                    segment.start0 + local
                    if self.strand == "+"
                    else segment.end0 - 1 - local
                )
        return None


def parse_gtf_attributes(value: str) -> dict[str, list[str]]:
    """Parse repeated GTF attributes without dropping tags."""
    output: dict[str, list[str]] = defaultdict(list)
    for field in value.strip().rstrip(";").split(";"):
        field = field.strip()
        if not field:
            continue
        key, quoted = field.split(" ", 1)
        assert quoted.startswith('"') and quoted.endswith('"'), field
        output[key].append(quoted[1:-1])
    return dict(output)


def build_transcript(
    *,
    transcript_id: str,
    gene_id: str,
    gene_name: str,
    strand: str,
    tags: Iterable[str],
    segments: Iterable[RawCdsSegment],
) -> TranscriptCds:
    """Build transcript-order CDS offsets and validate every GTF phase."""
    assert strand in {"+", "-"}
    ordered = sorted(
        segments,
        key=lambda segment: segment.start0,
        reverse=strand == "-",
    )
    assert ordered and all(segment.start0 < segment.end0 for segment in ordered)
    assert all(segment.phase in {0, 1, 2} for segment in ordered)
    coding_offset = (3 - ordered[0].phase) % 3
    output: list[CdsSegment] = []
    phase_consistent = True
    for segment in ordered:
        expected_phase = (3 - coding_offset % 3) % 3
        phase_consistent &= segment.phase == expected_phase
        output.append(
            CdsSegment(
                start0=segment.start0,
                end0=segment.end0,
                phase=segment.phase,
                coding_start=coding_offset,
            )
        )
        coding_offset += segment.end0 - segment.start0
    return TranscriptCds(
        transcript_id=transcript_id,
        gene_id=gene_id,
        gene_name=gene_name,
        strand=strand,
        tags=tuple(sorted(set(tags))),
        segments=tuple(output),
        phase_consistent=phase_consistent,
    )


def load_chr_cds_transcripts(gtf_path: Path, chrom: str) -> list[TranscriptCds]:
    """Load Ensembl CDS records, converting 1-based closed GTF at this boundary."""
    assert gtf_path.is_file() and chrom
    grouped: dict[str, dict[str, Any]] = {}
    with gzip.open(gtf_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            assert len(fields) == 9
            if fields[0] != chrom or fields[2] != "CDS":
                continue
            attributes = parse_gtf_attributes(fields[8])
            transcript_id = attributes["transcript_id"][0]
            record = grouped.setdefault(
                transcript_id,
                {
                    "gene_id": attributes["gene_id"][0],
                    "gene_name": attributes.get("gene_name", [""])[0],
                    "strand": fields[6],
                    "tags": [],
                    "segments": [],
                },
            )
            assert record["gene_id"] == attributes["gene_id"][0]
            assert record["strand"] == fields[6]
            record["tags"].extend(attributes.get("tag", []))
            start0 = int(fields[3]) - 1
            end0 = int(fields[4])
            assert start0 >= 0 and start0 < end0
            record["segments"].append(
                RawCdsSegment(start0=start0, end0=end0, phase=int(fields[7]))
            )
    transcripts = [
        build_transcript(
            transcript_id=transcript_id,
            gene_id=record["gene_id"],
            gene_name=record["gene_name"],
            strand=record["strand"],
            tags=record["tags"],
            segments=record["segments"],
        )
        for transcript_id, record in grouped.items()
    ]
    assert transcripts and len({item.transcript_id for item in transcripts}) == len(
        transcripts
    )
    return sorted(transcripts, key=lambda item: item.transcript_id)


def transcript_base(base: str, strand: str) -> str:
    assert base in "ACGT" and strand in {"+", "-"}
    return base if strand == "+" else base.translate(COMPLEMENT)


def annotate_transcript_hit(
    transcript: TranscriptCds,
    *,
    position0: int,
    ref: str,
    alt: str,
    fetch_base: Callable[[int, str], str],
) -> dict[str, Any] | None:
    """Annotate one SNV in one CDS, including exon-spanning codons."""
    offset = transcript.coding_offset(position0)
    if offset is None or not transcript.phase_consistent:
        return None
    codon_position0 = offset % 3
    codon_start = offset - codon_position0
    codon_positions = [
        transcript.genomic_position0(codon_start + index) for index in range(3)
    ]
    if any(position is None for position in codon_positions):
        return None
    ref_codon = "".join(
        fetch_base(int(position), transcript.strand) for position in codon_positions
    )
    assert len(ref_codon) == 3 and ref_codon in GENETIC_CODE
    assert ref_codon[codon_position0] == transcript_base(ref, transcript.strand)
    alt_codon_bases = list(ref_codon)
    alt_codon_bases[codon_position0] = transcript_base(alt, transcript.strand)
    alt_codon = "".join(alt_codon_bases)
    ref_aa = GENETIC_CODE[ref_codon]
    alt_aa = GENETIC_CODE[alt_codon]
    if ref_aa == alt_aa:
        consequence = NEGATIVE_CLASS
    elif ref_aa != "*" and alt_aa == "*":
        consequence = "stop_gained"
    elif ref_aa == "*" and alt_aa != "*":
        consequence = "stop_lost"
    else:
        consequence = POSITIVE_CLASS
    return {
        "transcript_id": transcript.transcript_id,
        "gene_id": transcript.gene_id,
        "gene_name": transcript.gene_name,
        "strand": transcript.strand,
        "codon_position": codon_position0 + 1,
        "ref_codon": ref_codon,
        "alt_codon": alt_codon,
        "ref_aa": ref_aa,
        "alt_aa": alt_aa,
        "amino_acid_change": f"{ref_aa}>{alt_aa}",
        "predicted_consequence": consequence,
        "is_mane_select": "MANE_Select" in transcript.tags,
        "is_ensembl_canonical": "Ensembl_canonical" in transcript.tags,
    }


def transcripts_by_panel_row(
    panel: pl.DataFrame, transcripts: list[TranscriptCds]
) -> dict[int, list[TranscriptCds]]:
    """Index only panel positions, avoiding a chromosome-wide per-base CDS index."""
    positions = sorted(set((panel["pos"] - 1).to_list()))
    rows_by_position: dict[int, list[TranscriptCds]] = defaultdict(list)
    for transcript in transcripts:
        for segment in transcript.segments:
            left = bisect.bisect_left(positions, segment.start0)
            right = bisect.bisect_left(positions, segment.end0)
            for position0 in positions[left:right]:
                rows_by_position[position0].append(transcript)
    return {
        int(row["panel_row"]): rows_by_position[int(row["pos"]) - 1]
        for row in panel.iter_rows(named=True)
    }


def unique_or_none(values: Iterable[Any]) -> Any | None:
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else None


def annotate_coding_panel(
    panel: pl.DataFrame,
    *,
    gtf_path: Path,
    fasta_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return one summary row per variant and one row per transcript CDS hit."""
    assert panel["panel_row"].is_sorted()
    assert panel["panel_row"].n_unique() == panel.height
    assert panel["chrom"].unique().to_list() == [CHROM]
    assert set(panel["consequence_cre"].unique()) == set(PAIR_CLASSES)
    genome = Genome(fasta_path, subset_chroms={CHROM})
    transcripts = load_chr_cds_transcripts(gtf_path, CHROM)
    assert sum(transcript.phase_consistent for transcript in transcripts) > 0
    indexed = transcripts_by_panel_row(panel, transcripts)

    def fetch_base(position0: int, strand: str) -> str:
        base = genome(CHROM, position0, position0 + 1, strand).upper()
        assert len(base) == 1 and base in "ACGT"
        return base

    hit_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for row in panel.iter_rows(named=True):
        position0 = int(row["pos"]) - 1
        assert fetch_base(position0, "+") == row["ref"]
        hits: list[dict[str, Any]] = []
        for transcript in indexed[int(row["panel_row"])]:
            hit = annotate_transcript_hit(
                transcript,
                position0=position0,
                ref=row["ref"],
                alt=row["alt"],
                fetch_base=fetch_base,
            )
            if hit is None:
                continue
            hit = {
                "panel_row": row["panel_row"],
                "panel_consequence": row["consequence_cre"],
                **hit,
            }
            hit["matches_panel_consequence"] = (
                hit["predicted_consequence"] == row["consequence_cre"]
            )
            hits.append(hit)
            hit_rows.append(hit)
        matching = [hit for hit in hits if hit["matches_panel_consequence"]]
        strands = sorted({hit["strand"] for hit in matching})
        codon_positions = sorted({hit["codon_position"] for hit in matching})
        amino_changes = sorted({hit["amino_acid_change"] for hit in matching})
        gene_names = sorted({hit["gene_name"] for hit in matching if hit["gene_name"]})
        ref_codons = sorted({hit["ref_codon"] for hit in matching})
        alt_codons = sorted({hit["alt_codon"] for hit in matching})
        summary_rows.append(
            {
                "panel_row": row["panel_row"],
                "cds_hit_count": len(hits),
                "matching_transcript_count": len(matching),
                "consensus_strand": unique_or_none(strands),
                "consensus_codon_position": unique_or_none(codon_positions),
                "consensus_amino_acid_change": unique_or_none(amino_changes),
                "consensus_gene_name": unique_or_none(gene_names),
                "consensus_ref_codon": unique_or_none(ref_codons),
                "consensus_alt_codon": unique_or_none(alt_codons),
                "matching_strands": ",".join(strands),
                "matching_codon_positions": ",".join(map(str, codon_positions)),
                "matching_amino_acid_changes": ",".join(amino_changes),
                "matching_gene_names": ",".join(gene_names),
                "matching_ref_codons": ",".join(ref_codons),
                "matching_alt_codons": ",".join(alt_codons),
                "has_mane_match": any(hit["is_mane_select"] for hit in matching),
                "has_canonical_match": any(
                    hit["is_ensembl_canonical"] for hit in matching
                ),
            }
        )
    summaries = panel.join(pl.DataFrame(summary_rows), on="panel_row", how="left")
    hits = pl.DataFrame(hit_rows)
    assert summaries.height == panel.height
    assert hits.height > 0 and hits["panel_row"].n_unique() <= panel.height
    return summaries, hits


def bsd_checksum(path: Path) -> tuple[int, int]:
    """Return the BSD checksum and 1-KiB block count used by Ensembl CHECKSUMS."""
    checksum = 0
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            size += len(chunk)
            for byte in chunk:
                checksum = ((checksum >> 1) + ((checksum & 1) << 15) + byte) & 0xFFFF
    return checksum, (size + 1023) // 1024


def assert_current_commit(value: str) -> None:
    """Require the manifest pin to equal the checkout that is executing."""
    assert_commit(value)
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert value == current, (value, current)


def reverse_complement(sequence: str) -> str:
    assert set(sequence) <= set("ACGT")
    return sequence.translate(COMPLEMENT)[::-1]


def oriented_substitution(ref: str, alt: str, strand: str) -> str:
    return f"{transcript_base(ref, strand)}>{transcript_base(alt, strand)}"


def comparable_pairs(positive: np.ndarray, strata: np.ndarray) -> int:
    assert positive.shape == strata.shape
    pairs = 0
    for stratum in np.unique(strata):
        selected = strata == stratum
        pairs += int(positive[selected].sum()) * int((~positive[selected]).sum())
    return pairs


def conditional_auc_metrics(
    scores: np.ndarray,
    positive: np.ndarray,
    blocks: np.ndarray,
    strata: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    assert scores.shape == positive.shape == blocks.shape == strata.shape
    pairs = comparable_pairs(positive, strata)
    assert pairs > 0
    low, high = bootstrap_matched_substitution_auc(
        scores,
        positive,
        strata,
        blocks,
        seed=seed,
    )
    return {
        "auroc": matched_substitution_auc(scores, positive, strata),
        "auroc_ci95_low": low,
        "auroc_ci95_high": high,
        "comparable_pairs": pairs,
        "strata_with_both_labels": sum(
            (positive[strata == stratum].any())
            and ((~positive[strata == stratum]).any())
            for stratum in np.unique(strata)
        ),
    }


def overall_auc_metrics(
    scores: np.ndarray,
    positive: np.ndarray,
    blocks: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    confidence = bootstrap_pairwise_metrics(
        scores, positive, blocks, seed=seed, samples=BOOTSTRAPS
    )
    return {
        "auroc": float(roc_auc_score(positive, scores)),
        "auroc_ci95_low": confidence["test_auroc_ci95_low"],
        "auroc_ci95_high": confidence["test_auroc_ci95_high"],
        "comparable_pairs": int(positive.sum() * (~positive).sum()),
        "strata_with_both_labels": 1,
    }


def phase_baseline_scores(annotated: pl.DataFrame) -> np.ndarray:
    """Estimate P(missense | codon position) on discovery and apply it unchanged."""
    assert annotated["consensus_codon_position"].null_count() == 0
    discovery = annotated.filter(pl.col("split") == "discovery")
    rates = {
        int(row["consensus_codon_position"]): float(row["missense_rate"])
        for row in discovery.group_by("consensus_codon_position")
        .agg(
            (pl.col("consequence_cre") == POSITIVE_CLASS).mean().alias("missense_rate")
        )
        .iter_rows(named=True)
    }
    assert set(rates) == {1, 2, 3}
    return np.asarray(
        [rates[int(position)] for position in annotated["consensus_codon_position"]]
    )


def selected_feature_score_frames(
    *,
    extraction_dir: Path,
    extraction_manifest: dict[str, Any],
    selected_rows: np.ndarray,
    annotated: pl.DataFrame,
) -> dict[tuple[str, int], pl.DataFrame]:
    """Load each needed arm once and materialize strand-aware feature scores."""
    assert annotated.height == len(selected_rows)
    strand = annotated["consensus_strand"].to_numpy()
    assert set(np.unique(strand)) == {"+", "-"}
    output: dict[tuple[str, int], pl.DataFrame] = {}
    for arm in sorted({spec[0] for spec in SELECTED_FEATURES}):
        paths = {
            "forward": extraction_dir / arm / "sae_activations_forward.parquet",
            "reverse_complement": extraction_dir
            / arm
            / "sae_activations_reverse_complement.parquet",
        }
        for path in paths.values():
            relative = str(path.relative_to(extraction_dir))
            assert relative in extraction_manifest["artifacts"]
            assert (
                sha256_file(path)
                == extraction_manifest["artifacts"][relative]["sha256"]
            )
        forward = load_selected_activations(
            paths["forward"], selected_rows=selected_rows, columns=D_SAE
        )
        reverse = load_selected_activations(
            paths["reverse_complement"],
            selected_rows=selected_rows,
            columns=D_SAE,
        )
        views = make_views(forward.delta, reverse.delta)
        for selected_arm, selected_view, feature_id, direction in SELECTED_FEATURES:
            if selected_arm != arm:
                continue
            forward_signed = direction * column_values(
                forward.delta, np.arange(len(selected_rows)), feature_id
            )
            reverse_signed = direction * column_values(
                reverse.delta, np.arange(len(selected_rows)), feature_id
            )
            signed_mean = direction * column_values(
                views["signed_mean"], np.arange(len(selected_rows)), feature_id
            )
            max_abs = direction * column_values(
                views["max_abs"], np.arange(len(selected_rows)), feature_id
            )
            forward_abs = np.abs(forward_signed)
            reverse_abs = np.abs(reverse_signed)
            coding_aligned_signed = np.where(
                strand == "+", forward_signed, reverse_signed
            )
            coding_anti_signed = np.where(strand == "+", reverse_signed, forward_signed)
            coding_aligned_abs = np.where(strand == "+", forward_abs, reverse_abs)
            coding_anti_abs = np.where(strand == "+", reverse_abs, forward_abs)
            feature_label = f"{arm} f{feature_id} {selected_view.replace('_', ' ')}"
            output[(arm, feature_id)] = annotated.with_columns(
                pl.lit(feature_label).alias("feature_label"),
                pl.lit(selected_view).alias("selected_view"),
                pl.lit(feature_id).alias("feature_id"),
                pl.Series("forward_signed", forward_signed),
                pl.Series("reverse_complement_signed", reverse_signed),
                pl.Series("signed_mean", signed_mean),
                pl.Series("max_abs", max_abs),
                pl.Series("forward_abs", forward_abs),
                pl.Series("reverse_complement_abs", reverse_abs),
                pl.Series("coding_aligned_signed", coding_aligned_signed),
                pl.Series("coding_anti_aligned_signed", coding_anti_signed),
                pl.Series("coding_aligned_abs", coding_aligned_abs),
                pl.Series("coding_anti_aligned_abs", coding_anti_abs),
            )
        del forward, reverse, views
    assert len(output) == len(SELECTED_FEATURES)
    return output


def score_metric_rows(
    frame: pl.DataFrame,
    *,
    score_column: str,
    method: str,
    seed: int,
) -> list[dict[str, Any]]:
    test = frame.filter(pl.col("split") == "test")
    assert test["consensus_codon_position"].null_count() == 0
    scores = test[score_column].to_numpy()
    positive = test["consequence_cre"].to_numpy() == POSITIVE_CLASS
    blocks = test["block_id"].to_numpy()
    phase = test["consensus_codon_position"].cast(pl.String).to_numpy()
    substitution = test["transcript_substitution"].to_numpy()
    phase_substitution = np.asarray(
        [f"{p}|{s}" for p, s in zip(phase, substitution, strict=True)]
    )
    common = {
        "method": method,
        "rows": test.height,
        "positive_rows": int(positive.sum()),
        "negative_rows": int((~positive).sum()),
    }
    return [
        {
            **common,
            "comparison": "overall",
            **overall_auc_metrics(scores, positive, blocks, seed=seed),
        },
        {
            **common,
            "comparison": "within codon position",
            **conditional_auc_metrics(
                scores, positive, blocks, phase, seed=seed + 10_000
            ),
        },
        {
            **common,
            "comparison": "within position + transcript ref→alt",
            **conditional_auc_metrics(
                scores,
                positive,
                blocks,
                phase_substitution,
                seed=seed + 20_000,
            ),
        },
    ]


def stratified_metric_rows(
    frame: pl.DataFrame,
    *,
    score_column: str,
    feature_label: str,
    group_column: str,
    seed: int,
) -> list[dict[str, Any]]:
    test = frame.filter(pl.col("split") == "test")
    rows: list[dict[str, Any]] = []
    for group_value in test[group_column].drop_nulls().unique().sort().to_list():
        selected = test.filter(pl.col(group_column) == group_value)
        positive = selected["consequence_cre"].to_numpy() == POSITIVE_CLASS
        if not positive.any() or positive.all():
            continue
        blocks = selected["block_id"].to_numpy()
        if min(len(np.unique(blocks[positive])), len(np.unique(blocks[~positive]))) < 2:
            continue
        scores = selected[score_column].to_numpy()
        rows.append(
            {
                "feature_label": feature_label,
                "selected_view": score_column,
                "group_column": group_column,
                "group_value": str(group_value),
                "rows": selected.height,
                "positive_rows": int(positive.sum()),
                "negative_rows": int((~positive).sum()),
                **overall_auc_metrics(
                    scores,
                    positive,
                    blocks,
                    seed=seed + sum(map(ord, str(group_value))),
                ),
            }
        )
    return rows


def plot_coding_controls(metrics: pl.DataFrame, output_dir: Path) -> None:
    comparison_order = [
        "overall",
        "within codon position",
        "within position + transcript ref→alt",
    ]
    method_order = metrics["method"].unique(maintain_order=True).to_list()
    frame = metrics.with_columns(
        pl.col("method").cast(pl.Enum(method_order)),
        pl.col("comparison").cast(pl.Enum(comparison_order)),
    ).to_pandas()
    palette = {
        "overall": "#4C78A8",
        "within codon position": "#F58518",
        "within position + transcript ref→alt": "#54A24B",
    }
    grid = sns.relplot(
        data=frame,
        x="auroc",
        y="method",
        hue="comparison",
        style="comparison",
        hue_order=comparison_order,
        style_order=comparison_order,
        kind="scatter",
        s=85,
        palette=palette,
        height=5.4,
        aspect=1.35,
    )
    axis = grid.ax
    for row in metrics.iter_rows(named=True):
        axis.errorbar(
            row["auroc"],
            row["method"],
            xerr=np.asarray(
                [
                    [max(0.0, row["auroc"] - row["auroc_ci95_low"])],
                    [max(0.0, row["auroc_ci95_high"] - row["auroc"])],
                ]
            ),
            fmt="none",
            capsize=2.5,
            linewidth=1,
            color=palette[row["comparison"]],
            alpha=0.7,
        )
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xlim(0.38, 0.98)
    grid.set_axis_labels("Held-out missense vs synonymous AUROC", "")
    if grid.legend is not None:
        grid.legend.set_title("Comparison")
    grid.figure.suptitle(
        "Codon position is a strong shortcut; SAE signal persists after matching\n"
        "error bars = genomic-block bootstrap 95% CI",
        y=1.04,
    )
    for suffix in ("png", "svg"):
        grid.figure.savefig(
            output_dir / f"coding_controls.{suffix}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(grid.figure)


def plot_orientation(metrics: pl.DataFrame, output_dir: Path) -> None:
    frame = metrics.to_pandas()
    view_order = [
        "FWD",
        "RC",
        "mean/max",
        "coding aligned",
        "coding anti-aligned",
    ]
    palette = {
        "FWD": "#4C78A8",
        "RC": "#F58518",
        "mean/max": "#54A24B",
        "coding aligned": "#E45756",
        "coding anti-aligned": "#B279A2",
    }
    grid = sns.relplot(
        data=frame,
        x="feature_short",
        y="auroc",
        hue="view_label",
        style="view_label",
        hue_order=view_order,
        style_order=view_order,
        kind="scatter",
        s=80,
        palette=palette,
        height=4.8,
        aspect=1.45,
    )
    axis = grid.ax
    for row in metrics.iter_rows(named=True):
        axis.errorbar(
            row["feature_short"],
            row["auroc"],
            yerr=np.asarray(
                [
                    [max(0.0, row["auroc"] - row["auroc_ci95_low"])],
                    [max(0.0, row["auroc_ci95_high"] - row["auroc"])],
                ]
            ),
            fmt="none",
            capsize=2.5,
            linewidth=1,
            color=palette[row["view_label"]],
            alpha=0.65,
        )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_ylim(0.35, 0.9)
    grid.set_axis_labels("Selected feature", "Held-out AUROC")
    if grid.legend is not None:
        grid.legend.set_title("Orientation view")
    grid.figure.suptitle(
        "Transcription-strand alignment does not improve the selected features\n"
        "error bars = genomic-block bootstrap 95% CI",
        y=1.04,
    )
    for suffix in ("png", "svg"):
        grid.figure.savefig(
            output_dir / f"coding_orientation.{suffix}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(grid.figure)


def analyze_coding_semantics(
    *,
    extraction_dir: Path,
    extraction_manifest_path: Path,
    pairwise_metrics_path: Path,
    panel_path: Path,
    gtf_path: Path,
    fasta_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert extraction_dir.is_dir() and extraction_manifest_path.is_file()
    assert pairwise_metrics_path.is_file() and panel_path.is_file()
    assert gtf_path.is_file() and fasta_path.is_file() and not output_dir.exists()
    analysis_commit = os.environ.get("ANALYSIS_COMMIT", "")
    assert_current_commit(analysis_commit)
    assert gtf_path.stat().st_size == GTF_BYTES
    assert bsd_checksum(gtf_path) == (GTF_FTP_BSD_CHECKSUM, GTF_FTP_BLOCKS)
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    assert extraction_manifest["issue"] == ISSUE
    assert sha256_file(panel_path) == extraction_manifest["panel"]["sha256"]
    pairwise_metrics = pl.read_parquet(pairwise_metrics_path)
    for arm, selected_view, feature_id, direction in SELECTED_FEATURES:
        selected = pairwise_metrics.filter(
            (pl.col("arm") == arm) & (pl.col("view") == selected_view)
        )
        assert selected.height == 1
        assert selected["feature_id"][0] == feature_id
        assert selected["direction"][0] == direction

    panel = pl.read_parquet(panel_path)
    pair_panel = panel.filter(pl.col("consequence_cre").is_in(PAIR_CLASSES))
    assert pair_panel.height == 1_024
    selected_rows = pair_panel["panel_row"].to_numpy().astype(np.int64)
    assert np.all(selected_rows[:-1] < selected_rows[1:])
    annotated, transcript_hits = annotate_coding_panel(
        pair_panel, gtf_path=gtf_path, fasta_path=fasta_path
    )
    assert annotated["consensus_strand"].null_count() == 0
    assert annotated["consensus_codon_position"].null_count() <= 5
    assert annotated["matching_transcript_count"].min() > 0
    contexts = extract_contexts(annotated, fasta_path)
    coding_contexts = np.asarray(
        [
            context if strand == "+" else reverse_complement(context)
            for context, strand in zip(
                contexts, annotated["consensus_strand"], strict=True
            )
        ]
    )
    annotated = annotated.with_columns(
        pl.Series("genomic_forward_context_31bp", contexts),
        pl.Series("transcript_oriented_context_31bp", coding_contexts),
        pl.Series(
            "transcript_substitution",
            [
                oriented_substitution(ref, alt, strand)
                for ref, alt, strand in zip(
                    annotated["ref"],
                    annotated["alt"],
                    annotated["consensus_strand"],
                    strict=True,
                )
            ],
        ),
    )
    complete = annotated.filter(pl.col("consensus_codon_position").is_not_null())
    feature_frames = selected_feature_score_frames(
        extraction_dir=extraction_dir,
        extraction_manifest=extraction_manifest,
        selected_rows=selected_rows,
        annotated=annotated,
    )

    metric_rows: list[dict[str, Any]] = []
    phase_scores = phase_baseline_scores(complete)
    phase_frame = complete.with_columns(pl.Series("phase_only", phase_scores))
    metric_rows.extend(
        score_metric_rows(
            phase_frame,
            score_column="phase_only",
            method="codon position only",
            seed=RANDOM_SEED,
        )
    )
    stratified_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    score_outputs: list[pl.DataFrame] = []
    top_outputs: list[pl.DataFrame] = []
    for feature_index, (arm, selected_view, feature_id, _) in enumerate(
        SELECTED_FEATURES
    ):
        frame = feature_frames[(arm, feature_id)].filter(
            pl.col("consensus_codon_position").is_not_null()
        )
        feature_label = frame["feature_label"][0]
        short = f"f{feature_id}\n{arm}"
        metric_rows.extend(
            score_metric_rows(
                frame,
                score_column=selected_view,
                method=feature_label,
                seed=RANDOM_SEED + (feature_index + 1) * 100_000,
            )
        )
        stratified_rows.extend(
            stratified_metric_rows(
                frame,
                score_column=selected_view,
                feature_label=feature_label,
                group_column="consensus_codon_position",
                seed=RANDOM_SEED + (feature_index + 1) * 100_000,
            )
        )
        stratified_rows.extend(
            stratified_metric_rows(
                frame,
                score_column=selected_view,
                feature_label=feature_label,
                group_column="consensus_strand",
                seed=RANDOM_SEED + (feature_index + 1) * 200_000,
            )
        )
        if selected_view == "signed_mean":
            orientation_views = (
                ("forward_signed", "FWD"),
                ("reverse_complement_signed", "RC"),
                ("signed_mean", "mean/max"),
                ("coding_aligned_signed", "coding aligned"),
                ("coding_anti_aligned_signed", "coding anti-aligned"),
            )
        else:
            assert selected_view == "max_abs"
            orientation_views = (
                ("forward_abs", "FWD"),
                ("reverse_complement_abs", "RC"),
                ("max_abs", "mean/max"),
                ("coding_aligned_abs", "coding aligned"),
                ("coding_anti_aligned_abs", "coding anti-aligned"),
            )
        test = frame.filter(pl.col("split") == "test")
        positive = test["consequence_cre"].to_numpy() == POSITIVE_CLASS
        blocks = test["block_id"].to_numpy()
        for score_column, view_label in orientation_views:
            orientation_rows.append(
                {
                    "feature_label": feature_label,
                    "feature_short": short,
                    "selected_view": selected_view,
                    "score_column": score_column,
                    "view_label": view_label,
                    "rows": test.height,
                    **overall_auc_metrics(
                        test[score_column].to_numpy(),
                        positive,
                        blocks,
                        seed=RANDOM_SEED
                        + (feature_index + 1) * 300_000
                        + sum(map(ord, score_column)),
                    ),
                }
            )
        score_outputs.append(frame)
        top_outputs.append(
            frame.filter(pl.col("split") == "test")
            .sort(selected_view, descending=True)
            .head(32)
            .with_row_index("feature_rank", offset=1)
            .select(
                [
                    "feature_label",
                    "feature_id",
                    "selected_view",
                    "feature_rank",
                    "panel_row",
                    "chrom",
                    "pos",
                    "ref",
                    "alt",
                    "consequence_cre",
                    "block_id",
                    "consensus_strand",
                    "consensus_codon_position",
                    "consensus_gene_name",
                    "consensus_ref_codon",
                    "consensus_alt_codon",
                    "consensus_amino_acid_change",
                    "matching_gene_names",
                    "matching_ref_codons",
                    "matching_alt_codons",
                    "transcript_substitution",
                    "transcript_oriented_context_31bp",
                    "forward_signed",
                    "reverse_complement_signed",
                    "signed_mean",
                    "max_abs",
                    "coding_aligned_signed",
                    "coding_anti_aligned_signed",
                ]
            )
        )

    metrics = pl.DataFrame(metric_rows)
    orientation_metrics = pl.DataFrame(orientation_rows)
    stratified_metrics = pl.DataFrame(stratified_rows)
    feature_scores = pl.concat(score_outputs)
    top_variants = pl.concat(top_outputs)
    assert metrics.height == (len(SELECTED_FEATURES) + 1) * 3
    assert orientation_metrics.height == len(SELECTED_FEATURES) * 5
    assert top_variants.height == len(SELECTED_FEATURES) * 32

    output_dir.mkdir(parents=True)
    annotated.write_parquet(output_dir / "coding_annotations.parquet")
    transcript_hits.write_parquet(output_dir / "transcript_hits.parquet")
    feature_scores.write_parquet(output_dir / "feature_scores.parquet")
    metrics.write_parquet(output_dir / "coding_control_metrics.parquet")
    orientation_metrics.write_parquet(output_dir / "orientation_metrics.parquet")
    stratified_metrics.write_parquet(output_dir / "stratified_metrics.parquet")
    top_variants.write_parquet(output_dir / "top_variants.parquet")
    plot_coding_controls(metrics, output_dir)
    plot_orientation(orientation_metrics, output_dir)

    phase_counts = (
        annotated.group_by(["consequence_cre", "consensus_codon_position"])
        .len()
        .sort(["consequence_cre", "consensus_codon_position"], nulls_last=True)
    )
    result = {
        "analysis_commit": analysis_commit,
        "issue": ISSUE,
        "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
        "pairwise_metrics_sha256": sha256_file(pairwise_metrics_path),
        "panel_sha256": sha256_file(panel_path),
        "gtf": {
            "release": GTF_RELEASE,
            "url": "https://ftp.ensembl.org/pub/release-109/gtf/homo_sapiens/Homo_sapiens.GRCh38.109.gtf.gz",
            "bytes": gtf_path.stat().st_size,
            "sha256": sha256_file(gtf_path),
            "ftp_bsd_checksum": GTF_FTP_BSD_CHECKSUM,
            "ftp_blocks": GTF_FTP_BLOCKS,
        },
        "rows": annotated.height,
        "rows_with_matching_transcript": int(
            (annotated["matching_transcript_count"] > 0).sum()
        ),
        "rows_with_consensus_strand": annotated.height
        - annotated["consensus_strand"].null_count(),
        "rows_with_consensus_codon_position": annotated.height
        - annotated["consensus_codon_position"].null_count(),
        "phase_counts": phase_counts.to_dicts(),
        "protocol": {
            "source_label": "VEP 109.1 --most_severe",
            "coordinates": "panel 1-based; GTF 1-based closed converted to 0-based half-open at parse boundary",
            "transcript_filter": "retain CDS hits whose reconstructed standard-code consequence matches the panel label; require consensus for strand/phase analyses",
            "conditional_auc": "pair-weighted AUROC within codon position, then within codon position plus transcript-oriented ref>alt",
            "feature_selection": "three features fixed by the prior untouched-test pairwise analysis; no reselection",
            "block_bootstraps": BOOTSTRAPS,
            "status": "exploratory semantic discriminator specified after sequence controls",
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
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--extraction-manifest", type=Path, required=True)
    parser.add_argument("--pairwise-metrics", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = analyze_coding_semantics(
        extraction_dir=args.extraction_dir,
        extraction_manifest_path=args.extraction_manifest,
        pairwise_metrics_path=args.pairwise_metrics,
        panel_path=args.panel,
        gtf_path=args.gtf,
        fasta_path=args.fasta,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
