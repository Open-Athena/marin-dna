"""Build and evaluate a compact GRCh38 interpretation panel for exp418.

The panel deliberately separates fitting (chromosome 20), feature selection
(chromosome 21), and final reporting (chromosome 22). GTF coordinates are
converted from 1-based closed to 0-based half-open exactly once, at parse time.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
from huggingface_hub import snapshot_download
from marin_dna.data.dna import reverse_complement
from marin_dna.data.genome import Genome
from marin_dna.model.sae import (
    M51_HIDDEN_SIZE,
    M51GenomicWindow,
    load_frozen_m51,
    run_m51_with_activations,
)
from sae_lens.saes.sae import SAE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from launch import BLOCK_INDEX, D_SAE, MODEL_ID, MODEL_REVISION, SEED

WINDOW_BP = 255
FOCAL_INDEX = 127
BASELINE_BP = 31
SPLIT_CHROMS = {"train": "20", "validation": "21", "test": "22"}
SPLICE_TASKS = ("donor", "acceptor")
NUCLEOTIDES = "ACGT"
MAX_INTRON_BP = 50_000
DEFAULT_SPLICE_PAIRS = 512
DEFAULT_NUCLEOTIDES_PER_BASE = 128
DEFAULT_BATCH_SIZE = 16
REFERENCE_FASTA_URI = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)
REFERENCE_GTF_URL = (
    "https://ftp.ensembl.org/pub/release-115/gtf/homo_sapiens/"
    "Homo_sapiens.GRCh38.115.gtf.gz"
)

assert WINDOW_BP == 2 * FOCAL_INDEX + 1
assert BASELINE_BP % 2 == 1


@dataclass(frozen=True, order=True)
class Intron:
    chrom: str
    start: int
    end: int
    strand: Literal["+", "-"]

    def __post_init__(self) -> None:
        assert self.start >= 0
        assert self.end > self.start
        assert self.strand in ("+", "-")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_gtf_attributes(text: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(" ", 1)
        attributes[key] = value.strip().strip('"')
    return attributes


def introns_from_gtf(path: Path, chroms: set[str]) -> list[Intron]:
    """Read protein-coding exons and return unique introns.

    GTF start/end fields are 1-based closed. Exons are converted at this file
    boundary to 0-based half-open as ``start0 = start1 - 1, end0 = end1``.
    """

    transcript_exons: dict[
        str, tuple[str, Literal["+", "-"], list[tuple[int, int]]]
    ] = {}
    saw_grch38 = False
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if line.startswith("#"):
                saw_grch38 |= "genome-build GRCh38" in line
                continue
            fields = line.rstrip("\n").split("\t")
            assert len(fields) == 9
            chrom, _, feature, start_text, end_text, _, strand, _, attrs_text = fields
            if chrom not in chroms or feature != "exon":
                continue
            attrs = _parse_gtf_attributes(attrs_text)
            if attrs.get("transcript_biotype") != "protein_coding":
                continue
            transcript_id = attrs["transcript_id"]
            assert strand in ("+", "-")
            start = int(start_text) - 1
            end = int(end_text)
            assert 0 <= start < end
            if transcript_id not in transcript_exons:
                transcript_exons[transcript_id] = (
                    chrom,
                    strand,
                    [],
                )
            tx_chrom, tx_strand, exons = transcript_exons[transcript_id]
            assert tx_chrom == chrom
            assert tx_strand == strand
            exons.append((start, end))
    assert saw_grch38, "GTF header does not identify GRCh38"
    assert transcript_exons, "no protein-coding exons found"

    introns: set[Intron] = set()
    for chrom, strand, exons in transcript_exons.values():
        ordered = sorted(set(exons))
        for left, right in pairwise(ordered):
            intron_start = left[1]
            intron_end = right[0]
            if intron_end > intron_start:
                introns.add(Intron(chrom, intron_start, intron_end, strand))
    assert introns, "no introns derived from protein-coding transcripts"
    return sorted(introns)


def focal_reference_coordinate(intron: Intron, task: str) -> int:
    """Return the reference base centered in a transcript-oriented window."""

    assert task in SPLICE_TASKS
    if task == "donor":
        return intron.start if intron.strand == "+" else intron.end - 1
    return intron.end - 1 if intron.strand == "+" else intron.start


def oriented_index_to_reference(intron: Intron, index: int) -> int:
    assert 0 <= index < intron.end - intron.start
    if intron.strand == "+":
        return intron.start + index
    return intron.end - 1 - index


def _window_row(
    genome: Genome,
    *,
    split: str,
    intron: Intron,
    task: str,
    label: int,
    focal: int,
    pair_id: str,
) -> dict[str, Any] | None:
    start = focal - FOCAL_INDEX
    end = start + WINDOW_BP
    chrom_size = genome.chroms[intron.chrom]
    if start < 0 or end > chrom_size:
        return None
    sequence = genome(intron.chrom, start, end, intron.strand).upper()
    assert len(sequence) == WINDOW_BP
    if set(sequence) - set(NUCLEOTIDES):
        return None
    motif = sequence[FOCAL_INDEX : FOCAL_INDEX + 2]
    if task == "acceptor":
        motif = sequence[FOCAL_INDEX - 1 : FOCAL_INDEX + 1]
    expected = "GT" if task == "donor" else "AG"
    if motif != expected:
        # Ensembl includes real non-canonical splice sites. This panel asks a
        # narrower question with motif-matched canonical GT/AG negatives.
        return None
    return {
        "row_id": f"{pair_id}:{label}",
        "split": split,
        "chrom": intron.chrom,
        "task": task,
        "label": label,
        "class_base": None,
        "strand": intron.strand,
        "start": start,
        "end": end,
        "focal_position": focal,
        "pair_id": pair_id,
        "sequence": sequence,
    }


def _decoy_focal(
    genome: Genome,
    intron: Intron,
    task: str,
    annotated_focals: set[int],
) -> int | None:
    intron_bp = intron.end - intron.start
    if intron_bp > MAX_INTRON_BP or intron_bp < 2 * FOCAL_INDEX + 4:
        return None
    sequence = genome(intron.chrom, intron.start, intron.end, intron.strand).upper()
    assert len(sequence) == intron_bp
    motif = "GT" if task == "donor" else "AG"
    candidates: list[int] = []
    for index in range(FOCAL_INDEX, intron_bp - FOCAL_INDEX - 1):
        if sequence[index : index + 2] != motif:
            continue
        focal_index = index if task == "donor" else index + 1
        focal = oriented_index_to_reference(intron, focal_index)
        if focal not in annotated_focals:
            candidates.append(focal)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda focal: hashlib.sha256(
            f"{SEED}|{task}|{intron}|{focal}".encode()
        ).digest(),
    )


def _splice_rows(
    genome: Genome,
    introns: Sequence[Intron],
    *,
    split: str,
    task: str,
    pair_count: int,
) -> list[dict[str, Any]]:
    candidates = [intron for intron in introns if intron.chrom == SPLIT_CHROMS[split]]
    annotated_focals = {
        focal_reference_coordinate(intron, task) for intron in candidates
    }
    candidates.sort(
        key=lambda intron: hashlib.sha256(
            f"{SEED}|{split}|{task}|{intron}".encode()
        ).digest()
    )
    rows: list[dict[str, Any]] = []
    used_positive_focals: set[tuple[str, int]] = set()
    used_decoy_focals: set[tuple[str, int]] = set()
    for intron in candidates:
        positive_focal = focal_reference_coordinate(intron, task)
        positive_key = (intron.strand, positive_focal)
        if positive_key in used_positive_focals:
            continue
        decoy_focal = _decoy_focal(genome, intron, task, annotated_focals)
        if decoy_focal is None:
            continue
        decoy_key = (intron.strand, decoy_focal)
        if decoy_key in used_decoy_focals:
            continue
        pair_id = hashlib.sha256(f"{split}|{task}|{intron}".encode()).hexdigest()[:20]
        positive = _window_row(
            genome,
            split=split,
            intron=intron,
            task=task,
            label=1,
            focal=positive_focal,
            pair_id=pair_id,
        )
        negative = _window_row(
            genome,
            split=split,
            intron=intron,
            task=task,
            label=0,
            focal=decoy_focal,
            pair_id=pair_id,
        )
        if positive is None or negative is None:
            continue
        used_positive_focals.add(positive_key)
        used_decoy_focals.add(decoy_key)
        rows.extend((positive, negative))
        if len(rows) == 2 * pair_count:
            break
    assert len(rows) == 2 * pair_count, (
        split,
        task,
        len(rows) // 2,
        pair_count,
    )
    assert sum(row["label"] for row in rows) == pair_count
    assert len({row["pair_id"] for row in rows}) == pair_count
    assert len({(row["strand"], row["start"], row["end"]) for row in rows}) == len(rows)
    return rows


def _nucleotide_rows(
    genome: Genome,
    *,
    split: str,
    per_base: int,
) -> list[dict[str, Any]]:
    chrom = SPLIT_CHROMS[split]
    rng = random.Random(f"{SEED}|{split}|nucleotide")
    selected: dict[str, list[dict[str, Any]]] = {base: [] for base in NUCLEOTIDES}
    seen: set[int] = set()
    attempts = 0
    while min(len(values) for values in selected.values()) < per_base:
        attempts += 1
        assert attempts < 2_000_000, (split, {k: len(v) for k, v in selected.items()})
        focal = rng.randrange(FOCAL_INDEX, genome.chroms[chrom] - FOCAL_INDEX)
        if focal in seen:
            continue
        seen.add(focal)
        start = focal - FOCAL_INDEX
        end = start + WINDOW_BP
        sequence = genome(chrom, start, end, "+").upper()
        if set(sequence) - set(NUCLEOTIDES):
            continue
        base = sequence[FOCAL_INDEX]
        if len(selected[base]) >= per_base:
            continue
        row_id = f"{split}:nucleotide:{chrom}:{focal}"
        selected[base].append(
            {
                "row_id": row_id,
                "split": split,
                "chrom": chrom,
                "task": "nucleotide",
                "label": 0,
                "class_base": base,
                "strand": "+",
                "start": start,
                "end": end,
                "focal_position": focal,
                "pair_id": row_id,
                "sequence": sequence,
            }
        )
    rows = [row for base in NUCLEOTIDES for row in selected[base]]
    assert len(rows) == len(NUCLEOTIDES) * per_base
    return rows


def prepare_panel(
    *,
    fasta_path: Path,
    gtf_path: Path,
    output_path: Path,
    splice_pairs: int,
    nucleotides_per_base: int,
) -> dict[str, Any]:
    assert fasta_path.exists()
    assert Path(f"{fasta_path}.fai").exists()
    assert Path(f"{fasta_path}.gzi").exists()
    assert gtf_path.exists()
    assert splice_pairs > 0
    assert nucleotides_per_base > 0
    genome = Genome(fasta_path, subset_chroms=set(SPLIT_CHROMS.values()))
    assert set(genome.chroms) == set(SPLIT_CHROMS.values())
    introns = introns_from_gtf(gtf_path, set(SPLIT_CHROMS.values()))

    rows: list[dict[str, Any]] = []
    for split in SPLIT_CHROMS:
        for task in SPLICE_TASKS:
            rows.extend(
                _splice_rows(
                    genome,
                    introns,
                    split=split,
                    task=task,
                    pair_count=splice_pairs,
                )
            )
        rows.extend(
            _nucleotide_rows(
                genome,
                split=split,
                per_base=nucleotides_per_base,
            )
        )
    # Splice rows intentionally carry null ``class_base`` while nucleotide
    # rows carry A/C/G/T; inspect the full panel when inferring that column.
    frame = pl.DataFrame(rows, infer_schema_length=None).sort(
        ["split", "task", "row_id"]
    )
    assert frame.height == 3 * (
        2 * len(SPLICE_TASKS) * splice_pairs + len(NUCLEOTIDES) * nucleotides_per_base
    )
    assert frame["row_id"].n_unique() == frame.height
    assert (
        frame.select(
            pl.struct("split", "task", "label", "chrom", "start", "end", "strand")
            .n_unique()
            .alias("unique_loci")
        ).item()
        == frame.height
    )
    assert frame.filter(pl.col("end") - pl.col("start") != WINDOW_BP).is_empty()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    panel_hash = _sha256(output_path)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "coordinate_system": "0-based half-open",
        "gtf_boundary_conversion": "start0 = start1 - 1; end0 = end1",
        "reference": {
            "assembly": "GRCh38.p14",
            "ensembl_release": 115,
            "fasta_uri": REFERENCE_FASTA_URI,
            "fasta_local_name": fasta_path.name,
            "gtf_url": REFERENCE_GTF_URL,
            "gtf_sha256": _sha256(gtf_path),
        },
        "splits": SPLIT_CHROMS,
        "splice_pairs_per_task_per_split": splice_pairs,
        "nucleotide_rows_per_base_per_split": nucleotides_per_base,
        "row_count": frame.height,
        "panel_sha256": panel_hash,
        "panel_path": str(output_path),
        "protein_coding_unique_introns": len(introns),
    }
    _write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def _one_hot_31bp(sequences: Sequence[str]) -> np.ndarray:
    radius = BASELINE_BP // 2
    mapping = {base: index for index, base in enumerate(NUCLEOTIDES)}
    output = np.zeros((len(sequences), BASELINE_BP * 4), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        assert len(sequence) == WINDOW_BP
        subsequence = sequence[FOCAL_INDEX - radius : FOCAL_INDEX + radius + 1]
        assert len(subsequence) == BASELINE_BP
        for position, base in enumerate(subsequence):
            output[row, position * 4 + mapping[base]] = 1.0
    assert np.all(output.reshape(len(sequences), BASELINE_BP, 4).sum(axis=2) == 1)
    return output


def _sparse_average_precision_all(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Exact AP for every non-negative sparse feature, including zero ties."""

    assert scores.ndim == 2
    assert labels.shape == (scores.shape[0],)
    assert np.isfinite(scores).all()
    assert np.all(scores >= 0)
    positives = int(labels.sum())
    assert 0 < positives < len(labels)
    prevalence = positives / len(labels)
    result = np.empty(scores.shape[1], dtype=np.float64)
    for feature in range(scores.shape[1]):
        column = scores[:, feature]
        active = np.flatnonzero(column > 0)
        if not len(active):
            result[feature] = prevalence
            continue
        order = active[np.argsort(-column[active], kind="stable")]
        sorted_scores = column[order]
        sorted_labels = labels[order]
        true_positives = np.cumsum(sorted_labels)
        group_end = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
        endpoints = np.flatnonzero(group_end)
        previous_tp = np.r_[0, true_positives[endpoints[:-1]]]
        group_positives = true_positives[endpoints] - previous_tp
        precision = true_positives[endpoints] / (endpoints + 1)
        ap = float(np.sum(precision * group_positives / positives))
        zero_positives = positives - int(true_positives[-1])
        ap += prevalence * zero_positives / positives
        result[feature] = ap
    return result


def _best_raw_dimension(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[int, int, float]:
    assert scores.shape[1] == M51_HIDDEN_SIZE
    best_dimension = -1
    best_sign = 0
    best_ap = -1.0
    for dimension in range(scores.shape[1]):
        for sign in (1, -1):
            ap = float(average_precision_score(labels, sign * scores[:, dimension]))
            if ap > best_ap:
                best_dimension = dimension
                best_sign = sign
                best_ap = ap
    assert best_dimension >= 0 and best_sign in (1, -1)
    return best_dimension, best_sign, best_ap


def _f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    assert len(precision) == len(recall) == len(thresholds) + 1
    if not len(thresholds):
        return float(scores[0])
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def _score_metrics(
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, float]:
    threshold = _f1_threshold(validation_labels, validation_scores)
    return {
        "validation_average_precision": float(
            average_precision_score(validation_labels, validation_scores)
        ),
        "threshold_selected_on_validation": threshold,
        "test_average_precision": float(
            average_precision_score(test_labels, test_scores)
        ),
        "test_auroc": float(roc_auc_score(test_labels, test_scores)),
        "test_f1": float(f1_score(test_labels, test_scores >= threshold)),
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    assert left.shape == right.shape
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _null_average_precision(
    labels: np.ndarray, scores: np.ndarray, *, permutations: int = 100
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values = np.array(
        [
            average_precision_score(rng.permutation(labels), scores)
            for _ in range(permutations)
        ]
    )
    return {
        "permutations": permutations,
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
    }


def _bootstrap_ap_difference(
    labels: np.ndarray,
    sae_scores: np.ndarray,
    raw_scores: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int = 500,
) -> dict[str, float]:
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(SEED)
    differences: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) != 2:
            continue
        differences.append(
            float(average_precision_score(sampled_labels, sae_scores[indices]))
            - float(average_precision_score(sampled_labels, raw_scores[indices]))
        )
    assert len(differences) >= int(0.95 * replicates)
    values = np.asarray(differences)
    return {
        "replicates": len(differences),
        "mean_sae_minus_raw": float(values.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


@torch.inference_mode()
def _extract_embeddings(
    rows: Sequence[dict[str, Any]],
    *,
    frozen: Any,
    sae: SAE,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_batches: list[np.ndarray] = []
    feature_batches: list[np.ndarray] = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        encoded = frozen.tokenizer(
            [row["sequence"] for row in batch],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded["attention_mask"].to("cuda")
        windows = [
            M51GenomicWindow(
                chrom=row["chrom"],
                start=int(row["start"]),
                end=int(row["end"]),
                strand=row["strand"],
            )
            for row in batch
        ]
        _, activation_batch = run_m51_with_activations(
            frozen,
            input_ids,
            attention_mask,
            windows,
            block_index=BLOCK_INDEX,
        )
        raw = activation_batch.activations[:, FOCAL_INDEX, :].float()
        features = sae.encode(raw)
        assert raw.shape == (len(batch), M51_HIDDEN_SIZE)
        assert features.shape == (len(batch), D_SAE)
        assert torch.isfinite(raw).all() and torch.isfinite(features).all()
        assert torch.all(features >= 0)
        raw_batches.append(raw.cpu().numpy())
        feature_batches.append(features.cpu().numpy())
    raw_output = np.concatenate(raw_batches)
    feature_output = np.concatenate(feature_batches)
    assert raw_output.shape == (len(rows), M51_HIDDEN_SIZE)
    assert feature_output.shape == (len(rows), D_SAE)
    return raw_output, feature_output


def _task_labels(rows: Sequence[dict[str, Any]], task: str) -> np.ndarray:
    if task.startswith("nucleotide_"):
        base = task.rsplit("_", 1)[1]
        assert base in NUCLEOTIDES
        return np.asarray([row["class_base"] == base for row in rows], dtype=np.int8)
    assert task in SPLICE_TASKS
    return np.asarray([row["label"] for row in rows], dtype=np.int8)


def _task_rows(rows: Sequence[dict[str, Any]], split: str, task: str) -> np.ndarray:
    panel_task = "nucleotide" if task.startswith("nucleotide_") else task
    return np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["split"] == split and row["task"] == panel_task
        ],
        dtype=np.int64,
    )


def _reverse_complement_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["sequence"] = reverse_complement(row["sequence"])
        copied["strand"] = "-" if row["strand"] == "+" else "+"
        output.append(copied)
    return output


def _results_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# exp418 compact GRCh38 interpretation",
        "",
        "Feature selection used chromosome 21 only; all displayed performance is on untouched chromosome 22.",
        "",
        "| Task | SAE feature | SAE AP | Raw AP | 31-bp baseline AP | SAE - raw bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, result in results["tasks"].items():
        bootstrap = result["sae_minus_raw_bootstrap"]
        lines.append(
            f"| {task} | {result['sae']['feature']} | "
            f"{result['sae']['test_average_precision']:.4f} | "
            f"{result['raw']['test_average_precision']:.4f} | "
            f"{result['baseline_31bp']['test_average_precision']:.4f} | "
            f"[{bootstrap['ci95_low']:.4f}, {bootstrap['ci95_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "Splice negatives carry the same focal GT/AG motif and come from the same intron as their positive. Coordinates are 0-based half-open.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_panel(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    sae_path: Path,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    assert panel_path.exists() and panel_manifest_path.exists() and sae_path.exists()
    panel_manifest = json.loads(panel_manifest_path.read_text())
    assert panel_manifest["panel_sha256"] == _sha256(panel_path)
    rows = pl.read_parquet(panel_path).to_dicts()
    assert rows
    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    sae = SAE.load_from_disk(sae_path, device="cuda", dtype="float32")
    assert sae.cfg.architecture() == "jumprelu"
    assert sae.cfg.d_in == M51_HIDDEN_SIZE and sae.cfg.d_sae == D_SAE

    raw, features = _extract_embeddings(
        rows, frozen=frozen, sae=sae, batch_size=batch_size
    )
    test_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["split"] == "test"]
    )
    test_rows = [rows[index] for index in test_indices]
    rc_raw, rc_features = _extract_embeddings(
        _reverse_complement_rows(test_rows),
        frozen=frozen,
        sae=sae,
        batch_size=batch_size,
    )
    global_to_rc = {
        global_index: local for local, global_index in enumerate(test_indices)
    }

    task_results: dict[str, Any] = {}
    tasks = list(SPLICE_TASKS) + [f"nucleotide_{base}" for base in NUCLEOTIDES]
    for task in tasks:
        train_idx = _task_rows(rows, "train", task)
        validation_idx = _task_rows(rows, "validation", task)
        test_idx = _task_rows(rows, "test", task)
        train_rows = [rows[index] for index in train_idx]
        validation_rows = [rows[index] for index in validation_idx]
        test_task_rows = [rows[index] for index in test_idx]
        train_labels = _task_labels(train_rows, task)
        validation_labels = _task_labels(validation_rows, task)
        test_labels = _task_labels(test_task_rows, task)
        assert set(train_labels) == set(validation_labels) == set(test_labels) == {0, 1}

        feature_ap = _sparse_average_precision_all(
            features[validation_idx], validation_labels
        )
        feature = int(np.argmax(feature_ap))
        assert np.isclose(
            feature_ap[feature],
            average_precision_score(
                validation_labels, features[validation_idx, feature]
            ),
        )
        raw_dimension, raw_sign, _ = _best_raw_dimension(
            raw[validation_idx], validation_labels
        )

        baseline = LogisticRegression(
            C=1.0,
            max_iter=2_000,
            random_state=SEED,
            solver="liblinear",
        )
        baseline.fit(
            _one_hot_31bp([row["sequence"] for row in train_rows]),
            train_labels,
        )
        baseline_validation = baseline.predict_proba(
            _one_hot_31bp([row["sequence"] for row in validation_rows])
        )[:, 1]
        baseline_test = baseline.predict_proba(
            _one_hot_31bp([row["sequence"] for row in test_task_rows])
        )[:, 1]

        sae_validation = features[validation_idx, feature]
        sae_test = features[test_idx, feature]
        raw_validation = raw_sign * raw[validation_idx, raw_dimension]
        raw_test = raw_sign * raw[test_idx, raw_dimension]
        sae_metrics = _score_metrics(
            validation_labels, sae_validation, test_labels, sae_test
        )
        raw_metrics = _score_metrics(
            validation_labels, raw_validation, test_labels, raw_test
        )
        baseline_metrics = _score_metrics(
            validation_labels, baseline_validation, test_labels, baseline_test
        )

        rc_idx = np.asarray([global_to_rc[index] for index in test_idx])
        contexts = []
        for local_index in np.argsort(-sae_test)[:10]:
            row = test_task_rows[int(local_index)]
            contexts.append(
                {
                    "row_id": row["row_id"],
                    "label": int(test_labels[local_index]),
                    "locus": f"{row['chrom']}:{row['start']}-{row['end']}({row['strand']})",
                    "center_61bp": row["sequence"][FOCAL_INDEX - 30 : FOCAL_INDEX + 31],
                    "activation": float(sae_test[local_index]),
                }
            )
        groups = np.asarray([row["pair_id"] for row in test_task_rows])
        task_results[task] = {
            "sae": {
                "feature": feature,
                "active_fraction_validation": float(np.mean(sae_validation > 0)),
                "active_fraction_test": float(np.mean(sae_test > 0)),
                "reverse_complement_pearson_test": _pearson(
                    sae_test, rc_features[rc_idx, feature]
                ),
                "shuffled_label_null_test_ap": _null_average_precision(
                    test_labels, sae_test
                ),
                **sae_metrics,
            },
            "raw": {
                "dimension": raw_dimension,
                "sign": raw_sign,
                "reverse_complement_pearson_test": _pearson(
                    raw_test, raw_sign * rc_raw[rc_idx, raw_dimension]
                ),
                **raw_metrics,
            },
            "baseline_31bp": baseline_metrics,
            "sae_minus_raw_bootstrap": _bootstrap_ap_difference(
                test_labels, sae_test, raw_test, groups
            ),
            "top_test_contexts": contexts,
        }

    results = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "block_index": BLOCK_INDEX,
        },
        "sae": {
            "path": str(sae_path),
            "weights_sha256": _sha256(sae_path / "sae_weights.safetensors"),
            "config_sha256": _sha256(sae_path / "cfg.json"),
        },
        "panel": panel_manifest,
        "selection_protocol": {
            "baseline_fit": "chromosome 20",
            "sae_feature_raw_dimension_and_threshold_selection": "chromosome 21",
            "final_test": "chromosome 22",
            "sae_feature_direction": "positive activation only",
            "raw_dimension_direction": "positive or negative, selected on chromosome 21",
        },
        "runtime": {
            "gpu_name": torch.cuda.get_device_name(0),
            "batch_size": batch_size,
        },
        "tasks": task_results,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "results.json", results)
    (output_dir / "RESULTS.md").write_text(_results_markdown(results))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--fasta", type=Path, required=True)
    prepare.add_argument("--gtf", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--splice-pairs", type=int, default=DEFAULT_SPLICE_PAIRS)
    prepare.add_argument(
        "--nucleotides-per-base", type=int, default=DEFAULT_NUCLEOTIDES_PER_BASE
    )
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--panel", type=Path, required=True)
    evaluate.add_argument("--panel-manifest", type=Path, required=True)
    evaluate.add_argument("--sae", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_panel(
            fasta_path=args.fasta,
            gtf_path=args.gtf,
            output_path=args.output,
            splice_pairs=args.splice_pairs,
            nucleotides_per_base=args.nucleotides_per_base,
        )
    else:
        result = evaluate_panel(
            panel_path=args.panel,
            panel_manifest_path=args.panel_manifest,
            sae_path=args.sae,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
