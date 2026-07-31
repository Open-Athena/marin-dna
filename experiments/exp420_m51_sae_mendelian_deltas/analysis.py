"""Matched ref-to-alt m5.1 SAE analysis for issue 420.

The Hugging Face dataset stores VCF-style 1-based variant positions. They are
converted once, at the FASTA boundary, to a 0-based center coordinate. Every
interval created after that conversion is 0-based half-open.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import polars as pl
import torch
from datasets import load_dataset
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
from sklearn.metrics import average_precision_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ISSUE = 420
SEED = 288
WINDOW_BP = 255
FOCAL_INDEX = 127
CONTEXT_BP = 41
BLOCK_INDEX = 9
D_SAE = 15_360
MODEL_ID = "marin-dna/marin-dna-exp135-m5.1"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
DATASET_ID = "bolinas-dna/evals_mendelian_traits"
DATASET_REVISION = "4aed58e50c5dea0b878a665007af2ef9e5108e9f"
RESULT_PREFIX = (
    "s3://oa-bolinas/snakemake/analysis/evals_v2/results/"
    "{kind}/mix-v0.9-p1B-i24-exp135-m5.1-step-59158/mendelian_traits.parquet"
)
REFERENCE_FASTA_URI = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)
ORIENTATIONS = ("forward", "reverse_complement")
SPACES = ("sae", "raw")
DISCOVERY_CHROMS = frozenset({"5", "7", "9", "13", "15", "17", "19", "21"})
VALIDATION_CHROMS = frozenset({"1", "3"})
TEST_CHROMS = frozenset({"11", "X"})
CHROM_SPLITS = {
    "discovery": DISCOVERY_CHROMS,
    "validation": VALIDATION_CHROMS,
    "test": TEST_CHROMS,
}
DESCRIPTIVE_SUBSETS = frozenset({"mature_miRNA_variant"})
KNOWN_NUCLEOTIDE_FEATURES = {"A": 1120, "C": 14562, "G": 11528, "T": 9728}
TOP_CANDIDATES = 5
BOOTSTRAPS = 5_000
PERMUTATIONS = 5_000
NUCLEOTIDES = frozenset("ACGT")

assert WINDOW_BP == 2 * FOCAL_INDEX + 1
assert CONTEXT_BP % 2 == 1
assert M51_HIDDEN_SIZE == 1_920


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in (SEED, *parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def split_for_chrom(chrom: str) -> str:
    matches = [name for name, chroms in CHROM_SPLITS.items() if chrom in chroms]
    assert len(matches) == 1, (chrom, matches)
    return matches[0]


def variant_sequences(reference_sequence: str, ref: str, alt: str) -> tuple[str, str]:
    """Validate a centered reference window and return ref/alt sequences."""

    reference_sequence = reference_sequence.upper()
    ref = ref.upper()
    alt = alt.upper()
    assert len(reference_sequence) == WINDOW_BP
    assert set(reference_sequence) <= NUCLEOTIDES
    assert len(ref) == len(alt) == 1
    assert ref in NUCLEOTIDES and alt in NUCLEOTIDES and ref != alt
    assert reference_sequence[FOCAL_INDEX] == ref
    alternate = (
        reference_sequence[:FOCAL_INDEX] + alt + reference_sequence[FOCAL_INDEX + 1 :]
    )
    assert len(alternate) == WINDOW_BP
    assert alternate[FOCAL_INDEX] == alt
    assert sum(a != b for a, b in zip(reference_sequence, alternate, strict=True)) == 1
    return reference_sequence, alternate


def _validate_panel(frame: pl.DataFrame) -> None:
    required = {
        "chrom",
        "pos",
        "ref",
        "alt",
        "label",
        "subset",
        "match_group",
        "split",
        "llr_fwd",
        "llr_rc",
        "minus_llr_avg",
        "probe_score",
    }
    assert required <= set(frame.columns), required - set(frame.columns)
    assert frame.height == 16_140
    assert (
        frame.select(pl.struct("chrom", "pos", "ref", "alt").n_unique()).item()
        == frame.height
    )
    assert frame["match_group"].n_unique() == 1_614
    assert frame.filter(pl.col("pos") <= 0).is_empty()
    assert frame.filter(pl.col("ref").str.len_chars() != 1).is_empty()
    assert frame.filter(pl.col("alt").str.len_chars() != 1).is_empty()
    assert frame.filter(~pl.col("ref").is_in(sorted(NUCLEOTIDES))).is_empty()
    assert frame.filter(~pl.col("alt").is_in(sorted(NUCLEOTIDES))).is_empty()
    assert frame.filter(pl.col("ref") == pl.col("alt")).is_empty()
    required_nonnull = required - {"probe_score"}
    required_nulls = {column: frame[column].null_count() for column in required_nonnull}
    assert sum(required_nulls.values()) == 0, required_nulls
    missing_probe = frame.filter(pl.col("probe_score").is_null())
    assert missing_probe.height == 40
    assert set(missing_probe["subset"]) == DESCRIPTIVE_SUBSETS
    groups = frame.group_by("match_group").agg(
        pl.len().alias("n"),
        pl.col("label").sum().alias("positives"),
        pl.col("chrom").n_unique().alias("chroms"),
        pl.col("subset").n_unique().alias("subsets"),
        pl.col("split").n_unique().alias("splits"),
    )
    assert groups.filter(
        (pl.col("n") != 10)
        | (pl.col("positives") != 1)
        | (pl.col("chroms") != 1)
        | (pl.col("subsets") != 1)
        | (pl.col("splits") != 1)
    ).is_empty()
    assert set(frame["split"].unique()) == set(CHROM_SPLITS)
    observed_chroms = set(frame["chrom"].unique())
    assert observed_chroms == set().union(*CHROM_SPLITS.values())
    for column in ("llr_fwd", "llr_rc", "minus_llr_avg"):
        assert np.isfinite(frame[column].to_numpy()).all(), column
    assert np.isfinite(frame["probe_score"].drop_nulls().to_numpy()).all()


def prepare_panel(output_path: Path) -> dict[str, Any]:
    assert not output_path.exists()
    dataset = load_dataset(
        DATASET_ID,
        split="train",
        revision=DATASET_REVISION,
    )
    base = pl.from_arrow(dataset.data.table).with_columns(
        pl.col("chrom").cast(pl.String),
        pl.col("ref").str.to_uppercase(),
        pl.col("alt").str.to_uppercase(),
        pl.col("label").cast(pl.Int8),
    )
    keys = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]
    assert base.select(pl.struct(keys).n_unique()).item() == base.height

    score_uri = RESULT_PREFIX.format(kind="scores")
    probe_uri = RESULT_PREFIX.format(kind="probe")
    key_normalization = (
        pl.col("chrom").cast(pl.String),
        pl.col("ref").str.to_uppercase(),
        pl.col("alt").str.to_uppercase(),
        pl.col("label").cast(pl.Int8),
    )
    scores = pl.read_parquet(
        score_uri, columns=keys + ["llr_fwd", "llr_rc"]
    ).with_columns(*key_normalization)
    probes = pl.read_parquet(probe_uri, columns=keys + ["probe_score"]).with_columns(
        *key_normalization
    )
    assert (
        scores.select(pl.struct(keys).n_unique()).item() == scores.height == base.height
    )
    assert (
        probes.select(pl.struct(keys).n_unique()).item() == probes.height == base.height
    )
    frame = (
        base.join(scores, on=keys, how="left", validate="1:1")
        .join(probes, on=keys, how="left", validate="1:1")
        .with_columns(
            (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("minus_llr_avg"),
            pl.col("chrom")
            .map_elements(split_for_chrom, return_dtype=pl.String)
            .alias("split"),
        )
        .sort(["subset", "match_group", "label", "chrom", "pos"])
    )
    _validate_panel(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path, compression="zstd")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "coordinate_boundary": "dataset pos1 -> pos0 = pos1 - 1",
        "internal_coordinate_system": "0-based half-open",
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": "train",
        },
        "baseline_uris": {"scores": score_uri, "probe": probe_uri},
        "reference_fasta_uri": REFERENCE_FASTA_URI,
        "chromosome_splits": {
            name: sorted(chroms) for name, chroms in CHROM_SPLITS.items()
        },
        "descriptive_only_subsets": sorted(DESCRIPTIVE_SUBSETS),
        "row_count": frame.height,
        "match_groups": frame["match_group"].n_unique(),
        "panel_sha256": _sha256(output_path),
    }
    _write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def matched_contrasts(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive-minus-nine-controls contrasts in stable group order."""

    assert scores.ndim in (1, 2)
    assert labels.shape == groups.shape == (scores.shape[0],)
    assert set(np.unique(labels)) == {0, 1}
    group_order = np.asarray(list(dict.fromkeys(groups.tolist())))
    output_shape = (len(group_order),) + scores.shape[1:]
    output = np.empty(output_shape, dtype=np.float32)
    for index, group in enumerate(group_order):
        selected = np.flatnonzero(groups == group)
        assert selected.shape == (10,)
        positive = selected[labels[selected] == 1]
        controls = selected[labels[selected] == 0]
        assert positive.shape == (1,) and controls.shape == (9,)
        output[index] = scores[positive[0]] - scores[controls].mean(axis=0)
    assert np.isfinite(output).all()
    return output, group_order


def standardized_means(contrasts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert contrasts.ndim == 2 and contrasts.shape[0] >= 2
    means = contrasts.mean(axis=0, dtype=np.float64)
    standard_errors = contrasts.std(axis=0, ddof=1, dtype=np.float64) / np.sqrt(
        contrasts.shape[0]
    )
    statistics = np.divide(
        means,
        standard_errors,
        out=np.zeros_like(means),
        where=standard_errors > 0,
    )
    assert np.isfinite(means).all() and np.isfinite(statistics).all()
    return means, statistics


def select_candidate(
    discovery: np.ndarray,
    validation: np.ndarray,
    *,
    top_k: int = TOP_CANDIDATES,
) -> tuple[int, int, bool, list[dict[str, Any]]]:
    """Rank on discovery and choose a direction-consistent validation feature."""

    assert discovery.ndim == validation.ndim == 2
    assert discovery.shape[1] == validation.shape[1]
    assert 0 < top_k <= discovery.shape[1]
    discovery_mean, discovery_t = standardized_means(discovery)
    ranked = np.lexsort((np.arange(discovery.shape[1]), -np.abs(discovery_t)))[:top_k]
    validation_mean, validation_t = standardized_means(validation[:, ranked])
    candidates: list[dict[str, Any]] = []
    consistent: list[tuple[float, int, int]] = []
    for rank, dimension in enumerate(ranked, start=1):
        direction = 1 if discovery_mean[dimension] >= 0 else -1
        oriented_validation_t = direction * validation_t[rank - 1]
        row = {
            "rank": rank,
            "dimension": int(dimension),
            "direction": direction,
            "discovery_mean": float(discovery_mean[dimension]),
            "discovery_t": float(discovery_t[dimension]),
            "validation_mean": float(validation_mean[rank - 1]),
            "validation_t": float(validation_t[rank - 1]),
            "direction_consistent": bool(oriented_validation_t > 0),
        }
        candidates.append(row)
        if oriented_validation_t > 0:
            consistent.append((float(oriented_validation_t), -rank, int(dimension)))
    replicated = bool(consistent)
    if replicated:
        _, _, selected = max(consistent)
    else:
        # Preserve a deterministic, discovery/validation-only choice while
        # marking the failed gate. Any subsequently displayed test metric for
        # this row is descriptive, not a replicated association.
        oriented = [row["direction"] * row["validation_t"] for row in candidates]
        selected = candidates[int(np.argmax(oriented))]["dimension"]
    direction = next(
        row["direction"] for row in candidates if row["dimension"] == selected
    )
    return selected, int(direction), replicated, candidates


def bootstrap_mean_interval(
    values: np.ndarray, *, seed: int, samples: int = BOOTSTRAPS
) -> tuple[float, float]:
    assert values.ndim == 1 and len(values) >= 2 and samples > 0
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    for offset in range(0, samples, 250):
        size = min(250, samples - offset)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        boot[offset : offset + size] = values[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def matched_permutation_pvalue(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    permutations: int = PERMUTATIONS,
) -> float:
    assert scores.ndim == 1
    observed, group_order = matched_contrasts(scores, labels, groups)
    statistic = abs(float(observed.mean()))
    group_values: list[np.ndarray] = []
    for group in group_order:
        values = scores[groups == group]
        assert values.shape == (10,)
        group_values.append(values)
    matrix = np.stack(group_values)
    group_sums = matrix.sum(axis=1)
    rng = np.random.default_rng(seed)
    exceed = 0
    for offset in range(0, permutations, 500):
        size = min(500, permutations - offset)
        chosen = rng.integers(0, 10, size=(size, len(group_order)))
        positives = np.take_along_axis(
            np.broadcast_to(matrix, (size, *matrix.shape)),
            chosen[:, :, None],
            axis=2,
        )[:, :, 0]
        contrasts = positives - (group_sums[None, :] - positives) / 9
        null_statistics = np.abs(contrasts.mean(axis=1))
        exceed += int(np.count_nonzero(null_statistics >= statistic))
    return float((exceed + 1) / (permutations + 1))


def _rows_for(frame: pl.DataFrame, *, split: str, subset: str) -> np.ndarray:
    return np.flatnonzero(
        (frame["split"].to_numpy() == split) & (frame["subset"].to_numpy() == subset)
    )


def _evaluate_fixed_dimension(
    matrix: np.ndarray,
    frame: pl.DataFrame,
    indices: np.ndarray,
    *,
    dimension: int,
    direction: int,
    seed: int,
) -> dict[str, Any]:
    labels = frame["label"].to_numpy()[indices].astype(np.int8)
    groups = frame["match_group"].to_numpy()[indices]
    oriented_scores = direction * np.asarray(matrix[indices, dimension])
    contrasts, _ = matched_contrasts(oriented_scores, labels, groups)
    low, high = bootstrap_mean_interval(contrasts, seed=seed)
    return {
        "test_groups": len(np.unique(groups)),
        "test_matched_mean": float(contrasts.mean()),
        "test_matched_ci95_low": low,
        "test_matched_ci95_high": high,
        "test_sign_stability": float(np.mean(contrasts > 0)),
        "test_average_precision": float(
            average_precision_score(labels, oriented_scores)
        ),
        "test_permutation_pvalue": matched_permutation_pvalue(
            oriented_scores,
            labels,
            groups,
            seed=seed + 1,
        ),
    }


def _baseline_metrics(
    scores: np.ndarray,
    frame: pl.DataFrame,
    indices: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    labels = frame["label"].to_numpy()[indices].astype(np.int8)
    groups = frame["match_group"].to_numpy()[indices]
    values = np.asarray(scores[indices], dtype=np.float64)
    contrasts, _ = matched_contrasts(values, labels, groups)
    low, high = bootstrap_mean_interval(contrasts, seed=seed)
    return {
        "test_groups": len(np.unique(groups)),
        "test_matched_mean": float(contrasts.mean()),
        "test_matched_ci95_low": low,
        "test_matched_ci95_high": high,
        "test_sign_stability": float(np.mean(contrasts > 0)),
        "test_average_precision": float(average_precision_score(labels, values)),
        "test_permutation_pvalue": matched_permutation_pvalue(
            values, labels, groups, seed=seed + 1
        ),
    }


def _substitution_scores(frame: pl.DataFrame, subset: str) -> np.ndarray:
    substitutions = frame.select(
        pl.concat_str("ref", pl.lit(">"), "alt").alias("substitution")
    )["substitution"].to_numpy()
    labels = frame["label"].to_numpy()
    training = (frame["subset"].to_numpy() == subset) & (
        frame["split"].to_numpy() != "test"
    )
    assert training.any()
    fallback = float((labels[training].sum() + 1) / (training.sum() + 2))
    rates: dict[str, float] = {}
    for substitution in sorted(set(substitutions[training])):
        selected = training & (substitutions == substitution)
        assert selected.any()
        rates[substitution] = float((labels[selected].sum() + 1) / (selected.sum() + 2))
    return np.asarray(
        [rates.get(value, fallback) for value in substitutions], dtype=np.float64
    )


def _extract_variant_batch(
    frame: pl.DataFrame,
    indices: Sequence[int],
    *,
    genome: Genome,
    frozen: Any,
    sae: SAE,
    orientation: Literal["forward", "reverse_complement"],
) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[str] = []
    windows: list[M51GenomicWindow] = []
    for index in indices:
        row = frame.row(index, named=True)
        pos0 = int(row["pos"]) - 1
        assert pos0 >= 0
        start = pos0 - FOCAL_INDEX
        end = pos0 + FOCAL_INDEX + 1
        assert end - start == WINDOW_BP
        reference = genome(row["chrom"], start, end, "+").upper()
        ref_sequence, alt_sequence = variant_sequences(
            reference, row["ref"], row["alt"]
        )
        strand = "+"
        if orientation == "reverse_complement":
            ref_sequence = reverse_complement(ref_sequence)
            alt_sequence = reverse_complement(alt_sequence)
            strand = "-"
            assert ref_sequence[FOCAL_INDEX] == reverse_complement(row["ref"])
            assert alt_sequence[FOCAL_INDEX] == reverse_complement(row["alt"])
        sequences.extend((ref_sequence, alt_sequence))
        windows.extend(
            (
                M51GenomicWindow(
                    chrom=row["chrom"], start=start, end=end, strand=strand
                ),
                M51GenomicWindow(
                    chrom=row["chrom"], start=start, end=end, strand=strand
                ),
            )
        )
    encoded = frozen.tokenizer(
        sequences,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")
    _, activation_batch = run_m51_with_activations(
        frozen,
        input_ids,
        attention_mask,
        windows,
        block_index=BLOCK_INDEX,
    )
    raw = activation_batch.activations[:, FOCAL_INDEX, :].float()
    features = sae.encode(raw)
    assert raw.shape == (2 * len(indices), M51_HIDDEN_SIZE)
    assert features.shape == (2 * len(indices), D_SAE)
    assert torch.isfinite(raw).all() and torch.isfinite(features).all()
    assert torch.all(features >= 0)
    raw_delta = raw[1::2] - raw[0::2]
    feature_delta = features[1::2] - features[0::2]
    return raw_delta.cpu().numpy(), feature_delta.cpu().numpy()


def _fill_delta_matrices(
    frame: pl.DataFrame,
    *,
    genome: Genome,
    frozen: Any,
    sae: SAE,
    temp_dir: Path,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for orientation in ORIENTATIONS:
        raw_path = temp_dir / f"{orientation}.raw.npy"
        sae_path = temp_dir / f"{orientation}.sae.npy"
        raw = np.lib.format.open_memmap(
            raw_path, mode="w+", dtype=np.float32, shape=(frame.height, M51_HIDDEN_SIZE)
        )
        features = np.lib.format.open_memmap(
            sae_path, mode="w+", dtype=np.float32, shape=(frame.height, D_SAE)
        )
        for offset in range(0, frame.height, batch_size):
            stop = min(offset + batch_size, frame.height)
            indices = list(range(offset, stop))
            raw_batch, feature_batch = _extract_variant_batch(
                frame,
                indices,
                genome=genome,
                frozen=frozen,
                sae=sae,
                orientation=orientation,
            )
            raw[offset:stop] = raw_batch
            features[offset:stop] = feature_batch
            if offset == 0 or stop == frame.height or stop % (batch_size * 25) == 0:
                print(
                    json.dumps(
                        {
                            "orientation": orientation,
                            "processed": stop,
                            "total": frame.height,
                        }
                    ),
                    flush=True,
                )
        raw.flush()
        features.flush()
        assert np.isfinite(raw).all() and np.isfinite(features).all()
        matrices[orientation] = {"raw": raw, "sae": features}
    return matrices


def _context_sequence(
    row: dict[str, Any], genome: Genome, orientation: str
) -> tuple[str, str]:
    pos0 = int(row["pos"]) - 1
    start = pos0 - FOCAL_INDEX
    end = pos0 + FOCAL_INDEX + 1
    ref_sequence, alt_sequence = variant_sequences(
        genome(row["chrom"], start, end, "+").upper(), row["ref"], row["alt"]
    )
    if orientation == "reverse_complement":
        ref_sequence = reverse_complement(ref_sequence)
        alt_sequence = reverse_complement(alt_sequence)
    radius = CONTEXT_BP // 2
    return (
        ref_sequence[FOCAL_INDEX - radius : FOCAL_INDEX + radius + 1],
        alt_sequence[FOCAL_INDEX - radius : FOCAL_INDEX + radius + 1],
    )


def _plot_summary(
    summary: pl.DataFrame, baselines: pl.DataFrame, output_dir: Path
) -> None:
    subsets = summary["subset"].unique(maintain_order=True).to_list()
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    colors = {
        "sae:forward": "#0072B2",
        "sae:reverse_complement": "#56B4E9",
        "raw:forward": "#D55E00",
        "raw:reverse_complement": "#E69F00",
    }
    x = np.arange(len(subsets))
    for key, color in colors.items():
        space, orientation = key.split(":")
        values = summary.filter(
            (pl.col("space") == space) & (pl.col("orientation") == orientation)
        ).sort(pl.col("subset").replace_strict(subsets, list(range(len(subsets)))))
        assert values.height == len(subsets)
        axes[0].plot(
            x,
            values["test_average_precision"],
            marker="o",
            linewidth=1.5,
            label=key,
            color=color,
        )
        axes[1].plot(
            x,
            values["test_sign_stability"],
            marker="o",
            linewidth=1.5,
            label=key,
            color=color,
        )
    for baseline, style in (
        ("minus_llr_avg", "--"),
        ("probe_score", ":"),
        ("substitution_class", "-."),
    ):
        values = baselines.filter(pl.col("baseline") == baseline).sort(
            pl.col("subset").replace_strict(subsets, list(range(len(subsets))))
        )
        assert values.height == len(subsets)
        axes[0].plot(
            x,
            values["test_average_precision"],
            color="black",
            linestyle=style,
            marker=".",
            linewidth=1,
            label=baseline,
        )
    for axis, ylabel in zip(
        axes,
        ("held-out row-level AUPRC", "held-out matched sign stability"),
        strict=True,
    ):
        axis.axhline(0.1 if axis is axes[0] else 0.5, color="grey", linewidth=0.8)
        axis.set_xticks(x, subsets, rotation=55, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_title("Predictive ranking after frozen selection")
    axes[1].set_title("Fraction of matched groups with positive oriented effect")
    figure.suptitle("exp420: Mendelian ref→alt effects on chr11 + chrX")
    figure.savefig(output_dir / "summary.png", dpi=180)
    figure.savefig(output_dir / "summary.svg")
    plt.close(figure)


def _results_markdown(summary: pl.DataFrame, baselines: pl.DataFrame) -> str:
    lines = [
        "# exp420 Mendelian variant SAE deltas",
        "",
        "All displayed feature and raw-dimension results are from the untouched chr11 + chrX test split. Candidate ranking used discovery chromosomes; validation chose among five fixed candidates.",
        "",
        "| Subset | Orientation | SAE feature | SAE AP | SAE matched mean (95% CI) | Raw dimension | Raw AP | LLR AP | Probe AP | Substitution AP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in summary["subset"].unique(maintain_order=True):
        for orientation in ORIENTATIONS:
            sae = summary.filter(
                (pl.col("subset") == subset)
                & (pl.col("orientation") == orientation)
                & (pl.col("space") == "sae")
            ).row(0, named=True)
            raw = summary.filter(
                (pl.col("subset") == subset)
                & (pl.col("orientation") == orientation)
                & (pl.col("space") == "raw")
            ).row(0, named=True)
            baseline = {
                row["baseline"]: row
                for row in baselines.filter(pl.col("subset") == subset).to_dicts()
            }
            lines.append(
                f"| {subset} | {orientation} | {sae['dimension']} | "
                f"{sae['test_average_precision']:.4f} | "
                f"{sae['test_matched_mean']:.4g} "
                f"[{sae['test_matched_ci95_low']:.4g}, {sae['test_matched_ci95_high']:.4g}] | "
                f"{raw['dimension']} | {raw['test_average_precision']:.4f} | "
                f"{baseline['minus_llr_avg']['test_average_precision']:.4f} | "
                f"{baseline['probe_score']['test_average_precision']:.4f} | "
                f"{baseline['substitution_class']['test_average_precision']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Chance AUPRC is 0.1 because every matched group contains one positive and nine controls. Forward and reverse-complement SAE IDs are separate feature spaces. Coordinates are 0-based half-open after the VCF-position boundary conversion.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def evaluate(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    fasta_path: Path,
    sae_path: Path,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert panel_path.exists() and panel_manifest_path.exists()
    assert fasta_path.exists() and Path(f"{fasta_path}.fai").exists()
    assert Path(f"{fasta_path}.gzi").exists() and sae_path.exists()
    assert batch_size > 0 and not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert len(experiment_commit) == 40
    assert all(character in "0123456789abcdef" for character in experiment_commit)
    started = time.monotonic()

    panel_manifest = json.loads(panel_manifest_path.read_text())
    assert panel_manifest["panel_sha256"] == _sha256(panel_path)
    frame = pl.read_parquet(panel_path)
    _validate_panel(frame)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    sae = SAE.load_from_disk(sae_path, device="cuda", dtype="float32")
    assert sae.cfg.architecture() == "jumprelu"
    assert sae.cfg.d_in == M51_HIDDEN_SIZE and sae.cfg.d_sae == D_SAE
    genome = Genome(fasta_path, subset_chroms=set(frame["chrom"].unique()))
    assert set(genome.chroms) == set(frame["chrom"].unique())

    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    subsets = sorted(set(frame["subset"]) - DESCRIPTIVE_SUBSETS)
    with tempfile.TemporaryDirectory(prefix="dense-", dir=output_dir) as temp_name:
        matrices = _fill_delta_matrices(
            frame,
            genome=genome,
            frozen=frozen,
            sae=sae,
            temp_dir=Path(temp_name),
            batch_size=batch_size,
        )
        labels_all = frame["label"].to_numpy().astype(np.int8)
        groups_all = frame["match_group"].to_numpy()
        for subset in subsets:
            discovery_indices = _rows_for(frame, split="discovery", subset=subset)
            validation_indices = _rows_for(frame, split="validation", subset=subset)
            test_indices = _rows_for(frame, split="test", subset=subset)
            assert len(discovery_indices) > 10
            assert len(validation_indices) > 10
            assert len(test_indices) > 10
            for baseline, score_values in (
                ("minus_llr_avg", frame["minus_llr_avg"].to_numpy()),
                ("probe_score", frame["probe_score"].to_numpy()),
                ("substitution_class", _substitution_scores(frame, subset)),
            ):
                baseline_rows.append(
                    {
                        "subset": subset,
                        "baseline": baseline,
                        **_baseline_metrics(
                            score_values,
                            frame,
                            test_indices,
                            seed=_seed("baseline", subset, baseline),
                        ),
                    }
                )
            for orientation in ORIENTATIONS:
                for space in SPACES:
                    matrix = matrices[orientation][space]
                    discovery, _ = matched_contrasts(
                        matrix[discovery_indices],
                        labels_all[discovery_indices],
                        groups_all[discovery_indices],
                    )
                    validation, _ = matched_contrasts(
                        matrix[validation_indices],
                        labels_all[validation_indices],
                        groups_all[validation_indices],
                    )
                    selected, direction, replicated, candidates = select_candidate(
                        discovery, validation
                    )
                    for candidate in candidates:
                        candidate_rows.append(
                            {
                                "subset": subset,
                                "orientation": orientation,
                                "space": space,
                                "selected": candidate["dimension"] == selected,
                                **candidate,
                            }
                        )
                    result = _evaluate_fixed_dimension(
                        matrix,
                        frame,
                        test_indices,
                        dimension=selected,
                        direction=direction,
                        seed=_seed("test", subset, orientation, space),
                    )
                    summary_rows.append(
                        {
                            "subset": subset,
                            "orientation": orientation,
                            "space": space,
                            "dimension": selected,
                            "direction": direction,
                            "known_nucleotide_base": next(
                                (
                                    base
                                    for base, feature in KNOWN_NUCLEOTIDE_FEATURES.items()
                                    if space == "sae" and feature == selected
                                ),
                                None,
                            ),
                            "validation_direction_consistent": replicated,
                            **result,
                        }
                    )
                    if space != "sae":
                        continue
                    source_scores = direction * np.asarray(matrix[:, selected])
                    for target_subset in subsets:
                        target_indices = _rows_for(
                            frame, split="test", subset=target_subset
                        )
                        target_labels = labels_all[target_indices]
                        target_groups = groups_all[target_indices]
                        target_contrasts, _ = matched_contrasts(
                            source_scores[target_indices],
                            target_labels,
                            target_groups,
                        )
                        cross_rows.append(
                            {
                                "source_subset": subset,
                                "target_subset": target_subset,
                                "orientation": orientation,
                                "feature": selected,
                                "direction": direction,
                                "test_average_precision": float(
                                    average_precision_score(
                                        target_labels, source_scores[target_indices]
                                    )
                                ),
                                "test_matched_mean": float(target_contrasts.mean()),
                                "test_sign_stability": float(
                                    np.mean(target_contrasts > 0)
                                ),
                            }
                        )
                    test_labels = labels_all[test_indices]
                    test_scores = source_scores[test_indices]
                    for label in (1, 0):
                        eligible_local = np.flatnonzero(test_labels == label)
                        ordered_local = eligible_local[
                            np.argsort(-test_scores[eligible_local], kind="stable")[:5]
                        ]
                        for context_rank, local_index in enumerate(
                            ordered_local, start=1
                        ):
                            global_index = int(test_indices[local_index])
                            row = frame.row(global_index, named=True)
                            ref_context, alt_context = _context_sequence(
                                row, genome, orientation
                            )
                            context_rows.append(
                                {
                                    "source_subset": subset,
                                    "orientation": orientation,
                                    "feature": selected,
                                    "direction": direction,
                                    "label": label,
                                    "rank": context_rank,
                                    "score": float(source_scores[global_index]),
                                    "chrom": row["chrom"],
                                    "pos1": row["pos"],
                                    "pos0": row["pos"] - 1,
                                    "ref": row["ref"],
                                    "alt": row["alt"],
                                    "match_group": row["match_group"],
                                    "ref_context": ref_context,
                                    "alt_context": alt_context,
                                }
                            )

    summary = pl.DataFrame(summary_rows).sort(["subset", "orientation", "space"])
    candidates = pl.DataFrame(candidate_rows).sort(
        ["subset", "orientation", "space", "rank"]
    )
    baselines = pl.DataFrame(baseline_rows).sort(["subset", "baseline"])
    cross = pl.DataFrame(cross_rows).sort(
        ["source_subset", "orientation", "target_subset"]
    )
    contexts = pl.DataFrame(context_rows).sort(
        ["source_subset", "orientation", "label", "rank"],
        descending=[False, False, True, False],
    )
    assert summary.height == len(subsets) * len(ORIENTATIONS) * len(SPACES)
    assert candidates.height == summary.height * TOP_CANDIDATES
    assert baselines.height == len(subsets) * 3
    assert cross.height == len(subsets) ** 2 * len(ORIENTATIONS)
    assert contexts.height == len(subsets) * len(ORIENTATIONS) * 10
    for name, table in (
        ("summary", summary),
        ("candidates", candidates),
        ("baselines", baselines),
        ("cross_subset", cross),
        ("contexts", contexts),
    ):
        assert table.null_count().sum_horizontal().sum() == (
            summary["known_nucleotide_base"].null_count() if name == "summary" else 0
        )
        table.write_parquet(output_dir / f"{name}.parquet", compression="zstd")
    _plot_summary(summary, baselines, output_dir)
    (output_dir / "RESULTS.md").write_text(_results_markdown(summary, baselines))
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", ""),
        "experiment_commit": experiment_commit,
        "elapsed_seconds": time.monotonic() - started,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "block_index": BLOCK_INDEX,
        },
        "sae": {
            "path": str(sae_path),
            "architecture": sae.cfg.architecture(),
            "d_in": sae.cfg.d_in,
            "d_sae": sae.cfg.d_sae,
            "weights_sha256": _sha256(sae_path / "sae_weights.safetensors"),
        },
        "panel": {
            "path": str(panel_path),
            "sha256": _sha256(panel_path),
            "rows": frame.height,
            "match_groups": frame["match_group"].n_unique(),
        },
        "protocol": {
            "coordinate_system": "0-based half-open after pos0 = pos1 - 1",
            "window_bp": WINDOW_BP,
            "focal_index": FOCAL_INDEX,
            "chromosome_splits": {
                name: sorted(chroms) for name, chroms in CHROM_SPLITS.items()
            },
            "top_candidates": TOP_CANDIDATES,
            "bootstraps": BOOTSTRAPS,
            "permutations": PERMUTATIONS,
            "orientations": list(ORIENTATIONS),
            "descriptive_only_subsets": sorted(DESCRIPTIVE_SUBSETS),
        },
    }
    _write_json(output_dir / "results.json", result)
    artifact_files = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        **result,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in artifact_files
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("evaluate")
    run.add_argument("--panel", type=Path, required=True)
    run.add_argument("--panel-manifest", type=Path, required=True)
    run.add_argument("--fasta", type=Path, required=True)
    run.add_argument("--sae", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_panel(args.output)
    else:
        result = evaluate(
            panel_path=args.panel,
            panel_manifest_path=args.panel_manifest,
            fasta_path=args.fasta,
            sae_path=args.sae,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
