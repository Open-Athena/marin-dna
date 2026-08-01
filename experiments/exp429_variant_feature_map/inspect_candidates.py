"""Inspect discovery-set sequence grammar for strong issue #429 spatial features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from marin_dna.data.dna import reverse_complement
from marin_dna.data.genome import Genome

from analyze import sha256_file, write_json
from analyze_spatial import (
    EXPECTED_FEATURES,
    EXPECTED_POSITIONS,
    EXPECTED_RADIUS,
    EXPECTED_ROWS,
    aligned_orientation_profile,
    oriented_profile,
    spatial_scores,
)
from sample_panel import assert_current_commit

ISSUE = 429
NUCLEOTIDES = ("A", "C", "G", "T")
VALIDATION_AP_THRESHOLD = 0.25
TOP_CONTEXTS = 128
EXPECTED_CANDIDATES = {
    ("splice_acceptor_variant", 11698),
    ("splice_donor_5th_base_variant", 11681),
    ("stop_gained", 3312),
    ("synonymous_variant", 6072),
}


def response_orientation(
    forward: np.ndarray,
    reverse_complement_aligned: np.ndarray,
    metric: Literal["focal", "local_max", "local_sum"],
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the strand and response position behind a max-absolute score."""

    assert forward.shape == reverse_complement_aligned.shape
    assert forward.ndim == 2 and forward.shape[1] % 2 == 1
    center = forward.shape[1] // 2
    if metric == "focal":
        forward_strength = np.abs(forward[:, center])
        reverse_strength = np.abs(reverse_complement_aligned[:, center])
        use_reverse = reverse_strength > forward_strength
        position = np.full(forward.shape[0], center, dtype=np.int64)
    elif metric == "local_max":
        forward_position = np.argmax(np.abs(forward), axis=1)
        reverse_position = np.argmax(np.abs(reverse_complement_aligned), axis=1)
        row = np.arange(forward.shape[0])
        forward_strength = np.abs(forward[row, forward_position])
        reverse_strength = np.abs(reverse_complement_aligned[row, reverse_position])
        use_reverse = reverse_strength > forward_strength
        position = np.where(use_reverse, reverse_position, forward_position)
    elif metric == "local_sum":
        forward_strength = np.abs(forward).sum(axis=1)
        reverse_strength = np.abs(reverse_complement_aligned).sum(axis=1)
        use_reverse = reverse_strength > forward_strength
        chosen = np.where(use_reverse[:, None], reverse_complement_aligned, forward)
        position = np.argmax(np.abs(chosen), axis=1)
    else:
        raise AssertionError(metric)
    assert use_reverse.shape == position.shape == (forward.shape[0],)
    assert (position >= 0).all() and (position < forward.shape[1]).all()
    return use_reverse, position.astype(np.int64)


def variant_context(
    genome: Genome,
    *,
    chrom: str,
    pos1: int,
    ref: str,
    alt: str,
    use_reverse: bool,
) -> tuple[str, str, str, str]:
    """Return validated REF/ALT contexts and alleles in the chosen orientation."""

    pos0 = pos1 - 1
    reference = genome(
        chrom, pos0 - EXPECTED_RADIUS, pos0 + EXPECTED_RADIUS + 1, "+"
    ).upper()
    assert len(reference) == EXPECTED_POSITIONS
    assert reference[EXPECTED_RADIUS] == ref
    alternate = reference[:EXPECTED_RADIUS] + alt + reference[EXPECTED_RADIUS + 1 :]
    oriented_ref = ref
    oriented_alt = alt
    if use_reverse:
        reference = reverse_complement(reference)
        alternate = reverse_complement(alternate)
        oriented_ref = reverse_complement(ref)
        oriented_alt = reverse_complement(alt)
    assert reference[EXPECTED_RADIUS] == oriented_ref
    assert alternate[EXPECTED_RADIUS] == oriented_alt
    assert sum(a != b for a, b in zip(reference, alternate, strict=True)) == 1
    return reference, alternate, oriented_ref, oriented_alt


def frequency_rows(contexts: pl.DataFrame) -> list[dict[str, Any]]:
    """Return base frequencies for top, remainder, and all discovery contexts."""

    rows: list[dict[str, Any]] = []
    for candidate in contexts["class"].unique(maintain_order=True):
        candidate_frame = contexts.filter(pl.col("class") == candidate)
        for subset, frame in (
            ("top", candidate_frame.filter(pl.col("is_top"))),
            ("remainder", candidate_frame.filter(~pl.col("is_top"))),
            ("all", candidate_frame),
        ):
            assert frame.height > 0
            for allele, column in (("ref", "ref_context"), ("alt", "alt_context")):
                sequences = frame[column].to_list()
                assert all(
                    len(sequence) == EXPECTED_POSITIONS for sequence in sequences
                )
                for position in range(EXPECTED_POSITIONS):
                    bases = [sequence[position] for sequence in sequences]
                    assert set(bases) <= set(NUCLEOTIDES)
                    for base in NUCLEOTIDES:
                        count = bases.count(base)
                        rows.append(
                            {
                                "class": candidate,
                                "feature_id": frame["feature_id"].item(0),
                                "subset": subset,
                                "allele": allele,
                                "relative_position": position - EXPECTED_RADIUS,
                                "base": base,
                                "count": count,
                                "total": len(bases),
                                "frequency": count / len(bases),
                            }
                        )
    return rows


def substitution_rows(contexts: pl.DataFrame) -> list[dict[str, Any]]:
    """Summarize oriented substitutions among top versus remaining examples."""

    rows: list[dict[str, Any]] = []
    for candidate in contexts["class"].unique(maintain_order=True):
        candidate_frame = contexts.filter(pl.col("class") == candidate)
        for subset, frame in (
            ("top", candidate_frame.filter(pl.col("is_top"))),
            ("remainder", candidate_frame.filter(~pl.col("is_top"))),
            ("all", candidate_frame),
        ):
            counts = (
                frame.group_by("oriented_ref", "oriented_alt")
                .len()
                .sort("len", descending=True)
            )
            for row in counts.iter_rows(named=True):
                rows.append(
                    {
                        "class": candidate,
                        "feature_id": frame["feature_id"].item(0),
                        "subset": subset,
                        "substitution": f"{row['oriented_ref']}>{row['oriented_alt']}",
                        "count": row["len"],
                        "total": frame.height,
                        "fraction": row["len"] / frame.height,
                    }
                )
    return rows


def plot_base_enrichment(frequencies: pl.DataFrame, output_dir: Path) -> None:
    """Plot ALT-base enrichment in top responses versus same-class remainder."""

    candidates = frequencies["class"].unique(maintain_order=True).to_list()
    figure, axes = plt.subplots(len(candidates), 1, figsize=(12, 2.5 * len(candidates)))
    axes = np.atleast_1d(axes)
    images = []
    for axis, candidate in zip(axes, candidates, strict=True):
        frame = frequencies.filter(
            (pl.col("class") == candidate) & (pl.col("allele") == "alt")
        )
        top = (
            frame.filter(pl.col("subset") == "top")
            .pivot(index="base", on="relative_position", values="frequency")
            .sort("base")
        )
        remainder = (
            frame.filter(pl.col("subset") == "remainder")
            .pivot(index="base", on="relative_position", values="frequency")
            .sort("base")
        )
        assert top["base"].to_list() == remainder["base"].to_list() == list(NUCLEOTIDES)
        positions = [
            str(value) for value in range(-EXPECTED_RADIUS, EXPECTED_RADIUS + 1)
        ]
        top_values = top.select(positions).to_numpy()
        remainder_values = remainder.select(positions).to_numpy()
        enrichment = np.log2((top_values + 1e-3) / (remainder_values + 1e-3))
        image = axis.imshow(
            enrichment,
            aspect="auto",
            cmap="coolwarm",
            vmin=-3,
            vmax=3,
            interpolation="nearest",
        )
        images.append(image)
        axis.axvline(EXPECTED_RADIUS, color="#222222", linewidth=0.8)
        axis.set_yticks(range(len(NUCLEOTIDES)), NUCLEOTIDES)
        axis.set_xticks(
            range(0, EXPECTED_POSITIONS, 3),
            range(-EXPECTED_RADIUS, EXPECTED_RADIUS + 1, 3),
        )
        feature_id = frame["feature_id"].item(0)
        axis.set_title(f"{candidate.removesuffix('_variant')} — feature {feature_id}")
    figure.supxlabel("Position relative to edited base (chosen response orientation)")
    figure.supylabel("Alternate base")
    figure.colorbar(
        images[0], ax=axes.tolist(), label="log2 frequency: top / same-class remainder"
    )
    figure.suptitle("Sequence grammar of validation-selected SAE responses")
    figure.subplots_adjust(left=0.08, right=0.9, top=0.94, bottom=0.07, hspace=0.55)
    figure.savefig(output_dir / "candidate_base_enrichment.svg", bbox_inches="tight")
    figure.savefig(
        output_dir / "candidate_base_enrichment.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def inspect_candidates(
    *,
    panel_path: Path,
    spatial_dir: Path,
    spatial_analysis_dir: Path,
    fasta_path: Path,
    output_dir: Path,
    top_contexts: int,
) -> dict[str, Any]:
    """Export discovery-only sequence hypotheses for strong spatial candidates."""

    assert top_contexts > 0 and not output_dir.exists()
    inspection_commit = os.environ.get("INSPECTION_COMMIT", "")
    assert_current_commit(inspection_commit)
    panel = pl.read_parquet(panel_path)
    assert panel.height == EXPECTED_ROWS
    spatial_manifest_path = spatial_dir / "manifest.json"
    spatial_analysis_manifest_path = spatial_analysis_dir / "manifest.json"
    spatial_manifest = json.loads(spatial_manifest_path.read_text())
    spatial_analysis_manifest = json.loads(spatial_analysis_manifest_path.read_text())
    assert spatial_manifest["panel"]["sha256"] == sha256_file(panel_path)
    assert spatial_analysis_manifest["panel_sha256"] == sha256_file(panel_path)
    selected_path = spatial_analysis_dir / "selected_spatial_features.parquet"
    assert spatial_analysis_manifest["artifacts"][selected_path.name][
        "sha256"
    ] == sha256_file(selected_path)
    selected = pl.read_parquet(selected_path).filter(
        (pl.col("validation_average_precision") >= VALIDATION_AP_THRESHOLD)
        & (pl.col("direction") == 1)
    )
    observed_candidates = set(selected.select("class", "dimension").iter_rows())
    assert observed_candidates == EXPECTED_CANDIDATES
    assert selected["orientation"].unique().to_list() == ["max_absolute"]
    feature_ids = spatial_manifest["selection"]["feature_ids"]
    assert len(feature_ids) == EXPECTED_FEATURES
    feature_index = {int(value): index for index, value in enumerate(feature_ids)}
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for orientation in ("forward", "reverse_complement"):
        for allele in ("ref", "alt"):
            path = spatial_dir / f"spatial_{allele}_{orientation}.npy"
            assert spatial_manifest["artifacts"][path.name]["sha256"] == sha256_file(
                path
            )
            array = np.load(path, mmap_mode="r")
            assert array.shape == (EXPECTED_ROWS, EXPECTED_POSITIONS, EXPECTED_FEATURES)
            arrays[(orientation, allele)] = array

    genome = Genome(fasta_path, subset_chroms={"21"})
    assert set(genome.chroms) == {"21"}
    discovery = panel["split"].to_numpy() == "discovery"
    labels = panel["consequence_cre"].to_numpy()
    contexts: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for candidate in selected.sort("class").iter_rows(named=True):
        feature_id = int(candidate["dimension"])
        index = feature_index[feature_id]
        forward = np.asarray(
            arrays[("forward", "alt")][:, :, index]
            - arrays[("forward", "ref")][:, :, index],
            dtype=np.float32,
        )
        reverse = np.asarray(
            arrays[("reverse_complement", "alt")][:, :, index]
            - arrays[("reverse_complement", "ref")][:, :, index],
            dtype=np.float32,
        )[:, ::-1]
        profile = oriented_profile(
            aligned_orientation_profile(forward, reverse[:, ::-1], "max_absolute"),
            transform=candidate["transform"],
            direction=int(candidate["direction"]),
        )
        scores = spatial_scores(profile, candidate["spatial_metric"])
        use_reverse, response_position = response_orientation(
            forward, reverse, candidate["spatial_metric"]
        )
        eligible = np.flatnonzero(discovery & (labels == candidate["class"]))
        assert len(eligible) == 1_024 and top_contexts < len(eligible)
        ordered = eligible[np.argsort(-scores[eligible], kind="stable")]
        top_rows = set(map(int, ordered[:top_contexts]))
        summary_rows.append(
            {
                "class": candidate["class"],
                "feature_id": feature_id,
                "transform": candidate["transform"],
                "spatial_metric": candidate["spatial_metric"],
                "validation_average_precision": candidate[
                    "validation_average_precision"
                ],
                "test_spatial_average_precision": candidate[
                    "test_spatial_average_precision"
                ],
                "top_contexts": top_contexts,
                "top_score_min": float(scores[ordered[top_contexts - 1]]),
                "top_score_median": float(np.median(scores[ordered[:top_contexts]])),
                "remainder_score_median": float(
                    np.median(scores[ordered[top_contexts:]])
                ),
                "top_reverse_fraction": float(use_reverse[list(top_rows)].mean()),
            }
        )
        rank = {int(panel_row): index + 1 for index, panel_row in enumerate(ordered)}
        for panel_row in eligible:
            row = panel.row(int(panel_row), named=True)
            ref_context, alt_context, oriented_ref, oriented_alt = variant_context(
                genome,
                chrom=row["chrom"],
                pos1=int(row["pos"]),
                ref=row["ref"],
                alt=row["alt"],
                use_reverse=bool(use_reverse[panel_row]),
            )
            contexts.append(
                {
                    "class": candidate["class"],
                    "feature_id": feature_id,
                    "transform": candidate["transform"],
                    "spatial_metric": candidate["spatial_metric"],
                    "panel_row": int(panel_row),
                    "rank": rank[int(panel_row)],
                    "is_top": int(panel_row) in top_rows,
                    "score": float(scores[panel_row]),
                    "response_orientation": (
                        "reverse_complement" if use_reverse[panel_row] else "forward"
                    ),
                    "response_relative_position": int(
                        response_position[panel_row] - EXPECTED_RADIUS
                    ),
                    "chrom": row["chrom"],
                    "pos": row["pos"],
                    "ref": row["ref"],
                    "alt": row["alt"],
                    "oriented_ref": oriented_ref,
                    "oriented_alt": oriented_alt,
                    "ref_context": ref_context,
                    "alt_context": alt_context,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    contexts_frame = pl.DataFrame(contexts).sort(["class", "rank"])
    summary_frame = pl.DataFrame(summary_rows).sort("class")
    frequencies_frame = pl.DataFrame(frequency_rows(contexts_frame)).sort(
        ["class", "subset", "allele", "relative_position", "base"]
    )
    substitutions_frame = pl.DataFrame(substitution_rows(contexts_frame)).sort(
        ["class", "subset", "count"], descending=[False, False, True]
    )
    assert contexts_frame.height == len(EXPECTED_CANDIDATES) * 1_024
    assert (
        contexts_frame.filter(pl.col("is_top")).height
        == len(EXPECTED_CANDIDATES) * top_contexts
    )
    contexts_frame.write_parquet(output_dir / "candidate_contexts.parquet")
    summary_frame.write_parquet(output_dir / "candidate_summary.parquet")
    frequencies_frame.write_parquet(output_dir / "candidate_base_frequencies.parquet")
    substitutions_frame.write_parquet(output_dir / "candidate_substitutions.parquet")
    plot_base_enrichment(frequencies_frame, output_dir)

    summary = {
        "issue": ISSUE,
        "inspection_commit": inspection_commit,
        "panel_sha256": sha256_file(panel_path),
        "spatial_manifest_sha256": sha256_file(spatial_manifest_path),
        "spatial_analysis_manifest_sha256": sha256_file(spatial_analysis_manifest_path),
        "fasta": {"path": str(fasta_path), "sha256": sha256_file(fasta_path)},
        "protocol": {
            "candidate_selection": f"validation AP >= {VALIDATION_AP_THRESHOLD} and positive direction; no test selection",
            "candidate_count": len(EXPECTED_CANDIDATES),
            "top_contexts": top_contexts,
            "sequence_split": "discovery only",
            "comparison": "top response versus remaining variants within the same consequence class",
            "orientation": "per-variant FWD/RC contributor to the frozen max-absolute spatial score",
        },
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
    parser.add_argument("--spatial-dir", type=Path, required=True)
    parser.add_argument("--spatial-analysis-dir", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-contexts", type=int, default=TOP_CONTEXTS)
    args = parser.parse_args()
    manifest = inspect_candidates(
        panel_path=args.panel,
        spatial_dir=args.spatial_dir,
        spatial_analysis_dir=args.spatial_analysis_dir,
        fasta_path=args.fasta,
        output_dir=args.output_dir,
        top_contexts=args.top_contexts,
    )
    print(json.dumps(manifest["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
