"""Analyze issue #473 official development-only evaluator outputs.

The official ``evals_v2`` workflow creates the score and metric parquets.  This
module adds only the preregistered paired policy analysis: AUPRC is primary,
Group SMD is secondary on compatible Mendelian match groups, and every center
minus full-window interval uses aligned match-group resamples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exp473_center_seeded_projection.eval_config import (
    ARM_DATASETS,
    CHECKPOINT_STEPS,
    model_name,
)
from exp473_center_seeded_projection.paired_metrics import paired_policy_bootstrap

DEFAULT_RESULTS_ROOT = "s3://oa-bolinas/snakemake/analysis/evals_v2/results"
S3_STORAGE_OPTIONS = {"client_kwargs": {"region_name": "us-east-2"}}
DEVELOPMENT_SPLIT = "train"
REGION_ARMS = {
    "cds": {
        "full_window": "cds_full_window",
        "center_1": "cds_center_1",
    },
    "enhancer": {
        "full_window": "enhancer_full_window",
        "center_1": "enhancer_center_1",
    },
}
MENDELIAN_SUBSETS = (
    "missense_variant",
    "synonymous_variant",
    "splicing",
    "3_prime_UTR_variant",
    "non_coding_transcript_exon_variant",
    "5_prime_UTR_variant",
    "tss_proximal",
    "distal",
)
ROW_IDENTITY_COLUMNS = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "subset",
    "match_group",
)

ParquetReader = Callable[[str], pd.DataFrame]


def score_uri(results_root: str, arm: str, step: int, dataset: str) -> str:
    return (
        f"{results_root.rstrip('/')}/scores/{model_name(arm, step)}/{dataset}.parquet"
    )


def metric_uri(results_root: str, arm: str, step: int, dataset: str) -> str:
    return (
        f"{results_root.rstrip('/')}/metrics/{model_name(arm, step)}/{dataset}.parquet"
    )


def read_parquet(uri: str) -> pd.DataFrame:
    options = S3_STORAGE_OPTIONS if uri.startswith("s3://") else None
    return pd.read_parquet(uri, storage_options=options)


def minus_llr_avg(frame: pd.DataFrame) -> np.ndarray:
    """Official Mendelian score: negate the mean raw FWD/RC LLR."""
    required = {"llr_fwd", "llr_rc"}
    missing = required - set(frame.columns)
    assert not missing, f"score bundle missing columns {sorted(missing)}"
    score = (
        -(
            frame["llr_fwd"].to_numpy(dtype=float)
            + frame["llr_rc"].to_numpy(dtype=float)
        )
        / 2.0
    )
    assert np.isfinite(score).all(), "minus_llr_avg contains non-finite values"
    return score


def validate_policy_pair(full: pd.DataFrame, center: pd.DataFrame) -> None:
    """Assert exact row alignment before any paired calculation."""
    for name, frame in (("full_window", full), ("center_1", center)):
        missing = set(ROW_IDENTITY_COLUMNS) - set(frame.columns)
        assert not missing, f"{name}: missing identity columns {sorted(missing)}"
    assert len(full) == len(center), (
        f"policy row count differs: full={len(full)} center={len(center)}"
    )
    full_identity = full[list(ROW_IDENTITY_COLUMNS)].reset_index(drop=True)
    center_identity = center[list(ROW_IDENTITY_COLUMNS)].reset_index(drop=True)
    assert full_identity.equals(center_identity), (
        "policy score bundles are not row-identical; paired analysis is invalid"
    )


def load_policy_pair(
    results_root: str,
    region: str,
    step: int,
    *,
    reader: ParquetReader = read_parquet,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, str]]:
    arms = REGION_ARMS[region]
    full_uri = score_uri(results_root, arms["full_window"], step, "mendelian_traits")
    center_uri = score_uri(results_root, arms["center_1"], step, "mendelian_traits")
    full = reader(full_uri)
    center = reader(center_uri)
    validate_policy_pair(full, center)
    observed = tuple(sorted(full["subset"].astype(str).unique()))
    assert observed == tuple(sorted(MENDELIAN_SUBSETS)), (
        f"Mendelian subset contract changed: expected {sorted(MENDELIAN_SUBSETS)}, "
        f"got {list(observed)}"
    )
    return full, center, (full_uri, center_uri)


def analyze_mendelian(
    results_root: str,
    *,
    n_bootstrap: int = 1_000,
    seed: int = 473,
    reader: ParquetReader = read_parquet,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Analyze all 2 regions × 10 steps × 8 Mendelian subsets."""
    point_parts: list[pd.DataFrame] = []
    sample_parts: list[pd.DataFrame] = []
    delta_parts: list[pd.DataFrame] = []
    inputs: list[str] = []
    for region in REGION_ARMS:
        for step in CHECKPOINT_STEPS:
            full, center, uris = load_policy_pair(
                results_root, region, step, reader=reader
            )
            inputs.extend(uris)
            for subset_index, subset in enumerate(MENDELIAN_SUBSETS):
                selected = full["subset"].astype(str) == subset
                full_subset = full.loc[selected].reset_index(drop=True)
                center_subset = center.loc[selected].reset_index(drop=True)
                scores = pd.DataFrame(
                    {
                        "full_window": minus_llr_avg(full_subset),
                        "center_1": minus_llr_avg(center_subset),
                    }
                )
                result = paired_policy_bootstrap(
                    full_subset["label"],
                    scores,
                    full_subset["match_group"],
                    n_bootstrap=n_bootstrap,
                    # Same subset seed at every region/step aligns trajectories.
                    seed=seed + subset_index,
                )
                context: dict[str, str | int] = {
                    "region": region,
                    "step": step,
                    "subset": subset,
                    "split": DEVELOPMENT_SPLIT,
                    "bootstrap_seed": seed + subset_index,
                }
                point_parts.append(result.point.assign(**context))
                sample_parts.append(result.samples.assign(**context))
                delta_parts.append(result.deltas.assign(**context))
    points = pd.concat(point_parts, ignore_index=True)
    samples = pd.concat(sample_parts, ignore_index=True)
    deltas = pd.concat(delta_parts, ignore_index=True)
    assert set(points["split"]) == {DEVELOPMENT_SPLIT}
    assert set(deltas["split"]) == {DEVELOPMENT_SPLIT}
    return points, samples, deltas, inputs


def collect_official_endpoints(
    results_root: str, *, reader: ParquetReader = read_parquet
) -> tuple[pd.DataFrame, list[str]]:
    """Collect official Complex (enhancer) and SGE (CDS) metric trajectories."""
    rows: list[pd.DataFrame] = []
    inputs: list[str] = []
    for region, arms in REGION_ARMS.items():
        dataset = "sge" if region == "cds" else "complex_traits"
        for policy, arm in arms.items():
            assert dataset in ARM_DATASETS[arm]
            for step in CHECKPOINT_STEPS:
                uri = metric_uri(results_root, arm, step, dataset)
                frame = reader(uri)
                assert "split" in frame
                assert set(frame["split"].astype(str)) == {DEVELOPMENT_SPLIT}, (
                    f"refusing non-development metrics from {uri}"
                )
                rows.append(
                    frame.assign(
                        region=region,
                        policy=policy,
                        step=step,
                        source_uri=uri,
                    )
                )
                inputs.append(uri)
    return pd.concat(rows, ignore_index=True), inputs


def seed_trigger_table(deltas: pd.DataFrame) -> pd.DataFrame:
    """Flag preregistered evidence that may justify a later additional seed."""
    rows: list[dict[str, Any]] = []
    for keys, group in deltas.groupby(["region", "subset", "metric"], sort=False):
        ordered = group.sort_values("step")
        values = ordered["delta_center_minus_full"].to_numpy(dtype=float)
        signs = np.sign(values)
        persistent = bool(np.any((signs[1:] == signs[:-1]) & (signs[1:] != 0)))
        final = ordered.iloc[-1]
        endpoint_excludes_zero = bool(final["ci_low"] > 0 or final["ci_high"] < 0)
        rows.append(
            {
                "region": keys[0],
                "subset": keys[1],
                "metric": keys[2],
                "two_consecutive_same_direction": persistent,
                "endpoint_interval_excludes_zero": endpoint_excludes_zero,
                "additional_seed_trigger": persistent or endpoint_excludes_zero,
            }
        )
    return pd.DataFrame(rows)


def plot_deltas(deltas: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Render one compact eight-subset trajectory figure per region/metric."""
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "auprc": "AUPRC delta",
        "group_smd": "Group SMD delta",
    }
    for region in REGION_ARMS:
        for metric in ("auprc", "group_smd"):
            fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
            for axis, subset in zip(axes.flat, MENDELIAN_SUBSETS, strict=True):
                cell = deltas[
                    (deltas["region"] == region)
                    & (deltas["metric"] == metric)
                    & (deltas["subset"] == subset)
                ].sort_values("step")
                assert cell["step"].tolist() == list(CHECKPOINT_STEPS)
                axis.axhline(0, color="#666666", linewidth=0.8)
                axis.fill_between(
                    cell["step"],
                    cell["ci_low"],
                    cell["ci_high"],
                    color="#0072B2",
                    alpha=0.16,
                    linewidth=0,
                )
                axis.plot(
                    cell["step"],
                    cell["delta_center_minus_full"],
                    color="#0072B2",
                    marker="o",
                    linewidth=1.8,
                    markersize=3.5,
                )
                axis.set_title(subset.replace("_", " "), fontsize=10)
                axis.grid(alpha=0.2, linewidth=0.6)
            for axis in axes[-1]:
                axis.set_xlabel("training step")
            fig.suptitle(
                f"Issue #473 {region}: center_1 − full_window {labels[metric]}",
                fontsize=14,
            )
            fig.text(
                0.5,
                0.012,
                "Development split only. Ribbons are paired 95% percentile "
                "intervals from aligned match-group bootstrap draws.",
                ha="center",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.035, 1, 0.97))
            stem = f"{region}_{metric}_paired_delta_trajectories"
            for extension, kwargs in (("svg", {}), ("png", {"dpi": 140})):
                path = plot_dir / f"{stem}.{extension}"
                fig.savefig(path, bbox_inches="tight", **kwargs)
                paths.append(path)
            plt.close(fig)
    return paths


def write_summary(
    deltas: pd.DataFrame,
    triggers: pd.DataFrame,
    output: Path,
) -> None:
    final = deltas[deltas["step"] == max(CHECKPOINT_STEPS)].copy()
    lines = [
        "# Issue #473 development evaluation",
        "",
        (
            "All results use the official `evals_v2` `train` split. No held-out "
            "even-autosome or chromosome-Y label, prediction, or aggregate metric "
            "is read. Positive deltas favor `center_1`."
        ),
        "",
        "## Final-checkpoint paired Mendelian deltas",
        "",
        "| Region | Subset | Metric | Delta | 95% interval | P(center better) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in final.sort_values(["region", "subset", "metric"]).itertuples():
        lines.append(
            f"| {row.region} | {row.subset} | {row.metric} | "
            f"{row.delta_center_minus_full:.6g} | "
            f"[{row.ci_low:.6g}, {row.ci_high:.6g}] | "
            f"{row.probability_center_better:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Additional-seed gate",
            "",
            (
                "This one-seed experiment remains exploratory. A flagged row records "
                "the preregistered evidence gate; it does not authorize or launch an "
                "additional seed."
            ),
            "",
            (
                f"Triggered cells: {int(triggers['additional_seed_trigger'].sum())} / "
                f"{len(triggers)}."
            ),
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_analysis(
    results_root: str,
    output_dir: Path,
    *,
    n_bootstrap: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    points, samples, deltas, score_inputs = analyze_mendelian(
        results_root, n_bootstrap=n_bootstrap, seed=seed
    )
    endpoints, metric_inputs = collect_official_endpoints(results_root)
    triggers = seed_trigger_table(deltas)

    outputs = {
        "paired_mendelian_metrics.parquet": points,
        "paired_mendelian_bootstrap_samples.parquet": samples,
        "paired_mendelian_deltas.parquet": deltas,
        "official_complex_sge_metrics.parquet": endpoints,
        "additional_seed_gate.parquet": triggers,
    }
    output_paths: list[Path] = []
    for name, frame in outputs.items():
        path = output_dir / name
        frame.to_parquet(path, index=False)
        output_paths.append(path)
    output_paths.extend(plot_deltas(deltas, output_dir))
    summary_path = output_dir / "summary.md"
    write_summary(deltas, triggers, summary_path)
    output_paths.append(summary_path)

    manifest = {
        "split": DEVELOPMENT_SPLIT,
        "held_out_access": False,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "score_inputs": sorted(set(score_inputs)),
        "metric_inputs": sorted(set(metric_inputs)),
        "outputs": {
            str(path.relative_to(output_dir)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(output_paths)
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {output_dir}: {len(points)} point rows, {len(deltas)} deltas, "
        f"{len(samples)} aligned bootstrap rows"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=473)
    args = parser.parse_args()
    run_analysis(
        args.results_root,
        args.output_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
