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
    validate_experiment_commit,
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
    "mature_miRNA_variant",
    "non_coding_transcript_exon_variant",
    "5_prime_UTR_variant",
    "tss_proximal",
    "distal",
)
PRESENTATION_SUBSETS = {
    "cds": ("missense_variant", "splicing", "synonymous_variant"),
    "enhancer": ("distal",),
}
OFFICIAL_ENDPOINT_DATASETS = {
    "cds": ("complex_traits", "sge"),
    "enhancer": ("complex_traits",),
}
OFFICIAL_PRESENTATION_SUBSETS = {
    ("cds", "complex_traits"): PRESENTATION_SUBSETS["cds"],
    ("cds", "sge"): ("missense_variant", "splicing"),
    ("enhancer", "complex_traits"): PRESENTATION_SUBSETS["enhancer"],
}
SUBSET_LABELS = {
    "missense_variant": "Missense variant",
    "synonymous_variant": "Synonymous variant",
    "splicing": "Splicing",
    "3_prime_UTR_variant": "3′ UTR variant",
    "mature_miRNA_variant": "Mature miRNA variant",
    "non_coding_transcript_exon_variant": "Non-coding transcript exon",
    "5_prime_UTR_variant": "5′ UTR variant",
    "tss_proximal": "TSS-proximal",
    "distal": "Distal",
}
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


def score_uri(
    results_root: str,
    arm: str,
    step: int,
    dataset: str,
    *,
    experiment_commit: str,
) -> str:
    name = model_name(arm, step, experiment_commit=experiment_commit)
    return f"{results_root.rstrip('/')}/scores/{name}/{dataset}.parquet"


def metric_uri(
    results_root: str,
    arm: str,
    step: int,
    dataset: str,
    *,
    experiment_commit: str,
) -> str:
    name = model_name(arm, step, experiment_commit=experiment_commit)
    return f"{results_root.rstrip('/')}/metrics/{name}/{dataset}.parquet"


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


def read_development_metric(uri: str, *, reader: ParquetReader) -> pd.DataFrame:
    """Refuse a score bundle unless its matching metric records ``train``."""
    frame = reader(uri)
    assert "split" in frame, f"metric bundle has no split provenance: {uri}"
    assert set(frame["split"].astype(str)) == {DEVELOPMENT_SPLIT}, (
        f"refusing non-development metrics from {uri}"
    )
    return frame


def load_policy_pair(
    results_root: str,
    region: str,
    step: int,
    *,
    experiment_commit: str,
    reader: ParquetReader = read_parquet,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, str], tuple[str, str]]:
    arms = REGION_ARMS[region]
    full_uri = score_uri(
        results_root,
        arms["full_window"],
        step,
        "mendelian_traits",
        experiment_commit=experiment_commit,
    )
    center_uri = score_uri(
        results_root,
        arms["center_1"],
        step,
        "mendelian_traits",
        experiment_commit=experiment_commit,
    )
    full = reader(full_uri)
    center = reader(center_uri)
    full_metric_uri = metric_uri(
        results_root,
        arms["full_window"],
        step,
        "mendelian_traits",
        experiment_commit=experiment_commit,
    )
    center_metric_uri = metric_uri(
        results_root,
        arms["center_1"],
        step,
        "mendelian_traits",
        experiment_commit=experiment_commit,
    )
    read_development_metric(full_metric_uri, reader=reader)
    read_development_metric(center_metric_uri, reader=reader)
    validate_policy_pair(full, center)
    observed = tuple(sorted(full["subset"].astype(str).unique()))
    assert observed == tuple(sorted(MENDELIAN_SUBSETS)), (
        f"Mendelian subset contract changed: expected {sorted(MENDELIAN_SUBSETS)}, "
        f"got {list(observed)}"
    )
    return full, center, (full_uri, center_uri), (full_metric_uri, center_metric_uri)


def analyze_mendelian(
    results_root: str,
    *,
    n_bootstrap: int = 1_000,
    seed: int = 473,
    experiment_commit: str,
    reader: ParquetReader = read_parquet,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Analyze all 2 regions × 9 steps × 8 Mendelian subsets."""
    experiment_commit = validate_experiment_commit(experiment_commit)
    point_parts: list[pd.DataFrame] = []
    sample_parts: list[pd.DataFrame] = []
    delta_parts: list[pd.DataFrame] = []
    inputs: list[str] = []
    metric_inputs: list[str] = []
    for region in REGION_ARMS:
        for step in CHECKPOINT_STEPS:
            full, center, uris, metric_uris = load_policy_pair(
                results_root,
                region,
                step,
                experiment_commit=experiment_commit,
                reader=reader,
            )
            inputs.extend(uris)
            metric_inputs.extend(metric_uris)
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
    return points, samples, deltas, inputs, metric_inputs


def collect_official_endpoints(
    results_root: str,
    *,
    experiment_commit: str,
    reader: ParquetReader = read_parquet,
) -> tuple[pd.DataFrame, list[str]]:
    """Collect official Complex (both regions) and SGE (CDS) trajectories."""
    experiment_commit = validate_experiment_commit(experiment_commit)
    rows: list[pd.DataFrame] = []
    inputs: list[str] = []
    for region, arms in REGION_ARMS.items():
        for dataset in OFFICIAL_ENDPOINT_DATASETS[region]:
            for policy, arm in arms.items():
                assert dataset in ARM_DATASETS[arm]
                for step in CHECKPOINT_STEPS:
                    uri = metric_uri(
                        results_root,
                        arm,
                        step,
                        dataset,
                        experiment_commit=experiment_commit,
                    )
                    frame = read_development_metric(uri, reader=reader)
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


def final_official_endpoint_table(endpoints: pd.DataFrame) -> pd.DataFrame:
    """Select comparable final endpoint rows for region-relevant subsets."""
    required = {
        "region",
        "dataset",
        "subset",
        "policy",
        "step",
        "value",
        "se",
        "score_type",
        "metric",
        "accession",
        "gene",
    }
    missing = required - set(endpoints.columns)
    assert not missing, f"official endpoints missing columns {sorted(missing)}"
    final = endpoints[endpoints["step"] == max(CHECKPOINT_STEPS)].copy()
    parts: list[pd.DataFrame] = []
    for (region, dataset), subsets in OFFICIAL_PRESENTATION_SUBSETS.items():
        cell = final[
            (final["region"] == region)
            & (final["dataset"] == dataset)
            & final["subset"].isin(subsets)
        ]
        if dataset == "complex_traits":
            cell = cell[cell["score_type"] == "abs_llr_avg"]
        else:
            cell = cell[
                (cell["metric"] == "AUPRC")
                & (cell["score_type"] == "minus_llr_avg")
                & (cell["accession"] == "_macro_avg_")
                & (cell["gene"] == "_macro_avg_")
            ]
        parts.append(cell)
    selected = pd.concat(parts, ignore_index=True)
    identity = ["region", "dataset", "subset"]
    full = selected[selected["policy"] == "full_window"][
        identity + ["value", "se"]
    ].rename(columns={"value": "full_window", "se": "full_window_se"})
    center = selected[selected["policy"] == "center_1"][
        identity + ["value", "se"]
    ].rename(columns={"value": "center_1", "se": "center_1_se"})
    result = full.merge(center, on=identity, validate="one_to_one")
    assert len(result) == sum(
        len(subsets) for subsets in OFFICIAL_PRESENTATION_SUBSETS.values()
    )
    result["delta_center_minus_full"] = (
        result["center_1"] - result["full_window"]
    )
    return result.sort_values(identity).reset_index(drop=True)


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


def select_presentation_subsets(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only region-relevant subsets for figures and written results."""
    required = {"region", "subset"}
    missing = required - set(frame.columns)
    assert not missing, f"presentation frame missing columns {sorted(missing)}"
    selected = pd.Series(False, index=frame.index)
    for region, subsets in PRESENTATION_SUBSETS.items():
        selected |= (frame["region"] == region) & frame["subset"].isin(subsets)
    result = frame.loc[selected].copy()
    assert not result.empty, "no region-relevant presentation rows"
    return result


def plot_deltas(deltas: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Render region-relevant trajectory subsets for each region and metric."""
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "auprc": "AUPRC difference",
        "group_smd": "Group SMD difference",
    }
    region_labels = {"cds": "CDS", "enhancer": "Enhancer"}
    for region in REGION_ARMS:
        subsets = PRESENTATION_SUBSETS[region]
        for metric in ("auprc", "group_smd"):
            fig, axes = plt.subplots(
                1,
                len(subsets),
                figsize=(4.2 * len(subsets), 5),
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            for axis, subset in zip(axes.flat, subsets, strict=True):
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
                    color="C0",
                    alpha=0.16,
                    linewidth=0,
                )
                axis.plot(
                    cell["step"],
                    cell["delta_center_minus_full"],
                    color="C0",
                    marker="o",
                    linewidth=1.8,
                    markersize=3.5,
                )
                axis.set_title(SUBSET_LABELS[subset])
                axis.set_box_aspect(1)
                axis.grid(alpha=0.2, linewidth=0.6)
            fig.supxlabel("Training step", y=0.11)
            fig.supylabel(f"Center 1 − full window {labels[metric]}")
            fig.suptitle(
                f"{region_labels[region]} paired {labels[metric]} trajectories",
            )
            fig.text(
                0.5,
                0.025,
                "Development split; paired 95% bootstrap intervals by match group.",
                ha="center",
            )
            fig.tight_layout(rect=(0.03, 0.17, 1, 0.93))
            stem = f"{region}_{metric}_paired_delta_trajectories"
            path = plot_dir / f"{stem}.svg"
            fig.savefig(path, bbox_inches="tight")
            paths.append(path)
            plt.close(fig)
    return paths


def write_summary(
    deltas: pd.DataFrame,
    triggers: pd.DataFrame,
    endpoints: pd.DataFrame,
    output: Path,
) -> None:
    presented_deltas = select_presentation_subsets(deltas)
    presented_triggers = select_presentation_subsets(triggers)
    official = final_official_endpoint_table(endpoints)
    final = presented_deltas[
        presented_deltas["step"] == max(CHECKPOINT_STEPS)
    ].copy()
    lines = [
        "# Issue #473 development evaluation",
        "",
        (
            "All results use the official `evals_v2` `train` split. No held-out "
            "even-autosome or chromosome-Y label, prediction, or aggregate metric "
            "is read. Positive deltas favor `center_1`."
        ),
        "",
        (
            "Only region-relevant subsets are presented: missense, splicing, "
            "and synonymous for CDS; distal for enhancer. The complete audit "
            "artifacts retain every registered Mendelian subset."
        ),
        "",
        "## Final-checkpoint relevant paired Mendelian deltas",
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
            "## Final-checkpoint relevant official endpoints",
            "",
            (
                "Complex uses official `abs_llr_avg` AUPRC. SGE uses official "
                "assay-macro `minus_llr_avg` AUPRC. Values are point estimate ± "
                "official bootstrap SE; the delta is center 1 minus full window."
            ),
            "",
            "| Region | Dataset | Subset | Full window | Center 1 | Delta |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in official.itertuples():
        lines.append(
            f"| {row.region} | {row.dataset} | {row.subset} | "
            f"{row.full_window:.6g} ± {row.full_window_se:.3g} | "
            f"{row.center_1:.6g} ± {row.center_1_se:.3g} | "
            f"{row.delta_center_minus_full:.6g} |"
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
                "Triggered relevant cells: "
                f"{int(presented_triggers['additional_seed_trigger'].sum())} / "
                f"{len(presented_triggers)}."
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
    experiment_commit: str,
    analysis_commit: str,
) -> None:
    experiment_commit = validate_experiment_commit(experiment_commit)
    analysis_commit = validate_experiment_commit(analysis_commit)
    output_dir.mkdir(parents=True, exist_ok=True)
    points, samples, deltas, score_inputs, mendelian_metric_inputs = analyze_mendelian(
        results_root,
        n_bootstrap=n_bootstrap,
        seed=seed,
        experiment_commit=experiment_commit,
    )
    endpoints, endpoint_metric_inputs = collect_official_endpoints(
        results_root, experiment_commit=experiment_commit
    )
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
    write_summary(deltas, triggers, endpoints, summary_path)
    output_paths.append(summary_path)

    manifest = {
        "experiment_commit": experiment_commit,
        "analysis_commit": analysis_commit,
        "split": DEVELOPMENT_SPLIT,
        "held_out_access": False,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "score_inputs": sorted(set(score_inputs)),
        "metric_inputs": sorted(set(mendelian_metric_inputs + endpoint_metric_inputs)),
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
    parser.add_argument("--experiment-commit", required=True)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()
    run_analysis(
        args.results_root,
        args.output_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        experiment_commit=args.experiment_commit,
        analysis_commit=args.analysis_commit,
    )


if __name__ == "__main__":
    main()
