"""Rebuild the issue #473 CDS comparison after the legacy-RoPE audit.

The original issue-specific evaluator loaded the reused issue #417 checkpoints
without the maintained ``evals_v2`` compatibility translation.  This module
combines the repaired canonical issue #417 outputs with the unaffected issue
#473 CDS center-1 outputs.  It emits audit tables and provenance only; it does
not make or publish an interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from exp473_center_seeded_projection.analyze_evals import (
    ROW_IDENTITY_COLUMNS,
    minus_llr_avg,
    read_development_metric,
    read_parquet,
    validate_policy_pair,
)
from exp473_center_seeded_projection.eval_config import (
    CHECKPOINT_STEPS,
    model_name,
    validate_experiment_commit,
)
from exp473_center_seeded_projection.paired_metrics import paired_policy_bootstrap

CANONICAL_RESULTS_ROOT = "s3://oa-bolinas/snakemake/analysis/evals_v2/results"
ISSUE_RESULTS_ROOT = (
    f"{CANONICAL_RESULTS_ROOT}/issue473/"
    "ae90f6d9e4b23ebe8fb1bd2314baa66cb82b37c1/development_eval"
)
DEVELOPMENT_SPLIT = "train"
MENDELIAN_SUBSETS = (
    "missense_variant",
    "splicing",
    "synonymous_variant",
)
SGE_SUBSETS = ("missense_variant", "splicing")


def join_uri(root: str, *parts: str) -> str:
    """Join either an object-store URI or a local filesystem root."""
    if "://" in root:
        return "/".join((root.rstrip("/"), *(part.strip("/") for part in parts)))
    return str(Path(root).joinpath(*parts))


def baseline_model_name(step: int) -> str:
    """Return the canonical repaired issue #417 model ID."""
    assert step in CHECKPOINT_STEPS
    return f"exp417-cds-combined-vertebrates-step-{step}"


def center_model_name(step: int, *, experiment_commit: str) -> str:
    """Return the immutable issue #473 CDS center-1 model ID."""
    return model_name("cds_center_1", step, experiment_commit=experiment_commit)


def artifact_uri(root: str, kind: str, model: str, dataset: str) -> str:
    """Build one score or metric artifact URI."""
    assert kind in {"scores", "metrics"}
    return join_uri(root, kind, model, f"{dataset}.parquet")


def _load_metric(uri: str) -> pd.DataFrame:
    return read_development_metric(uri, reader=read_parquet)


def analyze_mendelian(
    canonical_root: str,
    issue_root: str,
    *,
    experiment_commit: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Compute repaired paired Mendelian trajectories on exact shared rows."""
    point_parts: list[pd.DataFrame] = []
    sample_parts: list[pd.DataFrame] = []
    delta_parts: list[pd.DataFrame] = []
    inputs: list[str] = []
    for step in CHECKPOINT_STEPS:
        full_model = baseline_model_name(step)
        center_model = center_model_name(step, experiment_commit=experiment_commit)
        full_score_uri = artifact_uri(
            canonical_root, "scores", full_model, "mendelian_traits"
        )
        center_score_uri = artifact_uri(
            issue_root, "scores", center_model, "mendelian_traits"
        )
        full_metric_uri = artifact_uri(
            canonical_root, "metrics", full_model, "mendelian_traits"
        )
        center_metric_uri = artifact_uri(
            issue_root, "metrics", center_model, "mendelian_traits"
        )
        _load_metric(full_metric_uri)
        _load_metric(center_metric_uri)
        full = read_parquet(full_score_uri)
        center = read_parquet(center_score_uri)
        validate_policy_pair(full, center)
        assert tuple(full[list(ROW_IDENTITY_COLUMNS)].columns) == ROW_IDENTITY_COLUMNS
        inputs.extend(
            [full_score_uri, center_score_uri, full_metric_uri, center_metric_uri]
        )

        for subset_index, subset in enumerate(MENDELIAN_SUBSETS):
            selected = full["subset"].astype(str) == subset
            full_subset = full.loc[selected].reset_index(drop=True)
            center_subset = center.loc[selected].reset_index(drop=True)
            result = paired_policy_bootstrap(
                full_subset["label"],
                pd.DataFrame(
                    {
                        "full_window": minus_llr_avg(full_subset),
                        "center_1": minus_llr_avg(center_subset),
                    }
                ),
                full_subset["match_group"],
                n_bootstrap=n_bootstrap,
                seed=seed + subset_index,
            )
            context = {
                "step": step,
                "subset": subset,
                "dataset": "mendelian_traits",
                "split": DEVELOPMENT_SPLIT,
                "bootstrap_seed": seed + subset_index,
            }
            point_parts.append(result.point.assign(**context))
            sample_parts.append(result.samples.assign(**context))
            delta_parts.append(result.deltas.assign(**context))

    return (
        pd.concat(point_parts, ignore_index=True),
        pd.concat(sample_parts, ignore_index=True),
        pd.concat(delta_parts, ignore_index=True),
        inputs,
    )


def select_official_endpoint(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Select the preregistered region-relevant official AUPRC rows."""
    required = {"score_type", "subset", "value", "split"}
    missing = required - set(frame.columns)
    assert not missing, f"{dataset}: missing columns {sorted(missing)}"
    assert set(frame["split"].astype(str)) == {DEVELOPMENT_SPLIT}
    if dataset == "complex_traits":
        selected = frame[
            (frame["score_type"] == "abs_llr_avg")
            & (frame["subset"] == "missense_variant")
        ].copy()
    elif dataset == "sge":
        sge_required = {"metric", "accession", "gene"}
        sge_missing = sge_required - set(frame.columns)
        assert not sge_missing, f"sge: missing columns {sorted(sge_missing)}"
        selected = frame[
            (frame["score_type"] == "minus_llr_avg")
            & (frame["metric"] == "AUPRC")
            & (frame["accession"] == "_macro_avg_")
            & (frame["gene"] == "_macro_avg_")
            & frame["subset"].isin(SGE_SUBSETS)
        ].copy()
    else:
        raise ValueError(f"unsupported official endpoint dataset {dataset!r}")
    expected = 1 if dataset == "complex_traits" else len(SGE_SUBSETS)
    assert len(selected) == expected, (
        f"{dataset}: expected {expected} endpoint rows, got {len(selected)}"
    )
    return selected


def collect_official_endpoints(
    canonical_root: str,
    issue_root: str,
    *,
    experiment_commit: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Collect repaired Complex-trait and SGE trajectories."""
    rows: list[pd.DataFrame] = []
    inputs: list[str] = []
    for step in CHECKPOINT_STEPS:
        for policy, root, model in (
            ("full_window", canonical_root, baseline_model_name(step)),
            (
                "center_1",
                issue_root,
                center_model_name(step, experiment_commit=experiment_commit),
            ),
        ):
            for dataset in ("complex_traits", "sge"):
                uri = artifact_uri(root, "metrics", model, dataset)
                frame = _load_metric(uri)
                selected = select_official_endpoint(frame, dataset)
                rows.append(selected.assign(policy=policy, step=step, source_uri=uri))
                inputs.append(uri)
    result = pd.concat(rows, ignore_index=True)
    assert set(result["split"].astype(str)) == {DEVELOPMENT_SPLIT}
    return result, inputs


def write_summary(
    points: pd.DataFrame,
    deltas: pd.DataFrame,
    endpoints: pd.DataFrame,
    output: Path,
) -> None:
    """Write numeric audit tables without a scientific interpretation."""
    lines = [
        "# Issue #473 repaired CDS evaluation audit",
        "",
        (
            'These are development-only (`split="train"`) diagnostics. '
            "They replace the invalid CDS full-window values loaded without "
            "the maintained legacy-RoPE compatibility translation."
        ),
        "",
        "## Mendelian trajectories",
        "",
        (
            "Actual AUPRC and Group SMD values are shown for both arms. "
            "The paired interval is for center-1 minus full-window and uses "
            "1,000 aligned match-group bootstrap draws."
        ),
        "",
        "| Step | Subset | Metric | Full window | Center 1 | Delta | 95% interval |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    merged = points.pivot_table(
        index=["step", "subset", "metric"],
        columns="policy",
        values="value",
        aggfunc="first",
    ).reset_index()
    merged = merged.merge(
        deltas[
            ["step", "subset", "metric", "delta_center_minus_full", "ci_low", "ci_high"]
        ],
        on=["step", "subset", "metric"],
        validate="one_to_one",
    )
    for row in merged.sort_values(["subset", "metric", "step"]).itertuples():
        lines.append(
            f"| {row.step:,} | {row.subset} | {row.metric} | "
            f"{row.full_window:.6f} | {row.center_1:.6f} | "
            f"{row.delta_center_minus_full:+.6f} | "
            f"[{row.ci_low:.6f}, {row.ci_high:.6f}] |"
        )

    lines.extend(
        [
            "",
            "## Complex-trait and SGE AUPRC trajectories",
            "",
            "| Step | Dataset | Subset | Full window | Center 1 | Delta |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    official = endpoints.pivot_table(
        index=["step", "dataset", "subset"],
        columns="policy",
        values="value",
        aggfunc="first",
    ).reset_index()
    official["delta"] = official["center_1"] - official["full_window"]
    for row in official.sort_values(["dataset", "subset", "step"]).itertuples():
        lines.append(
            f"| {row.step:,} | {row.dataset} | {row.subset} | "
            f"{row.full_window:.6f} | {row.center_1:.6f} | {row.delta:+.6f} |"
        )
    lines.extend(
        [
            "",
            "No biological or projection-policy interpretation is made in this audit.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def sha256(path: Path) -> str:
    """Hash one local output."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_analysis(
    canonical_root: str,
    issue_root: str,
    output_dir: Path,
    *,
    experiment_commit: str,
    analysis_commit: str,
    n_bootstrap: int,
    seed: int,
) -> None:
    """Run the complete repaired CDS audit and write a hashed manifest."""
    experiment_commit = validate_experiment_commit(experiment_commit)
    analysis_commit = validate_experiment_commit(analysis_commit)
    output_dir.mkdir(parents=True, exist_ok=True)
    points, samples, deltas, mendelian_inputs = analyze_mendelian(
        canonical_root,
        issue_root,
        experiment_commit=experiment_commit,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    endpoints, endpoint_inputs = collect_official_endpoints(
        canonical_root,
        issue_root,
        experiment_commit=experiment_commit,
    )
    frames = {
        "paired_mendelian_points.parquet": points,
        "paired_mendelian_samples.parquet": samples,
        "paired_mendelian_deltas.parquet": deltas,
        "official_complex_sge_trajectories.parquet": endpoints,
    }
    outputs: list[Path] = []
    for name, frame in frames.items():
        path = output_dir / name
        frame.to_parquet(path, index=False)
        outputs.append(path)
    summary = output_dir / "summary.md"
    write_summary(points, deltas, endpoints, summary)
    outputs.append(summary)
    manifest = {
        "analysis_commit": analysis_commit,
        "experiment_commit": experiment_commit,
        "split": DEVELOPMENT_SPLIT,
        "held_out_evaluated": False,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "inputs": sorted(set(mendelian_inputs + endpoint_inputs)),
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(outputs)
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {output_dir}: {len(points)} Mendelian points, "
        f"{len(deltas)} paired deltas, {len(endpoints)} official endpoints"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", default=CANONICAL_RESULTS_ROOT)
    parser.add_argument("--issue-root", default=ISSUE_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-commit", required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=473)
    args = parser.parse_args()
    run_analysis(
        args.canonical_root,
        args.issue_root,
        args.output_dir,
        experiment_commit=args.experiment_commit,
        analysis_commit=args.analysis_commit,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
