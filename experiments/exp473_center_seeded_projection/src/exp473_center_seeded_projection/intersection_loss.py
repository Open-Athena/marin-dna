"""Paired chromosome-18 intersection loss for MarinDNA issue #473.

Scoring reuses the unchanged official ``evals_v2`` causal-LM kernel.  This
experiment-local module validates the producer's row identity, reconstructs
the case-aware training loss from official upper/lower log-likelihood atoms,
and bootstraps policy deltas over human anchors.  It does not read VEP labels,
predictions, or metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exp473_center_seeded_projection.analyze_evals import REGION_ARMS
from exp473_center_seeded_projection.eval_config import CHECKPOINT_STEPS

LOWERCASE_WEIGHT = 0.01
VALIDATION_CHROM = "chr18"
SPLIT = "chromosome_18_paired_intersection"
REGION_LABELS = {
    "cds": "cds",
    "enhancer": "ccre_enhancer_centered",
}
ARM_CONTEXT = {
    arm: {"region": region, "policy": policy}
    for region, policies in REGION_ARMS.items()
    for policy, arm in policies.items()
}
IDENTITY_COLUMNS = (
    "row_id",
    "query_name",
    "species",
    "source_chrom",
)
SCORED_COLUMNS = (
    *IDENTITY_COLUMNS,
    "region",
    "policy",
    "arm",
    "step",
    "split",
    "nll_numerator",
    "effective_tokens",
    "case_weighted_nll",
)


def validate_intersection_frame(frame: pd.DataFrame, *, region: str) -> pd.DataFrame:
    """Validate one producer-pinned intersection view and return row-id order."""
    if region not in REGION_LABELS:
        raise ValueError(f"unknown region {region!r}; expected {sorted(REGION_LABELS)}")
    required = {
        *IDENTITY_COLUMNS,
        "source_start",
        "source_end",
        "region_label",
        "sequence",
    }
    missing = required - set(frame.columns)
    assert not missing, f"intersection view missing columns {sorted(missing)}"
    assert len(frame) > 0, "intersection view is empty"
    assert frame["row_id"].notna().all() and frame["row_id"].is_unique
    assert not frame.duplicated(["query_name", "species"]).any()
    assert set(frame["source_chrom"].astype(str)) == {VALIDATION_CHROM}
    assert set(frame["region_label"].astype(str)) == {REGION_LABELS[region]}
    assert (frame["source_start"].astype(int) >= 0).all()
    assert (
        frame["source_end"].astype(int) - frame["source_start"].astype(int) == 255
    ).all()
    lengths = frame["sequence"].astype(str).str.len()
    assert (lengths == 255).all(), (
        f"intersection sequences must be 255 bp; got {sorted(lengths.unique())[:5]}"
    )
    return frame.sort_values("row_id").reset_index(drop=True)


def case_weighted_atoms(
    atoms: pd.DataFrame, *, lowercase_weight: float = LOWERCASE_WEIGHT
) -> pd.DataFrame:
    """Reconstruct case-aware NLL numerators and denominators per sequence."""
    assert 0 <= lowercase_weight <= 1
    required = {"ll_sum_upper", "ll_sum_lower", "n_upper", "n_lower"}
    missing = required - set(atoms.columns)
    assert not missing, f"official LL atoms missing columns {sorted(missing)}"
    numeric = atoms[list(required)].to_numpy(dtype=float)
    assert np.isfinite(numeric).all(), "official LL atoms contain non-finite values"
    result = atoms.copy()
    result["nll_numerator"] = -(
        result["ll_sum_upper"].astype(float)
        + lowercase_weight * result["ll_sum_lower"].astype(float)
    )
    result["effective_tokens"] = result["n_upper"].astype(
        float
    ) + lowercase_weight * result["n_lower"].astype(float)
    assert (result["nll_numerator"] >= 0).all()
    assert (result["effective_tokens"] > 0).all()
    result["case_weighted_nll"] = result["nll_numerator"] / result["effective_tokens"]
    assert np.isfinite(result["case_weighted_nll"]).all()
    return result


def score_intersection(
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    *,
    arm: str,
    step: int,
    batch_size: int,
    num_workers: int,
    torch_compile: bool,
) -> None:
    """Score one model on its policy-matched intersection sequences."""
    if arm not in ARM_CONTEXT:
        raise ValueError(f"unknown arm {arm!r}; expected {sorted(ARM_CONTEXT)}")
    if step not in CHECKPOINT_STEPS:
        raise ValueError(f"unexpected checkpoint step {step}")
    context = ARM_CONTEXT[arm]
    source = validate_intersection_frame(
        pd.read_parquet(input_path), region=context["region"]
    )
    scoring_input = pd.DataFrame(
        {
            "id": source["row_id"].astype(str),
            "seq": source["sequence"].astype(str),
        }
    )
    # Runtime-owned import: the Sky rule executes in the pinned official
    # evals_v2 environment, while this experiment's light tests remain CPU-only.
    from marin_dna_evals.ll_gap import compute_hf_ll_gap

    atoms = compute_hf_ll_gap(
        checkpoint_path=checkpoint_path,
        sequences=scoring_input,
        window_size=255,
        batch_size=batch_size,
        num_workers=num_workers,
        torch_compile=torch_compile,
    )
    assert atoms["id"].astype(str).tolist() == source["row_id"].astype(str).tolist()
    weighted = case_weighted_atoms(atoms)
    output = source[list(IDENTITY_COLUMNS)].copy()
    output["region"] = context["region"]
    output["policy"] = context["policy"]
    output["arm"] = arm
    output["step"] = step
    output["split"] = SPLIT
    for column in ["nll_numerator", "effective_tokens", "case_weighted_nll"]:
        output[column] = weighted[column].to_numpy()
    assert tuple(output.columns) == SCORED_COLUMNS
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(target, index=False)


def validate_scored_pair(
    full: pd.DataFrame, center: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require exact paired rows and policy metadata before loss comparison."""
    for name, frame, policy in (
        ("full_window", full, "full_window"),
        ("center_1", center, "center_1"),
    ):
        missing = set(SCORED_COLUMNS) - set(frame.columns)
        assert not missing, f"{name}: missing score columns {sorted(missing)}"
        assert set(frame["policy"].astype(str)) == {policy}
        assert set(frame["split"].astype(str)) == {SPLIT}
    full_ordered = full.sort_values("row_id").reset_index(drop=True)
    center_ordered = center.sort_values("row_id").reset_index(drop=True)
    assert full_ordered[list(IDENTITY_COLUMNS)].equals(
        center_ordered[list(IDENTITY_COLUMNS)]
    ), "intersection loss rows are not exactly paired"
    assert set(full_ordered["region"]) == set(center_ordered["region"])
    assert set(full_ordered["step"]) == set(center_ordered["step"])
    return full_ordered, center_ordered


def _interval(values: np.ndarray) -> tuple[float, float, float]:
    assert len(values) > 1 and np.isfinite(values).all()
    low, high = np.percentile(values, [2.5, 97.5])
    return float(np.std(values, ddof=1)), float(low), float(high)


def paired_loss_bootstrap(
    full: pd.DataFrame,
    center: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return token-weighted losses and aligned human-anchor bootstrap deltas."""
    assert n_bootstrap > 1
    full, center = validate_scored_pair(full, center)
    anchors = pd.Index(pd.unique(full["query_name"]), dtype=object)
    assert len(anchors) > 1
    anchor_index = anchors.get_indexer(full["query_name"])
    assert (anchor_index >= 0).all()
    n_anchors = len(anchors)
    generator = np.random.default_rng(seed)
    sampled = generator.integers(
        0, n_anchors, size=(n_bootstrap, n_anchors), dtype=np.int32
    )
    multiplicity = np.zeros((n_bootstrap, n_anchors), dtype=np.int32)
    np.add.at(
        multiplicity,
        (np.repeat(np.arange(n_bootstrap), n_anchors), sampled.ravel()),
        1,
    )

    point_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    draws_by_policy: dict[str, np.ndarray] = {}
    point_by_policy: dict[str, float] = {}
    for policy, frame in (("full_window", full), ("center_1", center)):
        numerator = np.bincount(
            anchor_index,
            weights=frame["nll_numerator"].to_numpy(dtype=float),
            minlength=n_anchors,
        )
        denominator = np.bincount(
            anchor_index,
            weights=frame["effective_tokens"].to_numpy(dtype=float),
            minlength=n_anchors,
        )
        draws = (multiplicity @ numerator) / (multiplicity @ denominator)
        point = float(numerator.sum() / denominator.sum())
        se, low, high = _interval(draws)
        draws_by_policy[policy] = draws
        point_by_policy[policy] = point
        point_rows.append(
            {
                "policy": policy,
                "metric": "case_weighted_nll",
                "value": point,
                "se": se,
                "ci_low": low,
                "ci_high": high,
                "n_anchors": n_anchors,
                "n_rows": len(frame),
                "bootstrap_unit": "human_anchor",
            }
        )
        sample_rows.extend(
            {
                "draw": draw,
                "policy": policy,
                "metric": "case_weighted_nll",
                "value": float(value),
            }
            for draw, value in enumerate(draws)
        )

    delta_draws = draws_by_policy["center_1"] - draws_by_policy["full_window"]
    se, low, high = _interval(delta_draws)
    delta = pd.DataFrame(
        [
            {
                "metric": "case_weighted_nll",
                "delta_center_minus_full": (
                    point_by_policy["center_1"] - point_by_policy["full_window"]
                ),
                "se": se,
                "ci_low": low,
                "ci_high": high,
                "probability_center_better": float(np.mean(delta_draws < 0)),
                "n_bootstrap": n_bootstrap,
                "bootstrap_unit": "human_anchor",
                "direction": "negative_favors_center_1",
            }
        ]
    )
    return pd.DataFrame(point_rows), pd.DataFrame(sample_rows), delta


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_loss_scores(
    score_paths: list[str],
    output_dir: str | Path,
    *,
    n_bootstrap: int,
    seed: int,
) -> None:
    """Analyze the complete 2-region × 2-policy × 10-checkpoint score matrix."""
    cells: dict[tuple[str, str, int], pd.DataFrame] = {}
    for path in score_paths:
        frame = pd.read_parquet(path)
        region_values = frame["region"].astype(str).unique()
        policy_values = frame["policy"].astype(str).unique()
        step_values = frame["step"].astype(int).unique()
        assert len(region_values) == len(policy_values) == len(step_values) == 1
        key = (str(region_values[0]), str(policy_values[0]), int(step_values[0]))
        assert key not in cells, f"duplicate intersection loss cell {key}"
        cells[key] = frame
    expected = {
        (region, policy, step)
        for region in REGION_ARMS
        for policy in ("full_window", "center_1")
        for step in CHECKPOINT_STEPS
    }
    assert set(cells) == expected, {
        "missing": sorted(expected - set(cells)),
        "unexpected": sorted(set(cells) - expected),
    }

    point_parts: list[pd.DataFrame] = []
    sample_parts: list[pd.DataFrame] = []
    delta_parts: list[pd.DataFrame] = []
    for region_index, region in enumerate(REGION_ARMS):
        for step in CHECKPOINT_STEPS:
            points, samples, deltas = paired_loss_bootstrap(
                cells[(region, "full_window", step)],
                cells[(region, "center_1", step)],
                n_bootstrap=n_bootstrap,
                # Reuse the region seed across checkpoints to align trajectories.
                seed=seed + region_index,
            )
            context = {
                "region": region,
                "step": step,
                "split": SPLIT,
                "bootstrap_seed": seed + region_index,
            }
            point_parts.append(points.assign(**context))
            sample_parts.append(samples.assign(**context))
            delta_parts.append(deltas.assign(**context))
    points = pd.concat(point_parts, ignore_index=True)
    samples = pd.concat(sample_parts, ignore_index=True)
    deltas = pd.concat(delta_parts, ignore_index=True)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    outputs = {
        "paired_loss_metrics.parquet": points,
        "paired_loss_bootstrap_samples.parquet": samples,
        "paired_loss_deltas.parquet": deltas,
    }
    paths: list[Path] = []
    for name, frame in outputs.items():
        path = target / name
        frame.to_parquet(path, index=False)
        paths.append(path)
    final = deltas[deltas["step"] == max(CHECKPOINT_STEPS)].sort_values("region")
    lines = [
        "# Issue #473 paired intersection loss",
        "",
        (
            "These chromosome-18 views contain unlabeled projection sequences, not "
            "held-out VEP labels, predictions, measurements, or metrics. Loss uses "
            "the training case weights (uppercase 1.0; lowercase 0.01). Negative "
            "`center_1 - full_window` deltas favor `center_1`."
        ),
        "",
        "| Region | Step | Delta | 95% interval | P(center better) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in final.itertuples():
        lines.append(
            f"| {row.region} | {row.step} | {row.delta_center_minus_full:.6g} | "
            f"[{row.ci_low:.6g}, {row.ci_high:.6g}] | "
            f"{row.probability_center_better:.3f} |"
        )
    summary = target / "summary.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.append(summary)
    manifest = {
        "split": SPLIT,
        "vep_held_out_access": False,
        "source_kind": "unlabeled chromosome-18 projection intersection",
        "lowercase_weight": LOWERCASE_WEIGHT,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "score_inputs": sorted(score_paths),
        "outputs": {
            str(path.relative_to(target)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths
        },
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--arm", choices=sorted(ARM_CONTEXT), required=True)
    score.add_argument("--step", type=int, choices=CHECKPOINT_STEPS, required=True)
    score.add_argument("--batch-size", type=int, default=128)
    score.add_argument("--num-workers", type=int, default=4)
    score.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--inputs", nargs="+", required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--n-bootstrap", type=int, default=1_000)
    analyze.add_argument("--seed", type=int, default=473)
    args = parser.parse_args()
    if args.command == "score":
        score_intersection(
            args.checkpoint,
            args.input,
            args.output,
            arm=args.arm,
            step=args.step,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            torch_compile=args.torch_compile,
        )
    else:
        analyze_loss_scores(
            args.inputs,
            args.output_dir,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
