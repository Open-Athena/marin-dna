"""Reconstruct the missing issue #417 validation curves with evals_v2.

This is a damage-control control experiment. It scores the existing #417
checkpoints on their own exact public chromosome-18 validation shards. The
legacy Transformers-5 checkpoint config is translated only in the ephemeral
downloaded copy, with fail-closed checks of the trained RoPE semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from exp473_center_seeded_projection.native_validation_replay import (
    EXPECTED_ROPE_SCALING,
    EXPECTED_ROPE_THETA,
    _canonical_rope_scaling,
    score_validation,
    validate_dual_schema_rope,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def translate_legacy_rope_config(checkpoint_path: str | Path) -> dict[str, Any]:
    """Materialize a checked Transformers-4 mirror in an ephemeral #417 copy."""
    config_path = Path(checkpoint_path) / "config.json"
    original_bytes = config_path.read_bytes()
    raw = json.loads(original_bytes)
    parameters = raw.get("rope_parameters")
    if not isinstance(parameters, dict):
        raise TypeError("legacy checkpoint lacks Transformers-5 rope_parameters")
    if "rope_theta" not in parameters:
        raise ValueError("legacy checkpoint rope_parameters lacks rope_theta")
    theta = float(parameters["rope_theta"])
    scaling = _canonical_rope_scaling(
        {key: value for key, value in parameters.items() if key != "rope_theta"}
    )
    if theta != EXPECTED_ROPE_THETA or scaling != EXPECTED_ROPE_SCALING:
        raise ValueError(
            f"unexpected trained RoPE semantics theta={theta}, scaling={scaling}"
        )

    had_top_level_mirror = "rope_theta" in raw or "rope_scaling" in raw
    if had_top_level_mirror:
        validate_dual_schema_rope(raw)
    else:
        raw["rope_theta"] = theta
        raw["rope_scaling"] = scaling
        validate_dual_schema_rope(raw)
        config_path.write_text(
            json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return {
        "mode": (
            "verified_existing_dual_schema"
            if had_top_level_mirror
            else "ephemeral_legacy_to_dual_schema"
        ),
        "original_config_sha256": _sha256_bytes(original_bytes),
        "translated_config_sha256": _sha256_bytes(config_path.read_bytes()),
        "rope_theta": theta,
        "rope_scaling": scaling,
    }


def score_control(
    checkpoint_path: str | Path,
    validation_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    arm: str,
    step: int,
    expected_rows: int,
    checkpoint_uri: str,
    validation_repo: str,
    validation_revision: str,
    validation_filename: str,
    batch_size: int,
    num_workers: int,
    torch_compile: bool,
) -> None:
    """Score one #417 checkpoint after a local, verified config translation."""
    translation = translate_legacy_rope_config(checkpoint_path)
    score_validation(
        checkpoint_path,
        validation_path,
        output_path,
        manifest_path,
        arm=arm,
        region="cds",
        step=step,
        native_wandb_loss=None,
        expected_rows=expected_rows,
        checkpoint_uri=checkpoint_uri,
        validation_repo=validation_repo,
        validation_revision=validation_revision,
        validation_filename=validation_filename,
        batch_size=batch_size,
        num_workers=num_workers,
        torch_compile=torch_compile,
    )
    target = Path(manifest_path)
    manifest = json.loads(target.read_text(encoding="utf-8"))
    manifest["legacy_rope_translation"] = translation
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def analyze_control(
    score_paths: list[str | Path],
    cell_manifest_paths: list[str | Path],
    output_dir: str | Path,
) -> None:
    """Assemble complete offline curves without experiment interpretation."""
    manifests = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in cell_manifest_paths
    ]
    if len(score_paths) != len(manifests):
        raise ValueError("score and cell-manifest counts differ")
    if len(manifests) < 4:
        raise ValueError("issue #417 control needs multiple checkpoints per arm")
    keys = [(str(item["arm"]), int(item["step"])) for item in manifests]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate issue #417 control cells")
    points = pd.DataFrame(
        [
            {
                "arm": str(item["arm"]),
                "step": int(item["step"]),
                "offline_evals_v2_nll": float(item["offline_evals_v2_nll"]),
                "validation_rows": int(item["validation_rows"]),
            }
            for item in manifests
        ]
    ).sort_values(["arm", "step"])
    if set(points["arm"]) != {"issue417_combined", "issue417_mammals_only"}:
        raise ValueError(f"unexpected control arms {sorted(set(points['arm']))}")
    if set(points["validation_rows"]) != {16_384}:
        raise ValueError("issue #417 validation row count changed")

    curve_rows: list[dict[str, Any]] = []
    for arm, curve in points.groupby("arm", sort=True):
        ordered = curve.sort_values("step").reset_index(drop=True)
        minimum_index = int(ordered["offline_evals_v2_nll"].idxmin())
        minimum = ordered.loc[minimum_index]
        first = ordered.iloc[0]
        terminal = ordered.iloc[-1]
        curve_rows.append(
            {
                "arm": arm,
                "first_step": int(first["step"]),
                "first_nll": float(first["offline_evals_v2_nll"]),
                "minimum_step": int(minimum["step"]),
                "minimum_nll": float(minimum["offline_evals_v2_nll"]),
                "terminal_step": int(terminal["step"]),
                "terminal_nll": float(terminal["offline_evals_v2_nll"]),
                "first_to_terminal_delta": float(
                    terminal["offline_evals_v2_nll"] - first["offline_evals_v2_nll"]
                ),
                "minimum_to_terminal_delta": float(
                    terminal["offline_evals_v2_nll"] - minimum["offline_evals_v2_nll"]
                ),
            }
        )
    curves = pd.DataFrame(curve_rows).sort_values("arm")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    points.to_parquet(target / "loss_points.parquet", index=False)
    curves.to_parquet(target / "curve_summary.parquet", index=False)

    lines = [
        "# Issue #417 exact-validation offline control",
        "",
        "Damage-control comparison only; no experiment interpretation.",
        "",
        (
            "Each value is case-weighted cross-entropy from the existing evals_v2 "
            "kernel on the checkpoint's own public, commit-pinned chromosome-18 "
            "validation shard."
        ),
        "",
        "## Curves",
        "",
        "| Arm | Step | Offline evals_v2 NLL |",
        "|---|---:|---:|",
    ]
    for row in points.itertuples(index=False):
        lines.append(f"| {row.arm} | {row.step:,} | {row.offline_evals_v2_nll:.9f} |")
    lines.extend(
        [
            "",
            "## Curve summary",
            "",
            "| Arm | Minimum | Terminal | Minimum-to-terminal change |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in curves.itertuples(index=False):
        lines.append(
            f"| {row.arm} | step {row.minimum_step:,}: {row.minimum_nll:.9f} | "
            f"step {row.terminal_step:,}: {row.terminal_nll:.9f} | "
            f"{row.minimum_to_terminal_delta:+.9f} |"
        )
    (target / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    output_manifest = {
        "purpose": "damage_control_issue417_validation_control",
        "interpretation_allowed": False,
        "vep_held_out_access": False,
        "cells": manifests,
        "score_paths": [str(path) for path in score_paths],
    }
    (target / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--validation", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--manifest", required=True)
    score.add_argument("--arm", required=True)
    score.add_argument("--step", type=int, required=True)
    score.add_argument("--expected-rows", type=int, required=True)
    score.add_argument("--checkpoint-uri", required=True)
    score.add_argument("--validation-repo", required=True)
    score.add_argument("--validation-revision", required=True)
    score.add_argument("--validation-filename", required=True)
    score.add_argument("--batch-size", type=int, default=128)
    score.add_argument("--num-workers", type=int, default=4)
    score.add_argument("--torch-compile", action="store_true")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--scores", nargs="+", required=True)
    analyze.add_argument("--cell-manifests", nargs="+", required=True)
    analyze.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "score":
        score_control(
            args.checkpoint,
            args.validation,
            args.output,
            args.manifest,
            arm=args.arm,
            step=args.step,
            expected_rows=args.expected_rows,
            checkpoint_uri=args.checkpoint_uri,
            validation_repo=args.validation_repo,
            validation_revision=args.validation_revision,
            validation_filename=args.validation_filename,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            torch_compile=args.torch_compile,
        )
    else:
        analyze_control(args.scores, args.cell_manifests, args.output_dir)


if __name__ == "__main__":
    main()
