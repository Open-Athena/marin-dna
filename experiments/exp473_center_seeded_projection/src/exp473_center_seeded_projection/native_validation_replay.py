"""Replay issue #473 native validation with the evals_v2 loss kernel.

This is a damage-control diagnostic, not an interpretation workflow. It reads
the exact public chromosome-18 validation shard used to build each Levanter
validation cache and calls the existing evals_v2 path
``run_ll_clm -> transform_ll_clm -> compute_ll_clm``. The only experiment-local
adapters are the known character-tokenizer loader and a fail-closed RoPE check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exp473_center_seeded_projection.development_eval import (
    load_issue473_tokenizer,
)

LOWERCASE_WEIGHT = 0.01
NATIVE_Z_LOSS_WEIGHT = 4.312883184368223e-06
SEQUENCE_LENGTH = 255
VALIDATION_CHROM = "chr18"
EXPECTED_ROPE_THETA = 500_000.0
EXPECTED_ROPE_SCALING: dict[str, Any] = {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8_192,
    "rope_type": "llama3",
}
VALIDATION_ID_COLUMNS = ("query_name", "species", "augmentation")


def download_public_validation_shard(
    repo_id: str,
    revision: str,
    filename: str,
    output_path: str | Path,
) -> None:
    """Download one immutable public Hugging Face validation shard."""
    if not repo_id.startswith("marin-dna/"):
        raise ValueError(
            f"validation source must be a public marin-dna repo: {repo_id}"
        )
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("validation revision must be a full lowercase commit SHA")
    if not filename.startswith("data/validation/") or not filename.endswith(
        ".jsonl.zst"
    ):
        raise ValueError(f"unexpected validation filename {filename!r}")
    from huggingface_hub import hf_hub_download

    source = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename=filename,
        token=False,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def validate_validation_frame(
    frame: pd.DataFrame,
    *,
    arm: str,
    region: str,
    expected_rows: int,
) -> pd.DataFrame:
    """Validate the exact native-validation source contract."""
    required = {
        *VALIDATION_ID_COLUMNS,
        "source_chrom",
        "source_start",
        "source_end",
        "region_label",
        "sequence",
        "clade",
        "family",
        "taxonomy_id",
    }
    missing = required - set(frame.columns)
    assert not missing, f"{arm}: validation shard missing columns {sorted(missing)}"
    assert len(frame) == expected_rows, (
        f"{arm}: expected {expected_rows:,} validation rows, got {len(frame):,}"
    )
    assert not frame.duplicated(list(VALIDATION_ID_COLUMNS)).any(), (
        f"{arm}: duplicate {VALIDATION_ID_COLUMNS}"
    )
    assert set(frame["source_chrom"].astype(str)) == {VALIDATION_CHROM}
    assert set(frame["region_label"].astype(str)) == {region}
    assert set(frame["augmentation"].astype(str)) == {"+"}
    assert (frame["source_start"].astype(int) >= 0).all()
    assert (
        frame["source_end"].astype(int) - frame["source_start"].astype(int)
        == SEQUENCE_LENGTH
    ).all()
    sequences = frame["sequence"].astype(str)
    assert (sequences.str.len() == SEQUENCE_LENGTH).all()
    observed_bases = set("".join(sequences.tolist()))
    assert observed_bases <= set("ACGTNacgtn"), (
        f"{arm}: unexpected sequence characters {sorted(observed_bases - set('ACGTNacgtn'))}"
    )
    result = frame.copy()
    result.insert(
        0,
        "row_id",
        result[list(VALIDATION_ID_COLUMNS)].astype(str).agg("|".join, axis=1),
    )
    assert result["row_id"].is_unique
    return result


def _canonical_rope_scaling(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"RoPE scaling must be a mapping, got {type(value).__name__}")
    result = dict(value)
    unexpected = set(result) - set(EXPECTED_ROPE_SCALING)
    if unexpected:
        raise ValueError(f"unexpected RoPE scaling fields {sorted(unexpected)}")
    return result


def validate_dual_schema_rope(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Require equivalent Transformers-5 and Transformers-4 RoPE metadata."""
    parameters = raw_config.get("rope_parameters")
    if not isinstance(parameters, dict):
        raise TypeError("checkpoint lacks Transformers-5 rope_parameters")
    if "rope_theta" not in parameters:
        raise ValueError("checkpoint rope_parameters lacks rope_theta")
    parameter_theta = float(parameters["rope_theta"])
    parameter_scaling = _canonical_rope_scaling(
        {key: value for key, value in parameters.items() if key != "rope_theta"}
    )
    if "rope_theta" not in raw_config or "rope_scaling" not in raw_config:
        raise ValueError(
            "checkpoint lacks Transformers-4 rope_theta/rope_scaling mirror"
        )
    top_theta = float(raw_config["rope_theta"])
    top_scaling = _canonical_rope_scaling(raw_config["rope_scaling"])
    if parameter_theta != top_theta:
        raise ValueError(
            f"conflicting RoPE theta: rope_parameters={parameter_theta}, top-level={top_theta}"
        )
    if parameter_scaling != top_scaling:
        raise ValueError(
            "conflicting RoPE scaling between rope_parameters and top-level rope_scaling"
        )
    if top_theta != EXPECTED_ROPE_THETA or top_scaling != EXPECTED_ROPE_SCALING:
        raise ValueError(
            f"unexpected trained RoPE semantics theta={top_theta}, scaling={top_scaling}"
        )
    return {"rope_theta": top_theta, "rope_scaling": top_scaling}


def load_verified_hf_model(checkpoint_path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a checkpoint only after raw and resolved RoPE semantics agree."""
    from transformers import AutoConfig, AutoModelForCausalLM

    checkpoint = Path(checkpoint_path)
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing checkpoint config {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    validated = validate_dual_schema_rope(raw)
    resolved = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    resolved_theta = float(resolved.rope_theta)
    resolved_scaling = _canonical_rope_scaling(resolved.rope_scaling)
    if (
        resolved_theta != validated["rope_theta"]
        or resolved_scaling != validated["rope_scaling"]
    ):
        raise ValueError(
            "installed Transformers resolves different RoPE semantics: "
            f"theta={resolved_theta}, scaling={resolved_scaling}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        config=resolved,
        trust_remote_code=True,
    )
    return model, {
        "producer_transformers_version": raw.get("transformers_version"),
        "resolved_transformers_version": __import__("transformers").__version__,
        **validated,
    }


def compute_evals_v2_atoms(
    checkpoint_path: str | Path,
    sequences: pd.DataFrame,
    *,
    batch_size: int,
    num_workers: int,
    torch_compile: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the existing evals_v2 CLM transform, runner, and loss kernel."""
    from datasets import Dataset
    from marin_dna_evals.model.runner import run_ll_clm

    assert list(sequences.columns) == ["id", "seq"]
    assert len(sequences) > 0
    assert (sequences["seq"].astype(str).str.len() == SEQUENCE_LENGTH).all()
    checkpoint = Path(checkpoint_path)
    tokenizer = load_issue473_tokenizer(checkpoint)
    model, model_metadata = load_verified_hf_model(checkpoint)
    prediction = np.asarray(
        run_ll_clm(
            model,
            tokenizer,
            Dataset.from_pandas(sequences[["seq"]], preserve_index=False),
            data_transform_on_the_fly=True,
            inference_kwargs={
                "per_device_eval_batch_size": batch_size,
                "torch_compile": torch_compile,
                "bf16_full_eval": True,
                "dataloader_num_workers": num_workers,
                "remove_unused_columns": False,
            },
        )
    )
    expected_shape = (len(sequences), 4)
    if prediction.ndim == 1 and prediction.shape[0] == len(sequences) * 4:
        prediction = prediction.reshape(expected_shape)
    assert prediction.shape == expected_shape, (
        f"evals_v2 LL output shape {prediction.shape} != {expected_shape}"
    )
    assert np.isfinite(prediction).all()
    atoms = pd.DataFrame(
        {
            "row_id": sequences["id"].astype(str).to_numpy(),
            "ll_sum_upper": prediction[:, 0].astype(np.float64),
            "ll_sum_lower": prediction[:, 1].astype(np.float64),
            "n_upper": prediction[:, 2].astype(np.int64),
            "n_lower": prediction[:, 3].astype(np.int64),
        }
    )
    assert ((atoms["n_upper"] + atoms["n_lower"]) == SEQUENCE_LENGTH).all()
    return atoms, model_metadata


def add_case_weighted_loss(atoms: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the exp473 target-weighted cross-entropy from LL atoms."""
    result = atoms.copy()
    result["nll_numerator"] = -(
        result["ll_sum_upper"].astype(float)
        + LOWERCASE_WEIGHT * result["ll_sum_lower"].astype(float)
    )
    result["effective_tokens"] = result["n_upper"].astype(
        float
    ) + LOWERCASE_WEIGHT * result["n_lower"].astype(float)
    assert (result["nll_numerator"] >= 0).all()
    assert (result["effective_tokens"] > 0).all()
    result["case_weighted_nll"] = result["nll_numerator"] / result["effective_tokens"]
    assert np.isfinite(result["case_weighted_nll"]).all()
    return result


def aggregate_case_weighted_loss(frame: pd.DataFrame) -> float:
    """Return the dataset-wide token-weighted NLL."""
    numerator = frame["nll_numerator"].to_numpy(dtype=np.float64).sum()
    denominator = frame["effective_tokens"].to_numpy(dtype=np.float64).sum()
    assert numerator >= 0 and denominator > 0
    return float(numerator / denominator)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_validation(
    checkpoint_path: str | Path,
    validation_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    arm: str,
    region: str,
    step: int,
    native_wandb_loss: float | None,
    expected_rows: int,
    checkpoint_uri: str,
    validation_repo: str,
    validation_revision: str,
    validation_filename: str,
    batch_size: int,
    num_workers: int,
    torch_compile: bool,
) -> None:
    """Score one exact validation shard and persist loss atoms plus provenance."""
    validation = Path(validation_path)
    source = validate_validation_frame(
        pd.read_json(validation, lines=True, compression="zstd"),
        arm=arm,
        region=region,
        expected_rows=expected_rows,
    )
    atoms, model_metadata = compute_evals_v2_atoms(
        checkpoint_path,
        pd.DataFrame(
            {"id": source["row_id"].astype(str), "seq": source["sequence"].astype(str)}
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        torch_compile=torch_compile,
    )
    assert atoms["row_id"].tolist() == source["row_id"].tolist()
    weighted = add_case_weighted_loss(atoms)
    metadata_columns = [
        "row_id",
        "query_name",
        "species",
        "clade",
        "family",
        "taxonomy_id",
        "source_start",
        "source_end",
    ]
    output = source[metadata_columns].copy()
    output["arm"] = arm
    output["region"] = region
    output["step"] = step
    output["lowercase_fraction"] = (
        source["sequence"]
        .astype(str)
        .map(
            lambda sequence: (
                sum(character.islower() for character in sequence) / len(sequence)
            )
        )
    )
    output["n_fraction"] = (
        source["sequence"].astype(str).str.upper().str.count("N") / SEQUENCE_LENGTH
    )
    for column in (
        "ll_sum_upper",
        "ll_sum_lower",
        "n_upper",
        "n_lower",
        "nll_numerator",
        "effective_tokens",
        "case_weighted_nll",
    ):
        output[column] = weighted[column].to_numpy()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(target, index=False)
    offline_loss = aggregate_case_weighted_loss(output)
    manifest = {
        "arm": arm,
        "region": region,
        "step": step,
        "checkpoint_uri": checkpoint_uri,
        "validation_source": {
            "repo": validation_repo,
            "revision": validation_revision,
            "filename": validation_filename,
            "sha256": _sha256(validation),
        },
        "validation_rows": len(output),
        "validation_chrom": VALIDATION_CHROM,
        "sequence_length": SEQUENCE_LENGTH,
        "loss_implementation": (
            "marin_dna_evals.model.runner.run_ll_clm -> "
            "marin_dna_evals.transforms.transform_ll_clm -> "
            "marin_dna_evals.model.scoring.compute_ll_clm"
        ),
        "lowercase_weight": LOWERCASE_WEIGHT,
        "native_z_loss_weight": NATIVE_Z_LOSS_WEIGHT,
        "offline_evals_v2_nll": offline_loss,
        "model": model_metadata,
    }
    if native_wandb_loss is not None:
        manifest["native_wandb_loss"] = native_wandb_loss
        manifest["offline_minus_native"] = offline_loss - native_wandb_loss
    manifest_target = Path(manifest_path)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def analyze_replay(
    score_paths: list[str | Path],
    cell_manifest_paths: list[str | Path],
    output_dir: str | Path,
) -> None:
    """Compare offline and native losses without making a research interpretation."""
    manifests = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in cell_manifest_paths
    ]
    assert len(score_paths) == len(manifests) == 6
    keys = [(item["arm"], int(item["step"])) for item in manifests]
    assert len(set(keys)) == len(keys)
    rows = [
        {
            "arm": item["arm"],
            "region": item["region"],
            "step": int(item["step"]),
            "offline_evals_v2_nll": float(item["offline_evals_v2_nll"]),
            "native_wandb_loss": float(item["native_wandb_loss"]),
            "offline_minus_native": float(item["offline_minus_native"]),
            "absolute_difference": abs(float(item["offline_minus_native"])),
            "validation_rows": int(item["validation_rows"]),
        }
        for item in manifests
    ]
    points = pd.DataFrame(rows).sort_values(["arm", "step"]).reset_index(drop=True)
    comparisons: list[dict[str, Any]] = []
    for arm, arm_points in points.groupby("arm", sort=True):
        ordered = arm_points.sort_values("step")
        assert len(ordered) == 2
        early, terminal = ordered.iloc[0], ordered.iloc[1]
        offline_delta = float(
            terminal["offline_evals_v2_nll"] - early["offline_evals_v2_nll"]
        )
        native_delta = float(terminal["native_wandb_loss"] - early["native_wandb_loss"])
        comparisons.append(
            {
                "arm": arm,
                "early_step": int(early["step"]),
                "terminal_step": int(terminal["step"]),
                "offline_delta": offline_delta,
                "native_delta": native_delta,
                "delta_difference": offline_delta - native_delta,
                "direction_matches": bool(
                    np.sign(offline_delta) == np.sign(native_delta)
                ),
            }
        )
        score_pair = [
            pd.read_parquet(path).sort_values("row_id").reset_index(drop=True)
            for path, item in zip(score_paths, manifests, strict=True)
            if item["arm"] == arm
        ]
        assert len(score_pair) == 2
        assert score_pair[0]["row_id"].equals(score_pair[1]["row_id"])
    deltas = pd.DataFrame(comparisons).sort_values("arm").reset_index(drop=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    points.to_parquet(output / "loss_points.parquet", index=False)
    deltas.to_parquet(output / "checkpoint_deltas.parquet", index=False)
    lines = [
        "# Issue #473 native-validation replay",
        "",
        "Damage-control comparison only; no experiment interpretation.",
        "",
        (
            "The offline value is case-weighted cross-entropy from the existing `evals_v2` kernel. "
            "The native W&B value also includes Levanter's configured z-loss penalty."
        ),
        "",
        "## Absolute values",
        "",
        "| Arm | Step | Offline evals_v2 NLL | Native W&B loss | Offline - native |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in points.itertuples(index=False):
        lines.append(
            f"| {row.arm} | {row.step:,} | {row.offline_evals_v2_nll:.9f} | "
            f"{row.native_wandb_loss:.9f} | {row.offline_minus_native:+.9f} |"
        )
    lines.extend(
        [
            "",
            "## Early-to-terminal change",
            "",
            "| Arm | Checkpoints | Offline delta | Native delta | Direction matches |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in deltas.itertuples(index=False):
        lines.append(
            f"| {row.arm} | {row.early_step:,} to {row.terminal_step:,} | "
            f"{row.offline_delta:+.9f} | {row.native_delta:+.9f} | "
            f"{'yes' if row.direction_matches else 'no'} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    combined_manifest = {
        "purpose": "damage_control_native_validation_replay",
        "interpretation_allowed": False,
        "cells": manifests,
        "all_direction_matches": bool(deltas["direction_matches"].all()),
        "max_absolute_offline_native_difference": float(
            points["absolute_difference"].max()
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(combined_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--repo", required=True)
    download.add_argument("--revision", required=True)
    download.add_argument("--filename", required=True)
    download.add_argument("--output", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--validation", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--manifest", required=True)
    score.add_argument("--arm", required=True)
    score.add_argument("--region", required=True)
    score.add_argument("--step", required=True, type=int)
    score.add_argument("--native-wandb-loss", required=True, type=float)
    score.add_argument("--expected-rows", required=True, type=int)
    score.add_argument("--checkpoint-uri", required=True)
    score.add_argument("--validation-repo", required=True)
    score.add_argument("--validation-revision", required=True)
    score.add_argument("--validation-filename", required=True)
    score.add_argument("--batch-size", required=True, type=int)
    score.add_argument("--num-workers", required=True, type=int)
    score.add_argument("--torch-compile", action="store_true")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--scores", nargs="+", required=True)
    analyze.add_argument("--cell-manifests", nargs="+", required=True)
    analyze.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "download":
        download_public_validation_shard(
            args.repo, args.revision, args.filename, args.output
        )
    elif args.command == "score":
        score_validation(
            args.checkpoint,
            args.validation,
            args.output,
            args.manifest,
            arm=args.arm,
            region=args.region,
            step=args.step,
            native_wandb_loss=args.native_wandb_loss,
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
        analyze_replay(args.scores, args.cell_manifests, args.output_dir)


if __name__ == "__main__":
    main()
