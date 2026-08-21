"""Validate the pinned evals_v2 Sky GPU runtime and its inference parity."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from marin_dna_evals.inference import compute_variant_scores


@dataclass(frozen=True)
class RuntimeSpec:
    """Exact software and hardware contract for the standard Sky GPU task."""

    image_id: str
    image_name: str
    operating_system: str
    driver_version: str
    pytorch_version: str
    compiled_cuda_version: str
    device_name: str


@dataclass(frozen=True)
class Tolerance:
    """Elementwise numerical tolerance for one score atom."""

    rtol: float
    atol: float


@dataclass(frozen=True)
class ParitySpec:
    """Fixed development inference cell and its PyTorch 2.8 baseline."""

    baseline_scores_uri: str
    baseline_sha256: str
    baseline_pytorch_version: str
    baseline_compiled_cuda_version: str
    baseline_driver_version: str
    checkpoint_name: str
    checkpoint_gcs_path: str
    context_size: int
    dataset_repo: str
    dataset_revision: str
    dataset_filename: str
    split: str
    batch_size: int
    num_workers: int
    data_transform_on_the_fly: bool
    torch_compile: bool
    bf16: bool
    return_embeddings: bool
    eval_accumulation_steps: int | None
    rc: bool
    identity_columns: tuple[str, ...]
    tolerances: dict[str, Tolerance]


@dataclass(frozen=True)
class ValidationSpec:
    """Complete runtime smoke-test and inference-parity contract."""

    runtime: RuntimeSpec
    parity: ParitySpec


def _require(condition: bool, message: str) -> None:
    """Raise even when Python assertions are disabled with ``-O``."""
    if not condition:
        raise AssertionError(message)


def load_validation_spec(path: str | Path) -> ValidationSpec:
    """Load and validate the checked-in GPU runtime contract."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    _require(isinstance(raw, dict), "GPU runtime config must be a mapping")
    runtime_raw = raw["runtime"]
    parity_raw = raw["parity"]
    baseline_raw = parity_raw["baseline"]
    checkpoint_raw = parity_raw["checkpoint"]
    dataset_raw = parity_raw["dataset"]
    inference_raw = parity_raw["inference"]
    tolerance_raw = parity_raw["tolerances"]

    tolerances = {
        str(column): Tolerance(rtol=float(values["rtol"]), atol=float(values["atol"]))
        for column, values in tolerance_raw.items()
    }
    _require(bool(tolerances), "at least one score-column tolerance is required")
    _require(
        all(t.rtol >= 0 and t.atol >= 0 for t in tolerances.values()),
        "score tolerances must be non-negative",
    )

    runtime = RuntimeSpec(
        image_id=str(runtime_raw["image_id"]),
        image_name=str(runtime_raw["image_name"]),
        operating_system=str(runtime_raw["operating_system"]),
        driver_version=str(runtime_raw["driver_version"]),
        pytorch_version=str(runtime_raw["pytorch_version"]),
        compiled_cuda_version=str(runtime_raw["compiled_cuda_version"]),
        device_name=str(runtime_raw["device_name"]),
    )
    parity = ParitySpec(
        baseline_scores_uri=str(baseline_raw["scores_uri"]),
        baseline_sha256=str(baseline_raw["sha256"]),
        baseline_pytorch_version=str(baseline_raw["pytorch_version"]),
        baseline_compiled_cuda_version=str(baseline_raw["compiled_cuda_version"]),
        baseline_driver_version=str(baseline_raw["driver_version"]),
        checkpoint_name=str(checkpoint_raw["name"]),
        checkpoint_gcs_path=str(checkpoint_raw["gcs_path"]),
        context_size=int(checkpoint_raw["context_size"]),
        dataset_repo=str(dataset_raw["repo"]),
        dataset_revision=str(dataset_raw["revision"]),
        dataset_filename=str(dataset_raw["filename"]),
        split=str(dataset_raw["split"]),
        batch_size=int(inference_raw["batch_size"]),
        num_workers=int(inference_raw["num_workers"]),
        data_transform_on_the_fly=bool(inference_raw["data_transform_on_the_fly"]),
        torch_compile=bool(inference_raw["torch_compile"]),
        bf16=bool(inference_raw["bf16"]),
        return_embeddings=bool(inference_raw["return_embeddings"]),
        eval_accumulation_steps=(
            None
            if inference_raw.get("eval_accumulation_steps") is None
            else int(inference_raw["eval_accumulation_steps"])
        ),
        rc=bool(inference_raw["rc"]),
        identity_columns=tuple(str(c) for c in parity_raw["identity_columns"]),
        tolerances=tolerances,
    )
    _require(
        len(parity.baseline_sha256) == 64,
        "baseline SHA-256 must have 64 hex digits",
    )
    _require(
        parity.batch_size > 0 and parity.num_workers >= 0,
        "batch size must be positive and worker count must be non-negative",
    )
    _require(
        parity.split == "train",
        "GPU parity must not access the held-out test split",
    )
    _require(
        parity.dataset_filename == f"{parity.split}.parquet",
        "GPU parity must load the development split's parquet file directly",
    )
    _require(parity.rc, "the production parity cell must score both strands")
    _require(parity.torch_compile, "the production parity cell must use torch.compile")
    _require(parity.bf16, "the production parity cell must use bf16")
    _require(
        parity.return_embeddings,
        "the production parity cell must return pooled embeddings",
    )
    if parity.eval_accumulation_steps is not None:
        _require(
            parity.eval_accumulation_steps > 0,
            "eval_accumulation_steps must be positive when configured",
        )
    return ValidationSpec(runtime=runtime, parity=parity)


def validate_runtime_metadata(
    spec: RuntimeSpec,
    *,
    pytorch_version: str,
    compiled_cuda_version: str | None,
    driver_version: str,
    device_name: str,
) -> dict[str, str]:
    """Assert that observed runtime metadata matches the checked-in tuple."""
    observed_torch = pytorch_version.partition("+")[0]
    expected = {
        "pytorch_version": spec.pytorch_version,
        "compiled_cuda_version": spec.compiled_cuda_version,
        "driver_version": spec.driver_version,
        "device_name": spec.device_name,
    }
    observed = {
        "pytorch_version": observed_torch,
        "compiled_cuda_version": str(compiled_cuda_version),
        "driver_version": driver_version,
        "device_name": device_name,
    }
    _require(
        observed == expected,
        f"GPU runtime mismatch: observed={observed}, expected={expected}",
    )
    return observed


def _nvidia_smi_metadata() -> tuple[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    _require(len(rows) == 1, f"expected exactly one GPU, found {rows}")
    driver_version, device_name = (part.strip() for part in rows[0].split(",", 1))
    return driver_version, device_name


def run_cuda_smoke(spec: RuntimeSpec) -> dict[str, Any]:
    """Exercise CUDA initialization and a finite bf16 matrix multiplication."""
    _require(torch.cuda.is_available(), "torch.cuda.is_available() is false")
    _require(
        torch.cuda.device_count() == 1,
        "the standard Sky task requires one GPU",
    )
    _require(torch.cuda.is_bf16_supported(), "CUDA device does not support bf16")

    driver_version, smi_device_name = _nvidia_smi_metadata()
    metadata = validate_runtime_metadata(
        spec,
        pytorch_version=torch.__version__,
        compiled_cuda_version=torch.version.cuda,
        driver_version=driver_version,
        device_name=smi_device_name,
    )
    _require(
        torch.cuda.get_device_name(0) == spec.device_name,
        f"torch CUDA device is not {spec.device_name}",
    )

    values = torch.arange(64, device="cuda", dtype=torch.float32).reshape(8, 8)
    values = (values / 64).to(torch.bfloat16)
    product = values @ values.T
    torch.cuda.synchronize()
    _require(
        product.dtype == torch.bfloat16,
        f"bf16 matmul returned {product.dtype}",
    )
    _require(
        bool(torch.isfinite(product).all().item()),
        "bf16 matmul returned non-finite values",
    )

    return {
        **metadata,
        "image_id": spec.image_id,
        "bf16_supported": True,
        "bf16_matmul_checksum": float(product.float().sum().item()),
    }


def read_verified_parquet(uri: str, expected_sha256: str) -> pd.DataFrame:
    """Read a parquet only after verifying its complete byte-level identity."""
    with fsspec.open(uri, "rb") as handle:
        payload = handle.read()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    _require(
        actual_sha256 == expected_sha256,
        f"baseline checksum mismatch for {uri}: "
        f"observed={actual_sha256}, expected={expected_sha256}",
    )
    return pd.read_parquet(io.BytesIO(payload))


def compare_score_frames(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    identity_columns: tuple[str, ...],
    tolerances: dict[str, Tolerance],
) -> dict[str, dict[str, float | int]]:
    """Require identical variants and tolerance-bounded finite score atoms."""
    _require(
        len(candidate) == len(baseline),
        f"row-count mismatch: candidate={len(candidate)}, baseline={len(baseline)}",
    )
    for column in (*identity_columns, *tolerances):
        _require(column in candidate, f"candidate missing column {column!r}")
        _require(column in baseline, f"baseline missing column {column!r}")

    candidate_identity = candidate.loc[:, list(identity_columns)].reset_index(drop=True)
    baseline_identity = baseline.loc[:, list(identity_columns)].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            candidate_identity,
            baseline_identity,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise AssertionError(
            "candidate variants do not match the pinned baseline"
        ) from error

    report: dict[str, dict[str, float | int]] = {}
    failed_columns: list[str] = []
    for column, tolerance in tolerances.items():
        candidate_values = candidate[column].to_numpy(dtype=np.float64)
        baseline_values = baseline[column].to_numpy(dtype=np.float64)
        _require(
            bool(np.isfinite(candidate_values).all()),
            f"candidate {column} contains non-finite values",
        )
        _require(
            bool(np.isfinite(baseline_values).all()),
            f"baseline {column} contains non-finite values",
        )

        absolute_difference = np.abs(candidate_values - baseline_values)
        close = np.isclose(
            candidate_values,
            baseline_values,
            rtol=tolerance.rtol,
            atol=tolerance.atol,
        )
        bad_rows = np.flatnonzero(~close)
        nonzero = np.abs(baseline_values) > 0
        max_relative_difference = (
            float(
                np.max(absolute_difference[nonzero] / np.abs(baseline_values[nonzero]))
            )
            if nonzero.any()
            else 0.0
        )
        report[column] = {
            "n_rows": len(candidate_values),
            "n_outside_tolerance": len(bad_rows),
            "mean_absolute_difference": float(absolute_difference.mean()),
            "median_absolute_difference": float(np.median(absolute_difference)),
            "p95_absolute_difference": float(np.quantile(absolute_difference, 0.95)),
            "p99_absolute_difference": float(np.quantile(absolute_difference, 0.99)),
            "max_absolute_difference": float(absolute_difference.max(initial=0.0)),
            "max_relative_difference": max_relative_difference,
            "rtol": tolerance.rtol,
            "atol": tolerance.atol,
        }
        if len(bad_rows) > 0:
            failed_columns.append(column)

    _require(
        not failed_columns,
        f"score parity failed for {failed_columns}:\n"
        f"{json.dumps(report, indent=2, sort_keys=True)}",
    )
    return report


def validate_pooled_embeddings(frame: pd.DataFrame) -> dict[str, int]:
    """Require finite, equally sized Float16 ref and alt embedding vectors."""
    for column in ("emb_ref", "emb_alt"):
        _require(column in frame, f"candidate missing column {column!r}")
    try:
        ref = np.stack(frame["emb_ref"].to_numpy())
        alt = np.stack(frame["emb_alt"].to_numpy())
    except ValueError as error:
        raise AssertionError("candidate embeddings have inconsistent vector sizes") from error
    _require(ref.ndim == 2 and ref.shape[1] > 0, f"invalid emb_ref shape {ref.shape}")
    _require(alt.shape == ref.shape, f"embedding shape mismatch: {ref.shape} != {alt.shape}")
    _require(ref.dtype == np.float16, f"emb_ref dtype must be float16, got {ref.dtype}")
    _require(alt.dtype == np.float16, f"emb_alt dtype must be float16, got {alt.dtype}")
    _require(bool(np.isfinite(ref).all()), "emb_ref contains non-finite values")
    _require(bool(np.isfinite(alt).all()), "emb_alt contains non-finite values")
    return {"n_rows": ref.shape[0], "hidden_size": ref.shape[1]}


def _download_checkpoint(gcs_path: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gcloud",
            "storage",
            "cp",
            "-r",
            f"{gcs_path.rstrip('/')}/*",
            f"{destination}/",
        ],
        check=True,
    )


def run_inference_parity(
    spec: ParitySpec,
    *,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Re-score the fixed train cell and compare it to the PyTorch 2.8 output."""
    dataset_path = hf_hub_download(
        repo_id=spec.dataset_repo,
        filename=spec.dataset_filename,
        repo_type="dataset",
        revision=spec.dataset_revision,
    )
    dataset = load_dataset(
        "parquet",
        data_files={spec.split: dataset_path},
        split=spec.split,
    ).to_pandas()

    def compute(checkpoint: Path) -> pd.DataFrame:
        scores = compute_variant_scores(
            checkpoint_path=checkpoint,
            dataset=dataset,
            genome_path=(
                "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/"
                "ensembl-release-115/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
            ),
            context_size=spec.context_size,
            batch_size=spec.batch_size,
            num_workers=spec.num_workers,
            data_transform_on_the_fly=spec.data_transform_on_the_fly,
            torch_compile=spec.torch_compile,
            bf16=spec.bf16,
            rc=spec.rc,
            return_embeddings=spec.return_embeddings,
            eval_accumulation_steps=spec.eval_accumulation_steps,
        )
        return pd.concat(
            [
                dataset.loc[:, list(spec.identity_columns)].reset_index(drop=True),
                scores,
            ],
            axis=1,
        )

    if checkpoint_path is None:
        with tempfile.TemporaryDirectory(prefix="marin-dna-evals-gpu-parity-") as tmp:
            local_checkpoint = Path(tmp) / spec.checkpoint_name
            _download_checkpoint(spec.checkpoint_gcs_path, local_checkpoint)
            candidate = compute(local_checkpoint)
    else:
        candidate = compute(Path(checkpoint_path))

    embedding_report = validate_pooled_embeddings(candidate)

    # Open the S3 baseline only after worker-backed inference completes.
    # fsspec owns an asyncio thread; opening S3 before PyTorch forks data-loader
    # workers leaves the children waiting forever on the inherited event loop.
    baseline = read_verified_parquet(spec.baseline_scores_uri, spec.baseline_sha256)
    comparison = compare_score_frames(
        candidate,
        baseline,
        identity_columns=spec.identity_columns,
        tolerances=spec.tolerances,
    )
    return {
        "checkpoint": spec.checkpoint_name,
        "dataset": spec.dataset_repo,
        "dataset_revision": spec.dataset_revision,
        "dataset_filename": spec.dataset_filename,
        "split": spec.split,
        "n_rows": len(candidate),
        "baseline_scores_uri": spec.baseline_scores_uri,
        "baseline_sha256": spec.baseline_sha256,
        "baseline_pytorch_version": spec.baseline_pytorch_version,
        "baseline_compiled_cuda_version": spec.baseline_compiled_cuda_version,
        "baseline_driver_version": spec.baseline_driver_version,
        "score_comparison": comparison,
        "embeddings": embedding_report,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/gpu_runtime_validation.yaml",
        help="Path to the checked-in GPU runtime validation contract.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("smoke", help="Validate CUDA metadata and a bf16 operation.")
    parity = subparsers.add_parser(
        "parity", help="Run the full fixed inference cell and compare its score atoms."
    )
    parity.add_argument(
        "--checkpoint-path",
        help="Use an existing local checkpoint instead of downloading the pinned GCS path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected validation and print a machine-readable result."""
    args = _build_parser().parse_args(argv)
    spec = load_validation_spec(args.config)
    runtime = run_cuda_smoke(spec.runtime)
    result: dict[str, Any] = {"runtime": runtime}
    if args.command == "parity":
        result["parity"] = run_inference_parity(
            spec.parity,
            checkpoint_path=args.checkpoint_path,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
