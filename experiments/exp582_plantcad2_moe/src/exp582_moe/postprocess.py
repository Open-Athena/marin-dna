"""Collect, reduce, and probe one completed exp582 final evaluation."""

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import fsspec
import jax
import wandb

from exp582_moe.evaluation_binding import CheckpointRole
from exp582_moe.gpu_smoke import SmokeConfig, dispatch
from exp582_moe.real_eval import fetch_file, read_bytes


@dataclasses.dataclass(frozen=True)
class PostprocessConfig(SmokeConfig):
    requests_uri: str = ""
    requests_sha256: str = ""
    artifact_root: str = ""
    analysis_wheel_uri: str = ""
    analysis_wheel_sha256: str = ""
    output_prefix: str = ""
    skip_probes: bool = False
    probe_only: bool = False
    reduced_analysis_root: str = ""
    reduced_analysis_wheel_sha256: str = ""
    checkpoint_role: CheckpointRole = "dna-trained"

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.nodes != 1
            or len(self.requests_sha256) != 64
            or len(self.analysis_wheel_sha256) != 64
            or any(
                not value.startswith("s3://")
                for value in (
                    self.artifact_root,
                    self.analysis_wheel_uri,
                    self.output_prefix,
                )
            )
        ):
            raise ValueError("Postprocessing requires pinned durable inputs")
        if self.skip_probes and self.probe_only:
            raise ValueError("Probe-only postprocessing cannot skip probes")
        if self.probe_only != self.reduced_analysis_root.startswith("s3://"):
            raise ValueError(
                "Probe-only postprocessing requires exactly one reduced analysis root"
            )
        if self.reduced_analysis_wheel_sha256 and (
            not self.probe_only or len(self.reduced_analysis_wheel_sha256) != 64
        ):
            raise ValueError(
                "A separately pinned reduction wheel applies only to probe-only postprocessing"
            )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 2**20):
            value.update(block)
    return value.hexdigest()


def download_tree(uri: str, output: Path) -> int:
    filesystem, root = fsspec.core.url_to_fs(uri)
    files = sorted(path for path in filesystem.find(root) if not path.endswith("/"))
    if not files:
        raise ValueError(f"No evaluation artifacts at {uri}")

    def download(path: str) -> None:
        target = output / path.removeprefix(root).lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        filesystem.get_file(path, str(target))

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(download, files))
    return len(files)


def download_verified(uri: str, output: Path, record: dict) -> None:
    filesystem, path = fsspec.core.url_to_fs(uri)
    output.parent.mkdir(parents=True, exist_ok=True)
    filesystem.get_file(path, str(output))
    if output.stat().st_size != record["bytes"] or digest(output) != record["sha256"]:
        raise ValueError(f"Reduced analysis differs from its receipt: {uri}")


def collect_reduced_predictions(
    config: PostprocessConfig, output: Path
) -> dict[str, str]:
    """Fetch only the validated AF reductions needed to fit the frozen probes."""
    receipt_uri = config.reduced_analysis_root.rstrip("/") + "/receipt.json"
    receipt_blob = read_bytes(receipt_uri)
    receipt = json.loads(receipt_blob)
    expected = {
        "condition": config.condition,
        "requests_sha256": config.requests_sha256,
        "analysis_wheel_sha256": (
            config.reduced_analysis_wheel_sha256 or config.analysis_wheel_sha256
        ),
        "source_artifacts": config.artifact_root,
        "probes_included": False,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Reduced analysis receipt does not match the probe inputs")
    if receipt.get("checkpoint_role", "dna-trained") != config.checkpoint_role:
        raise ValueError("Reduced analysis checkpoint role does not match")
    selected = {}
    for profile in ("context-8192", "context-2048"):
        relative = f"{profile}/af-predictions.parquet"
        record = receipt.get("files", {}).get(relative)
        if record is None:
            raise ValueError(f"Reduced analysis receipt omits {relative}")
        if not isinstance(record, dict):
            raise TypeError(f"Reduced analysis receipt has invalid {relative}")
        download_verified(
            config.reduced_analysis_root.rstrip("/") + "/" + relative,
            output / relative,
            record,
        )
        selected[relative] = record["sha256"]
    return {
        "root": config.reduced_analysis_root,
        "receipt_sha256": hashlib.sha256(receipt_blob).hexdigest(),
        "files": selected,
    }


def upload_tree(directory: Path, uri: str) -> dict[str, dict]:
    filesystem, root = fsspec.core.url_to_fs(uri)
    records = {}
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        target = root.rstrip("/") + "/" + relative
        expected = digest(path)
        if filesystem.exists(target):
            if filesystem.info(target)["size"] != path.stat().st_size:
                raise ValueError(f"Existing analysis size differs: {relative}")
        else:
            filesystem.put_file(str(path), target)
        value = hashlib.sha256()
        with filesystem.open(target, "rb") as stream:
            while block := stream.read(16 * 2**20):
                value.update(block)
        if value.hexdigest() != expected:
            raise ValueError(f"Analysis readback differs: {relative}")
        records[relative] = {"sha256": expected, "bytes": path.stat().st_size}
    return records


def run_postprocess(config: PostprocessConfig) -> None:
    run = wandb.init(
        entity="eric-czech",
        project="marin",
        id=config.run_id,
        name=config.run_id,
        group="exp582-plantcad2-moe-final-evaluation",
        resume="allow",
        tags=[
            "dna-exp582",
            "evaluation",
            "postprocess",
            "H100",
            config.condition,
            config.checkpoint_role,
        ],
        config=dataclasses.asdict(config),
    )
    run.summary.update({"evaluation/phase": "loading", "evaluation/error": None})
    try:
        payload = read_bytes(config.requests_uri)
        if hashlib.sha256(payload).hexdigest() != config.requests_sha256:
            raise ValueError("Postprocessing request digest differs")
        preparation = json.loads(payload)
        root = Path(tempfile.mkdtemp(prefix="exp582-postprocess-"))
        inputs = root / "inputs"
        for item in preparation["inputs"].values():
            for record in item["files"]:
                fetch_file(record, inputs)
        preparation["samples"] = str(inputs)
        preparation_path = inputs / "preparation.json"
        preparation_path.write_text(json.dumps(preparation, indent=2) + "\n")

        wheel_blob = read_bytes(config.analysis_wheel_uri)
        if hashlib.sha256(wheel_blob).hexdigest() != config.analysis_wheel_sha256:
            raise ValueError("Analysis wheel digest differs")
        wheel = root / "analysis.whl"
        wheel.write_bytes(wheel_blob)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root / "code")
        sys.path.insert(0, str(root / "code"))
        artifacts = root / "artifacts"
        if config.probe_only:
            run.summary["evaluation/phase"] = "collecting_embeddings"
            count = sum(
                download_tree(
                    f"{config.artifact_root}/{profile}/embeddings",
                    artifacts / profile / "embeddings",
                )
                for profile in ("context-8192", "context-2048")
            )
        else:
            run.summary["evaluation/phase"] = "collecting"
            count = download_tree(config.artifact_root, artifacts)
        run.summary["evaluation/downloaded_files"] = count

        output = root / "analysis"
        reduced_source = None
        if config.probe_only:
            run.summary["evaluation/phase"] = "loading_reductions"
            reduced_source = collect_reduced_predictions(config, output)
        else:
            from exp582_evals.full_metrics import reduce

            run.summary["evaluation/phase"] = "reducing"
            reduce(preparation_path, artifacts, output)
        if config.probe_only or not config.skip_probes:
            from exp582_evals.full_probe import fit_probes

            for profile in ("context-8192", "context-2048"):
                run.summary["evaluation/phase"] = f"probing_{profile}"
                fit_probes(
                    preparation_path,
                    artifacts / profile / "embeddings",
                    output / profile / "af-predictions.parquet",
                    output / profile / "probes",
                )
        records = upload_tree(output, config.output_prefix)
        receipt = {
            "schema": 2,
            "condition": config.condition,
            "checkpoint_role": config.checkpoint_role,
            "requests_sha256": config.requests_sha256,
            "analysis_wheel_sha256": config.analysis_wheel_sha256,
            "source_artifacts": config.artifact_root,
            "mode": "probe_only" if config.probe_only else "full",
            "probes_included": config.probe_only or not config.skip_probes,
            "reduced_analysis": reduced_source,
            "files": records,
        }
        receipt_blob = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
        receipt_uri = config.output_prefix.rstrip("/") + "/receipt.json"
        filesystem, path = fsspec.core.url_to_fs(receipt_uri)
        filesystem.pipe(path, receipt_blob)
        if filesystem.cat(path) != receipt_blob:
            raise ValueError("Analysis receipt readback differs")
        run.summary.update(
            {
                "evaluation/phase": "complete",
                "evaluation/output": config.output_prefix,
                "evaluation/receipt": receipt_uri,
                "evaluation/files": len(records),
            }
        )
        run.log({"run_progress": 1.0}, step=1)
        run.finish()
    except Exception as error:
        run.summary.update(
            {"evaluation/phase": "failed", "evaluation/error": str(error)}
        )
        run.finish(exit_code=1)
        raise


def run_postprocess_distributed(config: PostprocessConfig) -> None:
    """Run once within Iris's eight independent local Python processes."""
    iris_rank = os.environ.get("IRIS_MULTIGPU_PROCESS_INDEX")
    writer = int(iris_rank) == 0 if iris_rank is not None else jax.process_index() == 0
    if writer:
        run_postprocess(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "condition",
        "run-id",
        "cluster",
        "requests-uri",
        "requests-sha256",
        "artifact-root",
        "analysis-wheel-uri",
        "analysis-wheel-sha256",
        "output-prefix",
    ]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--reduced-analysis-root", default="")
    parser.add_argument("--reduced-analysis-wheel-sha256", default="")
    parser.add_argument(
        "--checkpoint-role",
        choices=("dna-trained", "language-base"),
        default="dna-trained",
    )
    dispatch(
        PostprocessConfig(**vars(parser.parse_args())),
        worker=run_postprocess_distributed,
        moe_implementation="sonic",
        timeout_hours=18,
    )


if __name__ == "__main__":
    main()
