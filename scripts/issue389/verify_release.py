#!/usr/bin/env python3
"""Fail-fast verification for the issue #389 m5.1 Hugging Face release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MANIFEST_PATH = HERE / "manifest.json"
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["schema_version"] == 1
    return manifest


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert isinstance(result, dict), (command, type(result))
    return result


def verify_tracked_artifacts(manifest: dict[str, Any]) -> None:
    artifacts = list(manifest["destination"]["release_files"].values())
    artifacts.append(manifest["collection"]["description"])
    for expected in artifacts:
        path = REPO_ROOT / expected["source"]
        assert path.is_file(), f"missing tracked release artifact: {path}"
        assert path.stat().st_size == expected["size"], path
        assert file_hash(path, "sha256") == expected["sha256"], path

    card = (HERE / "model_card.md").read_text(encoding="utf-8")
    for tag in manifest["destination"]["required_tags"]:
        assert f"  - {tag}\n" in card, f"missing model-card tag: {tag}"
    for key in (
        "training_script",
        "blog",
        "analysis_issue",
        "interpretation_issue",
    ):
        assert manifest["provenance"][key] in card, f"missing model-card link: {key}"
    for dataset in manifest["datasets"]:
        assert dataset["repo_id"] in card, dataset["repo_id"]
        assert dataset["revision"] in card, dataset["revision"]


def verify_cloud_sources(manifest: dict[str, Any]) -> None:
    source = manifest["source"]
    gcs_base = source["canonical_gcs_uri"]
    parsed_s3 = urlparse(source["transfer_s3_uri"])
    assert parsed_s3.scheme == "s3"
    s3_bucket = parsed_s3.netloc
    s3_prefix = parsed_s3.path.lstrip("/")

    total_size = 0
    for filename, expected in source["files"].items():
        gcs = run_json(
            [
                "gcloud",
                "storage",
                "objects",
                "describe",
                f"{gcs_base}/{filename}",
                "--format=json",
            ]
        )
        assert int(gcs["size"]) == expected["size"], filename
        assert gcs["generation"] == expected["gcs"]["generation"], filename
        assert gcs["md5_hash"] == expected["gcs"]["md5_base64"], filename
        assert gcs["crc32c_hash"] == expected["gcs"]["crc32c_base64"], filename

        s3 = run_json(
            [
                "aws",
                "s3api",
                "head-object",
                "--bucket",
                s3_bucket,
                "--key",
                f"{s3_prefix}/{filename}",
                "--checksum-mode",
                "ENABLED",
            ]
        )
        assert int(s3["ContentLength"]) == expected["size"], filename
        assert s3["ETag"].strip('"') == expected["s3"]["etag"], filename
        assert s3["ChecksumCRC64NVME"] == expected["s3"]["checksum_crc64nvme_base64"], (
            filename
        )
        total_size += expected["size"]

    assert total_size == source["total_checkpoint_bytes"]


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        assert 0 < header_size < path.stat().st_size
        header = json.loads(handle.read(header_size))
    assert isinstance(header, dict)
    return header, header_size


def verify_model_structure(checkpoint_dir: Path, manifest: dict[str, Any]) -> None:
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    model_expected = manifest["model"]
    assert config["architectures"] == [model_expected["architecture"]]
    assert config["model_type"] == model_expected["model_type"]
    for key, expected in model_expected["config"].items():
        assert config.get(key) == expected, (key, config.get(key), expected)

    header, header_size = read_safetensors_header(checkpoint_dir / "model.safetensors")
    metadata = header.pop("__metadata__", {})
    assert metadata == {"format": "pt"}
    assert len(header) == model_expected["tensor_count"]

    parameter_count = 0
    tensor_bytes = 0
    dtypes: set[str] = set()
    offsets: list[tuple[int, int]] = []
    for tensor in header.values():
        parameter_count += math.prod(tensor["shape"])
        start, end = tensor["data_offsets"]
        assert 0 <= start < end
        tensor_bytes += end - start
        offsets.append((start, end))
        dtypes.add(tensor["dtype"])

    assert parameter_count == model_expected["parameter_count"]
    assert tensor_bytes == model_expected["tensor_bytes"]
    assert dtypes == {model_expected["stored_dtype"]}
    assert min(start for start, _ in offsets) == 0
    assert max(end for _, end in offsets) == tensor_bytes
    assert (
        8 + header_size + tensor_bytes
        == (checkpoint_dir / "model.safetensors").stat().st_size
    )


def verify_tokenizer(checkpoint_dir: Path, manifest: dict[str, Any]) -> None:
    from transformers import AutoTokenizer

    expected = manifest["tokenizer"]
    tokenizer_json = json.loads(
        (checkpoint_dir / "tokenizer.json").read_text(encoding="utf-8")
    )
    assert tokenizer_json["model"]["vocab"] == expected["vocab"]
    assert tokenizer_json["normalizer"] == {"type": "Lowercase"}

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    assert len(tokenizer) == expected["vocab_size"]
    assert tokenizer.bos_token == expected["bos_token"]
    assert tokenizer.bos_token_id == expected["bos_token_id"]
    assert tokenizer.pad_token == expected["pad_token"]
    assert tokenizer.pad_token_id == expected["pad_token_id"]
    assert tokenizer.unk_token == expected["unk_token"]
    assert tokenizer.unk_token_id == expected["unk_token_id"]
    assert tokenizer.eos_token is None
    assert tokenizer.eos_token_id is None
    assert tokenizer("ACGT")["input_ids"] == [2, 3, 4, 5, 6]
    assert tokenizer("acgt")["input_ids"] == [2, 3, 4, 5, 6]
    assert tokenizer("N")["input_ids"] == [2, 1]


def verify_checkpoint_files(
    checkpoint_dir: Path,
    manifest: dict[str, Any],
    *,
    exact_inventory: bool,
) -> None:
    assert checkpoint_dir.is_dir(), checkpoint_dir
    expected_files = manifest["source"]["files"]
    observed = {path.name for path in checkpoint_dir.iterdir() if path.is_file()}
    if exact_inventory:
        assert observed == set(expected_files), (observed, set(expected_files))
    else:
        assert set(expected_files) <= observed

    for filename, expected in expected_files.items():
        path = checkpoint_dir / filename
        assert path.stat().st_size == expected["size"], path
        assert file_hash(path, "md5") == expected["md5_hex"], path
        assert file_hash(path, "sha256") == expected["sha256"], path

    verify_model_structure(checkpoint_dir, manifest)
    verify_tokenizer(checkpoint_dir, manifest)


def verify_inference(checkpoint_dir: Path, manifest: dict[str, Any]) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    expected = manifest["deterministic_inference"]
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    inputs = tokenizer(expected["sequence"], return_tensors="pt")
    assert inputs["input_ids"][0].tolist() == expected["input_ids"]

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == manifest["model"]["parameter_count"]
    )
    with torch.no_grad():
        observed = model(**inputs).logits[0, -1].float().tolist()

    assert len(observed) == len(expected["last_token_logits"])
    differences = [
        abs(actual - reference)
        for actual, reference in zip(
            observed, expected["last_token_logits"], strict=True
        )
    ]
    assert max(differences) <= expected["absolute_tolerance"], differences
    predicted_token_id = max(range(len(observed)), key=observed.__getitem__)
    assert predicted_token_id == expected["predicted_token_id"]
    assert (
        tokenizer.convert_ids_to_tokens(predicted_token_id)
        == expected["predicted_token"]
    )


def verify_datasets(manifest: dict[str, Any]) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for expected in manifest["datasets"]:
        info = api.dataset_info(expected["repo_id"], token=False)
        assert info.sha == expected["revision"], (
            expected["repo_id"],
            info.sha,
            expected["revision"],
        )
        assert not info.private, expected["repo_id"]
        assert not info.gated, expected["repo_id"]


def verify_collection(
    manifest: dict[str, Any],
    collection_slug: str,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    expected = manifest["collection"]
    collection = api.get_collection(collection_slug, token=False)
    assert collection.title == expected["title"]
    assert collection.owner == expected["namespace"]
    assert not collection.private
    expected_description = (REPO_ROOT / expected["description"]["source"]).read_text(
        encoding="utf-8"
    )
    assert (collection.description or "").strip() == expected_description.strip()

    observed_items = [
        (item.item_type, item.item_id, item.note)
        for item in sorted(collection.items, key=lambda item: item.position)
    ]
    expected_items = [
        (item["item_type"], item["item_id"], item["note"]) for item in expected["items"]
    ]
    assert observed_items == expected_items


def verify_public_hub(
    manifest: dict[str, Any],
    hub_dir: Path | None,
    collection_slug: str,
    run_inference: bool,
) -> None:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    destination = manifest["destination"]
    info = api.model_info(destination["repo_id"], token=False)
    assert not info.private
    assert not info.gated
    if destination["final_revision"] is not None:
        assert info.sha == destination["final_revision"]
    assert info.card_data is not None
    assert info.card_data.license == destination["license"]
    assert set(destination["required_tags"]) <= set(info.tags)

    observed_files = {sibling.rfilename for sibling in info.siblings}
    expected_files = (
        set(manifest["source"]["files"])
        | set(destination["release_files"])
        | set(destination["allowed_additional_files"])
    )
    assert observed_files == expected_files, (observed_files, expected_files)

    if hub_dir is not None:
        snapshot_download(
            repo_id=destination["repo_id"],
            revision=info.sha,
            local_dir=hub_dir,
            token=False,
        )
        verify_checkpoint_files(hub_dir, manifest, exact_inventory=False)
        for filename, expected in destination["release_files"].items():
            path = hub_dir / filename
            assert path.stat().st_size == expected["size"], path
            assert file_hash(path, "sha256") == expected["sha256"], path
        if run_inference:
            verify_inference(hub_dir, manifest)

    verify_collection(manifest, collection_slug)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Local five-file source checkpoint to verify.",
    )
    parser.add_argument(
        "--check-cloud",
        action="store_true",
        help="Verify GCS and evals_v2 S3 object metadata.",
    )
    parser.add_argument(
        "--check-datasets",
        action="store_true",
        help="Verify all twelve public dataset revisions.",
    )
    parser.add_argument(
        "--hub",
        action="store_true",
        help="Verify the public, ungated model repository and Collection.",
    )
    parser.add_argument(
        "--hub-dir",
        type=Path,
        help="Download and verify the public repository at this path.",
    )
    parser.add_argument(
        "--collection-slug",
        help="Published Collection slug; defaults to manifest.collection.slug.",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Run the deterministic CPU forward pass.",
    )
    args = parser.parse_args()
    if not any((args.checkpoint_dir, args.check_cloud, args.check_datasets, args.hub)):
        parser.error("select at least one verification target")
    if args.run_inference and args.checkpoint_dir is None and args.hub_dir is None:
        parser.error("--run-inference needs --checkpoint-dir or --hub-dir")
    if args.hub_dir is not None and not args.hub:
        parser.error("--hub-dir requires --hub")
    return args


def main() -> None:
    args = parse_args()
    manifest = load_manifest()
    verify_tracked_artifacts(manifest)

    if args.check_cloud:
        verify_cloud_sources(manifest)
    if args.check_datasets:
        verify_datasets(manifest)
    if args.checkpoint_dir is not None:
        verify_checkpoint_files(
            args.checkpoint_dir,
            manifest,
            exact_inventory=True,
        )
        if args.run_inference:
            verify_inference(args.checkpoint_dir, manifest)
    if args.hub:
        collection_slug = args.collection_slug or manifest["collection"]["slug"]
        assert collection_slug, "published Collection slug is required"
        verify_public_hub(
            manifest,
            args.hub_dir,
            collection_slug,
            args.run_inference,
        )

    print("issue #389 release verification passed")


if __name__ == "__main__":
    main()
