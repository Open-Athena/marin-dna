#!/usr/bin/env python3
"""Review-gated publication and verification for the MarinDNA v0.5 scaling ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MANIFEST_PATH = HERE / "manifest.json"
STATE_PATH = HERE / "release_state.json"
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert len(manifest["models"]) == 8
    return manifest


def load_state() -> dict[str, Any] | None:
    if not STATE_PATH.exists():
        return None
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    assert state["schema_version"] == 1
    return state


def save_state(state: dict[str, Any]) -> None:
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(STATE_PATH)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def run(
    command: list[str], *, capture_output: bool = True
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        check=True,
        capture_output=capture_output,
    )


def run_json(command: list[str]) -> dict[str, Any]:
    value = json.loads(run(command).stdout)
    assert isinstance(value, dict), (command, type(value))
    return value


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    assert parsed.scheme == "s3" and parsed.netloc and parsed.path
    return parsed.netloc, parsed.path.lstrip("/")


def s3_file_uri(model: dict[str, Any], filename: str) -> str:
    return f"{model['s3_uri']}/{filename}"


def source_files(
    manifest: dict[str, Any], model: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    overlap = set(model["files"]) & set(manifest["tokenizer"]["files"])
    assert not overlap, overlap
    return {**model["files"], **manifest["tokenizer"]["files"]}


def review_paths(manifest: dict[str, Any]) -> list[Path]:
    return [
        REPO_ROOT / manifest["m51_model"]["card"],
        *(REPO_ROOT / model["card"] for model in manifest["models"]),
    ]


def review_digest(manifest: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in review_paths(manifest):
        assert path.is_file(), path
        relative = path.relative_to(REPO_ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def verify_review_materials(manifest: dict[str, Any]) -> None:
    article_link = (
        "[A 1B standard Transformer rivals Evo 2 40B on variant effect prediction]"
        f"({manifest['provenance']['blog']})"
    )
    forbidden_sections = (
        "## Intended uses",
        "## Evaluation and caveats",
        "## Provenance",
    )
    for model in manifest["models"]:
        card = (REPO_ROOT / model["card"]).read_text(encoding="utf-8")
        assert model["repo_id"] in card
        assert model["gcs_uri"] in card
        assert model["s3_uri"] in card
        assert model["wandb_url"] in card
        assert f"{model['parameter_count']:,}" in card
        assert "step-215573" in card
        assert manifest["training"]["training_script"] in card
        assert manifest["training"]["data_definition"] in card
        assert article_link in card
        assert not any(section in card for section in forbidden_sections)
        for tag in manifest["required_tags"]:
            assert f"  - {tag}\n" in card

    m51 = manifest["m51_model"]
    card = (REPO_ROOT / m51["card"]).read_text(encoding="utf-8")
    assert m51["repo_id"] in card
    assert m51["training_script"] in card
    assert article_link in card
    assert not any(section in card for section in forbidden_sections)
    for tag in manifest["required_tags"]:
        assert f"  - {tag}\n" in card


def s3_head(uri: str) -> dict[str, Any]:
    bucket, key = parse_s3_uri(uri)
    return run_json(
        [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--checksum-mode",
            "ENABLED",
        ]
    )


def read_s3_bytes(uri: str) -> bytes:
    return run(["aws", "s3", "cp", uri, "-", "--no-progress"]).stdout


def download_s3_file(uri: str, destination: Path) -> None:
    assert destination.parent.is_dir()
    run(
        ["aws", "s3", "cp", uri, str(destination), "--no-progress"],
        capture_output=False,
    )


def presigned_range(uri: str, start: int, end: int) -> bytes:
    assert 0 <= start <= end
    signed_url = (
        run(["aws", "s3", "presign", uri, "--expires-in", "300"])
        .stdout.decode()
        .strip()
    )
    assert signed_url.startswith("https://")
    request = Request(signed_url, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=60) as response:
        content = response.read()
    assert len(content) == end - start + 1, (uri, start, end, len(content))
    return content


def read_safetensors_header(uri: str) -> tuple[dict[str, Any], int]:
    header_size = int.from_bytes(presigned_range(uri, 0, 7), "little")
    assert 0 < header_size < 100_000_000, (uri, header_size)
    header = json.loads(presigned_range(uri, 8, 7 + header_size))
    assert isinstance(header, dict)
    return header, header_size


def inspect_safetensors(model: dict[str, Any]) -> None:
    tensor_files = sorted(
        name for name in model["files"] if name.endswith(".safetensors")
    )
    assert tensor_files
    parameter_count = 0
    tensor_count = 0
    tensor_bytes = 0
    dtypes: set[str] = set()

    for filename in tensor_files:
        header, header_size = read_safetensors_header(s3_file_uri(model, filename))
        metadata = header.pop("__metadata__", {})
        assert metadata == {"format": "pt"}, (model["label"], filename, metadata)
        offsets: list[tuple[int, int]] = []
        shard_tensor_bytes = 0
        for tensor in header.values():
            parameter_count += math.prod(tensor["shape"])
            start, end = tensor["data_offsets"]
            assert 0 <= start < end
            offsets.append((start, end))
            shard_tensor_bytes += end - start
            dtypes.add(tensor["dtype"])
        assert min(start for start, _ in offsets) == 0
        assert max(end for _, end in offsets) == shard_tensor_bytes
        assert 8 + header_size + shard_tensor_bytes == model["files"][filename]["size"]
        tensor_bytes += shard_tensor_bytes
        tensor_count += len(header)

    assert parameter_count == model["parameter_count"], (
        model["label"],
        parameter_count,
    )
    assert tensor_count == model["tensor_count"], (model["label"], tensor_count)
    assert tensor_bytes == model["tensor_bytes"], (model["label"], tensor_bytes)
    assert dtypes == {model["stored_dtype"]}, (model["label"], dtypes)

    index_name = "model.safetensors.index.json"
    if len(tensor_files) == 1:
        assert index_name not in model["files"]
    else:
        index = json.loads(read_s3_bytes(s3_file_uri(model, index_name)))
        assert index["metadata"]["total_size"] == model["tensor_bytes"]
        assert len(index["weight_map"]) == model["tensor_count"]
        assert set(index["weight_map"].values()) == set(tensor_files)


def verify_model_config(manifest: dict[str, Any], model: dict[str, Any]) -> None:
    config = json.loads(read_s3_bytes(s3_file_uri(model, "config.json")))
    expected_global = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "qwen3",
        "max_position_embeddings": manifest["training"]["sequence_length"],
        "vocab_size": manifest["tokenizer"]["vocab_size"],
        "bos_token_id": manifest["tokenizer"]["bos_token_id"],
        "eos_token_id": manifest["tokenizer"]["eos_token_id"],
        "pad_token_id": manifest["tokenizer"]["pad_token_id"],
        "tie_word_embeddings": False,
    }
    for key, expected in {**expected_global, **model["config"]}.items():
        assert config.get(key) == expected, (
            model["label"],
            key,
            config.get(key),
            expected,
        )


def verify_tokenizer_content(manifest: dict[str, Any]) -> None:
    model = manifest["models"][0]
    tokenizer_json = json.loads(read_s3_bytes(s3_file_uri(model, "tokenizer.json")))
    assert tokenizer_json["model"]["vocab"] == manifest["tokenizer"]["vocab"]
    assert tokenizer_json["normalizer"] == {"type": "Lowercase"}


def verify_s3_inventory(manifest: dict[str, Any], model: dict[str, Any]) -> None:
    bucket, prefix = parse_s3_uri(model["s3_uri"])
    listing = run_json(
        [
            "aws",
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            f"{prefix}/",
        ]
    )
    observed = {
        entry["Key"].removeprefix(f"{prefix}/") for entry in listing.get("Contents", [])
    }
    expected = set(source_files(manifest, model)) | {".snakemake_timestamp"}
    assert observed == expected, (model["label"], observed, expected)

    for filename, expected_metadata in source_files(manifest, model).items():
        metadata = s3_head(s3_file_uri(model, filename))
        assert metadata["ContentLength"] == expected_metadata["size"], (
            model["label"],
            filename,
        )
        assert metadata["ETag"].strip('"') == expected_metadata["etag"], (
            model["label"],
            filename,
        )
        assert metadata["ChecksumCRC64NVME"] == expected_metadata["crc64nvme_base64"], (
            model["label"],
            filename,
        )


def verify_datasets(manifest: dict[str, Any]) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for dataset in manifest["datasets"]:
        info = api.dataset_info(dataset["repo_id"], token=False)
        assert info.sha == dataset["revision"], (
            dataset["repo_id"],
            info.sha,
            dataset["revision"],
        )
        assert not info.private, dataset["repo_id"]
        assert not info.gated, dataset["repo_id"]


def verify_source(manifest: dict[str, Any]) -> None:
    verify_review_materials(manifest)
    for model in manifest["models"]:
        print(f"verifying S3 inventory and tensors: {model['label']}", flush=True)
        verify_s3_inventory(manifest, model)
        verify_model_config(manifest, model)
        inspect_safetensors(model)
    verify_tokenizer_content(manifest)
    verify_datasets(manifest)


def expected_repo_files(manifest: dict[str, Any], model: dict[str, Any]) -> set[str]:
    return set(source_files(manifest, model)) | {
        ".gitattributes",
        "LICENSE",
        "README.md",
    }


def find_model_info(
    api: Any, repo_id: str, *, token: bool | None, files_metadata: bool = False
) -> Any | None:
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        return api.model_info(repo_id, token=token, files_metadata=files_metadata)
    except RepositoryNotFoundError:
        return None


def hub_weight_matches(info: Any, filename: str, size: int, sha256: str) -> bool:
    sibling = next((item for item in info.siblings if item.rfilename == filename), None)
    return bool(
        sibling is not None
        and sibling.size == size
        and sibling.lfs is not None
        and sibling.lfs.sha256 == sha256
    )


def upload_tracked_file(
    api: Any, model: dict[str, Any], local_path: Path, path_in_repo: str
) -> str:
    assert local_path.is_file(), local_path
    sha256 = file_sha256(local_path)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=model["repo_id"],
        commit_message=f"Add {path_in_repo}",
    )
    return sha256


def upload_m51_readme(
    api: Any, manifest: dict[str, Any], state: dict[str, Any]
) -> None:
    model = manifest["m51_model"]
    local_path = REPO_ROOT / model["card"]
    expected = {
        "size": local_path.stat().st_size,
        "sha256": file_sha256(local_path),
        "uploaded": True,
    }
    model_state = state.get("m51_model")
    if model_state is None or model_state.get("readme") != expected:
        info = find_model_info(api, model["repo_id"], token=True)
        assert info is not None and not info.private, model["repo_id"]
        sha256 = upload_tracked_file(api, model, local_path, "README.md")
        assert sha256 == expected["sha256"]
        state["m51_model"] = {
            "repo_id": model["repo_id"],
            "readme": expected,
        }
        save_state(state)

    info = api.model_info(model["repo_id"], token=False)
    assert not info.private and not info.gated
    state["m51_model"]["final_revision"] = info.sha
    save_state(state)


def upload_model(
    api: Any,
    manifest: dict[str, Any],
    model: dict[str, Any],
    state: dict[str, Any],
) -> None:
    model_state = state["models"].setdefault(
        model["label"],
        {
            "repo_id": model["repo_id"],
            "created": False,
            "files": {},
            "release_files": {},
        },
    )
    assert model_state["repo_id"] == model["repo_id"]
    existing = find_model_info(api, model["repo_id"], token=True, files_metadata=True)
    if existing is not None and not model_state["created"]:
        raise AssertionError(
            f"refusing to adopt untracked existing repository: {model['repo_id']}"
        )
    if existing is None:
        api.create_repo(model["repo_id"], private=True, repo_type="model")
        model_state["created"] = True
        save_state(state)
    else:
        assert existing.private or model_state.get("public"), model["repo_id"]

    card_path = REPO_ROOT / model["card"]
    license_path = REPO_ROOT / manifest["license"]["source"]
    for path_in_repo, local_path in (
        ("README.md", card_path),
        ("LICENSE", license_path),
    ):
        expected_sha = file_sha256(local_path)
        release_entry = model_state["release_files"].get(path_in_repo)
        if release_entry != {
            "size": local_path.stat().st_size,
            "sha256": expected_sha,
            "uploaded": True,
        }:
            sha256 = upload_tracked_file(api, model, local_path, path_in_repo)
            model_state["release_files"][path_in_repo] = {
                "size": local_path.stat().st_size,
                "sha256": sha256,
                "uploaded": True,
            }
            save_state(state)

    for filename, expected in source_files(manifest, model).items():
        entry = model_state["files"].get(filename)
        info = find_model_info(api, model["repo_id"], token=True, files_metadata=True)
        assert info is not None
        if (
            entry is not None
            and entry.get("uploaded") is True
            and filename.endswith(".safetensors")
            and hub_weight_matches(info, filename, expected["size"], entry["sha256"])
        ):
            continue
        if (
            entry is not None
            and entry.get("uploaded") is True
            and filename in {s.rfilename for s in info.siblings}
        ):
            continue

        with tempfile.TemporaryDirectory(prefix="marindna-hf-upload-") as temporary:
            local_path = Path(temporary) / filename
            download_s3_file(s3_file_uri(model, filename), local_path)
            assert local_path.stat().st_size == expected["size"], (
                model["label"],
                filename,
            )
            sha256 = file_sha256(local_path)
            model_state["files"][filename] = {
                "size": expected["size"],
                "sha256": sha256,
                "uploaded": False,
            }
            save_state(state)
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=model["repo_id"],
                commit_message=f"Upload final {model['label']} checkpoint: {filename}",
            )
            model_state["files"][filename]["uploaded"] = True
            save_state(state)

    info = api.model_info(model["repo_id"], token=True, files_metadata=True)
    observed = {sibling.rfilename for sibling in info.siblings}
    assert observed == expected_repo_files(manifest, model), (model["label"], observed)
    for filename in (name for name in model["files"] if name.endswith(".safetensors")):
        entry = model_state["files"][filename]
        assert hub_weight_matches(info, filename, entry["size"], entry["sha256"]), (
            model["label"],
            filename,
        )
    model_state["uploaded_revision"] = info.sha
    save_state(state)


def collection_tuple(item: Any) -> tuple[str, str, str | None]:
    return item.item_type, item.item_id, item.note


def desired_collection_items(
    manifest: dict[str, Any],
) -> list[tuple[str, str, str | None]]:
    existing = [tuple(item) for item in manifest["collection"]["existing_items"]]
    new_models = [
        (
            "model",
            model["repo_id"],
            manifest["collection"]["new_model_notes"][model["label"]],
        )
        for model in manifest["models"]
    ]
    return [existing[0], *new_models, *existing[1:]]


def verify_collection_baseline(manifest: dict[str, Any], collection: Any) -> None:
    new_ids = {model["repo_id"] for model in manifest["models"]}
    observed_existing = [
        collection_tuple(item)
        for item in sorted(collection.items, key=lambda item: item.position)
        if item.item_id not in new_ids
    ]
    expected_existing = [
        tuple(item) for item in manifest["collection"]["existing_items"]
    ]
    assert observed_existing == expected_existing, (
        observed_existing,
        expected_existing,
    )


def update_collection(api: Any, manifest: dict[str, Any]) -> None:
    expected = manifest["collection"]
    collection = api.get_collection(expected["slug"])
    assert collection.title == expected["title"]
    assert (collection.description or "").strip() == expected["description"]
    assert not collection.private
    verify_collection_baseline(manifest, collection)

    for model in manifest["models"]:
        api.add_collection_item(
            expected["slug"],
            item_id=model["repo_id"],
            item_type="model",
            note=expected["new_model_notes"][model["label"]],
            exists_ok=True,
        )

    desired = desired_collection_items(manifest)
    for position, (_, item_id, note) in enumerate(desired):
        collection = api.get_collection(expected["slug"])
        item = next(item for item in collection.items if item.item_id == item_id)
        if item.note != note or item.position != position:
            api.update_collection_item(
                expected["slug"],
                item.item_object_id,
                note=note,
                position=position,
            )


def publish(manifest: dict[str, Any], approved_digest: str) -> None:
    from huggingface_hub import HfApi

    verify_review_materials(manifest)
    observed_digest = review_digest(manifest)
    assert approved_digest == observed_digest, (approved_digest, observed_digest)
    verify_source(manifest)

    api = HfApi()
    identity = api.whoami()
    orgs = {org["name"] for org in identity.get("orgs", [])}
    assert manifest["namespace"] in orgs, (identity.get("name"), orgs)

    state = load_state()
    if state is None:
        state = {
            "schema_version": 1,
            "review_digest": observed_digest,
            "models": {},
            "collection_updated": False,
        }
        save_state(state)
    elif state["review_digest"] != observed_digest:
        state["review_digest"] = observed_digest
        state["collection_updated"] = False
        save_state(state)

    collection = api.get_collection(manifest["collection"]["slug"])
    verify_collection_baseline(manifest, collection)

    print(f"updating README: {manifest['m51_model']['repo_id']}", flush=True)
    upload_m51_readme(api, manifest, state)

    for model in manifest["models"]:
        print(f"uploading private repository: {model['repo_id']}", flush=True)
        upload_model(api, manifest, model, state)

    for model in manifest["models"]:
        api.update_repo_settings(model["repo_id"], private=False)
        info = api.model_info(model["repo_id"], token=False)
        assert not info.private and not info.gated
        model_state = state["models"][model["label"]]
        model_state["public"] = True
        model_state["final_revision"] = info.sha
        save_state(state)

    update_collection(api, manifest)
    state["collection_updated"] = True
    save_state(state)


def verify_public_repo(
    api: Any, manifest: dict[str, Any], model: dict[str, Any], state: dict[str, Any]
) -> None:
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer

    model_state = state["models"][model["label"]]
    info = api.model_info(model["repo_id"], token=False, files_metadata=True)
    assert not info.private and not info.gated
    assert info.sha == model_state["final_revision"], (
        model["label"],
        info.sha,
        model_state["final_revision"],
    )
    assert info.card_data is not None
    assert info.card_data.license == manifest["license"]["id"]
    assert set(manifest["required_tags"]) <= set(info.tags)
    assert {sibling.rfilename for sibling in info.siblings} == expected_repo_files(
        manifest, model
    )

    for filename in (name for name in model["files"] if name.endswith(".safetensors")):
        entry = model_state["files"][filename]
        assert hub_weight_matches(info, filename, entry["size"], entry["sha256"]), (
            model["label"],
            filename,
        )

    with tempfile.TemporaryDirectory(prefix="marindna-hf-public-small-") as temporary:
        local_dir = Path(temporary)
        small_expected = {
            **{
                name: model_state["files"][name]
                for name in source_files(manifest, model)
                if not name.endswith(".safetensors")
            },
            **model_state["release_files"],
        }
        for filename, expected in small_expected.items():
            path = Path(
                hf_hub_download(
                    repo_id=model["repo_id"],
                    filename=filename,
                    local_dir=local_dir,
                    token=False,
                )
            )
            assert path.stat().st_size == expected["size"], (model["label"], filename)
            assert file_sha256(path) == expected["sha256"], (model["label"], filename)

    tokenizer = AutoTokenizer.from_pretrained(model["repo_id"], token=False)
    assert len(tokenizer) == manifest["tokenizer"]["vocab_size"]
    assert tokenizer("ACGT")["input_ids"] == [2, 3, 4, 5, 6]
    assert tokenizer("N")["input_ids"] == [2, 1]
    assert tokenizer.eos_token_id is None

    config = AutoConfig.from_pretrained(model["repo_id"], token=False)
    for key, expected in model["config"].items():
        assert getattr(config, key) == expected, (model["label"], key)
    assert config.max_position_embeddings == manifest["training"]["sequence_length"]


def verify_public_m51_readme(
    api: Any, manifest: dict[str, Any], state: dict[str, Any]
) -> None:
    from huggingface_hub import hf_hub_download

    model = manifest["m51_model"]
    model_state = state["m51_model"]
    info = api.model_info(model["repo_id"], token=False)
    assert not info.private and not info.gated
    assert info.sha == model_state["final_revision"]
    with tempfile.TemporaryDirectory(prefix="marindna-hf-public-m51-") as temporary:
        path = Path(
            hf_hub_download(
                repo_id=model["repo_id"],
                filename="README.md",
                local_dir=temporary,
                token=False,
            )
        )
        assert path.stat().st_size == model_state["readme"]["size"]
        assert file_sha256(path) == model_state["readme"]["sha256"]


def verify_public_collection(api: Any, manifest: dict[str, Any]) -> None:
    expected = manifest["collection"]
    collection = api.get_collection(expected["slug"], token=False)
    assert collection.title == expected["title"]
    assert not collection.private
    assert (collection.description or "").strip() == expected["description"]
    observed = [
        collection_tuple(item)
        for item in sorted(collection.items, key=lambda item: item.position)
    ]
    assert observed == desired_collection_items(manifest), (
        observed,
        desired_collection_items(manifest),
    )


def verify_representative_load(model: dict[str, Any]) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with tempfile.TemporaryDirectory(prefix="marindna-hf-public-model-") as temporary:
        checkpoint = Path(temporary) / "checkpoint"
        snapshot_download(repo_id=model["repo_id"], local_dir=checkpoint, token=False)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        loaded = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        loaded.eval()
        assert (
            sum(parameter.numel() for parameter in loaded.parameters())
            == model["parameter_count"]
        )
        inputs = tokenizer("ACGTACGTACGT", return_tensors="pt")
        with torch.no_grad():
            logits = loaded(**inputs).logits
        assert logits.shape == (1, 13, 7)
        assert torch.isfinite(logits).all()


def verify_public(manifest: dict[str, Any], *, load_smallest: bool) -> None:
    from huggingface_hub import HfApi

    state = load_state()
    assert state is not None, "release_state.json is required"
    assert state["review_digest"] == review_digest(manifest)
    assert state["collection_updated"] is True
    api = HfApi()
    print(f"verifying public README: {manifest['m51_model']['repo_id']}", flush=True)
    verify_public_m51_readme(api, manifest, state)
    for model in manifest["models"]:
        print(f"verifying public repository: {model['repo_id']}", flush=True)
        verify_public_repo(api, manifest, model, state)
    verify_public_collection(api, manifest)
    verify_datasets(manifest)
    if load_smallest:
        verify_representative_load(manifest["models"][0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "review-digest", help="Validate review materials and print their SHA-256 gate."
    )
    subparsers.add_parser(
        "verify-source",
        help="Verify S3, configs, tensors, tokenizer, cards, and datasets.",
    )
    publish_parser = subparsers.add_parser(
        "publish", help="Publish the reviewed release and update the Collection."
    )
    publish_parser.add_argument("--approved-review-digest", required=True)
    public_parser = subparsers.add_parser(
        "verify-public", help="Verify the anonymous public release."
    )
    public_parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Skip the representative full 46M model load; metadata and tokenizer checks still run for all eight.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest()
    if args.command == "review-digest":
        verify_review_materials(manifest)
        print(review_digest(manifest))
    elif args.command == "verify-source":
        verify_source(manifest)
        print("scaling-ladder source verification passed")
    elif args.command == "publish":
        publish(manifest, args.approved_review_digest)
        print("scaling-ladder publication completed")
    elif args.command == "verify-public":
        verify_public(manifest, load_smallest=not args.skip_model_load)
        print("scaling-ladder public verification passed")
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
