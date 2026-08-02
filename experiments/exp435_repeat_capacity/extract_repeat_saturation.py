"""Extract selected-feature responses for frozen repeat saturation states."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from huggingface_hub import snapshot_download
from marin_dna.model.sae import M51_HIDDEN_SIZE, load_frozen_m51
from sae_lens.load_model import HookedProxyLM
from sae_lens.saes.sae import SAE

from common import ISSUE, assert_commit, sha256_file, write_json
from extract_common import (
    BLOCK_INDICES,
    D_SAE,
    FOCAL_INDEX,
    HOOK_NAMES,
    MODEL_ID,
    MODEL_REVISION,
    WINDOW_BP,
    arm_label,
    read_model_provenance,
)
from saturation_common import (
    FEATURES,
    MOTIF_ARCHIVE_MANIFEST_SHA256,
    MOTIF_ARCHIVE_RUN_ID,
    MUTATIONS_PER_CONTEXT,
    RUN_ID,
    STATES_PER_CONTEXT,
    VIEW_KEYS,
    build_state_table,
    qualifying_kmer_sets,
    select_contexts,
    verify_motif_archive,
)

assert M51_HIDDEN_SIZE == 1_920
REPORTED_TO_IMPLEMENTATION = {index + 1: index for index in BLOCK_INDICES}
FEATURE_IDS_BY_IMPLEMENTATION = {
    block_index: tuple(
        item.feature_id for item in FEATURES if item.block == block_index + 1
    )
    for block_index in BLOCK_INDICES
}
FEATURE_COLUMN_BY_IMPLEMENTATION = {
    block_index: {
        feature_id: column
        for column, feature_id in enumerate(FEATURE_IDS_BY_IMPLEMENTATION[block_index])
    }
    for block_index in BLOCK_INDICES
}


def model_path(*, block_index: int, models_root: Path) -> Path:
    path = models_root / arm_label(block_index)
    assert path.is_dir(), path
    return path


@torch.inference_mode()
def extract_raw_focal(
    sequences: Sequence[str], *, tokenizer: Any, model: HookedProxyLM
) -> dict[int, torch.Tensor]:
    encoded = tokenizer(
        list(sequences),
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_tensors="pt",
    )
    tokens = encoded["input_ids"].to("cuda")
    assert tokens.shape == (len(sequences), WINDOW_BP + 1)
    output, cache = model.run_with_cache(
        tokens,
        names_filter=list(HOOK_NAMES),
        stop_at_layer=max(BLOCK_INDICES) + 1,
    )
    assert output is None and set(cache) == set(HOOK_NAMES)
    raw: dict[int, torch.Tensor] = {}
    for block_index, hook_name in zip(BLOCK_INDICES, HOOK_NAMES, strict=True):
        captured = cache[hook_name]
        assert captured.shape == (len(sequences), WINDOW_BP + 1, M51_HIDDEN_SIZE)
        focal = captured[:, FOCAL_INDEX + 1, :].float()
        assert torch.isfinite(focal).all()
        raw[block_index] = focal
    return raw


def encode_selected_features(
    sae: SAE, raw: torch.Tensor, feature_ids: tuple[int, ...]
) -> torch.Tensor:
    """Evaluate selected JumpReLU columns with the exported exact formula."""

    assert not sae.training and raw.ndim == 2 and raw.shape[-1] == sae.cfg.d_in
    assert sae.cfg.architecture() == "jumprelu"
    assert sae.cfg.normalize_activations == "none"
    assert not sae.hook_z_reshaping_mode
    assert feature_ids and all(0 <= item < D_SAE for item in feature_ids)
    indices = torch.tensor(feature_ids, device=raw.device, dtype=torch.long)
    with torch.inference_mode():
        sae_in = sae.process_sae_in(raw)
        hidden_pre = sae_in @ sae.W_enc[:, indices] + sae.b_enc[indices]
        base_acts = sae.activation_fn(hidden_pre)
        selected = base_acts * (hidden_pre > sae.threshold[indices]).to(base_acts.dtype)
    assert selected.shape == (raw.shape[0], len(feature_ids))
    assert torch.isfinite(selected).all() and torch.all(selected >= 0)
    return selected


@torch.inference_mode()
def extract_batch(
    frame: pl.DataFrame,
    *,
    tokenizer: Any,
    model: HookedProxyLM,
    saes: dict[int, SAE],
    validate_formula: bool,
) -> np.ndarray:
    sequences = frame["sequence"].to_list()
    raw_layers = extract_raw_focal(sequences, tokenizer=tokenizer, model=model)
    activations = np.empty(frame.height, dtype=np.float32)
    assigned = np.zeros(frame.height, dtype=bool)
    target_blocks = frame["block"].to_numpy()
    target_features = frame["feature_id"].to_numpy()
    for block_index in BLOCK_INDICES:
        feature_ids = FEATURE_IDS_BY_IMPLEMENTATION[block_index]
        selected = encode_selected_features(
            saes[block_index], raw_layers[block_index], feature_ids
        )
        if validate_formula:
            expected = saes[block_index].encode(
                raw_layers[block_index][: min(4, frame.height)]
            )[:, list(feature_ids)]
            torch.testing.assert_close(
                selected[: expected.shape[0]], expected, rtol=1e-6, atol=1e-5
            )
        reported_block = block_index + 1
        for feature_id in feature_ids:
            mask = (target_blocks == reported_block) & (target_features == feature_id)
            if not mask.any():
                continue
            assert not assigned[mask].any()
            column = FEATURE_COLUMN_BY_IMPLEMENTATION[block_index][feature_id]
            values = selected[:, column].cpu().numpy()
            activations[mask] = values[mask]
            assigned[mask] = True
    assert assigned.all()
    assert np.isfinite(activations).all() and np.all(activations >= 0)
    return activations


def build_response_table(states: pl.DataFrame, activations: np.ndarray) -> pl.DataFrame:
    """Pair every mutant response with its context-specific baseline."""

    assert activations.shape == (states.height,)
    mutations = states.filter(~pl.col("is_baseline")).drop("sequence")
    state_indices = mutations["state_index"].to_numpy()
    baseline_indices = mutations["baseline_state_index"].to_numpy()
    baseline = activations[baseline_indices]
    mutant = activations[state_indices]
    assert np.all(baseline > 0)
    result = mutations.with_columns(
        pl.Series("baseline_activation", baseline, pl.Float32),
        pl.Series("mutant_activation", mutant, pl.Float32),
        pl.Series("delta", mutant - baseline, pl.Float32),
        pl.Series("abs_delta", np.abs(mutant - baseline), pl.Float32),
        pl.Series("relative_delta", (mutant - baseline) / baseline, pl.Float32),
        pl.Series("thresholded_to_zero", mutant == 0, pl.Boolean),
    )
    assert (
        result.height == (states.height // STATES_PER_CONTEXT) * MUTATIONS_PER_CONTEXT
    )
    assert result.select(
        pl.col("baseline_activation").is_finite().all(),
        pl.col("mutant_activation").is_finite().all(),
        pl.col("delta").is_finite().all(),
        pl.col("relative_delta").is_finite().all(),
    ).row(0) == (True, True, True, True)
    return result


def extract(
    *,
    motif_archive_root: Path,
    models_root: Path,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert batch_size > 0 and not output_dir.exists() and models_root.is_dir()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    assert os.environ.get("RUN_ID") == RUN_ID
    started = time.monotonic()
    archive, top_contexts, kmers = verify_motif_archive(motif_archive_root)
    contexts = select_contexts(top_contexts)
    kmer_sets = qualifying_kmer_sets(kmers)
    states = build_state_table(contexts, kmer_sets)
    output_dir.mkdir(parents=True)

    provenance: dict[str, Any] = {}
    saes: dict[int, SAE] = {}
    for block_index in BLOCK_INDICES:
        path = model_path(block_index=block_index, models_root=models_root)
        provenance[arm_label(block_index)] = read_model_provenance(
            path, block_index=block_index
        )
        sae = SAE.load_from_disk(path, device="cuda", dtype="float32")
        sae.requires_grad_(False)
        sae.eval()
        assert all(not parameter.requires_grad for parameter in sae.parameters())
        saes[block_index] = sae

    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    frozen.model.config.use_cache = False
    model = HookedProxyLM(frozen.model, frozen.tokenizer, hook_names=list(HOOK_NAMES))
    torch.cuda.reset_peak_memory_stats()

    activations = np.empty(states.height, dtype=np.float32)
    for offset in range(0, states.height, batch_size):
        stop = min(offset + batch_size, states.height)
        batch = states.slice(offset, stop - offset)
        activations[offset:stop] = extract_batch(
            batch,
            tokenizer=frozen.tokenizer,
            model=model,
            saes=saes,
            validate_formula=offset == 0,
        )
        if offset == 0 or stop == states.height or stop % (batch_size * 25) == 0:
            print(
                json.dumps(
                    {
                        "stage": "repeat_saturation",
                        "processed": stop,
                        "total": states.height,
                    }
                ),
                flush=True,
            )
    torch.cuda.synchronize()

    baseline = activations[::STATES_PER_CONTEXT]
    np.testing.assert_allclose(
        baseline,
        contexts["activation"].to_numpy(),
        rtol=1e-4,
        atol=1e-4,
    )
    responses = build_response_table(states, activations)
    context_path = output_dir / "contexts.parquet"
    state_path = output_dir / "sequence_states.parquet"
    response_path = output_dir / "mutation_responses.parquet"
    contexts.write_parquet(context_path, compression="zstd")
    states.write_parquet(state_path, compression="zstd")
    responses.write_parquet(response_path, compression="zstd")
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (context_path, state_path, response_path)
    }
    elapsed = time.monotonic() - started
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": RUN_ID,
        "analysis_status": "post_hoc_repeat_motif_saturation",
        "experiment_commit": experiment_commit,
        "elapsed_seconds": elapsed,
        "sequences_per_second": states.height / elapsed,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "reported_blocks": [index + 1 for index in BLOCK_INDICES],
            "implementation_block_indices": list(BLOCK_INDICES),
            "torch_compile": False,
            "torch_compile_reason": "validated dynamic SAE-Lens hook/cache path is eager",
        },
        "saes": provenance,
        "input": {
            "motif_archive_run_id": MOTIF_ARCHIVE_RUN_ID,
            "motif_archive_manifest_sha256": MOTIF_ARCHIVE_MANIFEST_SHA256,
            "motif_archive_objects": archive["object_count_excluding_this_manifest"],
        },
        "protocol": {
            "features": [
                {"block": item.block, "feature_id": item.feature_id}
                for item in FEATURES
            ],
            "views": len(VIEW_KEYS),
            "contexts_per_view": contexts.height // len(VIEW_KEYS),
            "motif_radius": 31,
            "states_per_context": STATES_PER_CONTEXT,
            "mutations_per_context": MUTATIONS_PER_CONTEXT,
            "feature_encoding": "exact selected-column JumpReLU formula",
            "shared_forward_layers": len(BLOCK_INDICES),
            "batch_size_sequences": batch_size,
            "sae_dtype": "float32",
        },
        "outputs": {
            "contexts": contexts.height,
            "sequence_states": states.height,
            "mutation_responses": responses.height,
            "motif_loss_responses": responses.filter(pl.col("motif_loss")).height,
            "neutral_responses": responses.filter(pl.col("neutral")).height,
            "thresholded_to_zero": int(responses["thresholded_to_zero"].sum()),
            "maximum_baseline_replay_absolute_error": float(
                np.max(np.abs(baseline - contexts["activation"].to_numpy()))
            ),
        },
        "artifacts": artifacts,
    }
    result_path = output_dir / "results.json"
    write_json(result_path, result)
    result["artifacts"]["results.json"] = {
        "bytes": result_path.stat().st_size,
        "sha256": sha256_file(result_path),
    }
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motif-archive-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    result = extract(
        motif_archive_root=args.motif_archive_root,
        models_root=args.models_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
