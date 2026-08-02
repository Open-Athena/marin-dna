"""Extract feature-1662 responses to the frozen saturation panel."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
from huggingface_hub import snapshot_download
from marin_dna.data.dna import reverse_complement
from marin_dna.data.genome import Genome
from marin_dna.model.sae import M51_HIDDEN_SIZE, load_frozen_m51
from sae_lens.load_model import HookedProxyLM
from sae_lens.saes.sae import SAE

from common import (
    ISSUE,
    MODEL_ID,
    MODEL_REVISION,
    assert_commit,
    sha256_file,
    write_json,
)
from extract_focal import arm_label, model_path, read_model_provenance
from prepare_feature1662_saturation import validate_design
from saturation_common import (
    BLOCK_INDEX,
    FEATURE_ID,
    FOCAL_INDEX,
    ORIENTATIONS,
    POSITIONS,
    RUN_ID,
    WINDOW_BP,
)

from saturation_states import (
    MUTATIONS_PER_CONTEXT,
    STATES_PER_CONTEXT,
    build_response_table,
    enumerate_context_states,
)

HOOK_NAME = f"model.layers.{BLOCK_INDEX}"

assert BLOCK_INDEX == 18
assert M51_HIDDEN_SIZE == 1_920


def build_state_table(
    design: pl.DataFrame, *, genome: Genome
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Materialize exact reference contexts and their frozen saturation states."""

    assert design["context_index"].to_list() == list(range(design.height))
    state_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    for row in design.iter_rows(named=True):
        position0 = int(row["pos"]) - 1
        start0 = position0 - FOCAL_INDEX
        end0 = position0 + FOCAL_INDEX + 1
        assert start0 >= 0 and end0 - start0 == WINDOW_BP
        sequence = genome(str(row["chrom"]), start0, end0, "+").upper()
        assert len(sequence) == WINDOW_BP
        baseline_state_index = len(state_rows)
        states = enumerate_context_states(row, sequence)
        for state in states:
            state["state_index"] = len(state_rows)
            state["baseline_state_index"] = baseline_state_index
            state_rows.append(state)
        context_rows.append(
            {
                **row,
                "window_start0": start0,
                "window_end0": end0,
                "reference_sequence": sequence,
                "baseline_state_index": baseline_state_index,
            }
        )
    contexts = pl.DataFrame(context_rows)
    states = pl.DataFrame(state_rows).select(
        "state_index",
        "baseline_state_index",
        "context_index",
        "is_baseline",
        "genomic_offset",
        "transcript_offset",
        "genomic_ref",
        "genomic_alt",
        "edited_codon_position",
        "alternate_codon",
        "alternate_amino_acid",
        "consequence",
        "sequence",
    )
    assert contexts.height == design.height
    assert states.height == contexts.height * STATES_PER_CONTEXT
    assert states["state_index"].to_list() == list(range(states.height))
    counts = states.group_by("context_index").len()["len"].unique().to_list()
    assert counts == [STATES_PER_CONTEXT]
    assert states.filter(pl.col("is_baseline")).height == contexts.height
    assert states.filter(~pl.col("is_baseline")).height == (
        contexts.height * MUTATIONS_PER_CONTEXT
    )
    return contexts, states


def encode_selected_feature(sae: SAE, raw: torch.Tensor) -> torch.Tensor:
    """Encode feature 1662 with the exact exported JumpReLU formula."""

    assert not sae.training and raw.ndim == 2 and raw.shape[-1] == sae.cfg.d_in
    assert sae.cfg.architecture() == "jumprelu"
    assert sae.cfg.normalize_activations == "none"
    assert not sae.hook_z_reshaping_mode
    with torch.inference_mode():
        sae_in = sae.process_sae_in(raw)
        hidden_pre = sae_in @ sae.W_enc[:, FEATURE_ID] + sae.b_enc[FEATURE_ID]
        base_acts = sae.activation_fn(hidden_pre)
        selected = base_acts * (hidden_pre > sae.threshold[FEATURE_ID]).to(
            base_acts.dtype
        )
    assert selected.shape == raw.shape[:-1]
    assert torch.isfinite(selected).all() and torch.all(selected >= 0)
    return selected


@torch.inference_mode()
def extract_feature_batch(
    sequences: Sequence[str],
    *,
    tokenizer: Any,
    model: HookedProxyLM,
    sae: SAE,
    validate_formula: bool,
) -> np.ndarray:
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
        names_filter=[HOOK_NAME],
        stop_at_layer=BLOCK_INDEX + 1,
    )
    assert output is None and set(cache) == {HOOK_NAME}
    captured = cache[HOOK_NAME]
    assert captured.shape == (len(sequences), WINDOW_BP + 1, M51_HIDDEN_SIZE)
    raw = captured[:, FOCAL_INDEX + 1, :].float()
    assert torch.isfinite(raw).all()
    selected = encode_selected_feature(sae, raw)
    if validate_formula:
        expected = sae.encode(raw[: min(4, len(sequences))])[:, FEATURE_ID]
        torch.testing.assert_close(
            selected[: expected.shape[0]], expected, rtol=1e-6, atol=1e-5
        )
    return selected.cpu().numpy().astype(np.float32, copy=False)


def extract_orientation(
    states: pl.DataFrame,
    *,
    orientation: Literal["forward", "reverse_complement"],
    tokenizer: Any,
    model: HookedProxyLM,
    sae: SAE,
    batch_size: int,
) -> np.ndarray:
    assert orientation in ORIENTATIONS and batch_size > 0
    activations = np.empty(states.height, dtype=np.float32)
    for offset in range(0, states.height, batch_size):
        stop = min(offset + batch_size, states.height)
        sequences = states["sequence"].slice(offset, stop - offset).to_list()
        if orientation == "reverse_complement":
            sequences = [reverse_complement(sequence) for sequence in sequences]
        activations[offset:stop] = extract_feature_batch(
            sequences,
            tokenizer=tokenizer,
            model=model,
            sae=sae,
            validate_formula=offset == 0,
        )
        if offset == 0 or stop == states.height or stop % (batch_size * 25) == 0:
            print(
                json.dumps(
                    {
                        "stage": "saturation_extraction",
                        "orientation": orientation,
                        "processed": stop,
                        "total": states.height,
                    }
                ),
                flush=True,
            )
    assert np.isfinite(activations).all() and np.all(activations >= 0)
    return activations


def extract(
    *,
    design_dir: Path,
    fasta_path: Path,
    models_root: Path,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert batch_size > 0 and not output_dir.exists()
    assert fasta_path.is_file() and Path(f"{fasta_path}.fai").is_file()
    assert Path(f"{fasta_path}.gzi").is_file()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    assert os.environ.get("RUN_ID") == RUN_ID
    design_manifest, design = validate_design(design_dir)
    started = time.monotonic()

    label = arm_label(BLOCK_INDEX)
    sae_path = model_path(block_index=BLOCK_INDEX, models_root=models_root)
    sae_provenance = read_model_provenance(sae_path, block_index=BLOCK_INDEX)
    sae = SAE.load_from_disk(sae_path, device="cuda", dtype="float32")
    sae.requires_grad_(False)
    sae.eval()
    assert all(not parameter.requires_grad for parameter in sae.parameters())

    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    frozen.model.config.use_cache = False
    model = HookedProxyLM(frozen.model, frozen.tokenizer, hook_names=[HOOK_NAME])
    chroms = set(design["chrom"].cast(pl.String).unique().to_list())
    genome = Genome(fasta_path, subset_chroms=chroms)
    assert set(genome.chroms) == chroms

    output_dir.mkdir(parents=True)
    contexts, states = build_state_table(design, genome=genome)
    context_path = output_dir / "contexts.parquet"
    state_path = output_dir / "sequence_states.parquet"
    contexts.write_parquet(context_path, compression="zstd")
    states.write_parquet(state_path, compression="zstd")

    torch.cuda.reset_peak_memory_stats()
    activations = {
        orientation: extract_orientation(
            states,
            orientation=orientation,
            tokenizer=frozen.tokenizer,
            model=model,
            sae=sae,
            batch_size=batch_size,
        )
        for orientation in ORIENTATIONS
    }
    responses = build_response_table(states, activations)
    response_path = output_dir / "feature1662_responses.parquet"
    responses.write_parquet(response_path, compression="zstd")

    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (context_path, state_path, response_path)
    }
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": RUN_ID,
        "analysis_status": "post_hoc_mechanistic_perturbation",
        "experiment_commit": experiment_commit,
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16"},
        "sae": {
            "arm": label,
            "reported_layer": BLOCK_INDEX + 1,
            "implementation_block_index": BLOCK_INDEX,
            "feature_id": FEATURE_ID,
            "dtype": "float32",
            "provenance": sae_provenance,
        },
        "design": {
            "run_id": design_manifest["run_id"],
            "manifest_sha256": sha256_file(design_dir / "manifest.json"),
            "experiment_commit": design_manifest["experiment_commit"],
            "contexts": contexts.height,
        },
        "protocol": {
            "window_bp": WINDOW_BP,
            "focal_index": FOCAL_INDEX,
            "saturation_radius": max(POSITIONS),
            "mutations_per_context": MUTATIONS_PER_CONTEXT,
            "orientations": list(ORIENTATIONS),
            "batch_size_sequences": batch_size,
            "torch_compile": False,
            "torch_compile_reason": (
                "the pinned dynamic hook-cache path is validated in eager mode"
            ),
            "feature_encoding": "exact selected-feature JumpReLU formula",
        },
        "outputs": {
            "contexts": contexts.height,
            "sequence_states": states.height,
            "mutation_responses": responses.height,
            "nonzero_responses": int((responses["abs_delta"] > 0).sum()),
        },
        "artifacts": artifacts,
    }
    write_json(output_dir / "results.json", result)
    result["artifacts"]["results.json"] = {
        "bytes": (output_dir / "results.json").stat().st_size,
        "sha256": sha256_file(output_dir / "results.json"),
    }
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    result = extract(
        design_dir=args.design_dir,
        fasta_path=args.fasta,
        models_root=args.models_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
