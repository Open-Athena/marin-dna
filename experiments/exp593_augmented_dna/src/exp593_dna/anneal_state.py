"""Preserve native optimizer/global counters with independent annealing clocks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any
from uuid import uuid4

import fsspec
from exp586_moe.state import restore_dna_checkpoint, validate_restored_training_state
from levanter.checkpoint import (
    discover_checkpoint_candidates,
    load_checkpoint,
    save_checkpoint,
)
from rigging.filesystem.storage_path import StoragePath

from exp593_dna.anneal import OWNER, SOURCE_BINDING_SHA256, AnnealTrial
from exp593_dna.anneal_tokenizer import verify_source_tokenizer
from exp593_dna.pipeline import Encoder
from exp593_dna.recovery import canonical_json
from exp593_dna.schedule import TokenClock
from exp593_dna.state import checkpoint_template


@dataclass(frozen=True)
class AnnealClock:
    stage: TokenClock = field(default_factory=TokenClock)
    mixture: dict[str, int] = field(default_factory=dict)

    def advance(self, counts: dict[str, int], mixture: dict[str, int]) -> AnnealClock:
        total = dict(self.mixture)
        for key, value in mixture.items():
            if type(value) is not int or value < 0:
                raise ValueError("Invalid augmentation counter")
            total[key] = total.get(key, 0) + value
        return AnnealClock(self.stage.advance(**counts), total)


def approved_source_discrepancy(
    trial: AnnealTrial,
    clock: AnnealClock,
    *,
    other_pairs_match: bool,
    count: int,
    maximum: float,
) -> None:
    """Accept only the exact, user-approved legacy head discrepancy at stage zero."""
    from exp593_dna.anneal import TRIALS

    if (
        trial != TRIALS["545b"]
        or clock.stage.updates != 0
        or not other_pairs_match
        or count != 18_468_864
        or maximum != 0.00146484375
    ):
        raise ValueError("Unexpected native model/master discrepancy")


def imported_state_integrity(
    state: Any, trial: AnnealTrial, clock: AnnealClock
) -> dict[str, Any]:
    """Inspect without changing a single imported model, master, or optimizer leaf."""
    import jax
    import jax.numpy as jnp
    from exp586_moe.perf5_optimizer import model_matches_fp32_master
    from exp586_moe.state import state_all_finite

    if not bool(state_all_finite(state)):
        raise ValueError("Imported native training state contains nonfinite values")
    if bool(model_matches_fp32_master(state.params, state.master_params)):
        return {"all_finite": True, "compute_matches_master_cast": True}
    if trial.key != "545b" or clock.stage.updates:
        raise ValueError("Unexpected native model/master discrepancy")
    head = state.params.output_proj
    rounded = jax.sharding.reshard(
        state.master_params.output_proj.astype(head.dtype), head.sharding
    )
    # This replacement is solely a comparison tree. The actual state is returned
    # unchanged to the initial evaluator and first optimizer update.
    comparison = replace(state.params, output_proj=rounded)
    other_pairs_match = bool(model_matches_fp32_master(comparison, state.master_params))
    count = int(jnp.count_nonzero(head != rounded))
    maximum = float(
        jnp.max(jnp.abs(head.astype(jnp.float32) - rounded.astype(jnp.float32)))
    )
    approved_source_discrepancy(
        trial, clock, other_pairs_match=other_pairs_match, count=count, maximum=maximum
    )
    return {
        "all_finite": True,
        "compute_matches_master_cast": False,
        "native_state_preserved_verbatim": True,
        "legacy_output_head_mismatches": count,
        "legacy_output_head_max_abs_cast_difference": maximum,
        "approval": "user_preserve_exact_saved_state_2026-11-10",
    }


def validate_metadata(
    metadata: dict[str, Any], trial: AnnealTrial, bindings: dict[str, Any]
) -> AnnealClock:
    if (
        metadata.get("exp593_anneal_schema") != 1
        or metadata.get("exp593_bindings") != bindings
    ):
        raise ValueError("Annealing checkpoint scientific/source binding differs")
    record = metadata["exp593_anneal_clock"]
    clock = AnnealClock(TokenClock(**record["stage"]), record["mixture"])
    if (
        metadata["step"] != trial.source_update + clock.stage.updates
        or clock.stage.examples != 64 * clock.stage.updates
        or clock.stage.input_tokens > trial.horizon
    ):
        raise ValueError("Source/global/stage checkpoint clocks differ")
    if clock.mixture.get("annotation_guided", 0) != clock.stage.fim_examples:
        raise ValueError("An annealing FIM example was not annotation-guided")
    return clock


def restore_source(
    exemplar: Any, trial: AnnealTrial, mesh: Any, encoder: Encoder
) -> tuple[Any, AnnealClock, dict[str, Any]]:
    """Restore every native leaf; preserve Adam/Muon/global history verbatim."""
    tokenizer = verify_source_tokenizer(encoder)
    state, source_clock = restore_dna_checkpoint(
        exemplar,
        trial.source_checkpoint,
        mesh,
        metadata_sha256=trial.source_metadata_sha256,
        data_seed=0,
        scientific_config_sha256=SOURCE_BINDING_SHA256,
        tokenizer_sha256=tokenizer["source_tokenizer_sha256"],
    )
    evidence = validate_restored_training_state(state, source_clock)
    if (
        source_clock.updates != trial.source_update
        or source_clock.input_tokens != trial.source_tokens
        or source_clock.examples != trial.source_examples
    ):
        raise ValueError("The selected old 1T checkpoint clock differs")
    return (
        state,
        AnnealClock(),
        {**evidence, "source_clock": asdict(source_clock), "tokenizer": tokenizer},
    )


def discover(root: str, trial: AnnealTrial, bindings: dict[str, Any]) -> str | None:
    if not root.startswith(OWNER + "/annealing/"):
        raise ValueError("Annealing writes require their new experiment namespace")
    path = StoragePath(root) / "training.json"
    serialized = canonical_json(bindings)
    if path.exists():
        if path.read_text() != serialized:
            raise ValueError("Trial scientific configuration changed")
    else:
        path.write_text(serialized)
        if path.read_text() != serialized:
            raise ValueError("Trial binding readback failed")
    candidates = discover_checkpoint_candidates(root)
    for candidate in candidates:
        validate_metadata(candidate.metadata, trial, bindings)
    return candidates[-1].path if candidates else None


def restore(
    exemplar: Any, path: str, trial: AnnealTrial, mesh: Any, bindings: dict[str, Any]
) -> tuple[Any, AnnealClock]:
    metadata = json.loads((StoragePath(path) / "metadata.json").read_bytes())
    clock = validate_metadata(metadata, trial, bindings)
    state = load_checkpoint(
        checkpoint_template(exemplar), path, mesh=mesh, allow_partial=False
    )
    if (
        int(state.step) != trial.source_update + clock.stage.updates
        or int(state.opt_state.count) != int(state.step)
        or state.master_params is None
        or state.ema_params is not None
    ):
        raise ValueError("Restored full-state/global optimizer clock differs")
    return state, clock


def destination(root: str, trial: AnnealTrial, clock: AnnealClock) -> str:
    path = f"{root}/step-{trial.source_update + clock.stage.updates}"
    fs, key = fsspec.core.url_to_fs(path)
    fs.invalidate_cache(key)
    if fs.exists(key + "/metadata.json"):
        raise ValueError("Refusing to overwrite a committed annealing checkpoint")
    return path + "-retry-" + uuid4().hex if fs.exists(key) else path


def save(
    state: Any,
    path: str,
    trial: AnnealTrial,
    clock: AnnealClock,
    bindings: dict[str, Any],
    *,
    permanent: bool,
) -> None:
    step = trial.source_update + clock.stage.updates
    if (
        int(state.step) != step
        or int(state.opt_state.count) != step
        or state.master_params is None
        or state.ema_params is not None
    ):
        raise ValueError("Cannot checkpoint incomplete full-state/global counters")
    metadata = {
        "step": step,
        "exp593_anneal_schema": 1,
        "exp593_anneal_clock": asdict(clock),
        "exp593_bindings": bindings,
    }
    validate_metadata(metadata, trial, bindings)
    if (StoragePath(path) / "metadata.json").exists():
        raise ValueError("Refusing to overwrite a committed checkpoint")
    save_checkpoint(
        state,
        step,
        path,
        is_temporary=not permanent,
        metadata={k: v for k, v in metadata.items() if k != "step"},
    )


def commit(
    root: str,
    path: str,
    trial: AnnealTrial,
    clock: AnnealClock,
    bindings: dict[str, Any],
    *,
    keep: int = 6,
) -> str:
    if keep < 1 or not path.startswith(root + "/step-"):
        raise ValueError("Unsafe annealing retention request")
    raw = (StoragePath(path) / "metadata.json").read_bytes()
    if validate_metadata(json.loads(raw), trial, bindings) != clock:
        raise ValueError("Committed annealing clocks differ")
    digest = hashlib.sha256(raw).hexdigest()
    text = canonical_json(
        {"checkpoint": path, "metadata_sha256": digest, "clock": asdict(clock)}
    )
    pointer = StoragePath(root) / "latest.json"
    pointer.write_text(text)
    if pointer.read_text() != text:
        raise ValueError("Annealing recovery pointer readback failed")
    temporary = [
        c
        for c in discover_checkpoint_candidates(root)
        if c.metadata.get("is_temporary")
    ]
    for candidate in temporary[:-keep]:
        validate_metadata(candidate.metadata, trial, bindings)
        if (
            not candidate.path.startswith(root + "/step-")
            or candidate.step >= trial.source_update + clock.stage.updates
        ):
            raise ValueError("Unsafe temporary checkpoint retention candidate")
        fs, key = fsspec.core.url_to_fs(candidate.path)
        fs.rm(key, recursive=True)
    return digest
