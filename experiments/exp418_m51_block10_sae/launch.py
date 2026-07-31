"""Train and evaluate the preregistered m5.1 block-10 SAE for issue 418."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import fsspec
import torch
from huggingface_hub import snapshot_download
from marin_dna.model.sae import (
    M51_HIDDEN_SIZE,
    M51_NUM_BLOCKS,
    M51PostBlockCapture,
    load_frozen_m51,
)
from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
from sae_lens.evals import EvalConfig, run_evals
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.load_model import HookedProxyLM
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAEConfig
from sae_lens.saes.sae import SAE, SAEMetadata
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.activations_store import ActivationsStore

from data import (
    CONTEXT_TOKENS,
    N_SOURCES,
    SOURCES,
    WINDOW_BP,
    BalancedBudget,
    balanced_budget,
    build_five_way_dataset,
    load_pinned_tokenizer,
    provenance_manifest,
)

ISSUE = 418
SEED = 288
MODEL_ID = "marin-dna/marin-dna-exp135-m5.1"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
MODEL_STEP = 59_158
SAELENS_REVISION = "8be14080485952f729ed58d674bcddf9778e0aa4"
MARIN_DNA_REVISION = "c4c1c86bbfc0bd58ff76dda3bac1d2acea856a33"
HUMAN_FASTA_URI = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)
HOOK_NAME = "model.layers.9"
BLOCK_INDEX = 9
D_SAE = 15_360
K = 64
TRAIN_BATCH_TOKENS = N_SOURCES * WINDOW_BP
BUFFER_CONTEXT_BATCHES = N_SOURCES
HELDOUT_GAP_PER_SOURCE = 1_000

assert BLOCK_INDEX < M51_NUM_BLOCKS
assert M51_HIDDEN_SIZE == 1_920
assert TRAIN_BATCH_TOKENS == 1_275


@dataclass(frozen=True)
class Tier:
    name: Literal["wiring", "micro"]
    budget: BalancedBudget
    eval_batches: int

    @property
    def optimizer_steps(self) -> int:
        assert self.budget.actual_activations % TRAIN_BATCH_TOKENS == 0
        return self.budget.actual_activations // TRAIN_BATCH_TOKENS


def tier_config(name: Literal["wiring", "micro"]) -> Tier:
    if name == "wiring":
        tier = Tier(name, balanced_budget(1_000_000), eval_batches=4)
    elif name == "micro":
        tier = Tier(name, balanced_budget(5_000_000), eval_batches=8)
    else:
        raise AssertionError(name)
    assert tier.optimizer_steps == tier.budget.windows_per_source
    return tier


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(_json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_finite_parameters(module: torch.nn.Module, label: str) -> None:
    for name, parameter in module.named_parameters():
        assert torch.isfinite(parameter).all(), f"{label}.{name} is not finite"


def _assert_finite_tree(value: Any, path: str = "metrics") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, (float, int)):
        assert torch.isfinite(torch.tensor(float(value))), f"{path} is not finite"


def _default_run_id(tier: Tier) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"dna-exp{ISSUE}-{tier.name}-seed{SEED}-{timestamp}"


def _runner_config(
    *,
    tier: Tier,
    output_path: Path | None,
    checkpoint_path: Path,
    run_name: str,
    bos_token_id: int,
    log_to_wandb: bool,
    n_checkpoints: int,
    resume_from_checkpoint: Path | None = None,
) -> LanguageModelSAERunnerConfig[BatchTopKTrainingSAEConfig]:
    metadata = SAEMetadata(
        issue=ISSUE,
        model_name=MODEL_ID,
        model_revision=MODEL_REVISION,
        model_step=MODEL_STEP,
        hook_name=HOOK_NAME,
        block_index=BLOCK_INDEX,
        report_block=BLOCK_INDEX + 1,
        dataset_path="exp418-pinned-five-way-stream",
        seed=SEED,
    )
    sae = BatchTopKTrainingSAEConfig(
        d_in=M51_HIDDEN_SIZE,
        d_sae=D_SAE,
        dtype="float32",
        device="cuda",
        normalize_activations="none",
        metadata=metadata,
        k=float(K),
        aux_loss_coefficient=1.0,
        rescale_acts_by_decoder_norm=True,
        topk_threshold_lr=0.01,
        decoder_init_norm=0.1,
    )
    logger = LoggingConfig(
        log_to_wandb=log_to_wandb,
        log_activations_store_to_wandb=False,
        log_optimizer_state_to_wandb=False,
        log_weights_to_wandb=True,
        wandb_project="marin",
        run_name=run_name,
        wandb_log_frequency=10,
        # Post-training evaluation uses fresh held-out stores. Avoid advancing the
        # active training iterator through runner-internal evaluations.
        eval_every_n_wandb_logs=100_000,
    )
    cfg = LanguageModelSAERunnerConfig(
        sae=sae,
        model_name=MODEL_ID,
        model_class_name="HookedProxyLM",
        hook_name=HOOK_NAME,
        dataset_path="exp418-pinned-five-way-stream",
        streaming=True,
        is_dataset_tokenized=True,
        context_size=CONTEXT_TOKENS,
        n_batches_in_buffer=BUFFER_CONTEXT_BATCHES,
        training_tokens=tier.budget.actual_activations,
        store_batch_size_prompts=N_SOURCES,
        train_batch_size_tokens=TRAIN_BATCH_TOKENS,
        disable_concat_sequences=True,
        sequence_separator_token="bos",
        activations_mixing_fraction=0.0,
        device="cuda",
        llm_device="cuda",
        act_store_device="cuda",
        seed=SEED,
        dtype="float32",
        prepend_bos=True,
        autocast=False,
        autocast_lm=False,
        compile_llm=False,
        compile_sae=False,
        lr=3e-4,
        lr_scheduler_name="constant",
        dead_feature_window=1_000,
        feature_sampling_window=2_000,
        n_eval_batches=1,
        logger=logger,
        n_checkpoints=n_checkpoints,
        checkpoint_path=str(checkpoint_path),
        save_final_checkpoint=False,
        output_path=None if output_path is None else str(output_path),
        resume_from_checkpoint=(
            None if resume_from_checkpoint is None else str(resume_from_checkpoint)
        ),
        model_from_pretrained_kwargs={"revision": MODEL_REVISION},
        exclude_special_tokens=[bos_token_id],
    )
    assert cfg.training_tokens % cfg.train_batch_size_tokens == 0
    assert cfg.activations_mixing_fraction == 0.0
    return cfg


@torch.inference_mode()
def _validate_real_hook(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: Any,
) -> dict[str, Any]:
    rows = list(dataset.take(N_SOURCES))
    assert [row["source"] for row in rows] == [source.name for source in SOURCES]
    input_ids = torch.tensor(
        [row["input_ids"] for row in rows], dtype=torch.long, device="cuda"
    )
    attention_mask = torch.ones_like(input_ids)
    assert input_ids.shape == (N_SOURCES, CONTEXT_TOKENS)
    assert torch.all(input_ids[:, 0] == tokenizer.bos_token_id)
    assert not torch.eq(input_ids[:, 1:], tokenizer.bos_token_id).any()
    assert not torch.eq(input_ids, tokenizer.pad_token_id).any()

    baseline = model(input_ids=input_ids, attention_mask=attention_mask).logits
    with M51PostBlockCapture(model, BLOCK_INDEX) as capture:
        observed = model(input_ids=input_ids, attention_mask=attention_mask).logits
        activations = capture.pop()
    assert torch.equal(baseline, observed), "read-only hook changed m5.1 logits"
    assert activations.shape == (N_SOURCES, CONTEXT_TOKENS, M51_HIDDEN_SIZE)
    assert torch.isfinite(activations).all()
    nucleotide_activations = activations[:, 1:, :]
    assert nucleotide_activations.shape == (N_SOURCES, WINDOW_BP, M51_HIDDEN_SIZE)
    return {
        "source_order": [row["source"] for row in rows],
        "record_ids": {row["source"]: row["record_id"] for row in rows},
        "unknown_tokens": {row["source"]: row["unknown_tokens"] for row in rows},
        "raw_hook_shape": list(activations.shape),
        "bos_filtered_shape": list(nucleotide_activations.shape),
        "logits_bitwise_identical": True,
        "activations_finite": True,
    }


def _activation_store(
    *,
    model: HookedProxyLM,
    dataset: Any,
    bos_token_id: int,
    total_tokens: int,
) -> ActivationsStore:
    return ActivationsStore(
        model=model,
        dataset=dataset,
        streaming=True,
        hook_name=HOOK_NAME,
        hook_head_index=None,
        context_size=CONTEXT_TOKENS,
        d_in=M51_HIDDEN_SIZE,
        n_batches_in_buffer=BUFFER_CONTEXT_BATCHES,
        total_training_tokens=total_tokens,
        store_batch_size_prompts=N_SOURCES,
        train_batch_size_tokens=TRAIN_BATCH_TOKENS,
        prepend_bos=True,
        normalize_activations="none",
        device=torch.device("cuda"),
        dtype="float32",
        cached_activations_path=None,
        model_kwargs=None,
        autocast_lm=False,
        dataset_trust_remote_code=False,
        seqpos_slice=(None,),
        exclude_special_tokens=torch.tensor(
            [bos_token_id], dtype=torch.long, device="cuda"
        ),
        disable_concat_sequences=True,
        sequence_separator_token="bos",
        activations_mixing_fraction=0.0,
        use_chat_formatting=False,
    )


def _feature_usage_summary(feature_density: list[float]) -> dict[str, Any]:
    density = torch.tensor(feature_density, dtype=torch.float64)
    assert density.shape == (D_SAE,)
    assert torch.isfinite(density).all()
    assert torch.all(density >= 0)
    total = density.sum().item()
    top_count = max(1, round(0.01 * D_SAE))
    top_share = (
        density.topk(top_count).values.sum().item() / total if total > 0 else 0.0
    )
    return {
        "heldout_inactive_count": int(torch.eq(density, 0).sum().item()),
        "heldout_inactive_fraction": float(torch.eq(density, 0).double().mean()),
        "top_1pct_activation_share": top_share,
        "density_quantiles": {
            str(q): torch.quantile(density, q).item()
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
        },
    }


@torch.inference_mode()
def _evaluate_export(
    *,
    training_sae: torch.nn.Module,
    export_path: Path,
    model: HookedProxyLM,
    tokenizer: Any,
    tier: Tier,
) -> dict[str, Any]:
    exported = SAE.load_from_disk(export_path, device="cuda", dtype="float32")
    assert exported.cfg.architecture() == "jumprelu"
    _assert_finite_parameters(training_sae, "training_sae")
    _assert_finite_parameters(exported, "exported_jumprelu")

    heldout_skip = tier.budget.windows_per_source + HELDOUT_GAP_PER_SOURCE
    reconstruction_store = _activation_store(
        model=model,
        dataset=build_five_way_dataset(tokenizer, skip_per_source=heldout_skip),
        bos_token_id=tokenizer.bos_token_id,
        total_tokens=tier.eval_batches * TRAIN_BATCH_TOKENS,
    )
    reconstruction_metrics, _ = run_evals(
        sae=exported,
        activation_store=reconstruction_store,
        model=model,
        activation_scaler=ActivationScaler(),
        eval_config=EvalConfig(
            batch_size_prompts=N_SOURCES,
            n_eval_reconstruction_batches=tier.eval_batches,
            compute_kl=True,
            compute_ce_loss=True,
        ),
        exclude_special_tokens=[tokenizer.bos_token_id],
        verbose=True,
    )

    feature_store = _activation_store(
        model=model,
        dataset=build_five_way_dataset(tokenizer, skip_per_source=heldout_skip),
        bos_token_id=tokenizer.bos_token_id,
        total_tokens=tier.eval_batches * TRAIN_BATCH_TOKENS,
    )
    scalar_metrics, feature_metrics = run_evals(
        sae=exported,
        activation_store=feature_store,
        model=model,
        activation_scaler=ActivationScaler(),
        eval_config=EvalConfig(
            batch_size_prompts=N_SOURCES,
            n_eval_sparsity_variance_batches=tier.eval_batches,
            compute_l2_norms=True,
            compute_sparsity_metrics=True,
            compute_variance_metrics=True,
            compute_featurewise_density_statistics=True,
        ),
        exclude_special_tokens=[tokenizer.bos_token_id],
        verbose=True,
    )
    assert "feature_density" in feature_metrics
    # SAELens also returns a 0/0 per-feature heuristic for features that never
    # fire on this small held-out panel.  Preserve the finite density vector;
    # report inactivity explicitly rather than serializing meaningless NaNs.
    feature_metrics = {"feature_density": feature_metrics["feature_density"]}

    boundary_store = _activation_store(
        model=model,
        dataset=build_five_way_dataset(tokenizer, skip_per_source=heldout_skip),
        bos_token_id=tokenizer.bos_token_id,
        total_tokens=TRAIN_BATCH_TOKENS,
    )
    activation_batch = boundary_store.get_filtered_llm_batch()
    assert activation_batch.shape == (TRAIN_BATCH_TOKENS, M51_HIDDEN_SIZE)
    batchtopk_acts = training_sae.encode(activation_batch)
    jumprelu_acts = exported.encode(activation_batch)
    batchtopk_support = batchtopk_acts > 0
    jumprelu_support = jumprelu_acts > 0
    batchtopk_l0 = batchtopk_support.sum(dim=-1).float().mean().item()
    jumprelu_l0 = jumprelu_support.sum(dim=-1).float().mean().item()

    metrics = {
        **reconstruction_metrics,
        **scalar_metrics,
        "threshold_boundary": {
            "batchtopk_heldout_l0": batchtopk_l0,
            "jumprelu_heldout_l0": jumprelu_l0,
            "jumprelu_minus_batchtopk_l0": jumprelu_l0 - batchtopk_l0,
            "support_mismatch_fraction": (
                torch.logical_xor(batchtopk_support, jumprelu_support)
                .float()
                .mean()
                .item()
            ),
        },
        "training_metadata": {
            "l0": training_sae.cfg.metadata.l0,
            "num_dead_features": training_sae.cfg.metadata.num_dead_features,
        },
        "feature_usage": _feature_usage_summary(feature_metrics["feature_density"]),
        "heldout": {
            "skip_per_source": heldout_skip,
            "gap_after_training_prefix_per_source": HELDOUT_GAP_PER_SOURCE,
            "eval_batches": tier.eval_batches,
            "eval_nucleotide_activations_per_metric_family": (
                tier.eval_batches * TRAIN_BATCH_TOKENS
            ),
        },
    }
    ce = metrics["model_performance_preservation"]
    metrics["model_performance_preservation"]["ce_loss_degradation"] = (
        ce["ce_loss_with_sae"] - ce["ce_loss_without_sae"]
    )
    _assert_finite_tree(metrics)
    return {"metrics": metrics, "feature_metrics": feature_metrics}


def _checkpoint_dirs(checkpoint_path: Path) -> list[Path]:
    result = [path for path in checkpoint_path.iterdir() if path.name.isdigit()]
    return sorted(result, key=lambda path: int(path.name))


def _read_json_uri(uri: str) -> dict[str, Any]:
    with fsspec.open(uri, "rt") as stream:
        value = json.load(stream)
    assert isinstance(value, dict)
    return value


def _validate_wiring_gate(uri: str) -> dict[str, Any]:
    manifest = _read_json_uri(uri)
    assert manifest["tier"]["name"] == "wiring"
    assert manifest["engineering_gate_passed"] is True
    assert manifest["fixed_config"]["model_revision"] == MODEL_REVISION
    assert manifest["fixed_config"]["saelens_revision"] == SAELENS_REVISION
    assert manifest["fixed_config"]["seed"] == SEED
    return {
        "manifest_uri": uri,
        "artifact_uri": manifest.get("artifact_uri"),
        "run_id": manifest["run_id"],
    }


def _upload_directory(local_dir: Path, destination_uri: str) -> None:
    fs, root = fsspec.core.url_to_fs(destination_uri)
    assert not fs.exists(root), f"refusing to overwrite artifact: {destination_uri}"
    for local_path in sorted(path for path in local_dir.rglob("*") if path.is_file()):
        relative = local_path.relative_to(local_dir).as_posix()
        remote_path = f"{root.rstrip('/')}/{relative}"
        parent = remote_path.rsplit("/", maxsplit=1)[0]
        fs.makedirs(parent, exist_ok=True)
        fs.put_file(str(local_path), remote_path)


def _results_markdown(manifest: dict[str, Any]) -> str:
    metrics = manifest["evaluation"]["metrics"]
    recon = metrics["reconstruction_quality"]
    performance = metrics["model_performance_preservation"]
    boundary = metrics["threshold_boundary"]
    usage = metrics["feature_usage"]
    return f"""# exp418 {manifest["tier"]["name"]} result

| Quantity | Value |
|---|---:|
| Nucleotide activations trained | {manifest["tier"]["budget"]["actual_activations"]:,} |
| Optimizer steps | {manifest["tier"]["optimizer_steps"]:,} |
| Explained variance | {recon["explained_variance"]:.6g} |
| Reconstruction MSE | {recon["mse"]:.6g} |
| CE loss degradation | {performance["ce_loss_degradation"]:.6g} |
| BatchTopK held-out L0 | {boundary["batchtopk_heldout_l0"]:.6g} |
| JumpReLU held-out L0 | {boundary["jumprelu_heldout_l0"]:.6g} |
| Held-out inactive features | {usage["heldout_inactive_fraction"]:.3%} |
| Top 1% activation share | {usage["top_1pct_activation_share"]:.3%} |
| Wall time (hours) | {manifest["runtime"]["wall_seconds"] / 3600:.4f} |

Engineering gate passed: **{manifest["engineering_gate_passed"]}**.
This is an engineering/health gate, not evidence of biological interpretation.
"""


def _dry_run_manifest(tier: Tier, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "tier": {**asdict(tier), "optimizer_steps": tier.optimizer_steps},
        "fixed_config": {
            "issue": ISSUE,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_step": MODEL_STEP,
            "hook_name": HOOK_NAME,
            "block_index": BLOCK_INDEX,
            "d_in": M51_HIDDEN_SIZE,
            "d_sae": D_SAE,
            "k": K,
            "seed": SEED,
            "saelens_revision": SAELENS_REVISION,
            "marin_dna_revision": MARIN_DNA_REVISION,
            "train_batch_tokens": TRAIN_BATCH_TOKENS,
            "activations_mixing_fraction": 0.0,
        },
        "data": provenance_manifest(),
        "interpretation_boundary": {
            "training": "pinned equal five-way cross-species stream",
            "post_training": "coordinate-clean held-out human GRCh38 panel",
            "human_fasta_uri": HUMAN_FASTA_URI,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    tier = tier_config(args.tier)
    run_id = args.run_id or _default_run_id(tier)
    dry_manifest = _dry_run_manifest(tier, run_id)
    if args.dry_run:
        return dry_manifest

    assert torch.cuda.is_available(), "exp418 training requires a CUDA accelerator"
    assert torch.cuda.device_count() == 1, "exp418 is preregistered for exactly one GPU"
    if not args.wandb_disabled:
        assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY is required"
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert len(experiment_commit) == 40 and all(
        character in "0123456789abcdef" for character in experiment_commit
    ), "EXPERIMENT_COMMIT must be the full lowercase Git commit SHA"
    wiring_manifest_uri = args.wiring_manifest_uri or os.environ.get(
        "WIRING_MANIFEST_URI"
    )
    wiring_prerequisite = None
    if tier.name == "micro":
        assert wiring_manifest_uri, (
            "--wiring-manifest-uri or WIRING_MANIFEST_URI is required: "
            "the micro-run cannot bypass the wiring gate"
        )
        wiring_prerequisite = _validate_wiring_gate(wiring_manifest_uri)

    local_dir = Path(args.local_root) / run_id
    assert not local_dir.exists(), f"refusing to overwrite local run: {local_dir}"
    local_dir.mkdir(parents=True)
    export_path = local_dir / "sae"
    checkpoint_path = local_dir / "checkpoints"

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    started = time.monotonic()
    snapshot_path = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    assert MODEL_REVISION in snapshot_path.parts, snapshot_path
    frozen = load_frozen_m51(snapshot_path, device="cuda", dtype=torch.bfloat16)
    tokenizer = load_pinned_tokenizer()
    assert tokenizer.get_vocab() == frozen.tokenizer.get_vocab()
    assert tokenizer.special_tokens_map == frozen.tokenizer.special_tokens_map
    model = HookedProxyLM(frozen.model, tokenizer, hook_names=[HOOK_NAME])

    hook_validation = _validate_real_hook(
        frozen.model,
        tokenizer,
        build_five_way_dataset(tokenizer),
    )

    checkpoint_resume: dict[str, Any]
    if tier.name == "wiring":
        first_cfg = _runner_config(
            tier=tier,
            output_path=None,
            checkpoint_path=checkpoint_path,
            run_name=f"{run_id}-checkpoint-source",
            bos_token_id=tokenizer.bos_token_id,
            log_to_wandb=not args.wandb_disabled,
            n_checkpoints=2,
        )
        first_runner = LanguageModelSAETrainingRunner(
            first_cfg,
            override_dataset=build_five_way_dataset(tokenizer),
            override_model=model,
        )
        first_sae = first_runner.run()
        _assert_finite_parameters(first_sae, "first_pass_sae")
        checkpoints = _checkpoint_dirs(checkpoint_path)
        assert len(checkpoints) == 2, checkpoints
        resume_path = checkpoints[-1]
        resume_samples = int(resume_path.name)
        assert resume_samples % (2 * TRAIN_BATCH_TOKENS) == 0, (
            "resume checkpoint must be at an empty passthrough-buffer boundary"
        )
        del first_sae, first_runner
        gc.collect()
        torch.cuda.empty_cache()

        resume_cfg = _runner_config(
            tier=tier,
            output_path=export_path,
            checkpoint_path=checkpoint_path,
            run_name=f"{run_id}-resume-validation",
            bos_token_id=tokenizer.bos_token_id,
            log_to_wandb=not args.wandb_disabled,
            n_checkpoints=0,
            resume_from_checkpoint=resume_path,
        )
        resumed_runner = LanguageModelSAETrainingRunner(
            resume_cfg,
            override_dataset=build_five_way_dataset(tokenizer),
            override_model=model,
        )
        training_sae = resumed_runner.run()
        checkpoint_resume = {
            "tested": True,
            "checkpoint": str(resume_path.relative_to(local_dir)),
            "checkpoint_training_samples": resume_samples,
            "checkpoint_optimizer_steps": resume_samples // TRAIN_BATCH_TOKENS,
            "resumed_to_training_samples": tier.budget.actual_activations,
            "buffer_boundary_clean": True,
        }
    else:
        cfg = _runner_config(
            tier=tier,
            output_path=export_path,
            checkpoint_path=checkpoint_path,
            run_name=run_id,
            bos_token_id=tokenizer.bos_token_id,
            log_to_wandb=not args.wandb_disabled,
            n_checkpoints=2,
        )
        runner = LanguageModelSAETrainingRunner(
            cfg,
            override_dataset=build_five_way_dataset(tokenizer),
            override_model=model,
        )
        training_sae = runner.run()
        checkpoint_resume = {
            "tested": False,
            "reason": "tested by the required wiring prerequisite",
            "wiring_prerequisite": wiring_prerequisite,
        }

    _assert_finite_parameters(training_sae, "final_training_sae")
    evaluation = _evaluate_export(
        training_sae=training_sae,
        export_path=export_path,
        model=model,
        tokenizer=tokenizer,
        tier=tier,
    )
    elapsed = time.monotonic() - started
    weights_path = export_path / "sae_weights.safetensors"
    config_path = export_path / "cfg.json"
    assert weights_path.exists()
    assert config_path.exists()

    destination_uri = None
    if not args.no_upload:
        artifact_prefix = args.artifact_prefix or os.environ.get("ARTIFACT_PREFIX")
        assert artifact_prefix, "--artifact-prefix or ARTIFACT_PREFIX is required"
        destination_uri = f"{artifact_prefix.rstrip('/')}/experiments/exp418/{run_id}"

    manifest = {
        **dry_manifest,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_commit": experiment_commit,
        "artifact_uri": destination_uri,
        "hook_validation": hook_validation,
        "checkpoint_resume": checkpoint_resume,
        "evaluation": evaluation,
        "runtime": {
            "wall_seconds": elapsed,
            "nucleotide_activations_per_second": (
                tier.budget.actual_activations / elapsed
            ),
            "accelerator_hours": elapsed / 3600,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "sae_lens": importlib.metadata.version("sae-lens"),
            "compute_provider": os.environ.get("COMPUTE_PROVIDER", "unrecorded"),
            "compute_instance_type": os.environ.get(
                "COMPUTE_INSTANCE_TYPE", "unrecorded"
            ),
            "skypilot_cluster": os.environ.get(
                "SKYPILOT_CLUSTER_NAME", "unrecorded"
            ),
        },
        "artifact_hashes": {
            "sae_weights.safetensors": _sha256(weights_path),
            "cfg.json": _sha256(config_path),
        },
        "engineering_gate_passed": True,
        "biological_interpretation_performed": False,
    }
    _assert_finite_tree(manifest["evaluation"])
    _write_json(local_dir / "manifest.json", manifest)
    _write_json(local_dir / "metrics.json", evaluation)
    (local_dir / "RESULTS.md").write_text(_results_markdown(manifest))

    if destination_uri is not None:
        _upload_directory(local_dir, destination_uri)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("wiring", "micro"), required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb-disabled", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--artifact-prefix")
    parser.add_argument("--wiring-manifest-uri")
    parser.add_argument("--local-root", default="/tmp/dna-exp418")
    args = parser.parse_args()
    print(json.dumps(_json_safe(run(args)), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
