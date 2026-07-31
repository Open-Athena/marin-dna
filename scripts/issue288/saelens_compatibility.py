"""Local SAELens compatibility spike for issue #288.

This deliberately uses a tiny randomly initialized Qwen3 causal LM.  It tests
the interfaces needed by an eventual m5.1 SAE experiment without downloading
the 1.12B-parameter checkpoint or launching accelerator resources.

Run from the repository root with the commit-pinned SAELens dependency::

    uv run --with \
      'sae-lens @ git+https://github.com/decoderesearch/SAELens@8be14080485952f729ed58d674bcddf9778e0aa4' \
      python scripts/issue288/saelens_compatibility.py
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import Dataset
from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
from sae_lens.evals import get_recons_loss
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.load_model import HookedProxyLM
from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from sae_lens.saes.sae import SAE, SAEMetadata, TrainStepInput
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.activations_store import ActivationsStore
from transformers import Qwen3Config, Qwen3ForCausalLM

SAELENS_COMMIT = "8be14080485952f729ed58d674bcddf9778e0aa4"
SEQUENCE_TOKENS = 256
NUCLEOTIDE_TOKENS = 255
BOS_TOKEN_ID = 1
PAD_TOKEN_ID = 0
VOCAB_SIZE = 8
HIDDEN_SIZE = 32
DICT_SIZE = 128
K = 4
BATCH_SIZE = 2
HOOK_NAME = "model.layers.0"


@dataclass(frozen=True)
class SpikeResults:
    saelens_commit: str
    hook_name: str
    hook_shape: tuple[int, ...]
    hook_is_logit_invariant: bool
    bos_tokens_removed: int
    nucleotide_activations: int
    batchtopk_total_nonzero: int
    batchtopk_mean_l0: float
    reconstruction_metrics_are_finite: bool
    checkpoint_type: str
    checkpoint_roundtrip_matches: bool
    checkpoint_active_set_matches: bool
    checkpoint_max_abs_diff: float
    checkpoint_training_nonzero: int
    checkpoint_inference_nonzero: int
    inference_is_partition_invariant: bool
    native_sequence_columns_supported: bool
    activation_store_reset_works: bool
    runner_completed: bool
    runner_training_l0: float
    runner_output_type: str
    runner_checkpoint_saved: bool
    runner_resume_completed: bool


class TinyDnaTokenizer:
    """Tokenizer metadata needed by SAELens for pretokenized toy inputs."""

    bos_token_id = BOS_TOKEN_ID
    pad_token_id = PAD_TOKEN_ID
    eos_token_id = 2
    all_special_ids = [PAD_TOKEN_ID, BOS_TOKEN_ID, eos_token_id]


def make_input_ids() -> torch.Tensor:
    generator = torch.Generator().manual_seed(288)
    input_ids = torch.randint(
        3,
        VOCAB_SIZE,
        (BATCH_SIZE, SEQUENCE_TOKENS),
        generator=generator,
        dtype=torch.long,
    )
    input_ids[:, 0] = BOS_TOKEN_ID
    assert torch.all(input_ids[:, 0] == BOS_TOKEN_ID)
    assert not torch.any(input_ids[:, 1:] == BOS_TOKEN_ID)
    assert not torch.any(input_ids == PAD_TOKEN_ID)
    return input_ids


def make_model() -> Qwen3ForCausalLM:
    torch.manual_seed(288)
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=SEQUENCE_TOKENS,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=TinyDnaTokenizer.eos_token_id,
        pad_token_id=PAD_TOKEN_ID,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


def make_store(
    model: HookedProxyLM,
    input_ids: torch.Tensor,
    *,
    exclude_bos: bool,
) -> ActivationsStore:
    dataset = Dataset.from_dict({"input_ids": input_ids.tolist()})
    excluded = torch.tensor([BOS_TOKEN_ID], dtype=torch.long) if exclude_bos else None
    return ActivationsStore(
        model=model,
        dataset=dataset,
        streaming=False,
        hook_name=HOOK_NAME,
        hook_head_index=None,
        context_size=SEQUENCE_TOKENS,
        d_in=HIDDEN_SIZE,
        n_batches_in_buffer=2,
        total_training_tokens=BATCH_SIZE,
        store_batch_size_prompts=BATCH_SIZE,
        train_batch_size_tokens=BATCH_SIZE * NUCLEOTIDE_TOKENS,
        prepend_bos=True,
        normalize_activations="none",
        device=torch.device("cpu"),
        dtype="float32",
        cached_activations_path=None,
        model_kwargs=None,
        autocast_lm=False,
        dataset_trust_remote_code=False,
        seqpos_slice=(None,),
        exclude_special_tokens=excluded,
        disable_concat_sequences=True,
        sequence_separator_token="bos",
        activations_mixing_fraction=0.0,
        use_chat_formatting=False,
    )


def native_sequence_columns_are_supported(model: HookedProxyLM) -> bool:
    """Return whether SAELens accepts MarinDNA's raw ``seq`` column directly."""

    dataset = Dataset.from_dict({"seq": ["A" * NUCLEOTIDE_TOKENS]})
    try:
        ActivationsStore(
            model=model,
            dataset=dataset,
            streaming=False,
            hook_name=HOOK_NAME,
            hook_head_index=None,
            context_size=SEQUENCE_TOKENS,
            d_in=HIDDEN_SIZE,
            n_batches_in_buffer=1,
            total_training_tokens=1,
            store_batch_size_prompts=1,
            train_batch_size_tokens=NUCLEOTIDE_TOKENS,
            prepend_bos=True,
            normalize_activations="none",
            device=torch.device("cpu"),
            dtype="float32",
            dataset_trust_remote_code=False,
            disable_concat_sequences=True,
        )
    except ValueError as error:
        assert "Dataset must have" in str(error)
        return False
    return True


def activation_store_reset_works(model: HookedProxyLM, input_ids: torch.Tensor) -> bool:
    """Exercise the public reset method before the input iterator is exhausted."""

    store = make_store(model, input_ids, exclude_bos=True)
    first = store.get_batch_tokens(batch_size=1)
    store.reset_input_dataset()
    after_reset = store.get_batch_tokens(batch_size=1)
    return torch.equal(first, after_reset)


def make_sae() -> BatchTopKTrainingSAE:
    metadata = SAEMetadata(
        model_name="tiny-random-qwen3",
        model_class_name="AutoModelForCausalLM",
        hook_name=HOOK_NAME,
        hook_head_index=None,
        context_size=SEQUENCE_TOKENS,
        prepend_bos=True,
        exclude_special_tokens=[BOS_TOKEN_ID],
    )
    config = BatchTopKTrainingSAEConfig(
        d_in=HIDDEN_SIZE,
        d_sae=DICT_SIZE,
        k=K,
        topk_threshold_lr=1.0,
        dtype="float32",
        device="cpu",
        normalize_activations="none",
        metadata=metadata,
    )
    return BatchTopKTrainingSAE(config)


def run_one_step_runner(
    model: HookedProxyLM,
    input_ids: torch.Tensor,
) -> tuple[bool, float, str, bool, bool]:
    """Run the public training runner for one 510-activation optimizer step."""

    dataset = Dataset.from_dict({"input_ids": input_ids.tolist()})
    sae_config = make_sae().cfg
    with tempfile.TemporaryDirectory(prefix="issue288-saelens-runner-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        runner_config = LanguageModelSAERunnerConfig(
            sae=sae_config,
            model_name="tiny-random-qwen3",
            model_class_name="AutoModelForCausalLM",
            hook_name=HOOK_NAME,
            dataset_path="override-pretokenized-dna",
            streaming=False,
            context_size=SEQUENCE_TOKENS,
            n_batches_in_buffer=2,
            training_tokens=BATCH_SIZE * NUCLEOTIDE_TOKENS,
            store_batch_size_prompts=BATCH_SIZE,
            train_batch_size_tokens=BATCH_SIZE * NUCLEOTIDE_TOKENS,
            prepend_bos=True,
            activations_mixing_fraction=0.0,
            device="cpu",
            dtype="float32",
            exclude_special_tokens=[BOS_TOKEN_ID],
            n_batches_for_norm_estimate=1,
            n_eval_batches=1,
            logger=LoggingConfig(log_to_wandb=False, wandb_id="issue288"),
            n_checkpoints=0,
            checkpoint_path=str(tmp_path / "checkpoints"),
            save_final_checkpoint=True,
            output_path=str(tmp_path / "output"),
            verbose=False,
        )
        runner = LanguageModelSAETrainingRunner(
            runner_config,
            override_dataset=dataset,
            override_model=model,
        )
        with torch.enable_grad():
            trained_sae = runner.run()

        training_l0 = float(trained_sae.cfg.metadata.l0)
        assert training_l0 == K
        output_sae = SAE.load_from_disk(tmp_path / "output", device="cpu")
        checkpoint_dir = (
            tmp_path
            / "checkpoints"
            / "issue288"
            / f"final_{BATCH_SIZE * NUCLEOTIDE_TOKENS}"
        )
        checkpoint_saved = checkpoint_dir.exists()
        assert checkpoint_saved

        resume_config = LanguageModelSAERunnerConfig(
            sae=make_sae().cfg,
            model_name="tiny-random-qwen3",
            model_class_name="AutoModelForCausalLM",
            hook_name=HOOK_NAME,
            dataset_path="override-pretokenized-dna",
            streaming=False,
            context_size=SEQUENCE_TOKENS,
            n_batches_in_buffer=2,
            training_tokens=2 * BATCH_SIZE * NUCLEOTIDE_TOKENS,
            store_batch_size_prompts=BATCH_SIZE,
            train_batch_size_tokens=BATCH_SIZE * NUCLEOTIDE_TOKENS,
            prepend_bos=True,
            activations_mixing_fraction=0.0,
            device="cpu",
            dtype="float32",
            exclude_special_tokens=[BOS_TOKEN_ID],
            n_batches_for_norm_estimate=1,
            n_eval_batches=1,
            logger=LoggingConfig(log_to_wandb=False, wandb_id="issue288-resume"),
            n_checkpoints=0,
            checkpoint_path=str(tmp_path / "resumed-checkpoints"),
            save_final_checkpoint=False,
            output_path=None,
            resume_from_checkpoint=str(checkpoint_dir),
            verbose=False,
        )
        resume_runner = LanguageModelSAETrainingRunner(
            resume_config,
            override_dataset=dataset,
            override_model=model,
        )
        with torch.enable_grad():
            resumed_sae = resume_runner.run()
        assert float(resumed_sae.cfg.metadata.l0) == K
        return True, training_l0, type(output_sae).__name__, checkpoint_saved, True


@torch.no_grad()
def run_spike() -> SpikeResults:
    input_ids = make_input_ids()
    raw_model = make_model()
    with torch.no_grad():
        logits_before_hook = raw_model(input_ids=input_ids).logits.clone()

    module_names = {name for name, _ in raw_model.named_modules()}
    assert HOOK_NAME in module_names, (
        f"expected Qwen3 hook {HOOK_NAME!r}; nearby names: "
        f"{sorted(name for name in module_names if 'layers.0' in name)}"
    )
    wrapped_model = HookedProxyLM(
        raw_model,
        TinyDnaTokenizer(),  # type: ignore[arg-type]
        hook_names=[HOOK_NAME],
    )
    logits_after_hook, cache = wrapped_model.run_with_cache(
        input_ids,
        names_filter=[HOOK_NAME],
        prepend_bos=False,
    )
    hook_activations = cache[HOOK_NAME]
    assert hook_activations.shape == (
        BATCH_SIZE,
        SEQUENCE_TOKENS,
        HIDDEN_SIZE,
    )
    hook_is_logit_invariant = torch.equal(logits_before_hook, logits_after_hook)
    assert hook_is_logit_invariant

    raw_store = make_store(wrapped_model, input_ids, exclude_bos=False)
    raw_activations, raw_tokens = raw_store.get_raw_llm_batch()
    assert raw_tokens is not None
    assert raw_activations.shape == (
        BATCH_SIZE * SEQUENCE_TOKENS,
        HIDDEN_SIZE,
    )
    assert torch.equal(raw_tokens.reshape_as(input_ids), input_ids)
    assert int((raw_tokens == BOS_TOKEN_ID).sum()) == BATCH_SIZE

    filtered_store = make_store(wrapped_model, input_ids, exclude_bos=True)
    filtered_activations = filtered_store.get_filtered_llm_batch()
    expected_nucleotide_activations = BATCH_SIZE * NUCLEOTIDE_TOKENS
    assert filtered_activations.shape == (
        expected_nucleotide_activations,
        HIDDEN_SIZE,
    )

    sae = make_sae()
    per_window_activations = filtered_activations.reshape(
        BATCH_SIZE, NUCLEOTIDE_TOKENS, HIDDEN_SIZE
    )
    feature_acts = sae.encode(per_window_activations)
    total_nonzero = int(torch.count_nonzero(feature_acts))
    expected_nonzero = K * expected_nucleotide_activations
    assert total_nonzero == expected_nonzero
    mean_l0 = total_nonzero / expected_nucleotide_activations
    assert mean_l0 == K

    step_output = sae.training_forward_pass(
        TrainStepInput(
            sae_in=per_window_activations,
            coefficients={},
            dead_neuron_mask=None,
            n_training_steps=0,
            is_logging_step=False,
        )
    )
    assert torch.isfinite(step_output.loss)
    assert sae.topk_threshold.item() > 0

    metrics = get_recons_loss(
        sae=sae,
        model=wrapped_model,
        activation_scaler=ActivationScaler(),
        batch_tokens=input_ids,
        compute_kl=True,
        compute_ce_loss=True,
        ignore_tokens=[BOS_TOKEN_ID],
    )
    reconstruction_metrics_are_finite = all(
        torch.isfinite(value).all().item() for value in metrics.values()
    )
    assert reconstruction_metrics_are_finite

    with tempfile.TemporaryDirectory(prefix="issue288-saelens-") as tmp_dir:
        sae.save_inference_model(tmp_dir)
        inference_sae = SAE.load_from_disk(tmp_dir, device="cpu")
        assert isinstance(inference_sae, JumpReLUSAE)

        training_features = sae.encode(per_window_activations)
        inference_features = inference_sae.encode(per_window_activations)
        checkpoint_active_set_matches = torch.equal(
            training_features != 0, inference_features != 0
        )
        checkpoint_max_abs_diff = float(
            (training_features - inference_features).abs().max()
        )
        checkpoint_roundtrip_matches = torch.allclose(
            training_features,
            inference_features,
            rtol=1e-5,
            atol=1e-6,
        )
        checkpoint_training_nonzero = int(torch.count_nonzero(training_features))
        checkpoint_inference_nonzero = int(torch.count_nonzero(inference_features))

        whole = inference_sae(per_window_activations)
        split = torch.cat(
            [
                inference_sae(per_window_activations[i : i + 1])
                for i in range(BATCH_SIZE)
            ],
            dim=0,
        )
        inference_is_partition_invariant = torch.equal(whole, split)
        assert inference_is_partition_invariant

        config_path = Path(tmp_dir) / "cfg.json"
        assert config_path.exists()

    reset_works = activation_store_reset_works(wrapped_model, input_ids)
    (
        runner_completed,
        runner_l0,
        runner_output_type,
        runner_checkpoint_saved,
        runner_resume_completed,
    ) = run_one_step_runner(wrapped_model, input_ids)

    return SpikeResults(
        saelens_commit=SAELENS_COMMIT,
        hook_name=HOOK_NAME,
        hook_shape=tuple(hook_activations.shape),
        hook_is_logit_invariant=hook_is_logit_invariant,
        bos_tokens_removed=BATCH_SIZE,
        nucleotide_activations=expected_nucleotide_activations,
        batchtopk_total_nonzero=total_nonzero,
        batchtopk_mean_l0=mean_l0,
        reconstruction_metrics_are_finite=reconstruction_metrics_are_finite,
        checkpoint_type=type(inference_sae).__name__,
        checkpoint_roundtrip_matches=checkpoint_roundtrip_matches,
        checkpoint_active_set_matches=checkpoint_active_set_matches,
        checkpoint_max_abs_diff=checkpoint_max_abs_diff,
        checkpoint_training_nonzero=checkpoint_training_nonzero,
        checkpoint_inference_nonzero=checkpoint_inference_nonzero,
        inference_is_partition_invariant=inference_is_partition_invariant,
        native_sequence_columns_supported=native_sequence_columns_are_supported(
            wrapped_model
        ),
        activation_store_reset_works=reset_works,
        runner_completed=runner_completed,
        runner_training_l0=runner_l0,
        runner_output_type=runner_output_type,
        runner_checkpoint_saved=runner_checkpoint_saved,
        runner_resume_completed=runner_resume_completed,
    )


if __name__ == "__main__":
    print(json.dumps(asdict(run_spike()), indent=2, sort_keys=True))
