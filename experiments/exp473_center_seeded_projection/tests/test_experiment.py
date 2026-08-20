from __future__ import annotations

from exp473_center_seeded_projection.experiment import (
    ARMS,
    BATCH_SIZE,
    HF_SAVE_STEPS,
    MODEL,
    SEED,
    SEQUENCE_LENGTH,
    TRAIN_STEPS,
    build_training,
)
from marin.execution.lazy import StepContext


def _set_required_env(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_PROJECT", "marin")
    monkeypatch.setenv(
        "MARIN_PREFIX",
        "gs://marin-us-east5/MarinDNA/exp473_center_seeded_projection",
    )
    for name in [
        "EXP473_CENTER1_CDS_REVISION",
        "EXP473_FULLWINDOW_ENHANCER_REVISION",
        "EXP473_CENTER1_ENHANCER_REVISION",
    ]:
        monkeypatch.setenv(name, "e" * 40)


def test_all_four_arms_materialize_the_matched_recipe(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    assert MODEL.max_seq_len == SEQUENCE_LENGTH
    assert MODEL.hidden_dim == 1_152
    assert MODEL.num_layers == 12
    assert MODEL.num_heads == 9

    for key, arm in ARMS.items():
        step = build_training(arm)
        pod = step.build_config(
            StepContext.for_fingerprint(
                runtime_arg_keys=step.runtime_args,
                deps=step.deps,
            )
        )
        train = pod.train_config
        assert train.trainer.train_batch_size == BATCH_SIZE
        assert train.trainer.num_train_steps == TRAIN_STEPS
        assert train.trainer.steps_per_eval == HF_SAVE_STEPS
        assert train.train_seq_len == SEQUENCE_LENGTH
        assert train.data_seed == SEED
        assert train.hf_save_steps == HF_SAVE_STEPS
        assert train.z_loss_weight == 4.312883184368223e-06
        assert len(step.deps) == 1
        assert key in step.name


def test_tokenized_handles_pin_hf_revisions(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    for arm in ARMS.values():
        step = build_training(arm)
        (cache,) = step.deps
        config = cache.build_config(
            StepContext.for_fingerprint(
                deps=cache.deps,
            ),
        )
        assert config.id == arm.hf_repo
        assert config.revision == arm.resolved_revision()
        assert len(config.revision) == 40
        assert config.format.text_key == "sequence"
        assert config.format.lowercase_weight == 0.01
