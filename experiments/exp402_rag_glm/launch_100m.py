"""Issue #402: train the gated 103.8M ortholog-RAG scaling rung.

This changes only the Qwen3 geometry relative to ``launch.py``. The immutable
corpus, token cache, one-billion-token horizon, token batch, AdamH transfer,
accelerator policy, and 1,000-step Hugging Face export cadence are identical.
"""

from dataclasses import replace

from fray.types import ResourceConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.train import EvalSuite, train_lm

from launch import (
    HF_SAVE_EVERY,
    MENDELIAN_TRAITS_RAG_255,
    OPTIMIZER,
    RAG_EVAL_EVERY,
    SEQ_LEN,
    TOKENIZER_PATH,
    TRAIN_BATCH_SIZE,
    TRAIN_HOST_CPU,
    TRAIN_HOST_RAM,
    TRAIN_REGIONS,
    TRAIN_STEPS,
    TRAIN_TPU,
    VOCAB_SIZE,
    online_eval_enabled,
    rag_tokenized_dataset,
)

CHECKPOINT_NAME = "checkpoints/dna-exp402-rag-h768-p104m-1b"
RUN_ID = "dna-exp402-rag-h768-p104M-1B"

# One modest size step above the completed 45.9M rung. All head dimensions and
# width ratios remain conventional; 11 layers lands at 103,838,976 parameters.
MODEL = Qwen3Config(
    max_seq_len=SEQ_LEN,
    hidden_dim=768,
    intermediate_dim=3_072,
    num_layers=11,
    num_heads=6,
    num_kv_heads=6,
    head_dim=128,
    use_sliding_window=False,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=False,
    tokenizer=TOKENIZER_PATH,
)
assert MODEL.total_trainable_params(VOCAB_SIZE) == 103_838_976


def build() -> ArtifactStep:
    """Reuse the frozen cache and assemble the scratch 103.8M training run."""
    dataset = rag_tokenized_dataset()
    training = train_lm(
        name=CHECKPOINT_NAME,
        model=MODEL,
        optimizer=OPTIMIZER,
        datasets={dataset: 1.0},
        batch_size=TRAIN_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=TRAIN_STEPS,
        z_loss_weight=1.0e-7,
        evals=(
            EvalSuite(tasks=(MENDELIAN_TRAITS_RAG_255,), every=RAG_EVAL_EVERY)
            if online_eval_enabled()
            else None
        ),
        resources=ResourceConfig.with_tpu(
            TRAIN_TPU,
            cpu=TRAIN_HOST_CPU,
            ram=TRAIN_HOST_RAM,
            disk="100g",
            regions=TRAIN_REGIONS,
        ),
        steps_per_eval=500,
        wandb_project="marin",
        wandb_group="dna-exp402-v1",
        run_id=RUN_ID,
        tags=("dna", "dna-exp402", "rag", "qwen3", "104M", "scratch", "scale-rung"),
    )
    original_build_config = training.build_config

    def build_config_with_early_hf_exports(ctx: StepContext):
        pod_config = original_build_config(ctx)
        return replace(
            pod_config,
            train_config=replace(
                pod_config.train_config,
                hf_save_steps=HF_SAVE_EVERY,
            ),
        )

    return replace(training, build_config=build_config_with_early_hf_exports)


if __name__ == "__main__":
    experiment_main(build)()
