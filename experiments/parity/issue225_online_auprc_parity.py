# Copyright The Marin Authors / MarinDNA Authors
# SPDX-License-Identifier: Apache-2.0

"""Online-vs-offline AUPRC parity check for the migrated mendelian VEP task (#225).

Trains a **tiny** Qwen3 (not 1B — we only need a non-degenerate model, fast) for
**10 steps** on the smallest v3 subset, with the ``mendelian_traits_255`` lm-eval
task firing in-training. This exercises the *real in-training setting* (the path
that reported chance in #187), now on FWD/RC-avg + AUPRC. The run:

  * logs ``lm_eval/mendelian_traits_255/<subset>/<fwd|rc|avg>/auprc`` cells to
    WandB at steps 5 and 10 (online AUPRC), and
  * saves a reloadable HF checkpoint at ``<output_path>/hf/step-{5,10}/``.

We then run ``snakemake/analysis/evals_v2/`` on the step-10 HF checkpoint (one
small GPU sky node) and confirm **online AUPRC == offline AUPRC per subset,
within bootstrap SE**. At 10 steps the model is ~untrained, so both sit near the
AUPRC baseline (≈0.1 ``_global_`` on the 1:9 dataset) — the test is *agreement
between the two code paths on the same checkpoint* + that the in-training path
runs end-to-end and emits AUPRC cells (it would have asserted-crashed under the
old PA aggregator on the 1:9 dataset).

Tiny model, not 1B: model size is irrelevant to a scoring-path/plumbing parity
check, and a tiny transformer cuts JIT-compile + step time so we iterate fast.
The scoring-relevant knobs are kept fixed: ``tokenizer-char-bos`` + 255 bp
window (256 tokens with BOS), same as the real per-region runs.

Launch from a CPU box with an iris tunnel open (see ``experiments/README.md``):

    uv run iris --cluster=marin job run \\
        --no-wait --user gonzalo --job-name issue225-auprc-parity \\
        --cpu 1 --memory 2g --extra marin --region us-east5 \\
        -e WANDB_API_KEY "$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')" \\
        -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \\
        -- python experiments/parity/issue225_online_auprc_parity.py
"""

import os

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from marin.processing.tokenize import lm_mixture_data_config
from marin.training.training import TrainLmOnPodConfig

# Reuse the per-region builders verbatim where we can — these imports double as a
# sanity check on the per-region module's importability.
from experiments.exp187_per_region import (
    TPU_TYPES,
    TRAIN_DATASETS,
    TRAIN_FORMAT,
    _build_optimizer,
    _eval_harness_config,
    _model_seq_len,
    _run_train_with_marin_dna_imports,
    _tokenize,
    _train_remote_env_vars,
)

EXP_ISSUE = 225
VERSION = "v0.1"

# Smallest v3 subset → fastest first-time fetch + tokenize, and the tokenize
# cache is shared with exp187 (same name + format) so this is a cache-hit.
STRATEGY = "v3_tss_region_and_utr5"
DATASET = TRAIN_DATASETS[STRATEGY]

# Tiny Qwen3 — geometry chosen for speed, not skill. head_dim = 128 / 2 = 64.
HIDDEN_DIM = 128
INTERMEDIATE_DIM = 512
NUM_LAYERS = 2
NUM_HEADS = 2
NUM_KV_HEADS = 2

# Small batch — a tiny model + 10 steps needs no more, and it keeps the step
# fast. v5p-8 data-parallel shards this fine with allow_nondivisible_batch_size.
BATCH_SIZE = 256

NUM_TRAIN_STEPS = 10
# Eval + HF-save at steps 5 and 10. NB: keep these strictly LESS than
# NUM_TRAIN_STEPS (5 != 10) — exp160 saw HF saves silently never land when
# ``hf_save_steps == num_train_steps`` (the final-step callback didn't fire);
# 5 makes step 10 a 2× multiple, which the smoke test verified does save.
EVAL_AND_SAVE_STEPS = 5

WANDB_PROJECT = "marin"
RUN_NAME = f"dna-exp{EXP_ISSUE}-online-auprc-parity-tiny-{VERSION}"


def _build_tiny_model_config() -> Qwen3Config:
    return Qwen3Config(
        hidden_dim=HIDDEN_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        max_seq_len=_model_seq_len(),
        rope=Llama3RotaryEmbeddingsConfig(),
    )


def _build_single_component_mixture():
    """Just the one training subset — the LL-gap val recipes aren't needed for
    this parity check, so we skip them (fewer tokenize jobs)."""
    component = _tokenize(
        f"marin_dna-zoonomia-v1-{STRATEGY}-char-bos", DATASET, TRAIN_FORMAT
    )
    return lm_mixture_data_config(
        components={STRATEGY: component},
        weights={STRATEGY: 1.0},
    )


def _build_train_step() -> ExecutorStep:
    tags = (
        "dna",
        "marin_dna",
        f"exp{EXP_ISSUE}",
        "parity",
        "online-auprc",
        VERSION,
        f"region={STRATEGY}",
        "scale=tiny",
        f"bs={BATCH_SIZE}",
        f"steps={NUM_TRAIN_STEPS}",
    )

    inner = TrainLmConfig(
        data=_build_single_component_mixture(),
        model=_build_tiny_model_config(),
        train_seq_len=_model_seq_len(),
        optimizer=_build_optimizer(),
        eval_harness=_eval_harness_config(),
        eval_harness_steps=EVAL_AND_SAVE_STEPS,
        # hf_save_path is auto-set by marin to ``<output_path>/hf`` since
        # output_path is set on the pod config; we only override the cadence.
        hf_save_steps=EVAL_AND_SAVE_STEPS,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=list(tags),
                group=f"dna-exp{EXP_ISSUE}-{VERSION}",
                name=RUN_NAME,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=NUM_TRAIN_STEPS,
            steps_per_eval=EVAL_AND_SAVE_STEPS,
            checkpointer=CheckpointerConfig(
                keep=[dict(every=EVAL_AND_SAVE_STEPS)],
            ),
            mesh=MeshConfig(axes={"replica": 1, "data": -1, "model": 1}),
            allow_nondivisible_batch_size=True,
        ),
    )
    pod_config = TrainLmOnPodConfig(
        train_config=inner,
        resources=ResourceConfig.with_tpu(TPU_TYPES, ram="300g"),
        output_path=this_output_path(),
    )
    return ExecutorStep(
        name=os.path.join("checkpoints", RUN_NAME),
        fn=remote(
            _run_train_with_marin_dna_imports,
            resources=ResourceConfig.with_tpu(TPU_TYPES, ram="300g"),
            pip_dependency_groups=["marin", "tpu"],
            env_vars=_train_remote_env_vars(),
        ),
        config=pod_config,
    )


def main() -> None:
    executor_main(
        steps=[_build_train_step()],
        description=(
            f"DNA MarinDNA exp{EXP_ISSUE} — tiny 10-step online-vs-offline AUPRC "
            f"parity check on {STRATEGY} {VERSION}"
        ),
    )


if __name__ == "__main__":
    main()
