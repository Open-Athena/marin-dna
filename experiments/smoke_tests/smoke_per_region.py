# Copyright The Marin Authors / Bolinas Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the per-region pipeline (issue #187, Workstream A.0).

Runs one 50-step Qwen3-1B training on the smallest v3 subset
(``v3_tss_region_and_utr5``, 8 M rows) to validate the load-bearing pieces
of the new pipeline cheaply (~10 min wall-clock on v5p-8, mostly JIT compile)
before committing the ~80 min per-region canary.

What it validates:
  * ``HfTokenizeConfig`` resolves ``bolinas-dna/zoonomia-v1-v3_tss_region_and_utr5``
    + the 5 ``zoonomia-v1-val_*`` recipes.
  * Train loop runs end-to-end on v5p-8.
  * ``MENDELIAN_TRAITS_255`` lm-eval task fires at step 25 + 50, emits
    ``_macro_avg_/avg/pairwise_accuracy``, ``_global_/avg/pairwise_accuracy``,
    and per-subset cells under ``lm_eval/mendelian_traits_255/…`` in WandB.
  * The 5 LL-gap traces appear in WandB.
  * HF checkpoint lands at ``<output_path>/hf/step-25/`` and ``step-50/`` with
    ``config.json`` + ``model.safetensors`` + ``tokenizer.json``. Sanity-load
    via ``AutoModelForCausalLM.from_pretrained`` from a small CPU job afterward.

Smoke acceptance criteria are pipeline-level — 50 steps doesn't train anything
meaningful, so eval metrics ~0.5 ± noise are fine. Any non-exception is a pass.

The smoke test reuses the per-region builders so cache-hits land for the
downstream canary (tokenize cache is keyed on dataset + format, both of which
match the real per-region script for ``v3_tss_region_and_utr5``).

Launch:

    uv run iris --cluster=marin job run \\
        --no-wait --user gonzalo --job-name exp187-smoke \\
        --cpu 1 --memory 2g --extra marin --region us-east5 \\
        -e WANDB_API_KEY "$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')" \\
        -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \\
        -- python experiments/smoke_tests/smoke_per_region.py
"""

import os
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from marin.training.training import TrainLmOnPodConfig

# Reuse builders from the per-region script. These imports double as a sanity
# check on the per-region module's importability.
from experiments.per_region.exp187_per_region import (
    BATCH_SIZE,
    EXP_ISSUE,
    TPU_TYPES,
    TRAIN_DATASETS,
    VERSION,
    _build_data_mixture,
    _build_model_config,
    _build_optimizer,
    _eval_harness_config,
    _run_train_with_marin_dna_imports,
    _train_remote_env_vars,
)

# Smoke-test overrides. Everything else (model, optimizer, mixture, eval task,
# val recipes) inherits from the per-region script.
SMOKE_NUM_TRAIN_STEPS = 50
SMOKE_EVALS_PER_RUN = 2  # eval at step 25 + step 50
SMOKE_HF_SAVE_STEPS = 25  # HF checkpoint at step 25 + step 50
SMOKE_CHECKPOINT_TIME_INTERVAL = timedelta(minutes=5)

# Pick the smallest v3 subset for the fastest first-time HF fetch + tokenize.
SMOKE_STRATEGY = "v3_tss_region_and_utr5"
SMOKE_DATASET = TRAIN_DATASETS[SMOKE_STRATEGY]


def _smoke_train_step() -> ExecutorStep:
    steps_per_eval = max(1, SMOKE_NUM_TRAIN_STEPS // SMOKE_EVALS_PER_RUN)
    run_name = f"dna-exp{EXP_ISSUE}-smoke-{SMOKE_STRATEGY}-{VERSION}"
    tags = (
        "dna",
        "marin_dna",
        f"exp{EXP_ISSUE}",
        "smoke",
        "per-region",
        VERSION,
        f"region={SMOKE_STRATEGY}",
        "scale=1b",
        f"bs={BATCH_SIZE}",
        f"steps={SMOKE_NUM_TRAIN_STEPS}",
    )

    inner = TrainLmConfig(
        data=_build_data_mixture(SMOKE_STRATEGY, SMOKE_DATASET),
        model=_build_model_config(),
        train_seq_len=_build_model_config().max_seq_len,
        optimizer=_build_optimizer(),
        eval_harness=_eval_harness_config(),
        eval_harness_steps=steps_per_eval,
        # hf_save_path auto-set by marin from pod_config.output_path; we just
        # override the default save cadence. See per-region script for the
        # full explanation.
        hf_save_steps=SMOKE_HF_SAVE_STEPS,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=list(tags),
                group=f"dna-exp{EXP_ISSUE}-smoke-{VERSION}",
                name=run_name,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=SMOKE_NUM_TRAIN_STEPS,
            steps_per_eval=steps_per_eval,
            checkpointer=CheckpointerConfig(
                save_interval=SMOKE_CHECKPOINT_TIME_INTERVAL,
                keep=[dict(every=SMOKE_HF_SAVE_STEPS)],
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
        name=os.path.join("checkpoints", run_name),
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
        steps=[_smoke_train_step()],
        description=(
            f"DNA Bolinas exp{EXP_ISSUE} smoke test — 50-step per-region "
            f"pipeline validation on {SMOKE_STRATEGY} {VERSION}"
        ),
    )


if __name__ == "__main__":
    main()
