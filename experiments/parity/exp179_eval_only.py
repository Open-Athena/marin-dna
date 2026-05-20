# Copyright The Marin Authors / MarinDNA Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval-only run of the online lm_eval VEP scorer on exp166-p1B — #179 parity
check against the offline ``snakemake/analysis/evals_v2/`` rc-averaged scores.

We construct ``EvalHarnessMainConfig`` directly with an explicit
``Qwen3Config`` and call ``run_eval_harness_main`` ourselves rather than going
through ``LevanterLmEvalEvaluator``: that wrapper's
``HFCheckpointConverter.from_hf(model_path)`` walks levanter's model registry
and instantiates each candidate's ``hf_checkpoint_converter()``;
``GemmaConfig`` defaults to ``reference_checkpoint="google/gemma-2b"``
(gated), and the ``_infer_tokenizer`` call 401s. The dna-branch precedent at
``marin@dna:experiments/dna/smoke_tests/eval_traitgym.py`` uses the same shape.

Launch from a CPU box with iris CLI authed to the marin cluster:

    uv run iris --cluster=marin job run \\
        --no-wait --user gonzalo --job-name exp179-eval-only \\
        --cpu 1 --memory 2g --extra marin --region us-east5 \\
        -e WANDB_API_KEY "$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')" \\
        -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \\
        -- python experiments/parity/exp179_eval_only.py
"""

import dataclasses
import logging
import os

import jmp
import levanter.eval_harness as eval_harness
from fray.cluster import ResourceConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.evaluation.evaluation_config import convert_to_levanter_task_config
from marin.execution.executor import ExecutorStep, executor_main
from marin.execution.remote import remote

from marin_dna.levanter.defaults import dna_effective_seq_len
from marin_dna.pipelines.evals.lm_eval.task_configs import MENDELIAN_TRAITS_255

logger = logging.getLogger(__name__)


MODEL_NAME = "exp166-p1B-step-16398"
MODEL_PATH = "bolinas-dna/exp166-p1B-step-16398"
TOKENIZER = "bolinas-dna/tokenizer-char-bos"

# Qwen3 1B geometry from marin@dna:experiments/dna/exp166_zoonomia_1ep_scaling.py
# via CompletedAdamHHeuristic._build_model_config(hidden_size=1920). Inlined
# to avoid pulling in the heuristic / scaling-sweep dependency tree.
DNA_BASE_SEQ_LEN = 255
HIDDEN_DIM = 1920
INTERMEDIATE_DIM = HIDDEN_DIM * 4
NUM_HEADS = HIDDEN_DIM // 128
NUM_LAYERS = 19

TPU_TYPES: tuple[str, ...] = ("v5p-8",)

# `tpu` pulls libtpu via `marin[tpu]`; without it JAX falls back to CPU and
# `trainer.initialize()` raises "No accelerator found" — which iris misreads
# as a bad-node signature and retries forever.
_EVAL_DEPENDENCY_GROUPS = ["marin", "tpu"]

WANDB_PROJECT = "marin"
WANDB_RUN_NAME = "dna-exp179-mendelian-traits-rc-eval-only"
WANDB_TAGS = ("dna", "exp179", "mendelian-traits-rc-eval-only", "parity")


def _build_model_config(seq_len: int) -> Qwen3Config:
    return Qwen3Config(
        hidden_dim=HIDDEN_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
        max_seq_len=seq_len,
        rope=Llama3RotaryEmbeddingsConfig(),
    )


@dataclasses.dataclass(frozen=True)
class _EvalConfig:
    """ExecutorStep requires a config dataclass; the eval is fully parameterized
    by module-level constants."""


def _run_eval_harness_only(_config: _EvalConfig) -> None:
    # This import MUST be inside the function body. Iris's
    # `Entrypoint.from_callable` cloudpickles `__main__` by-value, which
    # captures function bytecode but does not re-import the module on the
    # worker. A module-top `import marin_dna.pipelines.evals.lm_eval` therefore
    # never fires on the TPU pod, and the lm-eval / levanter monkeypatches
    # never install.
    import marin_dna.pipelines.evals.lm_eval  # noqa: F401

    seq_len = dna_effective_seq_len(DNA_BASE_SEQ_LEN, TOKENIZER)
    eval_config = eval_harness.EvalHarnessMainConfig(
        eval_harness=eval_harness.LmEvalHarnessConfig(
            task_spec=convert_to_levanter_task_config([MENDELIAN_TRAITS_255]),
            log_samples=False,
        ),
        tokenizer=MODEL_PATH,  # exp166's HF repo bundles its tokenizer
        checkpoint_path=MODEL_PATH,
        checkpoint_is_hf=True,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=list(WANDB_TAGS),
                name=WANDB_RUN_NAME,
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            per_device_eval_parallelism=64,
        ),
        model=_build_model_config(seq_len),
    )
    eval_harness.run_eval_harness_main(eval_config)


def main() -> None:
    # HF parquet-manifest fetch is slow at the default 10s timeout; concurrent
    # uv syncs on one VM serialize on a uv lock with a 300s default. See
    # `experiments/README.md`.
    env_vars: dict[str, str] = {
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "UV_LOCK_TIMEOUT": "7200",
    }
    if "WANDB_API_KEY" in os.environ:
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    step = ExecutorStep(
        name=f"evaluation/lm_evaluation_harness_levanter/exp179_{MODEL_NAME}",
        fn=remote(
            _run_eval_harness_only,
            resources=ResourceConfig.with_tpu(TPU_TYPES),
            pip_dependency_groups=_EVAL_DEPENDENCY_GROUPS,
            env_vars=env_vars,
        ),
        config=_EvalConfig(),
    )
    executor_main(
        steps=[step],
        description=(
            "exp179 eval-only — mendelian_traits_255 (FWD/RC/AVG) on "
            "exp166-p1B; parity-check vs snakemake/analysis/evals_v2/."
        ),
    )


if __name__ == "__main__":
    main()
