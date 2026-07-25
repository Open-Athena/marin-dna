"""Run a bounded online-harness smoke against the completed 104M checkpoint.

This is deliberately eval-only: it loads the saved Levanter checkpoint and
executes the same paired, cached-prefix Mendelian task that training would have
attached as a callback. Offline scoring remains the source of the full curves.
"""

from __future__ import annotations

import os

from lm_eval_compat import stub_unused_transformers5_multimodal_adapter

stub_unused_transformers5_multimodal_adapter()

import jmp
import levanter.tracker
from haliax.partitioning import ResourceAxis
from levanter.eval_harness import (
    EvalHarnessMainConfig,
    LmEvalHarnessConfig,
    run_eval_harness_main,
)
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.evaluation.evaluation_config import convert_to_levanter_task_config
from marin_dna.pipelines.evals.lm_eval.task_configs import MENDELIAN_TRAITS_RAG_255

from launch import TOKENIZER_PATH
from launch_100m import MODEL

CHECKPOINT_PATH_ENV = "EXP402_ONLINE_CHECKPOINT_PATH"
MAX_EXAMPLES_ENV = "EXP402_ONLINE_MAX_EXAMPLES"
RUN_ID_ENV = "EXP402_ONLINE_RUN_ID"
DEFAULT_MAX_EXAMPLES = 32
DEFAULT_RUN_ID = "dna-exp402-online-parity-p104M-final-smoke1"
RAG_EVAL_BATCH_SIZE = 16
_TOKEN_AXES = (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)


def build_config(*, checkpoint_path: str, max_examples: int, run_id: str) -> EvalHarnessMainConfig:
    """Build the standalone Levanter harness config without starting JAX."""
    assert checkpoint_path
    assert max_examples > 0
    assert max_examples <= 256
    assert max_examples % 2 == 0, "keep both strand rows for every limited variant"
    assert run_id.startswith("dna-exp402-")
    task_spec = convert_to_levanter_task_config([MENDELIAN_TRAITS_RAG_255])
    assert len(task_spec) == 1
    return EvalHarnessMainConfig(
        eval_harness=LmEvalHarnessConfig(
            task_spec=task_spec,
            max_examples=max_examples,
            max_length=2_048,
            log_samples=True,
            bootstrap_iters=0,
        ),
        tokenizer=TOKENIZER_PATH,
        checkpoint_path=checkpoint_path,
        checkpoint_is_hf=False,
        trainer=TrainerConfig(
            id=run_id,
            tracker=WandbConfig(
                project="marin",
                name=run_id,
                group="dna-exp402-v1",
                tags=[
                    "dna",
                    "dna-exp402",
                    "rag",
                    "104M",
                    "online-eval",
                    "parity-smoke",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=RAG_EVAL_BATCH_SIZE,
            per_device_parallelism=4,
            per_device_eval_parallelism=4,
            allow_nondivisible_batch_size=True,
            num_train_steps=1,
            mesh=MeshConfig(
                axes={"replica": 1, "data": -1, "model": 1},
                compute_mapping={"token": _TOKEN_AXES, "token_repeat": _TOKEN_AXES},
            ),
            require_accelerator=True,
            log_jaxprs=False,
            log_xla_hlo=False,
        ),
        model=MODEL,
    )


def finish_tracker_if_initialized() -> None:
    """Flush W&B only if Levanter initialized a global tracker."""
    try:
        tracker = levanter.tracker.current_tracker()
    except RuntimeError as error:
        assert "No global tracker set" in str(error)
        return
    tracker.finish()


def main() -> None:
    checkpoint_path = os.environ.get(CHECKPOINT_PATH_ENV)
    assert checkpoint_path, f"set {CHECKPOINT_PATH_ENV} to a Levanter checkpoint root"
    max_examples = int(os.environ.get(MAX_EXAMPLES_ENV, str(DEFAULT_MAX_EXAMPLES)))
    run_id = os.environ.get(RUN_ID_ENV, DEFAULT_RUN_ID)
    config = build_config(
        checkpoint_path=checkpoint_path,
        max_examples=max_examples,
        run_id=run_id,
    )
    try:
        outputs = run_eval_harness_main(config)
        assert outputs is not None
        assert "results" in outputs
        assert "mendelian_traits_rag_255" in outputs["results"]
    finally:
        finish_tracker_if_initialized()


if __name__ == "__main__":
    main()
