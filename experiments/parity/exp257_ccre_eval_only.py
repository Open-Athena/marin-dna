# Copyright The Marin Authors / MarinDNA Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue #257 — reproduce the ONLINE in-training mendelian eval on the exact
exp232 ``v4_ccre_non_promoter`` step-4999 model, via the real levanter lm_eval
path (not a from-scratch approximation).

Context: the in-training online metric logged ``distal/fwd/auprc = 0.2507`` at
the final step, but the offline ``evals_v2`` pipeline (and three from-scratch
reproductions) score the exported step-4999 checkpoint at **0.158** on that
exact cell — while matching online on 28/30 other (subset, strand) cells. So the
divergence is isolated to ``distal/fwd``. This run answers: does the *actual*
levanter lm_eval path reproduce 0.251 on this model?

  * Loads the **native levanter checkpoint** (the exact training state the
    in-training eval ran on — ``checkpoint_is_hf=False``, ``latest_checkpoint_path``
    resolves ``…/checkpoints`` → ``step-4999``).
  * Runs ``mendelian_traits_255`` (the same task, FWD/RC/AVG + AUPRC).
  * Monkeypatches the aggregation to dump every ``distal`` per-(variant, strand)
    raw LLR to stdout (``DISTAL_ITEM {...}``) so we can diff per-variant against
    the offline scores parquet and localize *which* distal variants diverge.

Adapted from ``experiments/parity/exp179_eval_only.py`` (geometry 1B→0.25B;
HF→native checkpoint; + the distal dump). Launch from a CPU box with gcloud
authed (``--cluster=marin`` auto-tunnels):

    uv run --extra marin iris --cluster=marin job run \\
        --no-wait --user gonzalo --job-name exp257-ccre-online-repro \\
        --cpu 1 --memory 2g --extra marin --region us-east5 \\
        -e WANDB_API_KEY "$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')" \\
        -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \\
        -- python experiments/parity/exp257_ccre_eval_only.py
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

MODEL_NAME = "exp232-v4_ccre_non_promoter-step-4999"
# Native levanter checkpoint root (the in-training state). load_checkpoint via
# latest_checkpoint_path(…/checkpoints) resolves to step-4999 (the latest).
CHECKPOINT_PATH = (
    "gs://marin-us-east5/checkpoints/"
    "dna-exp232-zoonomia-v1-0p25b-v4_ccre_non_promoter-v0.1-feca83/checkpoints"
)
TOKENIZER = "bolinas-dna/tokenizer-char-bos"

# exp232 Qwen3-0.25B geometry (hidden=1152, layers=12, heads=9, head_dim=128) —
# verbatim from experiments/exp232_per_region.py::_build_model_config so the
# native checkpoint's array structure matches exactly.
DNA_BASE_SEQ_LEN = 255
HIDDEN_DIM = 1152
INTERMEDIATE_DIM = HIDDEN_DIM * 4  # 4608
NUM_HEADS = HIDDEN_DIM // 128  # 9
NUM_LAYERS = 12
INITIALIZER_RANGE = 0.02

# v6e-4 (us-east5-b) is data-local to the checkpoint + mendelian data and, per
# `iris cluster status`, idle (10 ready / 0 demand) vs the heavily-contended
# v5p-8 us-east5-a pool (20 ready / 42 demand). A 0.25B eval needs nothing
# bigger; levanter reshards the v5p-8-saved checkpoint onto 4 chips on load.
# Env-overridable (comma-separated) to re-route without editing.
TPU_TYPES: tuple[str, ...] = tuple(
    t.strip() for t in os.getenv("TPU_TYPE", "v6e-4").split(",") if t.strip()
)
_EVAL_DEPENDENCY_GROUPS = ["marin", "tpu"]

WANDB_PROJECT = "marin"
WANDB_RUN_NAME = "dna-exp232-ccre-step4999-online-repro-issue257"
WANDB_TAGS = ("dna", "exp232", "issue257", "online-repro", "parity")


def _build_model_config(seq_len: int) -> Qwen3Config:
    return Qwen3Config(
        hidden_dim=HIDDEN_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
        max_seq_len=seq_len,
        rope=Llama3RotaryEmbeddingsConfig(),
        initializer_range=INITIALIZER_RANGE,
    )


def _install_llr_dump() -> None:
    """Wrap the AUPRC aggregation to write every per-(variant, strand) RAW LLR to
    a GCS JSONL (``DUMP_GCS`` env) — durable, unlike pod stdout which never reaches
    the launcher's iris logs / wandb console. For per-variant offline diffing."""
    import json
    import sys

    from marin_dna.pipelines.evals.lm_eval import dna_vep_llr_eval as dv

    # lm_eval resolves the task's `!function dna_vep_llr_eval.DnaVepLlrEvalTask`
    # by importing a TOP-LEVEL `dna_vep_llr_eval` module — a *different* object
    # than this package import, so a naive class patch misses the copy lm_eval
    # actually uses. Pre-register THIS module under that name so lm_eval reuses
    # it (and the patch below applies to the aggregation lm_eval calls).
    sys.modules["dna_vep_llr_eval"] = dv

    orig = dv._AuprcAggregation.__call__

    def patched(self, items):
        rows = [
            {
                "llr": float(llr),
                "label": int(target),
                "subset": str(subset),
                "chrom": str(vid[0]),
                "pos": int(vid[1]),
                "ref": str(vid[2]),
                "alt": str(vid[3]),
                "match_group": int(mg),
                "strand": strand,
            }
            for llr, target, subset, vid, mg, strand in items
        ]
        dump_gcs = os.getenv("DUMP_GCS")
        if dump_gcs:
            try:
                import fsspec

                with fsspec.open(dump_gcs, "w") as f:
                    f.write("\n".join(json.dumps(r) for r in rows))
                print(f"[exp257] wrote {len(rows)} LLR rows -> {dump_gcs}", flush=True)
            except Exception as exc:
                print(f"[exp257] GCS dump FAILED ({dump_gcs}): {exc}", flush=True)
        else:
            print(f"[exp257] DUMP_GCS unset; {len(rows)} rows not written", flush=True)
        return orig(self, items)

    dv._AuprcAggregation.__call__ = patched


@dataclasses.dataclass(frozen=True)
class _EvalConfig:
    """ExecutorStep requires a config dataclass; eval is fully parameterized by
    module-level constants."""


def _run_eval_harness_only(_config: _EvalConfig) -> None:
    # MUST be inside the function body — iris cloudpickles __main__ by-value and
    # never re-imports the module on the worker, so the lm-eval/levanter
    # monkeypatches (and our distal dump) only install if triggered here.
    import marin_dna.pipelines.evals.lm_eval  # noqa: F401

    _install_llr_dump()

    # Issue #257: levanter's lm_eval hardcodes max_packed_segments=64 (sequence
    # packing w/ block-diagonal segment attention). Set MAX_PACKED_SEGMENTS=1 to
    # disable packing (each sequence forwarded alone, positions from 0) and test
    # whether the packed path is what makes distal/fwd score 0.25 vs the offline
    # kernel's 0.158.
    _mps = int(os.getenv("MAX_PACKED_SEGMENTS", "64"))
    if _mps != 64:
        _orig_worker_init = eval_harness._LmEvalHarnessWorker.__init__

        def _patched_worker_init(self, *a, **kw):
            kw["max_packed_segments"] = _mps
            print(f"[exp257] OVERRIDE max_packed_segments -> {_mps}", flush=True)
            return _orig_worker_init(self, *a, **kw)

        eval_harness._LmEvalHarnessWorker.__init__ = _patched_worker_init

    _tag = os.getenv("STEP_TAG", "")
    run_name = (
        WANDB_RUN_NAME
        + (f"-mps{_mps}" if _mps != 64 else "")
        + (f"-{_tag}" if _tag else "")
    )
    seq_len = dna_effective_seq_len(DNA_BASE_SEQ_LEN, TOKENIZER)
    eval_config = eval_harness.EvalHarnessMainConfig(
        eval_harness=eval_harness.LmEvalHarnessConfig(
            task_spec=convert_to_levanter_task_config([MENDELIAN_TRAITS_255]),
            log_samples=False,
        ),
        tokenizer=TOKENIZER,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_is_hf=False,  # native levanter checkpoint = exact training state
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=list(WANDB_TAGS),
                name=run_name,
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            per_device_eval_parallelism=64,
        ),
        model=_build_model_config(seq_len),
    )
    eval_harness.run_eval_harness_main(eval_config)


def main() -> None:
    # Propagate the packing knob to the TPU pod (the iris-job `-e` only sets it
    # on the launcher). Also fold it into the step name so the content-addressed
    # marin executor doesn't treat a different packing mode as the same
    # (already-done) step and skip re-execution.
    mps = os.getenv("MAX_PACKED_SEGMENTS", "64")
    step_tag = os.getenv("STEP_TAG", "")
    env_vars: dict[str, str] = {
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "UV_LOCK_TIMEOUT": "7200",
        "MAX_PACKED_SEGMENTS": mps,
        "STEP_TAG": step_tag,
        "DUMP_GCS": os.getenv("DUMP_GCS", ""),
    }
    if "WANDB_API_KEY" in os.environ:
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    # Fold packing mode + tag into the step name so the content-addressed marin
    # executor re-runs (rather than skipping a same-named, already-done step).
    name_suffix = ("" if mps == "64" else f"-mps{mps}") + (f"-{step_tag}" if step_tag else "")
    step = ExecutorStep(
        name=f"evaluation/lm_evaluation_harness_levanter/exp257_{MODEL_NAME}{name_suffix}",
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
            "issue #257 — reproduce online mendelian eval on exp232 ccre "
            "step-4999 (native ckpt) via levanter lm_eval; dump distal per-variant LLRs."
        ),
    )


if __name__ == "__main__":
    main()
