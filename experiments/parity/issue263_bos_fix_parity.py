# Copyright The Marin Authors / MarinDNA Authors
# SPDX-License-Identifier: Apache-2.0

"""Online↔offline parity check for the #263 [BOS] fix (validates #266).

The online (in-training levanter ``lm_eval``) VEP eval used to forward each
variant sequence WITHOUT the leading ``[BOS]`` the DNA gLMs train with, because
levanter's ``eval_harness._encode_batch`` hardcodes ``add_special_tokens=False``.
That dropped BOS was the root cause of #257: on exp232 ``v4_ccre_non_promoter``
step-4999, ``mendelian distal/fwd`` AUPRC read **0.250** online (no-BOS, OOD
artifact) vs **0.158** offline (with-BOS, faithful). PR #266 adds
``_install_bos_fix()`` to the lm_eval package ``__init__`` (prepends the
tokenizer's BOS id at ``_encode_batch``).

This runs the *real* levanter ``lm_eval`` path on that exact native checkpoint,
**with the fix active and packing disabled**, and asserts the online
``distal/fwd`` AUPRC now lands in the with-BOS regime (≈0.158), refuting the
no-BOS 0.250. It is the end-to-end, on-iris counterpart to the unit tests in
``tests/pipelines/evals/test_lm_eval_patch.py`` and the local
``scripts/issue257/bos_auprc.py``.

Two guards run on the TPU pod (fail-fast, visible in the iris logs):
  1. ``eval_harness._marin_dna_bos_patched`` — confirms the shipped ``marin_dna``
     actually carries the fix (catches a stale-workspace deploy).
  2. ``distal/fwd`` AUPRC ∈ [0.13, 0.19] — the with-BOS regime, excluding 0.250.

Packing is disabled (``max_packed_segments=1``): each sequence is forwarded alone
with a plain causal mask. No env knobs — this check is deliberately fixed.

Launch from a CPU box (this fix-branch worktree) with an iris tunnel open
(``--cluster=marin`` auto-tunnels); ``create_environment`` ships the local
workspace, so the pod installs *this* branch's fixed ``marin_dna``:

    uv run --extra marin iris --cluster=marin job run \\
        --no-wait --user gonzalo --job-name issue263-bos-fix-parity \\
        --cpu 1 --memory 2g --extra marin --region us-east5 \\
        -e WANDB_API_KEY "$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')" \\
        -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \\
        -- python experiments/parity/issue263_bos_fix_parity.py
"""

import dataclasses
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

MODEL_NAME = "exp232-v4_ccre_non_promoter-step-4999"
# Native levanter checkpoint root (the exact in-training state #257 ran on).
# load_checkpoint via latest_checkpoint_path(…/checkpoints) → step-4999.
CHECKPOINT_PATH = (
    "gs://marin-us-east5/checkpoints/"
    "dna-exp232-zoonomia-v1-0p25b-v4_ccre_non_promoter-v0.1-feca83/checkpoints"
)
TOKENIZER = "bolinas-dna/tokenizer-char-bos"

# exp232 Qwen3-0.25B geometry — verbatim from exp232_per_region so the native
# checkpoint's array structure matches exactly.
DNA_BASE_SEQ_LEN = 255
HIDDEN_DIM = 1152
INTERMEDIATE_DIM = HIDDEN_DIM * 4  # 4608
NUM_HEADS = HIDDEN_DIM // 128  # 9
NUM_LAYERS = 12
INITIALIZER_RANGE = 0.02

# v6e-4 is small enough for a 0.25B eval; iris schedules it wherever the pool has
# capacity (e.g. us-east1-d — a cross-region read of the us-east5 checkpoint,
# fine for a one-off). Env-overridable to re-route if contended (check
# `iris cluster status` first).
TPU_TYPES: tuple[str, ...] = tuple(
    t.strip() for t in os.getenv("TPU_TYPE", "v6e-4").split(",") if t.strip()
)
_EVAL_DEPENDENCY_GROUPS = ["marin", "tpu"]

# Parity band: offline (with-BOS) distal/fwd = 0.1589; the pre-fix online no-BOS
# value was 0.2501. A pass means we're clearly in the with-BOS regime.
WITH_BOS_REFERENCE = 0.1589
NO_BOS_REFERENCE = 0.2501
PARITY_LOW, PARITY_HIGH = 0.13, 0.19

WANDB_PROJECT = "marin"
WANDB_RUN_NAME = "dna-exp232-ccre-step4999-bos-fix-parity-issue263"
WANDB_TAGS = ("dna", "exp232", "issue263", "bos-fix", "parity")

# Captured on the pod by the tracker-log intercept (module global so the
# post-run parity assertion can read it).
_CAPTURED: dict[str, float] = {}


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


def _install_distal_fwd_capture() -> None:
    """Capture the ``distal/fwd`` AUPRC cell by intercepting the levanter tracker
    log payload. The aggregator pushes per-subset cells via
    ``levanter.tracker.log({f"{prefix}/distal/fwd/auprc": value, ...})``; patching
    that (resolved at call time) is robust to the ``dna_vep_llr_eval``
    module-identity issue a direct aggregator patch hits under lm_eval's
    top-level ``!function`` import."""
    import levanter.tracker as lt

    orig_log = lt.log

    def patched_log(metrics, *args, **kwargs):
        try:
            for k, v in dict(metrics).items():
                if k.endswith("distal/fwd/auprc"):
                    _CAPTURED["distal_fwd_auprc"] = float(v)
                    print(f"[parity] captured {k} = {float(v):.4f}", flush=True)
        except Exception:  # never let capture break the real logging
            pass
        return orig_log(metrics, *args, **kwargs)

    lt.log = patched_log


def _force_no_packing() -> None:
    """Disable sequence packing (``max_packed_segments=1``): each sequence is
    forwarded alone with a plain causal mask."""
    orig_init = eval_harness._LmEvalHarnessWorker.__init__

    def patched_init(self, *a, **kw):
        kw["max_packed_segments"] = 1
        print("[parity] max_packed_segments -> 1 (packing disabled)", flush=True)
        return orig_init(self, *a, **kw)

    eval_harness._LmEvalHarnessWorker.__init__ = patched_init


@dataclasses.dataclass(frozen=True)
class _EvalConfig:
    """ExecutorStep requires a config dataclass; eval is fully parameterized by
    module-level constants."""


def _run_eval_harness_only(_config: _EvalConfig) -> None:
    # MUST be inside the function body — iris cloudpickles __main__ by-value and
    # never re-imports this module on the worker, so the package import (which
    # installs the BOS fix) and our patches only fire if triggered here.
    import marin_dna.pipelines.evals.lm_eval  # noqa: F401  installs _install_bos_fix

    # Guard 1: the shipped marin_dna must actually carry the fix (catches a
    # stale-workspace deploy where the pod got pre-fix code).
    assert getattr(eval_harness, "_marin_dna_bos_patched", False), (
        "BOS fix (#263) not installed on the pod — the deployed marin_dna predates "
        "the fix. Launch from the fix-branch worktree so create_environment ships it."
    )
    print(
        "[parity] BOS fix present (eval_harness._marin_dna_bos_patched=True)",
        flush=True,
    )

    _force_no_packing()
    _install_distal_fwd_capture()

    seq_len = dna_effective_seq_len(DNA_BASE_SEQ_LEN, TOKENIZER)
    eval_config = eval_harness.EvalHarnessMainConfig(
        eval_harness=eval_harness.LmEvalHarnessConfig(
            task_spec=convert_to_levanter_task_config([MENDELIAN_TRAITS_255]),
            log_samples=False,
        ),
        tokenizer=TOKENIZER,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_is_hf=False,
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

    # Guard 2: parity. distal/fwd must be in the with-BOS regime, not 0.250.
    auprc = _CAPTURED.get("distal_fwd_auprc")
    assert auprc is not None, (
        "distal/fwd/auprc was not captured — the tracker-log payload key may have changed."
    )
    status = "PASS" if PARITY_LOW <= auprc <= PARITY_HIGH else "FAIL"
    print(
        f"\n[parity] ===== BOS-fix parity {status} =====\n"
        f"[parity] distal/fwd AUPRC = {auprc:.4f}\n"
        f"[parity]   with-BOS (offline, faithful): {WITH_BOS_REFERENCE}\n"
        f"[parity]   no-BOS   (pre-fix online):    {NO_BOS_REFERENCE}\n"
        f"[parity]   pass band: [{PARITY_LOW}, {PARITY_HIGH}]",
        flush=True,
    )
    assert PARITY_LOW <= auprc <= PARITY_HIGH, (
        f"BOS-fix parity FAILED: distal/fwd AUPRC {auprc:.4f} is outside the "
        f"with-BOS band [{PARITY_LOW}, {PARITY_HIGH}] (no-BOS artifact was {NO_BOS_REFERENCE})."
    )


def main() -> None:
    env_vars: dict[str, str] = {
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "UV_LOCK_TIMEOUT": "7200",
    }
    if "WANDB_API_KEY" in os.environ:
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    step = ExecutorStep(
        name=f"evaluation/lm_evaluation_harness_levanter/issue263_bos_fix_parity_{MODEL_NAME}",
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
            "issue #263 — online↔offline BOS-fix parity: run levanter lm_eval on "
            "exp232 ccre step-4999 (native ckpt) with the fix + no packing; assert "
            "distal/fwd AUPRC ≈ 0.158 (with-BOS), not 0.250 (no-BOS)."
        ),
    )


if __name__ == "__main__":
    main()
