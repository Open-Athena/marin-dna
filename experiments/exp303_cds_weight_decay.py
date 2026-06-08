# Copyright The Marin Authors / Bolinas Authors
# SPDX-License-Identifier: Apache-2.0

"""CDS Qwen3-0.25B specialist — weight-decay sweep (issue #303).

Follow-up to #232. In exp232 the ``v4_cds`` 0.25B specialist's missense Mendelian
VEP AUPRC peaks ~0.33 around step 3500-4000 and then declines to 0.309 by step
5000 (#232 §1, offline ``minus_llr_avg``) — the only arm whose own-region metric
turns *down* late in training. This experiment tests one intervention: raise the
optimizer weight decay on the ``v4_cds`` arm only, holding every other
hyperparameter byte-identical to exp232, and re-read the missense trajectory
offline. Two arms: ``WEIGHT_DECAY`` in {0.3, 0.5} vs the exp232 default of 0.1.

This is a clone of exp232 / exp255 that sweeps a single axis — ``WEIGHT_DECAY``
(env-overridable; baked into the run name + tags so each value lands a distinct
output path and wandb run) — and otherwise changes nothing. The training data is
the FAMILY-projection ``bolinas-dna/zoonomia-v1-v4_cds`` (same as exp232, NOT the
-order cohort of exp255), so the only difference vs exp232's cds arm is WD. With
levanter's fixed seed and identical data, the WD arms share exp232's step-0 init
and data order, diverging only through the WD term. exp232's ``v4_cds`` @ WD=0.1
is the matched baseline — no re-run.

Why WD is the clean knob: exp232's other optimizer hparams come from marin's
``CompletedAdamHHeuristic``, which resolves lr / beta1 / beta2 / epsilon /
max_grad_norm for (B, T) but does NOT tune weight decay (a fixed 0.1 default
carried from exp135/exp160). Sweeping WD perturbs none of the heuristic-resolved
values, so "keep everything else the same" is exact.

Geometry, optimizer (besides WD), compute, eval/checkpoint cadence, the 5
``val_*`` LL-gap recipes, and the ``mendelian_traits_255`` AUPRC eval are all
inherited verbatim from exp232/exp255 — see those scripts for the per-block
rationale; only the WD-axis comments are added here. The offline mendelian
comparison (BOS-faithful ``minus_llr_avg``) is the headline read; the online
in-training AUPRC is a live health check only (not BOS-faithful, #257).

HF checkpoint saving is enabled by setting ``hf_save_steps`` to match the eval
cadence (every 500 steps). ``hf_save_path`` is auto-set by marin's
``_update_config_to_use_out_path`` to ``<output_path>/hf`` whenever
``TrainLmOnPodConfig.output_path`` is set, so we don't pass it explicitly.
``hf_save_dtype`` stays None, preserving the param dtype (fp32 under our
``jmp.get_policy("p=f32,c=bfloat16")`` policy) — losslessly downstream-loadable at
bf16 via ``torch_dtype=torch.bfloat16``. Mechanism inherited from exp187/exp232
(verified end-to-end in exp187's smoke test).

Hardware: ``v6e-4`` + ``PDP=1024`` in ``us-east5`` is the cost-optimal target
(script defaults, from exp255 / PR #250). v6e-4 is 4 chips at ~2× the MFU of
v6e-8 → ~half the chip-hours for similar wall-clock, and v6e is abundant in
``us-east5-b`` (data-local to the us-east5 cache) while v5p stays capacity-
starved. ``PDP`` is the per-chip microbatch: on the 4-chip v6e-4 slice the full
per-chip batch is 8192/4 = 2048, which OOMs v6e's ~31 GB HBM, so ``PDP=1024``
grad-accumulates 2 microbatches of 1024/chip (~24 GB, fits) with the **effective
batch unchanged at 8192** (mathematically identical step). ``PDP=1024`` is a no-op
on any 8-chip slice, so it's safe on every 4-8-chip pool — a re-route needs only
``-e TPU_TYPE``. v6e is preemptible → babysit/relaunch per PR #250 (resume is
free: neither TPU type nor PDP is in the output-config hash).

Download/tokenize pattern: option 1 (``HfTokenizeConfig(id=<hf-name>)``). The cds
training partition and the 5 val recipes all reuse exp232's tokenizations
(identical cache names → cache hit); nothing tokenizes anew. WD does not affect
tokenization, so both WD arms share the one cds training cache.

Launch from a CPU box (``uv sync --extra marin --extra tpu`` first; ``gcloud``
authed), one job per WD arm — family cds data, v6e-4:

    uv run iris --cluster=marin job run \\
        --no-wait --user gonzalo --job-name exp303-cds-wd0p3-v6e4 \\
        --cpu 1 --memory 2g --extra marin --region us-east5 \\
        -e WANDB_API_KEY "$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')" \\
        -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \\
        -e WEIGHT_DECAY 0.3 -e TPU_TYPE v6e-4 -e PDP 1024 \\
        -- python experiments/exp303_cds_weight_decay.py

  (second arm: ``--job-name exp303-cds-wd0p5-v6e4 -e WEIGHT_DECAY 0.5``.)

Reference templates:
  * Direct parents — ``experiments/exp232_per_region.py`` (the WD=0.1 baseline +
    family cds data) and ``experiments/exp255_per_region_order.py`` (the v6e-4 /
    PDP / TPU_RAM hardware defaults and env-override scaffolding).
  * Tracking issue — #303 (does higher WD counter the late missense decline?).
  * Eval-task wiring — ``experiments/parity/exp179_eval_only.py``.
"""

import logging
import os
from datetime import timedelta
from functools import lru_cache

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.evaluation.evaluation_config import convert_to_levanter_task_config
from marin.execution import (
    ExecutorStep,
    ensure_versioned,
    executor_main,
    this_output_path,
)
from marin.execution.remote import remote
from marin.processing.tokenize import lm_mixture_data_config, tokenize
from marin.processing.tokenize.tokenize import HfTokenizeConfig
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from marin_dna.levanter.defaults import dna_effective_seq_len
from marin_dna.levanter.formats import DNALmDatasetFormat
from marin_dna.pipelines.evals.lm_eval.task_configs import MENDELIAN_TRAITS_255

# =============================================================================
# Constants — issue #303 CDS weight-decay sweep.
# =============================================================================

EXP_ISSUE = 303
VERSION = "v0.1"
TOKENIZER = "bolinas-dna/tokenizer-char-bos"
DNA_BASE_SEQ_LEN = 255  # bp (256 - 1 for BOS)

# Single arm: the v4 CDS partition on the 108-family v1 projection — exactly
# exp232's ``v4_cds`` training data (NOT exp255's -order cohort). The swept axis
# is WEIGHT_DECAY (below), not the dataset; exp232's v4_cds @ WD=0.1 is the
# matched baseline (no re-run). Keep the dict + SWEEP_DATASETS scaffolding for
# structural parity with exp232/exp255 (it defaults to the one cds entry).
TRAIN_DATASETS: dict[str, str] = {
    "v4_cds": "bolinas-dna/zoonomia-v1-v4_cds",
}

# Five region-specific validation recipes from PR #171, tokenized functional +
# nonfunctional for the LL-gap signal. Byte-identical to exp187/exp232 — the
# identical names reuse exp232's tokenization caches (no re-tokenize).
VAL_DATASETS: tuple[tuple[str, str], ...] = (
    ("val_cds", "bolinas-dna/zoonomia-v1-val_cds"),
    ("val_utr3", "bolinas-dna/zoonomia-v1-val_utr3"),
    ("val_ncrna", "bolinas-dna/zoonomia-v1-val_ncrna"),
    ("val_enhancer", "bolinas-dna/zoonomia-v1-val_enhancer"),
    ("val_tss_pc", "bolinas-dna/zoonomia-v1-val_tss_pc"),
)

# Training masks lowercase positions to 1% loss weight. Zoonomia datasets use
# ``sequence`` as the text field (vs the older ``seq`` default for genomes-v5).
TRAIN_FORMAT = DNALmDatasetFormat(text_key="sequence", lowercase_weight=0.01)

# Validation tokenization specs — the two terms of the LL gap (matches exp160 /
# exp166; deliberately skips the "default" matched-to-training variant).
VAL_SPECS: tuple[tuple[str, DNALmDatasetFormat], ...] = (
    ("functional", DNALmDatasetFormat(uppercase_weight=1.0, lowercase_weight=0.0)),
    ("nonfunctional", DNALmDatasetFormat(uppercase_weight=0.0, lowercase_weight=1.0)),
)

# Qwen3-0.25B head_dim=128 (1152 / 9 = 128). The ``~255M`` rung of exp109's
# calibrated scaling ladder; geometry from marin's
# ``CompletedAdamHHeuristic._build_model_config(1152)``: num_layers=12,
# intermediate=hidden×4, n_heads=hidden//128. Identical to exp232.
HIDDEN_DIM = 1152
INTERMEDIATE_DIM = HIDDEN_DIM * 4  # 4608
NUM_HEADS = HIDDEN_DIM // 128  # 9
NUM_LAYERS = 12

BATCH_SIZE = 8192
# TPU type is env-overridable (``TPU_TYPE``, comma-separated for fallbacks). The
# default ``v6e-4`` (exp255 / PR #250): 4 chips at ~2× the MFU of v6e-8 → ~half
# the chip-hours for similar wall-clock, and v6e is abundant in ``us-east5-b``
# (data-local to the us-east5 cache) while v5p stays capacity-starved. Re-route
# with ``-e TPU_TYPE v6e-8`` / ``v5p-8`` / ``v4-8`` (all 8-chip; launch v4-8 with
# ``--region us-central2``). Resume is free across re-routes — neither TPU type
# nor PDP is in the output-config hash (PR #250).
TPU_TYPES: tuple[str, ...] = tuple(
    t.strip() for t in os.getenv("TPU_TYPE", "v6e-4").split(",") if t.strip()
)

# Per-device microbatch (`per_device_parallelism`), env-overridable via ``PDP``;
# default 1024. On the 4-chip v6e-4 slice the full per-chip batch is 8192/4 =
# 2048, which OOMs v6e's ~31 GB HBM; ``PDP=1024`` grad-accumulates 2 microbatches
# of 1024/chip (~24 GB, fits) with the **effective batch unchanged at 8192**
# (train_batch_size fixed → mathematically identical step). PDP=1024 is a no-op on
# any 8-chip slice, so it's safe on every 4-8-chip pool — a re-route needs only
# ``-e TPU_TYPE``. The v6e OOM is activations, not weights, so the microbatch is
# the lever (gradient_checkpointing is already on by default).
PER_DEVICE_PARALLELISM: int = int(os.getenv("PDP", "1024"))

# Host RAM request for the TPU pod, env-overridable via ``TPU_RAM`` (default
# ``300g``, inherited from exp166's 1B/4B). Host RAM buffers the data loader (the
# model + optimizer states live in TPU HBM). DO NOT shrink this below the job's
# working set to ease scheduling: a 40g ask OOM-kills 0.25B training around step
# ~141 (exit 137) — the failure is MASKED on preemptible/contended pools (each
# attempt is preempted before the OOM point, so it looks like preemption churn)
# and only surfaces on a roomy pool with long windows. If 300g won't schedule
# under contention, wait for a roomier pool / off-peak rather than under-request.
# ``resources`` is not in the marin output-path hash, so this is resume-safe.
TPU_RAM: str = os.getenv("TPU_RAM", "300g")

# Weight decay — THE swept axis for #303 (env-overridable). exp232/exp255 hold it
# at the 0.1 exp135/exp160 default; here we sweep {0.3, 0.5} (one value per iris
# job, passed ``-e WEIGHT_DECAY``). It's the one optimizer hparam the AdamH
# heuristic does not resolve, so changing it leaves every other hparam exactly as
# in exp232. Baked into the run name + tags (WD_TAG) so each value lands a
# distinct output path / checkpoint dir / wandb run.
WEIGHT_DECAY: float = float(os.getenv("WEIGHT_DECAY", "0.1"))
# Filename-safe rendering, e.g. 0.3 -> "wd0p3", 0.5 -> "wd0p5", 0.1 -> "wd0p1".
WD_TAG: str = f"wd{WEIGHT_DECAY:g}".replace(".", "p")

# 5K steps × 8192 batch × 256 tokens/seq ≈ 10.5B tokens — same compute & data
# schedule as exp232's cds arm (0.71 epochs of the family v4_cds partition).
NUM_TRAIN_STEPS = 5_000

# Optimizer hparams from marin PR #5530's exp166 transferred-hparam table
# (resolved by ``CompletedAdamHHeuristic`` for B=8192, T=5.73e10). Inherited
# verbatim from exp232 — only WEIGHT_DECAY (above) is swept. The (B=8192,
# T≈1.05e10) regime caveat from exp232's Q1 carries over unchanged.
LEARNING_RATE = 0.00430097
BETA1 = 0.66756
BETA2 = 0.952222
EPSILON = 6.77142e-15
MAX_GRAD_NORM = 0.995188
Z_LOSS_WEIGHT = 4.312883184368223e-06
INITIALIZER_RANGE = 0.02
WARMUP_FRACTION = 0.1
DECAY_FRACTION = 0.2
LR_SCHEDULE = "linear"
MIN_LR_RATIO = 0.0

# Eval + checkpoint cadence. EVALS_PER_RUN = 10 → first eval at step 500; every
# eval step has a paired reloadable HF checkpoint for offline analysis. Identical
# to exp232 → apples-to-apples trajectory comparison against the WD=0.1 baseline.
EVALS_PER_RUN = 10
CHECKPOINTS_PER_RUN = 10
# Levanter resume-checkpoint cadence 10min (exp255): under frequent v6e
# preemption a 1h interval can lose a whole window. Resume-safe (not in the marin
# output-path hash). HF checkpoints stay every 500 steps (``hf_save_steps``).
CHECKPOINT_TIME_INTERVAL = timedelta(minutes=10)

WANDB_PROJECT = "marin"
WANDB_GROUP = f"dna-exp{EXP_ISSUE}-{VERSION}"

_EXPECTED_VOCAB_SIZE_WARNING = (
    f"Tokenizer {TOKENIZER!r} not found in _KNOWN_VOCAB_SIZES"
)
logging.getLogger("marin.processing.tokenize.data_configs").addFilter(
    lambda record: _EXPECTED_VOCAB_SIZE_WARNING not in record.getMessage()
)


# =============================================================================
# Environment overrides
# =============================================================================


def _selected_datasets() -> dict[str, str]:
    """Return the subset of TRAIN_DATASETS named in SWEEP_DATASETS (or all)."""
    raw = os.getenv("SWEEP_DATASETS")
    if not raw:
        return dict(TRAIN_DATASETS)
    requested = tuple(s.strip() for s in raw.split(","))
    invalid = [n for n in requested if n not in TRAIN_DATASETS]
    if invalid:
        raise ValueError(
            f"Invalid SWEEP_DATASETS {invalid}; available: {sorted(TRAIN_DATASETS)}"
        )
    return {n: TRAIN_DATASETS[n] for n in requested}


# =============================================================================
# Builders
# =============================================================================


@lru_cache(maxsize=1)
def _model_seq_len() -> int:
    """Model context size = base DNA seq len + special tokens (BOS)."""
    return dna_effective_seq_len(DNA_BASE_SEQ_LEN, TOKENIZER)


# Inherited verbatim from exp160 — small orchestrator footprint for the
# tokenize step (heavy work runs on zephyr workers).
_TOKENIZE_RESOURCES = ResourceConfig.with_cpu(cpu=1, ram="12g", disk="10g")


def _tokenize(
    name: str, dataset: str, dataset_format: DNALmDatasetFormat
) -> ExecutorStep:
    """Tokenize one HF dataset (option-1 path, same as exp160/exp166)."""
    config = HfTokenizeConfig(
        id=dataset,
        cache_path=this_output_path(),
        tokenizer=ensure_versioned(TOKENIZER),
        format=dataset_format,
    )
    return ExecutorStep(
        name=os.path.join("tokenized", name),
        description=f"Tokenize {dataset!r} with the {TOKENIZER} tokenizer.",
        fn=remote(
            tokenize,
            resources=_TOKENIZE_RESOURCES,
            # bolinas-dna doesn't define a ``cpu`` extra; route to ``marin``
            # which transitively installs marin + jax + jmp + tokenizers.
            pip_dependency_groups=["marin"],
            env_vars={
                "TRANSFORMERS_NO_TORCH": "1",
                "TRANSFORMERS_NO_TORCHVISION": "1",
                "USE_TORCH": "0",
                "TORCH_DISABLE_GLOBAL_DEPS": "1",
                # huggingface_hub's default read_timeout=10s is too short for
                # bigger parquet manifests; bump to 120s.
                "HF_HUB_DOWNLOAD_TIMEOUT": "120",
                # Many concurrent zephyr workers share a uv cache; first build
                # serializes lm-eval (URL dep). Default 300s isn't enough.
                "UV_LOCK_TIMEOUT": "7200",
            },
        ),
        config=config,
    )


def _build_data_mixture(strategy: str, dataset: str) -> LmDataConfig:
    """One training component + cross-product of validation recipes × specs.

    The training component is keyed on ``strategy`` (``v4_cds``), so its tokenize
    cache name matches exp232's family cds cache (reused — WD doesn't affect
    tokenization); the val components keep the exp232 names so their caches are
    reused too. Nothing tokenizes anew.
    """
    components: dict[str, ExecutorStep] = {
        strategy: _tokenize(
            f"marin_dna-zoonomia-v1-{strategy}-char-bos", dataset, TRAIN_FORMAT
        ),
    }
    for region, val_dataset in VAL_DATASETS:
        for suffix, fmt in VAL_SPECS:
            key = f"{region}_{suffix}"
            components[key] = _tokenize(
                f"marin_dna-zoonomia-v1-{key}-char-bos", val_dataset, fmt
            )
    return lm_mixture_data_config(
        components=components,
        weights={strategy: 1.0},
    )


def _build_model_config() -> Qwen3Config:
    """Qwen3-0.25B head_dim=128 (hidden=1152, layers=12, heads=9). Identical to
    exp232 — geometry from marin's ``CompletedAdamHHeuristic._build_model_config(1152)``.
    """
    return Qwen3Config(
        hidden_dim=HIDDEN_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
        max_seq_len=_model_seq_len(),
        rope=Llama3RotaryEmbeddingsConfig(),
        initializer_range=INITIALIZER_RANGE,
    )


def _build_optimizer() -> AdamConfig:
    """AdamW config — identical to exp232 except ``weight_decay`` (the swept axis)."""
    return AdamConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        max_grad_norm=MAX_GRAD_NORM,
        warmup=WARMUP_FRACTION,
        decay=DECAY_FRACTION,
        lr_schedule=LR_SCHEDULE,
        min_lr_ratio=MIN_LR_RATIO,
    )


def _eval_harness_config() -> LmEvalHarnessConfig:
    """Wire the mendelian eval (AUPRC + Global + Macro Avg + FWD/RC strand
    averaging; online metric migrated from PA in #226). Same shape as exp179_eval_only.py."""
    return LmEvalHarnessConfig(
        task_spec=convert_to_levanter_task_config([MENDELIAN_TRAITS_255]),
    )


def _checkpointer(num_train_steps: int) -> CheckpointerConfig:
    return CheckpointerConfig(
        save_interval=CHECKPOINT_TIME_INTERVAL,
        keep=[dict(every=max(1, num_train_steps // CHECKPOINTS_PER_RUN))],
    )


def _hf_save_steps(num_train_steps: int) -> int:
    """Match HF-checkpoint cadence to eval cadence — every eval has a paired
    reloadable HF artifact under ``<this_output_path>/hf/step-<N>/``."""
    return max(1, num_train_steps // EVALS_PER_RUN)


def _run_train_with_marin_dna_imports(pod_config: TrainLmOnPodConfig) -> None:
    """Wrap ``run_levanter_train_lm`` with the in-function marin_dna imports the
    TPU worker needs.

    Iris's ``Entrypoint.from_callable`` cloudpickles ``__main__`` by-value,
    capturing function bytecode but NOT re-importing modules on the worker.
    Module-top imports in this script (``marin_dna.levanter.formats``,
    ``marin_dna.pipelines.evals.lm_eval.task_configs``) never fire on the TPU
    pod, so the side-effecting registrations (``@register_subclass("dna")``
    for ``DNALmDatasetFormat``; the lm-eval task-manager patch) never
    install. Trainer init then fails with
    ``ValueError: Unknown format DNALmDatasetFormat(...)`` when deserializing
    the data config (smoke5 confirmed this).

    Exp179_eval_only.py:91-96 has the same pattern for the eval path. The
    fix is the same here: bake the imports into the function body that the
    TPU worker actually runs.
    """
    import marin_dna.levanter.formats  # noqa: F401  # registers "dna" LmDatasetFormat
    import marin_dna.pipelines.evals.lm_eval  # noqa: F401  # patches lm_eval TaskManager

    run_levanter_train_lm(pod_config)


def _train_remote_env_vars() -> dict[str, str]:
    """Env vars baked into the train ``remote()`` call.

    Iris workers don't inherit ``-e`` flags from the launcher's ``iris job run``
    command — the orchestrator passes them to its own process tree, but child
    tasks (the TPU pod running ``run_levanter_train_lm``) get a fresh env.
    Per ``experiments/README.md`` and the working pattern in
    ``experiments/parity/exp179_eval_only.py``, capture the key vars from the
    launcher env here and bake them into the remote spec.

    Missing ``WANDB_API_KEY`` here was the cause of ``/gonzalo/exp187-smoke2``
    failing with ``wandb.UsageError: No API key configured`` after ~45 min of
    successful tokenize work.
    """
    env: dict[str, str] = {
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "UV_LOCK_TIMEOUT": "7200",
    }
    # Reject empty WANDB_API_KEY — the launch command's shell ordering can
    # silently substitute an empty value (``VAR=$(...) cmd -e VAR "$VAR"``
    # resolves ``$VAR`` *before* the prefix assignment). Wandb's UsageError
    # in that case is delayed until trainer init on the TPU pod, after ~45
    # min of tokenize work has succeeded — costly. Fail loud at launch time
    # instead.
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        env["WANDB_API_KEY"] = wandb_key
    else:
        raise RuntimeError(
            "WANDB_API_KEY is not set in the launcher's environment "
            "(or is empty). The recommended launch pattern is to use "
            "inline ``$(...)`` substitution in ``-e WANDB_API_KEY ...`` "
            "per experiments/README.md — not a ``VAR=... cmd`` prefix, "
            "which has a shell-ordering bug."
        )
    return env


def _build_train_step(strategy: str, dataset: str) -> ExecutorStep:
    steps_per_eval = max(1, NUM_TRAIN_STEPS // EVALS_PER_RUN)
    # WD_TAG makes each swept value structurally distinct: the run name → the
    # ExecutorStep name → the checkpoint output path, AND the wandb run id (which
    # derives from replicate_path=output_path), all carry the WD value. Without it
    # the two arms would collide on display name even if the optimizer-config hash
    # differed.
    run_name = f"dna-exp{EXP_ISSUE}-zoonomia-v1-0p25b-{strategy}-{WD_TAG}-{VERSION}"
    tags = (
        "dna",
        "marin_dna",
        f"exp{EXP_ISSUE}",
        "per-region",
        "zoonomia_v1_v4",
        "cohort=family",  # the 108-family projection (exp232 baseline), not -order
        VERSION,
        f"region={strategy}",
        "scale=0.25b",
        f"wd={WEIGHT_DECAY:g}",  # the swept axis — filterable in wandb
        f"bs={BATCH_SIZE}",
        f"steps={NUM_TRAIN_STEPS}",
    )

    inner = TrainLmConfig(
        data=_build_data_mixture(strategy, dataset),
        model=_build_model_config(),
        train_seq_len=_model_seq_len(),
        z_loss_weight=Z_LOSS_WEIGHT,
        optimizer=_build_optimizer(),
        eval_harness=_eval_harness_config(),
        eval_harness_steps=steps_per_eval,
        # HF checkpoint cadence — one save per eval step. ``hf_save_path`` is
        # auto-set by marin's ``_update_config_to_use_out_path`` to
        # ``<output_path>/hf`` (since ``output_path`` is set on the pod config
        # below); we only override the default ``hf_save_steps=10_000``.
        hf_save_steps=_hf_save_steps(NUM_TRAIN_STEPS),
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=list(tags),
                group=WANDB_GROUP,
                name=run_name,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            per_device_parallelism=PER_DEVICE_PARALLELISM,
            num_train_steps=NUM_TRAIN_STEPS,
            steps_per_eval=steps_per_eval,
            checkpointer=_checkpointer(NUM_TRAIN_STEPS),
            mesh=MeshConfig(axes={"replica": 1, "data": -1, "model": 1}),
            allow_nondivisible_batch_size=True,
        ),
    )
    pod_config = TrainLmOnPodConfig(
        train_config=inner,
        resources=ResourceConfig.with_tpu(TPU_TYPES, ram=TPU_RAM),
        output_path=this_output_path(),
    )
    # The remote() call here IS the TPU worker — Fray's RemoteCallable submits
    # the job with these resources verbatim (exp179_eval_only.py mirrors this).
    # pip_dependency_groups explicitly installs ``tpu`` so libtpu is present —
    # without it JAX silently falls back to CPU ("No accelerator found").
    return ExecutorStep(
        name=os.path.join("checkpoints", run_name),
        fn=remote(
            _run_train_with_marin_dna_imports,
            resources=ResourceConfig.with_tpu(TPU_TYPES, ram=TPU_RAM),
            pip_dependency_groups=["marin", "tpu"],
            env_vars=_train_remote_env_vars(),
        ),
        config=pod_config,
    )


def main() -> None:
    selected = _selected_datasets()
    steps = [
        _build_train_step(strategy, dataset) for strategy, dataset in selected.items()
    ]
    executor_main(
        steps=steps,
        description=(
            f"DNA Bolinas exp{EXP_ISSUE} CDS weight-decay sweep — "
            f"WD={WEIGHT_DECAY:g}, {len(selected)} arm(s) {VERSION}"
        ),
    )


if __name__ == "__main__":
    main()
