# Copyright The Marin Authors / Bolinas Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-region Qwen3-0.25B gLMs on the 6 zoonomia-v1-v4 subsets (issue #232).

Trains one Qwen3-0.25B model per region partition of ``bolinas-dna/zoonomia-v1-v1``
at matched data (5K steps × 8192 batch on v5p-8) and asks whether each region's
specialist wins its own region's VEP under the v4 labeling. Six arms total;
``SWEEP_DATASETS`` selects which arms run in a given iris job. Follow-up to #187
(which ran this at 1B on the v3 partitions); 0.25B here is purely the cheap
iteration size, not a scale probe. exp187's 1B/v3 run is the qualitative
cross-reference; this experiment compares the six v4 regional specialists against
each other only.

In-training validation:
  * 5 region-specific ``zoonomia-v1-val_*`` recipes (LL gap) from PR #171,
    each tokenized functional + nonfunctional. Drops ``val_utr5`` / ``val_promoter``
    in favor of the gene-centric ``val_tss_pc``.
  * ``mendelian_traits_255`` lm-eval task (PR #186) — per consequence subset +
    Global + Macro Avg + FWD/RC strand averaging. The online metric migrated to
    AUPRC + per-variant FWD/RC averaging in #226 (was a broken PairwiseAccuracy
    that read flat-at-chance in exp187); the in-training headline scalar is now
    ``_global_/avg/auprc``, and the offline evals_v2 leaderboard uses
    ``_macro_avg_/avg/auprc`` (per #161).

HF checkpoint saving is enabled by setting ``hf_save_steps`` to match the
eval cadence (every 500 steps). ``hf_save_path`` is auto-set by marin's
``_update_config_to_use_out_path`` to ``<output_path>/hf`` whenever
``TrainLmOnPodConfig.output_path`` is set, so we don't pass it explicitly.
``hf_save_dtype`` stays at the default ``None``, preserving the param dtype
(fp32 under our ``jmp.get_policy("p=f32,c=bfloat16")`` policy) — losslessly
downstream-loadable at bf16 via ``torch_dtype=torch.bfloat16``.

This is new vs exp160_parity, whose HF saves never landed because it left
``hf_save_steps`` at the default 10_000 (= the final step, which doesn't
trigger the callback because of the skip-step-0 guard). The mechanism is
inherited from exp187, which verified it end-to-end in its smoke test.

Hardware: ``v5p-8`` in ``us-east5-a`` (matches #166's actual stack per marin
PR #5530's current code, despite that PR body's stale ``v6e-4``). At 0.25B each
arm is ~4× cheaper in FLOPs than exp187's 1B; ``v6e-4`` would also fit now (it
OOM'd the 1B/4B in exp166) but ``v5p-8`` is the de-risked default.

Download/tokenize pattern: option 1 (``HfTokenizeConfig(id=<hf-name>)``), same
as exp135 / exp160 / exp166 / exp187. The 5 val recipes reuse exp187's
tokenizations (identical names); only the 6 v4 training partitions tokenize anew.

Launch from a CPU box with an iris tunnel open (see ``experiments/README.md``):

    SWEEP_DATASETS=v4_cds uv run iris --cluster=marin job run \\
        --no-wait --user gonzalo --job-name exp232-v4-cds \\
        --cpu 1 --memory 2g --extra marin --region us-east5 \\
        -e WANDB_API_KEY "$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')" \\
        -e HF_HUB_DOWNLOAD_TIMEOUT 120 -e UV_LOCK_TIMEOUT 7200 \\
        -e SWEEP_DATASETS v4_cds \\
        -- python experiments/exp232_per_region.py

Reference templates:
  * Direct parent — ``experiments/exp187_per_region.py`` (this is a
    clone with v3→v4 datasets and the 1B→0.25B geometry swap).
  * Eval-task wiring — ``experiments/parity/exp179_eval_only.py``.
  * Geometry ladder — marin ``exp109`` (``SCALING_HIDDEN_SIZES``) /
    ``exp166_zoonomia_1ep_scaling.py`` (``CompletedAdamHHeuristic``).
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
# Constants — issue #232 per-region experiment.
# =============================================================================

EXP_ISSUE = 232
VERSION = "v0.1"
TOKENIZER = "bolinas-dna/tokenizer-char-bos"
DNA_BASE_SEQ_LEN = 255  # bp (256 - 1 for BOS)

# Six v4 region partitions of bolinas-dna/zoonomia-v1-v1 (PR #228 / #227).
# bp-priority + window-majority labeling, PC-only TSS, ccre_flank=0. One arm per
# entry. SWEEP_DATASETS selects the subset to launch per iris job.
TRAIN_DATASETS: dict[str, str] = {
    "v4_cds": "bolinas-dna/zoonomia-v1-v4_cds",
    "v4_utr3": "bolinas-dna/zoonomia-v1-v4_utr3",
    "v4_ncrna_exon": "bolinas-dna/zoonomia-v1-v4_ncrna_exon",
    "v4_tss_region_and_utr5": "bolinas-dna/zoonomia-v1-v4_tss_region_and_utr5",
    "v4_ccre_non_promoter": "bolinas-dna/zoonomia-v1-v4_ccre_non_promoter",
    "v4_bg": "bolinas-dna/zoonomia-v1-v4_bg",
}

# Five region-specific validation recipes from PR #171, tokenized functional +
# nonfunctional for the LL-gap signal. Drops val_utr5 / val_promoter (both
# subsumed by the gene-centric ±255 bp val_tss_pc); keeps val_enhancer
# (chromatin-side, not subsumed).
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
# calibrated scaling ladder (1152→255M, 1408→476M, 1920→1.12B); the 1B that
# exp187 ran sits two rungs up at hidden=1920. Geometry from marin's
# ``CompletedAdamHHeuristic._build_model_config(1152)``: num_layers=12,
# intermediate=hidden×4, n_heads=hidden//128.
HIDDEN_DIM = 1152
INTERMEDIATE_DIM = HIDDEN_DIM * 4  # 4608
NUM_HEADS = HIDDEN_DIM // 128  # 9
NUM_LAYERS = 12

BATCH_SIZE = 8192
# TPU type is env-overridable (``TPU_TYPE``, comma-separated for flexible
# alternatives) so we can re-route between pools without editing the script —
# e.g. ``TPU_TYPE=v4-8`` to escape a ``v5p-us-east5`` provisioning outage. v4-8
# and v5p-8 share the same 8-chip / 2-VM topology, so the mesh + ram request and
# a saved checkpoint carry over unchanged (v4 lives in us-central2-b, so launch
# such arms with ``--region us-central2``; data stays cross-region in us-east5).
TPU_TYPES: tuple[str, ...] = tuple(
    t.strip() for t in os.getenv("TPU_TYPE", "v5p-8").split(",") if t.strip()
)

# Per-device microbatch (`per_device_parallelism`), env-overridable via ``PDP``.
# -1 (default) = full per-chip batch in one microbatch (no grad accum). levanter
# already runs ``gradient_checkpointing=True`` by default, yet on v6e (~31 GB HBM)
# the full per-chip batch of 8192/4 = 2048 still OOMs at ~43 GB. Set ``PDP=1024``
# to grad-accumulate (2 microbatches of 1024/chip on a 4-chip slice) → same
# activation memory as the v6e-8 run (~24 GB, fits) while the **effective batch
# stays 8192** (train_batch_size unchanged → mathematically identical step).
PER_DEVICE_PARALLELISM: int = int(os.getenv("PDP", "-1"))

# 5K steps × 8192 batch × 256 tokens/seq ≈ 10.5B tokens per arm (same data
# schedule as exp187). Asymmetric epoch counts across arms are by design (we
# hold compute fixed, not data); v4 post-RC rows shift the counts vs v3:
# v4_ccre_non_promoter 0.47 ep, v4_cds 0.71, v4_bg 1.08, v4_ncrna_exon 2.51,
# v4_utr3 3.25, v4_tss_region_and_utr5 3.63.
NUM_TRAIN_STEPS = 5_000

# Optimizer hparams from marin PR #5530's exp166 transferred-hparam table
# (resolved by ``CompletedAdamHHeuristic`` for B=8192, T=5.73e10 — exp166's
# full-epoch horizon).
#
# **CAVEAT** — our T = 5_000 × 8192 × 256 ≈ 1.05e10, ~5× smaller than exp166's
# horizon. The DNA-calibrated heuristic would resolve slightly different lr /
# beta2 / epsilon for our regime. We use exp166's values here as a deliberate
# starting point; the AdamH heuristic is size-independent (keyed on (B, T), both
# unchanged at 0.25B), so per-region runs share exp187's / exp166's optimizer
# config — only the model is smaller. If reviewers want regime-correct hparams,
# re-resolve via marin's CompletedAdamHHeuristic for (B=8192, T=1.05e10) and pin
# the resolved values here (this applies equally to the 1B). Tracked as Q1.
LEARNING_RATE = 0.00430097
BETA1 = 0.66756
BETA2 = 0.952222
EPSILON = 6.77142e-15
MAX_GRAD_NORM = 0.995188
Z_LOSS_WEIGHT = 4.312883184368223e-06
INITIALIZER_RANGE = 0.02
WEIGHT_DECAY = 0.1  # exp160 / exp135 default; the heuristic doesn't tune WD
WARMUP_FRACTION = 0.1
DECAY_FRACTION = 0.2
LR_SCHEDULE = "linear"
MIN_LR_RATIO = 0.0

# Eval + checkpoint cadence. EVALS_PER_RUN = 10 → first eval at step 500. We
# match HF-checkpoint cadence to eval cadence so every in-training eval has a
# reloadable HF artifact for offline analysis (e.g. re-running mendelian eval
# against a specific step).
EVALS_PER_RUN = 10
CHECKPOINTS_PER_RUN = 10
CHECKPOINT_TIME_INTERVAL = timedelta(hours=1)

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
    """One training component + cross-product of validation recipes × specs."""
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
    """Qwen3-0.25B head_dim=128 (hidden=1152, layers=12, heads=9).

    The ~255M rung of exp109's calibrated ladder; geometry from marin's
    ``CompletedAdamHHeuristic._build_model_config(1152)`` (num_layers=12,
    intermediate=4608, n_heads=9).
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
    Per ``experiments/README.md:59-67`` and the working pattern in
    ``experiments/parity/exp179_eval_only.py:124-130``, capture the key vars
    from the launcher env here and bake them into the remote spec.

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
    run_name = f"dna-exp{EXP_ISSUE}-zoonomia-v1-0p25b-{strategy}-{VERSION}"
    # Optional suffix for an intentional clean re-train. Suffixing run_name yields
    # a fresh output path (=> fresh checkpoint dir => trains from scratch) AND a
    # fresh wandb run-id — the id derives from replicate_path=output_path (below),
    # NOT the display name, so this is the only way to escape a prior run's wandb
    # step-barrier (levanter "cowardly refuses" to log a step below a resumed
    # run's max; a from-scratch retrain at the old path collides with a crashed
    # run's history). Empty by default → no effect on normal runs.
    _run_suffix = os.environ.get("RUN_NAME_SUFFIX", "")
    if _run_suffix:
        run_name = f"{run_name}-{_run_suffix}"
    tags = (
        "dna",
        "marin_dna",
        f"exp{EXP_ISSUE}",
        "per-region",
        "zoonomia_v1_v4",
        VERSION,
        f"region={strategy}",
        "scale=0.25b",
        f"bs={BATCH_SIZE}",
        f"steps={NUM_TRAIN_STEPS}",
    )

    # iris's Entrypoint.from_callable cloudpickles __main__ by-value, so a
    # module-top import of marin_dna.pipelines.evals.lm_eval doesn't fire on the
    # TPU pod. Per the eval-task wiring pattern in exp179_eval_only.py:91-96,
    # the in-function noqa import on the train worker happens inside
    # marin.training.training.run_levanter_train_lm — see that function for the
    # lm-eval task-manager monkeypatch's installation. We don't need to do
    # anything special here; the convert_to_levanter_task_config([MENDELIAN_TRAITS_255])
    # call serializes the task name + class path, which run_levanter_train_lm
    # resolves on the worker after it imports our task module.

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
        # below); we only need to override the default ``hf_save_steps=10_000``.
        # ``hf_save_dtype`` stays None → preserves param dtype (fp32 under our
        # ``p=f32,c=bfloat16`` jmp policy); downstream consumers can cast at
        # load time via ``from_pretrained(..., torch_dtype=torch.bfloat16)``.
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
        resources=ResourceConfig.with_tpu(TPU_TYPES, ram="300g"),
        output_path=this_output_path(),
    )
    # The remote() call here IS the TPU worker — Fray's RemoteCallable submits
    # the job with these resources verbatim. exp179_eval_only.py:140-145 mirrors
    # this pattern; the old exp160_parity / exp166 pattern of CPU-orchestrator +
    # TPU-pod-config relies on marin executor magic that no longer fires after
    # the PR #182 / #186 rebase (smoke4 ran on a CPU worker and JAX raised
    # "No accelerator found" at trainer.initialize). pip_dependency_groups
    # explicitly installs ``tpu`` so libtpu is present — without it JAX
    # silently falls back to CPU and we get the same error.
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
    selected = _selected_datasets()
    steps = [
        _build_train_step(strategy, dataset) for strategy, dataset in selected.items()
    ]
    executor_main(
        steps=steps,
        description=(
            f"DNA Bolinas exp{EXP_ISSUE} per-region Qwen3-0.25B sweep — "
            f"{len(selected)}/{len(TRAIN_DATASETS)} arms {VERSION}"
        ),
    )


if __name__ == "__main__":
    main()
